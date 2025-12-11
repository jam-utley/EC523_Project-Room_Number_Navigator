import os
# Fix for Qt crash on some Linux systems
os.environ["QT_QPA_PLATFORM"] = "xcb"
# Unset WAYLAND_DISPLAY to force X11 backend if QT_QPA_PLATFORM isn't enough
if "WAYLAND_DISPLAY" in os.environ:
    del os.environ["WAYLAND_DISPLAY"]

import cv2
import threading
import queue
import time
import torch
import re
import psutil
import numpy as np
import random
import glob
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

# =========================
# Config
# =========================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LORA_PATH = "./qwen2vl_BU_Sign_1/checkpoint_epoch_30"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR = "./BU_Sign/test"
IMG_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

# Simulation: 3 seconds at 30 FPS on a single image
FRAMES_PER_IMAGE = 90
ZOOM_FACTOR = 1.2
OUTPUT_SIZE = (640, 480)

# Qwen-style [BOX: x1, y1, x2, y2, "label"] with coords in 0–1000
BOX_RE = re.compile(
    r"\[BOX:\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*\"([^\"]+)\"\s*\]"
)

def parse_boxes(text):
    boxes = []
    for m in BOX_RE.finditer(text):
        x1, y1, x2, y2, label = m.groups()
        boxes.append((int(x1), int(y1), int(x2), int(y2), label))
    return boxes


# =========================
# Video Simulator (single image, 3s pan/zoom)
# =========================
class VideoSimulator:
    """
    Uses ONLY the first image in TEST_DIR, creates a 3-second (FRAMES_PER_IMAGE)
    pan/zoom clip at OUTPUT_SIZE. After 3 seconds, .read() returns False.
    """
    def __init__(self, img_dir, output_size=(640, 480)):
        self.img_paths = []
        for ext in IMG_EXTENSIONS:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_paths.sort()
        print(f"Found {len(self.img_paths)} images for simulation.")

        self.output_size = output_size
        self.current_image = None
        self.frame_counter = 0
        self.finished = False

        self._load_single_image()

    def _load_single_image(self):
        if not self.img_paths:
            print("No images found, finishing.")
            self.finished = True
            return

        # Use only the FIRST image
        path = self.img_paths[0]
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read {path}")
            self.finished = True
            return

        # Resize to be at least output size, keeping zoom factor
        h, w = img.shape[:2]
        target_w, target_h = self.output_size
        scale = max(target_w / w, target_h / h) * ZOOM_FACTOR  # ensure enough room to pan
        new_w, new_h = int(w * scale), int(h * scale)
        self.current_image = cv2.resize(img, (new_w, new_h))

        # Random pan start/end (fixed for full 3 s clip)
        self.pan_start_x = random.randint(0, new_w - target_w)
        self.pan_start_y = random.randint(0, new_h - target_h)
        self.pan_end_x = random.randint(0, new_w - target_w)
        self.pan_end_y = random.randint(0, new_h - target_h)

        self.frame_counter = 0
        print(f"[VideoSimulator] Using image: {path}")
        print(f"[VideoSimulator] Scaled to: {new_w}x{new_h}")
        print(f"[VideoSimulator] Pan from ({self.pan_start_x},{self.pan_start_y}) "
              f"to ({self.pan_end_x},{self.pan_end_y}) for {FRAMES_PER_IMAGE} frames")

    def read(self):
        # Stop after 3 seconds worth of frames
        if self.finished or self.current_image is None:
            return False, None

        if self.frame_counter >= FRAMES_PER_IMAGE:
            self.finished = True
            return False, None

        # Interpolate crop position over FRAMES_PER_IMAGE frames
        t = self.frame_counter / float(FRAMES_PER_IMAGE)
        # Ease in-out
        t = t * t * (3 - 2 * t)

        target_w, target_h = self.output_size

        cur_x = int(self.pan_start_x + (self.pan_end_x - self.pan_start_x) * t)
        cur_y = int(self.pan_start_y + (self.pan_end_y - self.pan_start_y) * t)

        frame = self.current_image[cur_y:cur_y + target_h, cur_x:cur_x + target_w]

        self.frame_counter += 1
        return True, frame

    def release(self):
        pass


# =========================
# VLM Predictor
# =========================
class VLMPredictor:
    def __init__(self):
        print(f"Loading base model: {MODEL_ID}...")

        # Limit visual tokens to speed up inference a bit
        # (recommended pattern from Qwen2-VL docs)
        min_pixels = 256 * 28 * 28
        max_pixels = 1024 * 28 * 28  # smaller than the default 1280*28*28 for speed

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )

        print(f"Loading LoRA adapter from {LORA_PATH}...")
        self.model = PeftModel.from_pretrained(self.model, LORA_PATH)
        self.model.eval()
        print("Model loaded successfully!")

    def predict(self, image: Image.Image):
        # Exact prompt structure you used for training
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Detect all objects in the image. Return exactly one line with this format:\n"
                        "[DETECTIONS: [BOX: x1, y1, x2, y2, \"label\"], "
                        "[BOX: x1, y1, x2, y2, \"label\"], ...]\n"
                        "No explanations, no extra text."
                    ),
                },
            ],
        }]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(text=text, images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=96,    # keep small; you only need one line of output
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(f"[DEBUG] Raw Output: {output_text}")
        boxes_norm = parse_boxes(output_text)   # coords in 0–1000
        return boxes_norm


# =========================
# Dashboard drawing
# =========================
def draw_dashboard(frame_raw, frame_infer, fps, inference_fps):
    h, w, _ = frame_raw.shape

    dashboard_w = w * 2
    dashboard_h = h + 150
    canvas = np.zeros((dashboard_h, dashboard_w, 3), dtype=np.uint8)

    canvas[0:h, 0:w] = frame_raw
    canvas[0:h, w:w * 2] = frame_infer

    cv2.putText(canvas, "Simulated Feed (Pan/Zoom)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(canvas, "Inference Feed (Async)", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    stats_y = h + 40
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    gpu_mem = 0.0
    gpu_max = 0.0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3
        gpu_max = torch.cuda.max_memory_allocated() / 1024 ** 3

    cv2.putText(canvas, f"Display FPS: {fps:.1f}", (20, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"Model FPS:   {inference_fps:.2f}", (20, stats_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(canvas, f"CPU Usage: {cpu_percent}%", (300, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(canvas, f"RAM Usage: {ram_percent}%", (300, stats_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.putText(canvas, f"VRAM Used: {gpu_mem:.1f} GB", (600, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(canvas, f"VRAM Peak: {gpu_max:.1f} GB", (600, stats_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.putText(canvas, "Press 'q' to quit", (dashboard_w - 200, dashboard_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    return canvas


# =========================
# Main
# =========================
def main():
    try:
        predictor = VLMPredictor()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Simulated 3-second pan video from one image
    cap = VideoSimulator(TEST_DIR, output_size=OUTPUT_SIZE)

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def inference_worker():
        while not stop_event.is_set():
            try:
                frame_bgr = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            t0 = time.perf_counter()
            boxes_norm = predictor.predict(pil_img)   # coords in 0–1000
            # Scale from [0,1000] normalized coords to pixel coords for THIS frame
            W, H = pil_img.size
            scaled_boxes = []
            for (x1, y1, x2, y2, label) in boxes_norm:
                sx1 = int(x1 / 1000.0 * W)
                sy1 = int(y1 / 1000.0 * H)
                sx2 = int(x2 / 1000.0 * W)
                sy2 = int(y2 / 1000.0 * H)
                scaled_boxes.append((sx1, sy1, sx2, sy2, label))

            t1 = time.perf_counter()
            fps_local = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0

            if not result_queue.full():
                result_queue.put((scaled_boxes, fps_local, frame_bgr))
            else:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                result_queue.put((scaled_boxes, fps_local, frame_bgr))

    thread = threading.Thread(target=inference_worker, daemon=True)
    thread.start()

    print("Starting Simulated Demo... Press 'q' to quit.")

    last_boxes = []
    last_infer_fps = 0.0
    last_infer_frame = None

    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            # 3-second clip finished
            break

        # Simulate 30 FPS playback speed
        time.sleep(1 / 30.0)

        t_now = time.time()
        display_fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 0 else 0.0
        t_prev = t_now

        if frame_queue.empty():
            frame_queue.put(frame.copy())

        try:
            boxes, fps_local, infer_frame = result_queue.get_nowait()
            last_boxes = boxes
            last_infer_fps = fps_local
            last_infer_frame = infer_frame
        except queue.Empty:
            pass

        if last_infer_frame is not None:
            display_infer = last_infer_frame.copy()
        else:
            display_infer = frame.copy()

        H, W = display_infer.shape[:2]
        for (x1, y1, x2, y2, label) in last_boxes:
            # Clamp to frame
            x1_clamped = max(0, min(W, x1))
            x2_clamped = max(0, min(W, x2))
            y1_clamped = max(0, min(H, y1))
            y2_clamped = max(0, min(H, y2))

            cv2.rectangle(display_infer, (x1_clamped, y1_clamped),
                          (x2_clamped, y2_clamped), (0, 255, 0), 2)
            cv2.putText(display_infer, label, (x1_clamped, y1_clamped - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        dashboard = draw_dashboard(frame, display_infer, display_fps, last_infer_fps)
        cv2.imshow("YOLO-VLM Simulated Demo", dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
