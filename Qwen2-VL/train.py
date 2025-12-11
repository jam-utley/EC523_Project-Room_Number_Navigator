#!/usr/bin/env python
"""
YOLO-VLM (No Flash-Attn): Multi-Object Structured Output
Model: Qwen/Qwen2-VL-2B-Instruct (4-bit QLoRA, eager attention)

Output format (single line):
[BOXES: [BOX: x1, y1, x2, y2, "txt"]; [BOX: ...]; ...]

Saves eval visuals to ./qwen2vl_multiobj_mnist/eval_vis/
Prints FPS measured over the eval set.
"""

import os
# New env var (replaces deprecated CUDA_ALLOC_CONF); still safe if older PyTorch
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import math, random, re, time, json
import wandb
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageDraw

from transformers import (
    AutoModelForVision2Seq,   # correct class for Qwen2-VL
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =========================
# Config
# =========================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Data complexity knobs
CANVAS_SIZE = 224
OBJ_MIN = 2
OBJ_MAX = 6
NUM_PROB = 0.6
DIGITS_PER_NUM_MIN = 2
DIGITS_PER_NUM_MAX = 5
DIGIT_SIZE_MIN = 22
DIGIT_SIZE_MAX = 40
GROUP_GAP_SMALL = (2, 6)
PLACEMENT_RETRIES = 60

TRAIN_SAMPLES = None  # Use full dataset
VAL_SAMPLES = None    # Use full dataset
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4    # Effective batch size = 32
EPOCHS = 20             # multiple epochs as requested
LEARNING_RATE = 2e-4    # Increased for larger effective batch
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01     # Increased from 0.0 for regularization

# LoRA kept small for 16 GB
LORA_R = 16             # Increased from 8
LORA_ALPHA = 32         # Increased from 16
LORA_DROPOUT = 0.1      # Increased from 0.05

OUT_DIR = "./qwen2vl_BU_Sign_3"
VIS_DIR = os.path.join(OUT_DIR, "eval_vis")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

INSTRUCTION = (
    "Detect all objects in the image. "
    "Return exactly one line with this format:\n"
    "[DETECTIONS: [BOX: x1, y1, x2, y2, \"label\"], [BOX: x1, y1, x2, y2, \"label\"], ...]\n"
    "No explanations, no extra text."
)

def set_seed(s: int = 42):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

set_seed(SEED)

# =========================
# Utilities
# =========================

def iou(b1, b2):
    x1,y1,x2,y2 = b1; X1,Y1,X2,Y2 = b2
    ix1,iy1,ix2,iy2 = max(x1,X1),max(y1,Y1),min(x2,X2),min(y2,Y2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    a1 = max(0,x2-x1)*max(0,y2-y1)
    a2 = max(0,X2-X1)*max(0,Y2-Y1)
    return inter/(a1+a2-inter+1e-9)

def bboxes_overlap(b1, b2, iou_thresh=0.15) -> bool:
    return iou(b1, b2) > iou_thresh

def union_bbox(boxes):
    xs1 = [b[0] for b in boxes]; ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]; ys2 = [b[3] for b in boxes]
    return (min(xs1), min(ys1), max(xs2), max(ys2))

def sanitize_box(b, W, H):
    """Sort corners and clamp to image bounds; drop degenerate boxes."""
    x1,y1,x2,y2 = b
    x1,x2 = sorted((x1,x2))
    y1,y2 = sorted((y1,y2))
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1,y1,x2,y2)

# =========================
# Dataset: JSON Metadata
# =========================

class JSONDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.metadata_path = os.path.join(self.data_dir, "metadata.json")
        with open(self.metadata_path, "r") as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.data_dir, item["image"])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        target_str = item["annotation"]
        
        # Parse GTs for evaluation
        gts = parse_all_boxes(target_str)
        
        return {"image": image, "target_str": target_str, "gts": gts}


class MultiObjectCanvasMnist(Dataset):
    def __init__(self, split="train", num_samples=1000, canvas=CANVAS_SIZE, seed=42):
        assert split in ["train", "val"]
        self.split = split
        self.num_samples = num_samples
        self.canvas = canvas
        set_seed(seed + (0 if split == "train" else 1))
        self.mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

    def __len__(self): return self.num_samples

    def _get_digit_img(self, label: int) -> Image.Image:
        while True:
            idx = random.randrange(0, len(self.mnist))
            img_t, lab = self.mnist[idx]
            if int(lab) == label:
                return transforms.ToPILImage()(img_t)

    def _place_object(self, placed_boxes, obj_w, obj_h):
        W, H = self.canvas, self.canvas
        for _ in range(PLACEMENT_RETRIES):
            x1 = random.randint(0, max(0, W - obj_w))
            y1 = random.randint(0, max(0, H - obj_h))
            x2, y2 = x1 + obj_w, y1 + obj_h
            candidate = (x1, y1, x2, y2)
            if all(not bboxes_overlap(candidate, pb) for pb in placed_boxes):
                return candidate
        return None

    def __getitem__(self, _):
        canvas = Image.new("L", (self.canvas, self.canvas), color=255)
        placed_boxes = []
        gts = []  # list of (bbox, txt)

        num_objects = random.randint(OBJ_MIN, OBJ_MAX)
        for _obj in range(num_objects):
            is_number = (random.random() < NUM_PROB)
            if is_number:
                n_digits = random.randint(DIGITS_PER_NUM_MIN, DIGITS_PER_NUM_MAX)
                digits = [random.randint(0, 9) for _ in range(n_digits)]
                sizes = [random.randint(DIGIT_SIZE_MIN, DIGIT_SIZE_MAX) for _ in range(n_digits)]
                total_w = sum(sizes) + sum(random.randint(*GROUP_GAP_SMALL) for _ in range(n_digits-1))
                max_h = max(sizes)

                band = self._place_object(placed_boxes, total_w, max_h)
                if band is None: continue
                x1, y1, x2, y2 = band

                digit_boxes = []
                cur_x = x1
                for d, s in zip(digits, sizes):
                    img = self._get_digit_img(d).resize((s, s), Image.BILINEAR)
                    canvas.paste(img, (cur_x, y1))
                    digit_boxes.append((cur_x, y1, cur_x + s, y1 + s))
                    if cur_x + s < x2:
                        cur_x += s
                        if cur_x < x2:
                            cur_x += random.randint(*GROUP_GAP_SMALL)

                final_bbox = union_bbox(digit_boxes)
                placed_boxes.append(final_bbox)
                txt = "".join(str(d) for d in digits)
                gts.append((final_bbox, txt))
            else:
                s = random.randint(DIGIT_SIZE_MIN, DIGIT_SIZE_MAX)
                band = self._place_object(placed_boxes, s, s)
                if band is None: continue
                x1, y1, x2, y2 = band
                d = random.randint(0, 9)
                img = self._get_digit_img(d).resize((s, s), Image.BILINEAR)
                canvas.paste(img, (x1, y1))
                placed_boxes.append((x1, y1, x2, y2))
                gts.append(((x1, y1, x2, y2), str(d)))

        canvas_rgb = canvas.convert("RGB")
        chunks = [f'[BOX: {bx}, {by}, {br}, {bb}, "{txt}"]' for ((bx,by,br,bb), txt) in gts]
        target_str = "[BOXES: " + "; ".join(chunks) + "]"
        return {"image": canvas_rgb, "target_str": target_str, "gts": gts}

# =========================
# Collator (per-sample processor call)
# =========================

@dataclass
class SingleShotCollator:
    processor: AutoProcessor
    instruction: str

    def _msgs(self, answer: str):
        return [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.instruction},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

    def __call__(self, batch: List[Dict[str,Any]]) -> Dict[str, torch.Tensor]:
        tok = self.processor.tokenizer
        input_ids_list, attn_masks_list, labels_list = [], [], []
        extra_lists: Dict[str, List[torch.Tensor]] = {}

        for ex in batch:
            img, target = ex["image"], ex["target_str"]
            full_str = tok.apply_chat_template(self._msgs(target), tokenize=False, add_generation_prompt=False)
            prompt_str = tok.apply_chat_template(self._msgs(""), tokenize=False, add_generation_prompt=False)

            enc_full = self.processor(text=full_str, images=img, return_tensors="pt", padding=False, truncation=True)
            enc_prompt = self.processor(text=prompt_str, images=img, return_tensors="pt", padding=False, truncation=True)

            ids = enc_full["input_ids"].squeeze(0)
            mask = enc_full["attention_mask"].squeeze(0)
            prompt_len = int(enc_prompt["attention_mask"].sum().item())

            labels = ids.clone()
            labels[:prompt_len] = -100

            input_ids_list.append(ids)
            attn_masks_list.append(mask)
            labels_list.append(labels)

            for k, v in enc_full.items():
                if k in ("input_ids", "attention_mask"): continue
                extra_lists.setdefault(k, []).append(v)

        max_len = max(t.size(0) for t in input_ids_list)
        pad_id = tok.pad_token_id or tok.eos_token_id

        def pad1d(x, val):
            if x.size(0) == max_len: return x
            return torch.cat([x, x.new_full((max_len-x.size(0),), val)], 0)

        out = {
            "input_ids": torch.stack([pad1d(t, pad_id) for t in input_ids_list], 0),
            "attention_mask": torch.stack([pad1d(t, 0) for t in attn_masks_list], 0),
            "labels": torch.stack([pad1d(t, -100) for t in labels_list], 0),
        }
        for k, lst in extra_lists.items():
            out[k] = torch.cat(lst, 0)
        return out

# =========================
# Model loader (Vision2Seq + QLoRA, eager attn, NO flash)
# =========================

def get_model_and_processor(model_id: str):
    print("=== Loading processor (use_fast=False) ===")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

    try:
        ip = processor.image_processor
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size = {"shortest_edge": 384}
        if hasattr(ip, "crop_size") and isinstance(ip.crop_size, dict):
            ip.crop_size = {"height": 384, "width": 384}
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = min(getattr(ip, "max_pixels", 512*512), 384*384)
    except Exception as e:
        print(f"[Init] image_processor tweak skipped: {e}")

    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    try:
        tok.model_max_length = min(getattr(tok, "model_max_length", 4096), 512)
    except: pass

    print("=== Loading Base Model (bfloat16, flash_attention_2) ===")
    # Removed quantization for speed
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    # model = prepare_model_for_kbit_training(model) # Not needed for 16-bit
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    candidates = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","qkv_proj","gate_up_proj"}
    present = set()
    for n,_ in model.named_modules():
        end = n.split(".")[-1]
        if end in candidates: present.add(end)
    target_modules = sorted(present) or ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]

    lcfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules
    )
    model = get_peft_model(model, lcfg)
    model.config.pad_token_id = tok.pad_token_id

    freeze_keys = ["vision", "visual", "mm_projector", "image_proj"]
    frozen = 0
    for n,p in model.named_parameters():
        if any(k in n for k in freeze_keys):
            p.requires_grad = False; frozen += 1
    print(f"[Init] Frozen vision params: {frozen}")
    model.print_trainable_parameters()

    return model, processor

# =========================
# Parsing + Matching (robust)
# =========================

# Allow negative coords; restrict label to digits string
# Allow negative coords; accept any label content inside quotes
BOX_RE = re.compile(r"\[BOX:\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*\"([^\"]+)\"\s*\]")

def parse_all_boxes(text: str):
    out = []
    for m in BOX_RE.finditer(text):
        x1,y1,x2,y2,txt = m.groups()
        out.append(((int(x1),int(y1),int(x2),int(y2)), txt))
    return out

def greedy_match(preds, gts, iou_thresh=0.5):
    used_g = set()
    matches = []
    for i, (pb, pt) in enumerate(preds):
        best_j, best_iou = -1, 0.0
        for j, (gb, gt) in enumerate(gts):
            if j in used_g: continue
            v = iou(pb, gb)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thresh:
            used_g.add(best_j)
            matches.append((i, best_j, best_iou))
    return matches

# =========================
# Visualization (with sanitize)
# =========================

def draw_vis(img: Image.Image, preds, gts, save_path: str):
    W, H = img.size
    vis = img.copy()
    d = ImageDraw.Draw(vis)

    # GT in green
    for (x1,y1,x2,y2), txt in gts:
        sb = sanitize_box((x1,y1,x2,y2), W, H)
        if sb is None: continue
        x1,y1,x2,y2 = sb
        d.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)
        d.text((x1, max(0,y1-12)), f"GT:{txt}", fill=(0,128,0))

    # Pred in red
    for (x1,y1,x2,y2), txt in preds:
        sb = sanitize_box((x1,y1,x2,y2), W, H)
        if sb is None: continue
        x1,y1,x2,y2 = sb
        d.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=2)
        d.text((x1, min(y2+2, H-12)), f"P:{txt}", fill=(255,0,0))

    vis.save(save_path)

# =========================
# Evaluation (with FPS + visuals)
# =========================

def evaluate(model, processor, val_ds: Dataset, num_samples: int = 50, save_vis_k: int = 12):
    model.eval()
    n = min(num_samples, len(val_ds))
    total_correct_txt = 0
    total_matched = 0
    total_gts = 0
    total_preds = 0
    iou_sum = 0.0

    tot_gen = 0.0

    print("\n=== EVALUATION ===")
    for i in range(n):
        ex = val_ds[i]
        img, gts = ex["image"], ex["gts"]
        W, H = img.size

        # sanitize GTs just in case (defensive)
        gts_s = []
        for (bx,by,br,bb), t in gts:
            sb = sanitize_box((bx,by,br,bb), W, H)
            if sb is not None:
                gts_s.append((sb, t))
        gts = gts_s

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": INSTRUCTION}
        ]}]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = processor(text=prompt, images=img, return_tensors="pt").to(DEVICE)

        t1 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=200,                 # allow multiple boxes
                do_sample=False,                    # deterministic; removes temp/top_p warnings
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
        t2 = time.perf_counter()
        tot_gen += (t2 - t1)

        gen_only = out_ids[:, enc["input_ids"].shape[1]:]
        text = processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        raw_preds = parse_all_boxes(text)

        # sanitize predictions
        preds = []
        for (bx,by,br,bb), t in raw_preds:
            sb = sanitize_box((bx,by,br,bb), W, H)
            if sb is not None:
                preds.append((sb, t))

        total_gts += len(gts)
        total_preds += len(preds)

        matches = greedy_match(preds, gts, iou_thresh=0.5)
        total_matched += len(matches)
        for ip, ig, v in matches:
            iou_sum += v
            if preds[ip][1] == gts[ig][1]:
                total_correct_txt += 1

        if i < save_vis_k:
            save_path = os.path.join(VIS_DIR, f"vis_{i:03d}.png")
            draw_vis(img, preds, gts, save_path)
            print(f"[{i}] saved {save_path}")
            print(f"Raw: {text[:200]}{'...' if len(text)>200 else ''}")

    precision = total_matched / max(total_preds, 1)
    recall = total_matched / max(total_gts, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    mean_iou = iou_sum / max(total_matched, 1)
    txt_acc_on_matches = total_correct_txt / max(total_matched, 1)

    avg_infer_s = tot_gen / max(n, 1)
    fps = 1.0 / avg_infer_s if avg_infer_s > 0 else 0.0

    print("\n=== Evaluation Summary ===")
    print(f"Samples:            {n}")
    print(f"Preds total:        {total_preds} | GT total: {total_gts}")
    print(f"Matched (IoU>=.5):  {total_matched}")
    print(f"Precision:          {precision:.3f}")
    print(f"Recall:             {recall:.3f}")
    print(f"F1:                 {f1:.3f}")
    print(f"Mean IoU:           {mean_iou:.3f}")
    print(f"Text acc (matched): {txt_acc_on_matches:.3f}")
    print(f"Avg infer time:     {avg_infer_s*1000:.1f} ms  |  FPS: {fps:.2f}")

    wandb.log({
        "val/precision": precision,
        "val/recall": recall,
        "val/f1": f1,
        "val/mean_iou": mean_iou,
        "val/text_acc": txt_acc_on_matches,
        "val/fps": fps,
    })
    
    # Log a few visual examples to W&B
    vis_images = []
    for i in range(min(3, n)):
        save_path = os.path.join(VIS_DIR, f"vis_{i:03d}.png")
        if os.path.exists(save_path):
            vis_images.append(wandb.Image(save_path, caption=f"Val Sample {i}"))
    if vis_images:
        wandb.log({"val/examples": vis_images})

    model.train()

# =========================
# Training
# =========================

def get_model_and_processor(model_id: str):
    # defined above; keep for clarity of flow
    return _get_model_and_processor_impl(model_id)

def _get_model_and_processor_impl(model_id: str):
    # same body as earlier get_model_and_processor; split to keep ordering clean
    print("=== Loading processor (use_fast=False) ===")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    try:
        ip = processor.image_processor
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size = {"shortest_edge": 384}
        if hasattr(ip, "crop_size") and isinstance(ip.crop_size, dict):
            ip.crop_size = {"height": 384, "width": 384}
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = min(getattr(ip, "max_pixels", 512*512), 384*384)
    except Exception as e:
        print(f"[Init] image_processor tweak skipped: {e}")

    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    try:
        tok.model_max_length = min(getattr(tok, "model_max_length", 4096), 512)
    except: pass

    print("=== Loading Base Model (bfloat16, flash_attention_2) ===")
    # Removed quantization for speed
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    # model = prepare_model_for_kbit_training(model) # Not needed for 16-bit
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    candidates = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","qkv_proj","gate_up_proj"}
    present = set()
    for n,_ in model.named_modules():
        end = n.split(".")[-1]
        if end in candidates: present.add(end)
    target_modules = sorted(present) or ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]

    lcfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
    model = get_peft_model(model, lcfg)
    model.config.pad_token_id = tok.pad_token_id

    freeze_keys = ["vision", "visual", "mm_projector", "image_proj"]
    frozen = 0
    for n,p in model.named_parameters():
        if any(k in n for k in freeze_keys):
            p.requires_grad = False; frozen += 1
    print(f"[Init] Frozen vision params: {frozen}")
    model.print_trainable_parameters()
    return model, processor

def train():
    print("############################################################")
    print("YOLO-VLM (No Flash-Attn): Multi-Object Structured Output")
    print("############################################################\n")

    print("=== STEP 1: DATA ===")
    # Add augmentation for training
    train_transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    
    train_ds = JSONDataset(data_dir="./BU_Sign", split="train", transform=train_transform)
    val_ds   = JSONDataset(data_dir="./BU_Sign", split="valid")
    print(f"Train {len(train_ds)} | Val {len(val_ds)}")
    print("Sample target:", train_ds[0]["target_str"])

    # Initialize W&B
    wandb.init(project="YOLO-VLM", name=f"run_{int(time.time())}")
    wandb.config.update({
        "model_id": MODEL_ID,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
    })

    model, processor = get_model_and_processor(MODEL_ID)
    collator = SingleShotCollator(processor=processor, instruction=INSTRUCTION)

    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    total_steps = max(1, EPOCHS * math.ceil(len(loader) / GRAD_ACCUM_STEPS))
    warmup_steps = int(WARMUP_RATIO * total_steps)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print("\n=== TRAIN ===")
    global_step = 0
    for ep in range(EPOCHS):
        print(f"\nEpoch {ep+1}/{EPOCHS}")
        model.train()
        for bidx, batch in enumerate(loader):
            for k in batch: batch[k] = batch[k].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                extra = {k:v for k,v in batch.items() if k not in ("input_ids","attention_mask","labels")}
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                    **extra,
                )
                loss = out.loss

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (bidx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if (global_step % 5 == 0):
                     print(f"Step {global_step}/{total_steps} | Loss {loss.item() * GRAD_ACCUM_STEPS:.4f}")
                     wandb.log({
                        "train/loss": loss.item() * GRAD_ACCUM_STEPS,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": ep + 1,
                        "train/global_step": global_step
                    })

        save_dir = os.path.join(OUT_DIR, f"checkpoint_epoch_{ep+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir); processor.save_pretrained(save_dir)
        print(f"Saved adapter + processor to {save_dir}")

        evaluate(model, processor, val_ds, num_samples=50, save_vis_k=12)

if __name__ == "__main__":
    train()
