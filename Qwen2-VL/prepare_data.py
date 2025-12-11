"""
YOLO-VLM Data Generator (Multi-object, digits & numbers)
=======================================================

- Each image has 1–6 groups.
- A group is either:
    * a single digit (label: digit name "zero".."nine"), or
    * a contiguous number of length 2–5 (label: "9867", etc.)
- One bounding box per group.
- Output format: [DETECTIONS: [BOX: x1, y1, x2, y2, "label"], ...]
"""

import os
import json
import random
from typing import List, Tuple

from PIL import Image
import numpy as np
from torchvision import datasets
from tqdm import tqdm


DIGIT_NAMES = ["zero","one","two","three","four","five","six","seven","eight","nine"]


class YOLOVLMDataGenerator:
    """Generates composite images with multiple digits and numbers."""

    def __init__(self, output_dir="data", canvas_size=384, seed=42):
        self.output_dir = output_dir
        self.canvas_size = canvas_size
        random.seed(seed)
        np.random.seed(seed)

        # MNIST (PIL images)
        self.mnist_train = datasets.MNIST(root="./mnist_data", train=True, download=True)
        self.mnist_test  = datasets.MNIST(root="./mnist_data", train=False, download=True)

    def _sample_digit_img(self, split="train", label: int = None) -> Image.Image:
        ds = self.mnist_train if split == "train" else self.mnist_test
        if label is None:
            idx = np.random.randint(0, len(ds))
            img, _ = ds[idx]
            return img
        # pick one with target == label
        while True:
            idx = np.random.randint(0, len(ds))
            img, y = ds[idx]
            if y == label:
                return img

    @staticmethod
    def _rect_intersects(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
            return False
        return True

    def _try_place_rect(self, w: int, h: int, occupied: List[Tuple[int,int,int,int]], max_tries=100) -> Tuple[int,int]:
        for _ in range(max_tries):
            x = np.random.randint(8, self.canvas_size - w - 8)
            y = np.random.randint(8, self.canvas_size - h - 8)
            rect = (x, y, x + w, y + h)
            if all(not self._rect_intersects(rect, r) for r in occupied):
                return x, y
        return None, None  # give up

    def _render_group(
        self,
        canvas: Image.Image,
        split: str,
        group_kind: str,
        num_len: int = 1,
        digit_spacing: int = 4
    ) -> Tuple[Tuple[int,int,int,int], str]:
        """
        Render a group on the canvas:
        - group_kind == "digit": one 28x28 digit, label is a digit name
        - group_kind == "number": horizontally placed 2..5 digits, label is the concatenated string
        Returns: (bbox, label)
        """
        if group_kind == "digit":
            d = np.random.randint(0, 10)
            img = self._sample_digit_img(split, d).convert("RGB")
            # paste size
            w = h = 28
            return self._paste_group(canvas, [img], label="{}".format(DIGIT_NAMES[d]), spacing=0)

        # number group
        digits = [np.random.randint(0, 10) for _ in range(num_len)]
        imgs = [self._sample_digit_img(split, d).convert("RGB") for d in digits]
        label = "".join(str(d) for d in digits)
        return self._paste_group(canvas, imgs, label=label, spacing=digit_spacing)

    def _paste_group(
        self, canvas: Image.Image, digit_imgs: List[Image.Image], label: str, spacing: int
    ) -> Tuple[Tuple[int,int,int,int], str]:
        """Place digit_imgs in a row with spacing, ensuring non-overlap with existing content."""
        occupied = getattr(canvas, "_occupied", [])
        # All MNIST are 28x28
        digit_w = digit_h = 28
        group_w = len(digit_imgs) * digit_w + (len(digit_imgs) - 1) * spacing
        group_h = digit_h

        x, y = self._try_place_rect(group_w, group_h, occupied, max_tries=200)
        if x is None:
            # If crowded, fall back to smaller figure near (0,0)
            x, y = 4, 4

        # Paste digits
        cx = x
        for img in digit_imgs:
            canvas.paste(img.convert("RGB"), (cx, y))
            cx += digit_w + spacing

        bbox = (x, y, x + group_w, y + group_h)
        occupied.append(bbox)
        canvas._occupied = occupied  # stash updated occupancy
        return bbox, label

    def _create_image(self, split: str, max_groups: int = 6) -> Tuple[Image.Image, str]:
        """
        Create one composite sample and return (canvas, annotation_string).
        """
        canvas = Image.new("RGB", (self.canvas_size, self.canvas_size), color=(255, 255, 255))
        canvas._occupied = []

        n_groups = np.random.randint(1, max_groups + 1)
        boxes_and_labels = []

        for _ in range(n_groups):
            # 60% chance make a multi-digit number (length 2..5), else single digit
            if np.random.rand() < 0.6:
                L = np.random.randint(2, 6)
                bbox, label = self._render_group(canvas, split, "number", num_len=L, digit_spacing=np.random.randint(2, 7))
            else:
                bbox, label = self._render_group(canvas, split, "digit")

            boxes_and_labels.append((bbox, label))

        # Build annotation string
        items = []
        for (x1, y1, x2, y2), lab in boxes_and_labels:
            items.append(f'[BOX: {x1}, {y1}, {x2}, {y2}, "{lab}"]')
        annotation = "[DETECTIONS: " + ", ".join(items) + "]"
        return canvas, annotation

    def generate_dataset(self, split="train", num_samples=2000):
        split_dir = os.path.join(self.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        metadata = []
        print(f"🎨 Generating {num_samples} {split} samples (multi-object)…")
        for i in tqdm(range(num_samples)):
            canvas, annotation = self._create_image(split=split)
            fname = f"{split}_{i:06d}.png"
            canvas.save(os.path.join(split_dir, fname))
            metadata.append({"image": fname, "annotation": annotation})

        with open(os.path.join(split_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Generated {num_samples} samples in {split_dir}")
        return metadata

    def validate_samples(self, num_samples=5):
        print("\n" + "=" * 60)
        print("📋 SAMPLE ANNOTATIONS (Format Validation)")
        print("=" * 60)
        for i in range(num_samples):
            canvas, annotation = self._create_image(split="train")
            print(f"Sample {i+1}: {annotation}")
        print("=" * 60 + "\n")


def main():
    print(
        """
    ╔════════════════════════════════════════════════════════╗
    ║  YOLO-VLM Multi-Object Automated Annotation Engine    ║
    ╚════════════════════════════════════════════════════════╝
    """
    )
    gen = YOLOVLMDataGenerator()

    gen.validate_samples(num_samples=5)

    train_meta = gen.generate_dataset(split="train", num_samples=3000)
    val_meta   = gen.generate_dataset(split="val",   num_samples=400)

    print("\n" + "=" * 60)
    print("🎉 DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Training samples: {len(train_meta)}")
    print(f"Validation samples: {len(val_meta)}")
    print("\nDataset structure:")
    print("  data/train/  - Training images + metadata.json")
    print("  data/val/    - Validation images + metadata.json")
    print("\n✅ Ready for training! Run: python train_and_validate.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
