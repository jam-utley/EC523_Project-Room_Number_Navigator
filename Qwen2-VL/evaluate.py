#!/usr/bin/env python
"""
Comprehensive Metrics Evaluation Script for YOLO-VLM
Evaluates: Precision, Recall, F1-Score, Text Accuracy, Inference Speed, Parse Success
Runs on both validation and test sets
"""

import os
import time
import json
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from typing import List, Tuple, Dict
import pandas as pd
from tqdm import tqdm

# =========================
# Config
# =========================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LORA_PATH = "./qwen2vl_BU_Sign_2/checkpoint_epoch_20"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IOU_THRESHOLD = 0.5  # IoU threshold for matching predictions to ground truth

# Regex for parsing [BOX: x1, y1, x2, y2, "label"]
BOX_RE = re.compile(r"\[BOX:\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*\"([^\"]+)\"\s*\]")

INSTRUCTION = (
    "Detect all objects in the image. "
    "Return exactly one line with this format:\n"
    "[DETECTIONS: [BOX: x1, y1, x2, y2, \"label\"], [BOX: x1, y1, x2, y2, \"label\"], ...]\n"
    "No explanations, no extra text."
)

# =========================
# Utility Functions
# =========================

def parse_boxes(text: str) -> List[Tuple[Tuple[int, int, int, int], str]]:
    """Parse boxes from model output"""
    boxes = []
    for m in BOX_RE.finditer(text):
        x1, y1, x2, y2, label = m.groups()
        boxes.append(((int(x1), int(y1), int(x2), int(y2)), label))
    return boxes


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two boxes"""
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2
    
    # Compute intersection
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    
    # Compute union
    box1_area = max(0, x2 - x1) * max(0, y2 - y1)
    box2_area = max(0, X2 - X1) * max(0, Y2 - Y1)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def greedy_match(preds: List, gts: List, iou_thresh: float = 0.5) -> List[Tuple[int, int, float]]:
    """Greedy matching between predictions and ground truth boxes"""
    used_gt = set()
    matches = []
    
    for i, (pred_box, pred_label) in enumerate(preds):
        best_j, best_iou = -1, 0.0
        for j, (gt_box, gt_label) in enumerate(gts):
            if j in used_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        
        if best_j >= 0 and best_iou >= iou_thresh:
            used_gt.add(best_j)
            matches.append((i, best_j, best_iou))
    
    return matches


def sanitize_box(box: Tuple[int, int, int, int], W: int, H: int):
    """Sanitize box coordinates to be within image bounds"""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))
    
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# =========================
# Model Loading
# =========================

def load_model():
    """Load the model and processor"""
    print(f"Loading base model: {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    
    print(f"Loading LoRA adapter from {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    print("Model loaded successfully!")
    
    return model, processor


# =========================
# Evaluation
# =========================

def evaluate_dataset(model, processor, data_dir, split_name):
    """Run comprehensive evaluation on a dataset split"""
    
    metadata_path = os.path.join(data_dir, split_name, "metadata.json")
    
    # Load data
    print(f"\nLoading {split_name} data from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    print(f"Found {len(data)} images")
    
    # Metrics accumulators
    total_tp = 0  # True Positives (matched predictions)
    total_fp = 0  # False Positives (unmatched predictions)
    total_fn = 0  # False Negatives (unmatched ground truths)
    total_correct_labels = 0  # Correctly predicted labels among matches
    total_matched = 0
    total_iou_sum = 0.0
    
    inference_times = []
    parse_success_count = 0
    parse_attempts = 0
    
    # Per-sample results
    results = []
    
    print(f"\n=== Running {split_name} Evaluation ===")
    for idx, item in enumerate(tqdm(data, desc=f"Evaluating {split_name}")):
        image_path = os.path.join(data_dir, split_name, item["image"])
        gt_annotation = item["annotation"]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        
        # Parse ground truth
        gt_boxes_raw = parse_boxes(gt_annotation)
        gt_boxes = []
        for (box, label) in gt_boxes_raw:
            sanitized = sanitize_box(box, W, H)
            if sanitized is not None:
                gt_boxes.append((sanitized, label))
        
        # Prepare input
        messages = [{
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION}
            ]
        }]
        
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        
        # Inference
        t_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
        t_end = time.perf_counter()
        
        inference_time = t_end - t_start
        inference_times.append(inference_time)
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse predictions
        parse_attempts += 1
        pred_boxes_raw = parse_boxes(output_text)
        
        # Check if parsing was successful (at least one box found or no GT boxes)
        if len(pred_boxes_raw) > 0 or len(gt_boxes) == 0:
            parse_success_count += 1
        
        # Sanitize predictions
        pred_boxes = []
        for (box, label) in pred_boxes_raw:
            sanitized = sanitize_box(box, W, H)
            if sanitized is not None:
                pred_boxes.append((sanitized, label))
        
        # Match predictions to ground truth
        matches = greedy_match(pred_boxes, gt_boxes, iou_thresh=IOU_THRESHOLD)
        
        # Update metrics
        num_matches = len(matches)
        num_preds = len(pred_boxes)
        num_gts = len(gt_boxes)
        
        tp = num_matches
        fp = num_preds - num_matches
        fn = num_gts - num_matches
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Label accuracy and IoU
        sample_iou_sum = 0.0
        correct_labels = 0
        for pred_idx, gt_idx, iou_val in matches:
            sample_iou_sum += iou_val
            if pred_boxes[pred_idx][1] == gt_boxes[gt_idx][1]:
                correct_labels += 1
        
        total_iou_sum += sample_iou_sum
        total_matched += num_matches
        total_correct_labels += correct_labels
        
        # Store per-sample result
        sample_result = {
            "image": item["image"],
            "num_gt": num_gts,
            "num_pred": num_preds,
            "num_matches": num_matches,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "correct_labels": correct_labels,
            "inference_time_ms": inference_time * 1000,
            "predicted_text": output_text[:200],  # First 200 chars
            "parse_success": len(pred_boxes_raw) > 0 or len(gt_boxes) == 0
        }
        results.append(sample_result)
    
    # Calculate final metrics
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-9)
    
    text_accuracy = total_correct_labels / max(total_matched, 1)
    mean_iou = total_iou_sum / max(total_matched, 1)
    
    avg_inference_time = np.mean(inference_times)
    inference_speed_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    parse_success_rate = parse_success_count / max(parse_attempts, 1)
    
    # Summary
    summary = {
        "model_id": MODEL_ID,
        "lora_path": LORA_PATH,
        "split": split_name,
        "num_samples": len(data),
        "iou_threshold": IOU_THRESHOLD,
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "text_accuracy": float(text_accuracy),
            "mean_iou": float(mean_iou),
            "inference_speed_fps": float(inference_speed_fps),
            "avg_inference_time_ms": float(avg_inference_time * 1000),
            "parse_success_rate": float(parse_success_rate)
        },
        "detailed_counts": {
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "total_matched_boxes": total_matched,
            "total_correct_labels": total_correct_labels,
            "total_predictions": total_tp + total_fp,
            "total_ground_truths": total_tp + total_fn
        },
        "per_sample_results": results
    }
    
    return summary


def print_metrics(summary, split_name):
    """Print metrics summary"""
    print("\n" + "="*60)
    print(f"{split_name.upper()} SET RESULTS")
    print("="*60)
    print(f"\nSamples: {summary['num_samples']}")
    print(f"IoU Threshold: {summary['iou_threshold']}")
    print("\n--- METRICS ---")
    print(f"Precision:          {summary['metrics']['precision']:.4f}")
    print(f"Recall:             {summary['metrics']['recall']:.4f}")
    print(f"F1-Score:           {summary['metrics']['f1_score']:.4f}")
    print(f"Text Accuracy:      {summary['metrics']['text_accuracy']:.4f}")
    print(f"Mean IoU:           {summary['metrics']['mean_iou']:.4f}")
    print(f"Inference Speed:    {summary['metrics']['inference_speed_fps']:.2f} FPS")
    print(f"Avg Inference Time: {summary['metrics']['avg_inference_time_ms']:.2f} ms")
    print(f"Parse Success Rate: {summary['metrics']['parse_success_rate']:.4f}")
    
    print("\n--- DETAILED COUNTS ---")
    print(f"True Positives:     {summary['detailed_counts']['total_true_positives']}")
    print(f"False Positives:    {summary['detailed_counts']['total_false_positives']}")
    print(f"False Negatives:    {summary['detailed_counts']['total_false_negatives']}")
    print(f"Total Predictions:  {summary['detailed_counts']['total_predictions']}")
    print(f"Total Ground Truth: {summary['detailed_counts']['total_ground_truths']}")
    print(f"Matched Boxes:      {summary['detailed_counts']['total_matched_boxes']}")
    print(f"Correct Labels:     {summary['detailed_counts']['total_correct_labels']}")


# =========================
# Main
# =========================

def main():
    print("="*60)
    print("YOLO-VLM Comprehensive Metrics Evaluation")
    print("="*60)
    
    # Load model
    model, processor = load_model()
    
    # Evaluate on both validation and test sets
    data_dir = "./BU_Sign"
    
    # Validation set (used during training)
    val_summary = evaluate_dataset(model, processor, data_dir, "valid")
    print_metrics(val_summary, "Validation")
    
    # Save validation results
    with open("metrics_validation.json", 'w') as f:
        json.dump(val_summary, f, indent=2)
    print(f"\n✓ Validation results saved to: metrics_validation.json")
    
    df_val = pd.DataFrame(val_summary['per_sample_results'])
    df_val.to_csv("metrics_validation.csv", index=False)
    print(f"✓ Per-sample validation results saved to: metrics_validation.csv")
    
    # Test set
    test_summary = evaluate_dataset(model, processor, data_dir, "test")
    print_metrics(test_summary, "Test")
    
    # Save test results
    with open("metrics_test.json", 'w') as f:
        json.dump(test_summary, f, indent=2)
    print(f"\n✓ Test results saved to: metrics_test.json")
    
    df_test = pd.DataFrame(test_summary['per_sample_results'])
    df_test.to_csv("metrics_test.csv", index=False)
    print(f"✓ Per-sample test results saved to: metrics_test.csv")
    
    # Create combined summary table
    print("\n" + "="*60)
    print("COMBINED METRICS SUMMARY TABLE")
    print("="*60)
    
    combined_table = pd.DataFrame([
        ["Validation", "Precision", f"{val_summary['metrics']['precision']:.4f}"],
        ["Validation", "Recall", f"{val_summary['metrics']['recall']:.4f}"],
        ["Validation", "F1-Score", f"{val_summary['metrics']['f1_score']:.4f}"],
        ["Validation", "Text Accuracy", f"{val_summary['metrics']['text_accuracy']:.4f}"],
        ["Validation", "Mean IoU", f"{val_summary['metrics']['mean_iou']:.4f}"],
        ["Validation", "Inference Speed (FPS)", f"{val_summary['metrics']['inference_speed_fps']:.2f}"],
        ["Validation", "Avg Inference Time (ms)", f"{val_summary['metrics']['avg_inference_time_ms']:.2f}"],
        ["Validation", "Parse Success Rate", f"{val_summary['metrics']['parse_success_rate']:.4f}"],
        ["", "", ""],
        ["Test", "Precision", f"{test_summary['metrics']['precision']:.4f}"],
        ["Test", "Recall", f"{test_summary['metrics']['recall']:.4f}"],
        ["Test", "F1-Score", f"{test_summary['metrics']['f1_score']:.4f}"],
        ["Test", "Text Accuracy", f"{test_summary['metrics']['text_accuracy']:.4f}"],
        ["Test", "Mean IoU", f"{test_summary['metrics']['mean_iou']:.4f}"],
        ["Test", "Inference Speed (FPS)", f"{test_summary['metrics']['inference_speed_fps']:.2f}"],
        ["Test", "Avg Inference Time (ms)", f"{test_summary['metrics']['avg_inference_time_ms']:.2f}"],
        ["Test", "Parse Success Rate", f"{test_summary['metrics']['parse_success_rate']:.4f}"],
    ], columns=["Dataset", "Metric", "Value"])
    
    print(combined_table.to_string(index=False))
    
    # Save combined summary
    combined_table.to_csv("metrics_combined_summary.csv", index=False)
    print(f"\n✓ Combined metrics summary saved to: metrics_combined_summary.csv")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
