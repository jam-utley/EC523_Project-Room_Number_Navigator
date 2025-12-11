# EC523 Project: Room Number Navigator

**Boston University EC523 - Deep Learning Final Project**

A comprehensive vision-language model (VLM) system for real-time room number and door sign detection. This project explores multiple approaches including YOLO object detection, Florence-2, and Qwen2-VL fine-tuning.

## 🎯 Project Overview

We developed an end-to-end system for detecting and reading door signs in real-time using vision-language models. The system combines:

- **Object Detection:** Fast localization with YOLOv7
- **Vision-Language Models:** Structured text generation with Qwen2-VL and Florence-2
- **Real-time Processing:** Optimized inference pipeline with async processing

## 📂 Repository Structure

```
├── Qwen2-VL/           # ⭐ Main implementation (1,543 LOC)
│   ├── train.py        # LoRA fine-tuning pipeline
│   ├── evaluate.py     # Metrics evaluation
│   ├── demo.py         # Real-time video demo
│   ├── annotate.py     # Interactive annotation tool
│   └── prepare_data.py # Dataset preparation
│
├── Florence2/          # Florence-2 experiments
│   ├── Finetuning_Florence2.ipynb
│   ├── Graphing_Florence2_TrainVal.ipynb
│   ├── Quantizing_Florence2.ipynb
│   └── QLoRA_Training/
│
└── Yolo/              # YOLOv7 baseline
    └── yolov7_door_plaque_training.ipynb
```

## 🚀 Quick Start

### Qwen2-VL (Recommended)

Our best-performing model uses Qwen2-VL with LoRA fine-tuning:

```bash
cd Qwen2-VL

# 1. Install dependencies
pip install torch transformers peft opencv-python pillow

# 2. Train model
python train.py --data-dir ./dataset --epochs 30

# 3. Evaluate
python evaluate.py --model-path ./checkpoints/checkpoint_epoch_30

# 4. Run demo
python demo.py --model-path ./checkpoints/checkpoint_epoch_30
```

See [Qwen2-VL/README.md](Qwen2-VL/README.md) for detailed documentation.

### Florence-2

Experimental Florence-2 fine-tuning (Jupyter notebooks):

```bash
cd Florence2
jupyter notebook Finetuning_Florence2.ipynb
```

### YOLOv7

Baseline object detection:

```bash
cd Yolo
jupyter notebook yolov7_door_plaque_training.ipynb
```

## 📊 Results Comparison

| Model | Precision | Recall | F1-Score | Inference Speed | Code Size |
|-------|-----------|--------|----------|-----------------|-----------|
| **Qwen2-VL** | **~87%** | **~82%** | **~85%** | **1.0-1.5 FPS** | **1,543 LOC** |
| Florence-2 | ~75% | ~70% | ~72% | 0.8-1.2 FPS | N/A |
| YOLOv7 | ~80% | ~75% | ~77% | 15+ FPS | N/A |

**Note:** Qwen2-VL provides the best balance of accuracy and text understanding, though YOLOv7 is faster for pure detection.

## 🎓 Key Features

### Qwen2-VL Implementation

- ✅ **Structured Text Generation:** Grammar-based output format
- ✅ **LoRA Fine-tuning:** Efficient parameter-efficient training
- ✅ **Custom Annotation Tool:** Interactive data labeling
- ✅ **Real-time Dashboard:** FPS tracking, system monitoring
- ✅ **Comprehensive Metrics:** P/R/F1, text accuracy, parse success

### Dataset

- **813 real-world images** of campus corridors
- Varied lighting, reflections, angles
- Multi-object scenes with door numbers, room signs
- YOLO format → VLM structured text conversion

## 🔬 Technical Details

### Architecture

**Base Model:** Qwen2-VL-2B-Instruct (2B parameters)  
**Fine-tuning:** LoRA (r=64, α=16, dropout=0.05)  
**Optimizer:** AdamW with cosine learning rate schedule  
**Precision:** BFloat16 with Flash Attention 2  
**Training Resolution:** 576×576  

### Output Format

```
[DETECTIONS: [BOX: x1, y1, x2, y2, "211"], [BOX: x1, y1, x2, y2, "ENG"], ...]
```

Coordinates normalized to [0, 1000] scale for consistency.

## 📝 Files Description

### Qwen2-VL (Core Implementation)



-`train.py`- Training pipeline with LoRA fine-tuning 
-`evaluate.py`- Metrics: P/R/F1, text accuracy, FPS 
- `demo.py` -  Video inference with async processing 
- `annotate.py`- Interactive annotation interface 
- `prepare_data.py` - YOLO → VLM format conversion 

### Florence-2 Notebooks

- `Finetuning_Florence2.ipynb` - Successful fine-tuning
- `Graphing_Florence2_TrainVal.ipynb` - Training visualization
- `Quantizing_Florence2.ipynb` - Model quantization attempts
- `QLoRA_Training/` - Experimental QLoRA training

### YOLO Notebooks

- `yolov7_door_plaque_training.ipynb` - YOLOv7 baseline training and evaluation

## 🛠️ Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision transformers peft
pip install opencv-python pillow numpy psutil

# Optional: Flash Attention 2 for faster inference
pip install flash-attn --no-build-isolation
```

### GPU Requirements

- **Minimum:** 8GB VRAM (training with batch size 2-4)
- **Recommended:** 16GB+ VRAM for optimal performance

## 📖 Usage Examples

### Training

```bash
python Qwen2-VL/train.py \
  --data-dir ./dataset \
  --output-dir ./checkpoints \
  --epochs 30 \
  --batch-size 4 \
  --learning-rate 1e-4
```

### Evaluation

```bash
python Qwen2-VL/evaluate.py \
  --model-path ./checkpoints/checkpoint_epoch_30 \
  --test-dir ./BU_Sign/test \
  --output metrics_report.json
```

### Real-time Demo

```bash
python Qwen2-VL/demo.py \
  --model-path ./checkpoints/checkpoint_epoch_30 \
  --video-dir ./BU_Sign/test
```

## 🤝 Team

Boston University EC523 Deep Learning Project

## 📄 License

MIT License - Feel free to use and modify for your projects!

## 🔗 References

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) - Base vision-language model
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [YOLOv7](https://github.com/WongKinYiu/yolov7) - Object detection baseline
- [Florence-2](https://huggingface.co/microsoft/Florence-2-large) - Alternative VLM

## 📧 Contact

For questions or collaboration, please open an issue or contact the team through Boston University.

---

**⭐ Star this repository if you find it useful!**
