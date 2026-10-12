# AlexNet — License Plate Verification Subsystem

## Overview
- **Team Role**: Teammate 1
- **Architecture**: AlexNet (Classic 8-Layer Deep Convolutional Neural Network)
- **Year / Authors**: 2012, Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **Primary Innovation**: 8-layer deep architecture, Rectified Linear Unit (ReLU) activations, Dropout (0.5) regularization, and overlapping max pooling.
- **Complexity**: ~57.01 Million Parameters (217.5 MB weight checkpoint).

## Key Files
- `model.py`: Modular AlexNet architecture definition with transfer learning.
- `train.py`: Independent training script for Teammate 1.
- `evaluate.py`: Diagnostic visualizer and evaluation runner.
- `run_reader.py`: Independent detection-to-reader ALPR pipeline runner.
- `generate_reader_cases.py`: Independent ALPR diagnostic test cases generator.
- Root runners: `run_cnn_alexnet.py`, `generate_alexnet_cases.py`, `run_reader_alexnet.py`, and `generate_alexnet_reader_cases.py`.

## Execution Commands

### Part 1: CNN Plate Verification
```bash
# 1. Train model
python AlexNet/train.py --epochs 3 --device cpu

# 2. Generate 5 diagnostic cases
python AlexNet/evaluate.py
```
Outputs: `models/alexnet_plate_classifier.pth`, `results/cnn_comparison_results.csv`, `results/alexnet/`.

### Part 2: End-to-End ALPR Reader Pipeline (EasyOCR + Quality Gate)
```bash
# 1. Run detection-to-reader pipeline
python AlexNet/run_reader.py --samples 40 --device cpu

# 2. Generate 5 ALPR diagnostic cases + master overview
python AlexNet/generate_reader_cases.py
```
Outputs: `results/reader_comparison_results.csv`, `results/alexnet_reader/`.

## Benchmark Highlights
- **Verification Accuracy**: 99.00% (F1-score: 0.9900)
- **CNN Forward Latency**: 2.78 ms / sample (359.5 FPS)
- **ALPR Gate Pass Rate**: 100.0%
- **ALPR End-to-End Latency**: 86.9 ms (11.5 FPS — 3.2x real-time speedup vs ResNet-50)

