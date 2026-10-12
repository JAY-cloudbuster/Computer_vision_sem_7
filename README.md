# Automated License Plate Recognition (ALPR) & Deep Vision Benchmark
> **Computer Vision (Semester 7) Final Academic Project**  
> An end-to-end, multi-stage Automatic License Plate Recognition pipeline benchmarking classical computer vision, handcrafted machine learning, deep CNN backbones, and optical character recognition.

---

## 🏛️ System Architecture

The pipeline processes high-resolution vehicle scenes through a hierarchical 4-stage computer vision workflow:

```
[ Raw Vehicle Image ]
         │
         ▼
[ Stage 1: Proposal Generator ] ──► Classical OpenCV (Morphology + Sobel) vs Pretrained YOLOv8
         │
         ▼
[ Stage 2: Deep CNN Quality Gate ] ──► AlexNet | VGG-16 | GoogLeNet | ResNet-50 | DenseNet-121
         │ (Filters out Non-Plate False Positives: Grilles, Emblems, Shadows)
         ▼
[ Stage 3: Plate Preprocessing ] ──► CLAHE Contrast Normalization + Bilateral Edge Denoising
         │
         ▼
[ Stage 4: OCR Text Reader ] ──► EasyOCR Engine + HSRP Syntax Cleaning + Position Disambiguation
         │
         ▼
[ Final ALPR Output & Report ] ──► Exact Match %, Levenshtein NED, Character Accuracy, End-to-End FPS
```

---

## 📁 Repository Directory Structure

```text
cV_project/
├── combined_dataset/            # 1,696 vehicle images with ground-truth labels.csv
│   ├── images/                  # High-resolution vehicle images
│   └── labels.csv               # Bounding boxes (xmin, ymin, xmax, ymax) and plate texts
│
├── data/                        # Benchmark dataset splits & cached feature crops
│   ├── annotations_test.json    # Test split detection annotations
│   ├── annotations_train.json   # Train split detection annotations
│   └── crop_dataset_224.npz     # Pre-cached 224x224 crops (Plates vs Non-Plates)
│
├── models/                      # Model checkpoints and weights
│   ├── resnet50_plate_classifier.pth   # Trained ResNet-50 candidate verifier (89.7 MB)
│   └── yolov8_plate.onnx               # Pretrained YOLOv8 detector (12.2 MB)
│
├── modules/                     # Modular vision & evaluation engines
│   ├── detector_classical.py    # OpenCV Sobel & morphological candidate detector
│   ├── detector_huggingface.py  # YOLOv8 ONNX detector wrapper
│   ├── evaluate_detection.py    # Detection IoU, Precision, Recall, and FPS metrics
│   ├── feature_extractor.py     # 10 Handcrafted Shape, Texture & Color features
│   ├── ml_classifiers.py        # Classical ML models & feature ablation framework
│   ├── cnn_models.py            # 5 Pretrained ImageNet CNN transfer learning builders
│   ├── cnn_individual_runner.py # CNN training, checkpointing & benchmark runner
│   ├── case_visualizer.py       # Detection diagnostic multi-panel visual generator
│   ├── reader_evaluator.py      # Preprocessing, EasyOCR, syntax cleaner & ALPR metrics
│   ├── reader_visualizer.py     # ALPR multi-panel visual case generator & overview
│   └── efficiency_tester.py     # Real-time video frame-skipping policies (k=1, 2, 3)
│
├── results/                     # Experimental outputs, benchmark CSVs & figures
│   ├── cnn_comparison_results.csv      # CNN Stage 1 verification benchmark
│   ├── detection_results.csv           # Classical OpenCV vs YOLOv8 detection metrics
│   ├── efficiency_frame_skipping.csv   # Frame-skipping throughput trade-off
│   ├── ml_ablation_results.csv         # Feature ablation across 5 ML algorithms
│   ├── reader_comparison_results.csv   # Stage 2 ALPR End-to-End Reader benchmark
│   ├── FINAL_BENCHMARK_REPORT.md       # Comprehensive markdown benchmark report
│   ├── figures/                        # Publication-quality benchmark comparison charts
│   ├── resnet50/                       # ResNet-50 detection diagnostic panels
│   └── resnet50_reader/                # ResNet-50 ALPR diagnostic panels & master overview
│
├── tests/                       # Unit tests and sanity checks
│   └── test_cnn.py              # CNN forward-pass tensor shape test
│
├── TEAM_INSTRUCTIONS.md         # Team Guide: Stage 1 CNN Detection Verification
├── TEAM_READER_INSTRUCTIONS.md  # Team Guide: Stage 2 ALPR Reader Pipeline
├── run_full_pipeline.py         # Master pipeline orchestrator (Steps 1 to 8)
└── .gitignore                   # Ignores large datasets, caches, logs, and OS files
```

---

## 👥 Architecture & Teammate Distribution

| Team Member | CNN Backbone | 1. Train CNN | 2. Detection Cases | 3. ALPR Reader Pipeline | 4. Reader Cases |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Satvic (Done)** | **ResNet-50** | `run_cnn_resnet50.py` | `generate_resnet50_cases.py` | `run_reader_resnet50.py` | `generate_resnet50_reader_cases.py` |
| **Teammate 1** | **AlexNet** | `run_cnn_alexnet.py` | `generate_alexnet_cases.py` | `run_reader_alexnet.py` | `generate_alexnet_reader_cases.py` |
| **Teammate 2** | **VGG-16** | `run_cnn_vgg16.py` | `generate_vgg16_cases.py` | `run_reader_vgg16.py` | `generate_vgg16_reader_cases.py` |
| **Teammate 3** | **GoogLeNet** | `run_cnn_googlenet.py` | `generate_googlenet_cases.py` | `run_reader_googlenet.py` | `generate_googlenet_reader_cases.py` |
| **Teammate 4** | **DenseNet-121** | `run_cnn_densenet121.py` | `generate_densenet121_cases.py` | `run_reader_densenet121.py` | `generate_densenet121_reader_cases.py` |

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
pip install torch torchvision opencv-python easyocr pandas scikit-learn xgboost matplotlib
```

### 2. Stage 1: CNN Candidate Verification
Train your assigned architecture and produce detection diagnostics:
```bash
# Example for ResNet-50:
python run_cnn_resnet50.py --epochs 3 --device cpu
python generate_resnet50_cases.py
```

### 3. Stage 2: End-to-End ALPR Reader
Run the OCR pipeline through your trained CNN gate and produce diagnostic graphics:
```bash
# Example for ResNet-50:
python run_reader_resnet50.py --samples 40 --device cpu
python generate_resnet50_reader_cases.py
```

---

## 📊 Benchmark Results

### Stage 1: Candidate Verification (CNN Quality Gate)
From `results/cnn_comparison_results.csv`:
* **ResNet-50**: 23.51M params | **99.5% Accuracy** | **0.995 F1-Score** | 10.5 ms latency (95.1 FPS on CPU)

### Stage 2: ALPR Detection-to-Reader
From `results/reader_comparison_results.csv`:
* **ResNet-50 Gate**: **96.0% Pass Rate** | **28.0% Exact Match** | **72.3% Character Accuracy** | **0.7271 Normalized Edit Distance** | 3.6 FPS End-to-End (CPU)

---

## 📜 Team Guides
- [TEAM_INSTRUCTIONS.md](file:///d:/academics/cV_project/TEAM_INSTRUCTIONS.md): Detailed instructions for training Stage 1 CNNs.
- [TEAM_READER_INSTRUCTIONS.md](file:///d:/academics/cV_project/TEAM_READER_INSTRUCTIONS.md): Detailed instructions for running Stage 2 ALPR Reader.
