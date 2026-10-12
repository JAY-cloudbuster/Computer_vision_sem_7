# License Plate Vision Benchmark - Team CNN Runner Guide

This document provides clear instructions for each team member to:
1. Run their assigned CNN architecture independently.
2. Generate their individual 5 side-by-side diagnostic test case images.
3. Preserve all Classical ML results.
4. Push their model checkpoint and visual results back to GitHub.

---

## 👥 Architecture & Script Assignment

| Member | Assigned CNN | 1. Training Script | 2. Visual Cases Script | Results Folder | Model Complexity | Primary Innovation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **You (Done)** | **ResNet-50** | `python run_cnn_resnet50.py` | `python generate_resnet50_cases.py` | `results/resnet50/` | 23.51 M params (89.7 MB) | Deep Residual Bottlenecks & Skip Connections |
| **Teammate 1** | **AlexNet** | `python run_cnn_alexnet.py` | `python generate_alexnet_cases.py` | `results/alexnet/` | 57.01 M params (217.5 MB) | Classic 8-layer deep CNN, ReLU, Dropout |
| **Teammate 2** | **VGG-16** | `python run_cnn_vgg16.py` | `python generate_vgg16_cases.py` | `results/vgg16/` | 134.27 M params (512.2 MB) | Deep homogeneous 3×3 convolutions stack |
| **Teammate 3** | **GoogLeNet** | `python run_cnn_googlenet.py` | `python generate_googlenet_cases.py` | `results/googlenet/` | 5.60 M params (21.4 MB) | Multi-scale Inception modules & GAP |
| **Teammate 4** | **DenseNet-121** | `python run_cnn_densenet121.py` | `python generate_densenet121_cases.py` | `results/densenet121/` | 6.96 M params (26.5 MB) | Dense feature concatenation & layer reuse |

---

## 🚀 Step-by-Step Instructions for Teammates

### Step 1: Clone or Pull the Latest Repository
Ensure you have the latest code, cached dataset, and scripts:
```bash
git pull origin main
```

### Step 2: Environment Setup
Ensure standard dependencies are installed:
```bash
pip install torch torchvision opencv-python pandas scikit-learn matplotlib
```

### Step 3: Run Your Training & Benchmark Script
Each script loads the pre-cached crop dataset (`data/crop_dataset_224.npz`), applies transfer learning, evaluates on the test split, saves weights to `models/`, and safely updates `results/cnn_comparison_results.csv` **without modifying other models or Classical ML results**.

* **Teammate 1 (AlexNet)**:
  ```bash
  python run_cnn_alexnet.py --epochs 3 --device cpu
  ```
* **Teammate 2 (VGG-16)**:
  ```bash
  python run_cnn_vgg16.py --epochs 3 --device cpu
  ```
* **Teammate 3 (GoogLeNet)**:
  ```bash
  python run_cnn_googlenet.py --epochs 3 --device cpu
  ```
* **Teammate 4 (DenseNet-121)**:
  ```bash
  python run_cnn_densenet121.py --epochs 3 --device cpu
  ```
*(Note: If your machine has an NVIDIA GPU, add `--device cuda` for faster training)*

---

### Step 4: Generate Your 5 Side-by-Side Test Case Images
Once your training script finishes and creates your checkpoint in `models/`, run your test case generator. This produces 5 presentation-quality diagnostic images showing:
1. Ground Truth Target
2. Classical OpenCV Proposals
3. Your CNN Verified Output (Kept vs Filtered Out)
4. Bottom Inset Strip with individual crop confidence scores

* **Teammate 1 (AlexNet)**:
  ```bash
  python generate_alexnet_cases.py
  ```
  *(Outputs saved to `results/alexnet/`)*

* **Teammate 2 (VGG-16)**:
  ```bash
  python generate_vgg16_cases.py
  ```
  *(Outputs saved to `results/vgg16/`)*

* **Teammate 3 (GoogLeNet)**:
  ```bash
  python generate_googlenet_cases.py
  ```
  *(Outputs saved to `results/googlenet/`)*

* **Teammate 4 (DenseNet-121)**:
  ```bash
  python generate_densenet121_cases.py
  ```
  *(Outputs saved to `results/densenet121/`)*

---

## 💾 Summary of Outputs Produced

Each teammate's run produces:
1. **Model Weights**: `models/<model>_plate_classifier.pth`
2. **Updated Results Row**: Updates its row in `results/cnn_comparison_results.csv`.
3. **5 Visual Test Cases + Overview**: Saved in `results/<model>/`:
   - `case_1_standard_frontal.jpg`
   - `case_2_grille_elimination.jpg`
   - `case_3_multi_candidate_clutter.jpg`
   - `case_4_street_perspective.jpg`
   - `case_5_rear_bumper_shadows.jpg`
   - `<model>_all_5_cases_overview.jpg`

Existing Classical ML results (`results/ml_ablation_results.csv`), detection benchmarks (`results/detection_results.csv`), and frame-skipping tests (`results/efficiency_frame_skipping.csv`) remain preserved.

---

## 📤 Git Commands to Commit & Push Your Work

After completing Steps 3 and 4:

```bash
# 1. Check changed files
git status

# 2. Add your model checkpoint, comparison CSV, and your results folder
git add models/ results/cnn_comparison_results.csv results/<YOUR_MODEL_FOLDER>/

# 3. Commit with a clear message
git commit -m "feat(cnn): benchmark and test cases for <YOUR_MODEL_NAME>"

# 4. Pull and push
git pull origin main --rebase
git push origin main
```

---

## 📊 Regenerating Final Comparison Figures & Master Report
When all teammates have pushed their results:
```bash
python -c "from run_full_pipeline import generate_visual_figures; generate_visual_figures()"
```
This updates the overall comparison chart `results/figures/cnn_parameters_vs_fps.png` with everyone's finalized numbers.
