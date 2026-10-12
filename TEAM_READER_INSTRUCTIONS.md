# End-to-End ALPR Benchmark - Team Reader Runner Guide

This document provides step-by-step instructions for each team member to:
1. Run the end-to-end **Detection-to-Reader (ALPR)** pipeline with their assigned CNN backbone.
2. Filter candidate proposals through their CNN Quality Gate.
3. Perform Optical Character Recognition (OCR) with **EasyOCR** and syntax post-processing.
4. Generate 5 diagnostic side-by-side test case panels + master overview graphic.
5. Push their ALPR results and graphics to GitHub without overwriting teammates' work.

---

## 👥 Architecture & Script Assignment

| Member | Assigned CNN Backbone | 1. Train CNN (If not done) | 2. Run ALPR Reader Pipeline | 3. Generate Diagnostic Cases | Results Folder |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Satvic (Done)** | **ResNet-50** | `python run_cnn_resnet50.py` | `python run_reader_resnet50.py` | `python generate_resnet50_reader_cases.py` | `results/resnet50_reader/` |
| **Teammate 1** | **AlexNet** | `python run_cnn_alexnet.py` | `python run_reader_alexnet.py` | `python generate_alexnet_reader_cases.py` | `results/alexnet_reader/` |
| **Teammate 2** | **VGG-16** | `python run_cnn_vgg16.py` | `python run_reader_vgg16.py` | `python generate_vgg16_reader_cases.py` | `results/vgg16_reader/` |
| **Teammate 3** | **GoogLeNet** | `python run_cnn_googlenet.py` | `python run_reader_googlenet.py` | `python generate_googlenet_reader_cases.py` | `results/googlenet_reader/` |
| **Teammate 4** | **DenseNet-121** | `python run_cnn_densenet121.py` | `python run_reader_densenet121.py` | `python generate_densenet121_reader_cases.py` | `results/densenet121_reader/` |

---

## 🚀 Step-by-Step Instructions for Teammates

### Step 1: Pull Latest Repository
Ensure your branch has the latest reader modules and pre-cached dataset:
```bash
git pull origin main
```

### Step 2: Environment Check
Ensure EasyOCR and PyTorch are installed:
```bash
pip install easyocr torch torchvision opencv-python pandas
```

---

### Step 3: Run Your ALPR Reader Pipeline

Each runner loads your trained model from `models/<arch>_plate_classifier.pth`, runs the 2-stage Quality-Gate + OCR pipeline, and safely updates `results/reader_comparison_results.csv` without modifying other members' records.

*(Note: If you have not trained your CNN yet, run `python run_cnn_<your_model>.py --epochs 3` first!)*

* **Teammate 1 (AlexNet)**:
  ```bash
  python run_reader_alexnet.py --samples 40 --device cpu
  ```
* **Teammate 2 (VGG-16)**:
  ```bash
  python run_reader_vgg16.py --samples 40 --device cpu
  ```
* **Teammate 3 (GoogLeNet)**:
  ```bash
  python run_reader_googlenet.py --samples 40 --device cpu
  ```
* **Teammate 4 (DenseNet-121)**:
  ```bash
  python run_reader_densenet121.py --samples 40 --device cpu
  ```
*(If you have an NVIDIA GPU, add `--device cuda` for faster inference)*

---

### Step 4: Generate Your 5 Diagnostic Test Case Graphics

This script produces 5 presentation-quality diagnostic images showing:
1. **Scene Panel**: Full vehicle scene with detection box & your CNN gate probability score.
2. **Crop & Enhancement Panel**: Raw plate crop vs CLAHE + Bilateral filtered crop.
3. **ALPR Result Card**: Ground Truth text, Predicted OCR text, character-level visual alignment diff (green for matches, red for discrepancies), Levenshtein NED, and latency breakdown.
4. **Master Overview**: All 5 cases stitched into a single high-resolution comparison chart.

* **Teammate 1 (AlexNet)**:
  ```bash
  python generate_alexnet_reader_cases.py
  ```
  *(Outputs saved to `results/alexnet_reader/`)*

* **Teammate 2 (VGG-16)**:
  ```bash
  python generate_vgg16_reader_cases.py
  ```
  *(Outputs saved to `results/vgg16_reader/`)*

* **Teammate 3 (GoogLeNet)**:
  ```bash
  python generate_googlenet_reader_cases.py
  ```
  *(Outputs saved to `results/googlenet_reader/`)*

* **Teammate 4 (DenseNet-121)**:
  ```bash
  python generate_densenet121_reader_cases.py
  ```
  *(Outputs saved to `results/densenet121_reader/`)*

---

## 💾 Summary of Outputs Produced

Each teammate's run produces:
1. **Updated Row in Master CSV**: `results/reader_comparison_results.csv`
2. **5 Diagnostic Test Cases + Overview Graphic** in `results/<your_model>_reader/`:
   - `case_1_karnataka_frontal.jpg` (KA19TR02)
   - `case_2_maharashtra_sedan.jpg` (MH20CS9817)
   - `case_3_kerala_compact.jpg` (KL05AK3300)
   - `case_4_maharashtra_urban.jpg` (MH12JC2813)
   - `case_5_uttarpradesh_highres.jpg` (UP50AS4535)
   - `<your_model>_all_5_reader_cases_overview.jpg`

---

## 📤 Git Commands to Commit & Push Your Work

After completing Steps 3 and 4:

```bash
# 1. Check changed files
git status

# 2. Add your reader results folder and the comparison CSV
git add results/reader_comparison_results.csv results/<YOUR_MODEL>_reader/

# 3. Commit with a clear message
git commit -m "feat(reader): ALPR benchmark and test cases for <YOUR_MODEL_NAME>"

# 4. Pull and push
git pull origin main --rebase
git push origin main
```
