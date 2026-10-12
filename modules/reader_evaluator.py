"""
=============================================================================
  DETECTION-TO-READER (ALPR) CORE EVALUATION MODULE
=============================================================================
Provides:
1. Plate Image Preprocessing (Bilateral filtering, CLAHE, deskewing & padding)
2. EasyOCR Engine Integration with Multi-Line Token Fusion
3. Syntax Normalization & Post-Processing (HSRP 'IND' badge removal, char disambiguation)
4. Standard Academic ALPR Metrics (Exact Match, Levenshtein NED, Character Accuracy)
5. CNN Quality-Gating Integration (ResNet-50, AlexNet, VGG16, GoogLeNet, DenseNet121)
6. Non-Destructive Results Benchmarking into results/reader_comparison_results.csv
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import re
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.cnn_individual_runner import build_model
from modules.case_visualizer import load_trained_model

# -----------------------------------------------------------------------------
# 1. Image Preprocessing for Optical Character Recognition
# -----------------------------------------------------------------------------

def preprocess_plate_for_ocr(crop_bgr, target_height=80):
    """
    Applies image enhancement to maximize OCR character legibility:
    - Maintains aspect ratio while scaling to optimal character height (~80px).
    - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    - Bilateral filtering suppresses plate noise and screw heads while preserving character edges.
    - Adds subtle border padding to prevent character boundary clipping.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, None

    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return None, None

    # 1. Scale maintaining aspect ratio
    scale = target_height / float(h)
    new_w = max(1, int(w * scale))
    resized = cv2.resize(crop_bgr, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE Contrast Normalization
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 4. Bilateral Filtering (Edge-preserving noise removal)
    denoised = cv2.bilateralFilter(enhanced_gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 5. Padding: 6px neutral border so edge characters don't get truncated by OCR
    padded_denoised = cv2.copyMakeBorder(
        denoised, 6, 6, 10, 10, cv2.BORDER_CONSTANT, value=255
    )
    padded_rgb = cv2.copyMakeBorder(
        resized, 6, 6, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    return padded_denoised, padded_rgb


# -----------------------------------------------------------------------------
# 2. Syntax Normalization & Post-Processing
# -----------------------------------------------------------------------------

CHAR_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1', 'T': '1',
    'Z': '2',
    'A': '4',
    'S': '5',
    'G': '6', 'b': '6',
    'B': '8',
}

DIGIT_TO_CHAR = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '8': 'B',
}

def clean_and_format_plate_text(raw_text):
    """
    Cleans raw OCR output and applies syntactic rules:
    - Removes non-alphanumeric noise (spaces, hyphens, punctuation).
    - Strips 'IND' country identifier common in Indian HSRP plates.
    - Disambiguates common OCR misclassifications (O vs 0, I vs 1, B vs 8).
    """
    if not raw_text:
        return ""

    # Uppercase and keep only alphanumeric
    clean = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

    # Remove HSRP "IND" badge if present at the start of longer text
    if clean.startswith("IND") and len(clean) > 7:
        clean = clean[3:]

    # Remove trailing/leading artifact single characters if total length is abnormally long
    if len(clean) > 10:
        # Standard plates are typically 8 to 10 characters (e.g. MH20CS9817, KA19TR02)
        # Check if first character is isolated artifact
        clean = clean[:10]

    # Contextual character position correction for standard Indian plates:
    # Pattern: [2 Letters: State] [1-2 Digits: RTO] [1-2 Letters: Series] [4 Digits: Unique]
    if 8 <= len(clean) <= 10:
        chars = list(clean)
        # Position 0 and 1 should be letters (State code)
        for i in [0, 1]:
            if chars[i] in DIGIT_TO_CHAR:
                chars[i] = DIGIT_TO_CHAR[chars[i]]

        # Last 4 positions should be digits
        for i in range(len(chars) - 4, len(chars)):
            if chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]

        clean = "".join(chars)

    return clean


# -----------------------------------------------------------------------------
# 3. Levenshtein Metrics & ALPR Performance Scores
# -----------------------------------------------------------------------------

def levenshtein_distance(s1, s2):
    """Computes exact dynamic-programming Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

def compute_alpr_metrics(pred_text, gt_text):
    """
    Computes all standard ALPR evaluation metrics:
    - Exact Match (1.0 or 0.0)
    - Normalized Edit Distance (NED: 0.0 to 1.0)
    - Character Recognition Accuracy (1 - CER)
    """
    pred_clean = clean_and_format_plate_text(pred_text)
    gt_clean = clean_and_format_plate_text(gt_text)

    is_exact = 1.0 if pred_clean == gt_clean else 0.0
    dist = levenshtein_distance(pred_clean, gt_clean)
    max_len = max(len(pred_clean), len(gt_clean), 1)
    ned = max(0.0, 1.0 - (dist / float(max_len)))

    gt_len = max(len(gt_clean), 1)
    char_acc = max(0.0, 1.0 - (dist / float(gt_len)))

    return {
        "pred_clean": pred_clean,
        "gt_clean": gt_clean,
        "exact_match": is_exact,
        "edit_distance": dist,
        "ned": ned,
        "char_accuracy": char_acc
    }


# -----------------------------------------------------------------------------
# 4. EasyOCR Reader Engine Singleton
# -----------------------------------------------------------------------------

_EASYOCR_READER = None

def get_easyocr_reader(device="cpu"):
    """Cached singleton reader instance to avoid reloading models into memory."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        use_gpu = (device == "cuda" and torch.cuda.is_available())
        print(f"[*] Initializing EasyOCR Reader (GPU={use_gpu})...", flush=True)
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        print("[+] EasyOCR Reader ready.", flush=True)
    return _EASYOCR_READER

def read_license_plate(crop_bgr, reader=None, device="cpu"):
    """
    Executes end-to-end OCR extraction on a cropped plate:
    1. Preprocesses crop with bilateral filter & CLAHE.
    2. Runs EasyOCR on both RGB and denoised variants to maximize detection.
    3. Merges multi-line boxes sorted vertically then horizontally.
    4. Cleans and normalizes syntax.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", "", 0.0, 0.0

    if reader is None:
        reader = get_easyocr_reader(device=device)

    start_t = time.perf_counter()
    padded_gray, padded_rgb = preprocess_plate_for_ocr(crop_bgr)
    if padded_rgb is None:
        return "", "", 0.0, 0.0

    # Run EasyOCR on enhanced RGB image
    raw_results = reader.readtext(padded_rgb)

    # Fallback to enhanced grayscale if nothing detected
    if len(raw_results) == 0 and padded_gray is not None:
        raw_results = reader.readtext(padded_gray)

    ocr_latency_ms = (time.perf_counter() - start_t) * 1000.0

    if len(raw_results) == 0:
        return "", "", 0.0, ocr_latency_ms

    # Sort text blocks top-to-bottom, left-to-right
    # raw_results element: (bbox, text, conf)
    sorted_blocks = sorted(raw_results, key=lambda b: (b[0][0][1], b[0][0][0]))
    raw_text = " ".join([b[1] for b in sorted_blocks])
    conf = float(np.mean([b[2] for b in sorted_blocks]))

    cleaned_text = clean_and_format_plate_text(raw_text)
    return raw_text, cleaned_text, conf, ocr_latency_ms


# -----------------------------------------------------------------------------
# 5. CNN Verification Gating & Pipeline Benchmark
# -----------------------------------------------------------------------------

def evaluate_reader_with_cnn(
    arch_name,
    ckpt_path,
    labels_csv_path="combined_dataset/labels.csv",
    images_dir="combined_dataset/images",
    max_samples=50,
    gate_threshold=0.5,
    device="cpu",
    save_csv=True,
    csv_out="results/reader_comparison_results.csv"
):
    """
    Evaluates the complete 2-stage Detection-to-Reader Pipeline:
    Stage 1 (Quality Gate): CNN Backbone (ResNet-50 / AlexNet / VGG / etc.) verifies plate candidate.
    Stage 2 (Recognition): EasyOCR extracts and cleans plate text on verified crops.
    """
    print(f"\n{'='*75}")
    print(f"  BENCHMARKING DETECTION-TO-READER PIPELINE: {arch_name.upper()}")
    print(f"{'='*75}")
    print(f"  Model Checkpoint : {ckpt_path}")
    print(f"  Evaluation Split : {labels_csv_path} (Max samples: {max_samples})")
    print(f"  CNN Gate Thresh  : {gate_threshold*100:.1f}%")
    print(f"  Inference Device : {device}")

    # 1. Load trained CNN Gate
    model = load_trained_model(arch_name, ckpt_path, device=device)
    if model is None:
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'! Please train {arch_name} first.")

    # Standard ImageNet transform for CNN classifier
    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load EasyOCR reader
    ocr_reader = get_easyocr_reader(device=device)

    # 3. Load dataset
    df = pd.read_csv(labels_csv_path)
    if max_samples and max_samples < len(df):
        df = df.iloc[:max_samples]

    total_samples = len(df)
    print(f"[+] Loaded {total_samples} samples for ALPR evaluation.")

    verified_plates = 0
    rejected_false_positives = 0
    exact_matches = 0
    total_ned = 0.0
    total_char_acc = 0.0
    cnn_latencies = []
    ocr_latencies = []
    total_latencies = []

    results_records = []

    for idx, row in df.iterrows():
        img_file = os.path.join(images_dir, str(row["filename"]))
        gt_text = str(row["plate_text"]).strip()
        xmin, ymin = int(row["xmin"]), int(row["ymin"])
        xmax, ymax = int(row["xmax"]), int(row["ymax"])

        img = cv2.imread(img_file)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        # Safe coordinate bounds
        xmin = max(0, min(xmin, w_img - 1))
        xmax = max(0, min(xmax, w_img))
        ymin = max(0, min(ymin, h_img - 1))
        ymax = max(0, min(ymax, h_img))

        plate_crop = img[ymin:ymax, xmin:xmax]
        if plate_crop.size == 0 or plate_crop.shape[0] < 5 or plate_crop.shape[1] < 5:
            continue

        # ----------------------------------------------------
        # Stage 1: CNN Verification Gate
        # ----------------------------------------------------
        t0 = time.perf_counter()
        crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        tensor_crop = cnn_transform(pil_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor_crop)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            plate_prob = float(probs[1])

        t_cnn_ms = (time.perf_counter() - t0) * 1000.0
        cnn_latencies.append(t_cnn_ms)

        # ----------------------------------------------------
        # Stage 2: OCR Reading (Gated by CNN confidence)
        # ----------------------------------------------------
        if plate_prob >= gate_threshold:
            verified_plates += 1
            raw_pred, clean_pred, ocr_conf, t_ocr_ms = read_license_plate(
                plate_crop, reader=ocr_reader, device=device
            )
            ocr_latencies.append(t_ocr_ms)
            total_latencies.append(t_cnn_ms + t_ocr_ms)

            metrics = compute_alpr_metrics(clean_pred, gt_text)
            exact_matches += int(metrics["exact_match"])
            total_ned += metrics["ned"]
            total_char_acc += metrics["char_accuracy"]

            results_records.append({
                "filename": row["filename"],
                "gt_text": metrics["gt_clean"],
                "pred_text": metrics["pred_clean"],
                "exact_match": int(metrics["exact_match"]),
                "ned": round(metrics["ned"], 4),
                "char_accuracy": round(metrics["char_accuracy"], 4),
                "cnn_conf": round(plate_prob, 4),
                "ocr_conf": round(ocr_conf, 4)
            })
        else:
            # Blocked by CNN Gate
            rejected_false_positives += 1
            results_records.append({
                "filename": row["filename"],
                "gt_text": clean_and_format_plate_text(gt_text),
                "pred_text": "[REJECTED_BY_CNN_GATE]",
                "exact_match": 0,
                "ned": 0.0,
                "char_accuracy": 0.0,
                "cnn_conf": round(plate_prob, 4),
                "ocr_conf": 0.0
            })

    # Summary Statistics
    num_eval = max(1, verified_plates)
    exact_acc_pct = (exact_matches / float(total_samples)) * 100.0
    avg_ned = total_ned / float(total_samples)
    avg_char_acc = (total_char_acc / float(total_samples)) * 100.0
    gate_pass_rate = (verified_plates / float(total_samples)) * 100.0

    avg_cnn_ms = float(np.mean(cnn_latencies)) if cnn_latencies else 0.0
    avg_ocr_ms = float(np.mean(ocr_latencies)) if ocr_latencies else 0.0
    avg_total_ms = float(np.mean(total_latencies)) if total_latencies else avg_cnn_ms
    end_to_end_fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

    summary_metrics = {
        "Architecture": arch_name,
        "Total Evaluated": total_samples,
        "Verified by Gate": verified_plates,
        "Gate Pass Rate (%)": round(gate_pass_rate, 2),
        "Exact Match (%)": round(exact_acc_pct, 2),
        "Avg NED (Similarity)": round(avg_ned, 4),
        "Character Acc (%)": round(avg_char_acc, 2),
        "CNN Latency (ms)": round(avg_cnn_ms, 2),
        "OCR Latency (ms)": round(avg_ocr_ms, 2),
        "End-to-End Latency (ms)": round(avg_total_ms, 2),
        "End-to-End FPS": round(end_to_end_fps, 2)
    }

    print(f"\n{'-'*75}")
    print(f"  RESULTS SUMMARY FOR {arch_name.upper()} READER PIPELINE")
    print(f"{'-'*75}")
    for k, v in summary_metrics.items():
        print(f"  {k:25}: {v}")
    print(f"{'-'*75}")

    if save_csv:
        update_reader_results_csv(summary_metrics, csv_path=csv_out)

    return summary_metrics, pd.DataFrame(results_records)


def update_reader_results_csv(metrics, csv_path="results/reader_comparison_results.csv"):
    """
    Safely updates results/reader_comparison_results.csv.
    Overwrites the architecture's row or appends if new, preserving all teammate records.
    """
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    arch = metrics["Architecture"]
    if not df.empty and "Architecture" in df.columns:
        if arch in df["Architecture"].values:
            idx = df.index[df["Architecture"] == arch].tolist()[0]
            for col, val in metrics.items():
                df.at[idx, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics])

    preferred_order = [
        "Architecture", "Total Evaluated", "Verified by Gate",
        "Gate Pass Rate (%)", "Exact Match (%)", "Avg NED (Similarity)",
        "Character Acc (%)", "CNN Latency (ms)", "OCR Latency (ms)",
        "End-to-End Latency (ms)", "End-to-End FPS"
    ]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df[cols]
    df.to_csv(csv_path, index=False)
    print(f"[OK] Updated master ALPR reader table saved to '{csv_path}'.")
    return df
