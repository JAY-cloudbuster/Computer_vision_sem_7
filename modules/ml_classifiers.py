"""
Classical Machine Learning Module for License Plate vs Non-Plate Verification.
Evaluates 5 Classifiers:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest
5. XGBoost

Performs Automated Feature Ablation Analysis across:
- Shape Only (4 features)
- Texture Only (3 features)
- Color Only (3 features)
- Shape + Texture (7 features)
- All 10 Features Combined
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.feature_extractor import PlateFeatureExtractor, FEATURE_NAMES
from modules.detector_classical import ClassicalPlateDetector
from modules.evaluate_detection import compute_iou

FEATURE_SETS = {
    "Shape Only (4)": [0, 1, 2, 3],
    "Texture Only (3)": [4, 5, 6],
    "Color Only (3)": [7, 8, 9],
    "Shape + Texture (7)": [0, 1, 2, 3, 4, 5, 6],
    "All 10 Features": list(range(10))
}

def build_crop_dataset(annotations, max_images=None):
    """
    Extracts positive plate crops and negative false-positive proposals from images.
    Returns:
        crops: list of BGR images
        labels: list of int (1 = plate, 0 = non-plate)
    """
    if max_images is not None:
        annotations = annotations[:max_images]

    detector = ClassicalPlateDetector()
    crops = []
    labels = []

    for item in annotations:
        img = cv2.imread(item["filepath"])
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        gt_boxes = item.get("bboxes", [])

        # 1. Positive crops: Ground-truth plates
        for bx, by, bw, bh in gt_boxes:
            bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)
            bx = max(0, min(bx, w_img - 1))
            by = max(0, min(by, h_img - 1))
            bw = max(1, min(bw, w_img - bx))
            bh = max(1, min(bh, h_img - by))
            crop = img[by:by+bh, bx:bx+bw]
            if crop.size > 0 and crop.shape[0] >= 5 and crop.shape[1] >= 5:
                crops.append(crop)
                labels.append(1)

        # 2. Negative crops: Candidate boxes from Classical detector that have IoU < 0.15 with any GT plate
        c_boxes = detector.detect(img)
        for cand in c_boxes:
            cx, cy, cw, ch = [int(v) for v in cand]
            max_iou = 0.0
            for g in gt_boxes:
                iou = compute_iou(cand, g)
                if iou > max_iou:
                    max_iou = iou

            if max_iou < 0.15:  # Pure negative proposal (grille, headlight, road noise)
                cx = max(0, min(cx, w_img - 1))
                cy = max(0, min(cy, h_img - 1))
                cw = max(1, min(cw, w_img - cx))
                ch = max(1, min(ch, h_img - cy))
                neg_crop = img[cy:cy+ch, cx:cx+cw]
                if neg_crop.size > 0 and neg_crop.shape[0] >= 5 and neg_crop.shape[1] >= 5:
                    crops.append(neg_crop)
                    labels.append(0)

    print(f"[+] Built crop dataset: {sum(labels)} Positives (Plates), {len(labels) - sum(labels)} Negatives (Non-Plates).")
    return crops, np.array(labels, dtype=np.int32)

def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42)
    }

def run_ml_ablation_benchmark(X, y, test_size=0.25, random_state=42):
    """
    Evaluates all 5 classifiers across all feature subsets.
    Uses 5-fold stratified cross-validation for robust evaluation.
    Returns: DataFrame containing performance metrics.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    
    results = []
    
    # Use 5-fold stratified CV for robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for feat_set_name, feat_indices in FEATURE_SETS.items():
        # Select feature subset
        X_sub = X[:, feat_indices]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)

        classifiers = get_classifiers()
        for clf_name, clf in classifiers.items():
            # Measure train time on full data
            t0 = time.perf_counter()
            clf.fit(X_scaled, y)
            train_time_ms = (time.perf_counter() - t0) * 1000.0

            # Cross-validated predictions for robust evaluation
            cv_preds = cross_val_predict(
                get_classifiers()[clf_name], X_scaled, y, cv=cv
            )

            # Measure inference latency
            t0 = time.perf_counter()
            _ = clf.predict(X_scaled[:10])
            infer_time_ms = ((time.perf_counter() - t0) * 1000.0) / 10

            acc = accuracy_score(y, cv_preds)
            prec = precision_score(y, cv_preds, zero_division=0)
            rec = recall_score(y, cv_preds, zero_division=0)
            f1 = f1_score(y, cv_preds, zero_division=0)

            results.append({
                "Feature Subset": feat_set_name,
                "Num Features": len(feat_indices),
                "Algorithm": clf_name,
                "Accuracy": round(float(acc), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1-Score": round(float(f1), 4),
                "Train Time (ms)": round(float(train_time_ms), 2),
                "Latency/Sample (ms)": round(float(infer_time_ms), 4)
            })

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    anno_path = "data/annotations_test.json"
    if not os.path.exists(anno_path):
        print("Run data/download_dataset.py first.")
        exit(1)

    with open(anno_path) as f:
        annos = json.load(f)

    print("[*] Generating plate & non-plate crops from annotations...")
    crops, labels = build_crop_dataset(annos, max_images=50)

    print("[*] Extracting 10 handcrafted features for all crops...")
    extractor = PlateFeatureExtractor()
    X = extractor.extract_batch(crops)
    y = labels

    print("[*] Running 5 ML Algorithms across 5 Feature Combinations (25 runs)...")
    results_df = run_ml_ablation_benchmark(X, y)

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/ml_ablation_results.csv", index=False)
    print("\n[OK] Results saved to 'results/ml_ablation_results.csv'.")

    # Display Top 10 Configurations by F1-Score
    print("\nTop 10 Configurations by F1-Score:")
    top10 = results_df.sort_values(by="F1-Score", ascending=False).head(10)
    print(top10[["Feature Subset", "Algorithm", "Accuracy", "F1-Score", "Train Time (ms)"]].to_string(index=False))
