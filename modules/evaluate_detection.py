"""
Detection Evaluator: Computes IoU, Precision, Recall, F1, Detection Rate,
Inference Latency, and FPS for Classical vs. Pretrained Detectors.
"""

import os
import sys
import json
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def compute_iou(boxA, boxB):
    """
    Computes IoU between two boxes in [x, y, w, h] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] * boxA[3])
    areaB = max(0.0, boxB[2] * boxB[3])
    union_area = areaA + areaB - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def evaluate_detector_on_dataset(detector, annotations, iou_threshold=0.5, name="Detector"):
    """
    Evaluates a detector instance on a list of annotations.
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []
    latencies = []

    for item in annotations:
        img_path = item["filepath"]
        gt_boxes = item.get("bboxes", [])
        total_gt += len(gt_boxes)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Measure inference latency
        t0 = time.perf_counter()
        if hasattr(detector, "detect"):
            res = detector.detect(img)
            pred_boxes = res[0] if isinstance(res, tuple) else res
        else:
            pred_boxes = detector(img)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

        # Match predicted boxes to ground truth
        matched_gt = set()
        matched_pred = set()

        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= iou_threshold and best_g_idx >= 0:
                total_tp += 1
                matched_gt.add(best_g_idx)
                matched_pred.add(p_idx)
                all_ious.append(best_iou)

        # Unmatched predictions are False Positives
        total_fp += (len(pred_boxes) - len(matched_pred))
        # Unmatched ground truths are False Negatives
        total_fn += (len(gt_boxes) - len(matched_gt))

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    detection_rate = total_tp / (total_gt + 1e-6)
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    avg_latency = np.mean(latencies) if len(latencies) > 0 else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "method": name,
        "total_images": len(annotations),
        "total_gt": total_gt,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "detection_rate": round(float(detection_rate), 4),
        "mean_iou": round(float(mean_iou), 4),
        "avg_latency_ms": round(float(avg_latency), 2),
        "fps": round(float(fps), 1)
    }

if __name__ == "__main__":
    from modules.detector_classical import ClassicalPlateDetector
    from modules.detector_huggingface import HuggingFacePlateDetector

    anno_path = "data/annotations_test.json"
    if not os.path.exists(anno_path):
        print(f"Error: {anno_path} not found. Run data/download_dataset.py first.")
        exit(1)

    with open(anno_path, "r") as f:
        annos = json.load(f)[:30]  # Evaluate on first 30 images for fast testing

    print(f"[*] Evaluating Classical OpenCV vs Pretrained Hugging Face YOLO on {len(annos)} images...")

    classical = ClassicalPlateDetector()
    hf_detector = HuggingFacePlateDetector()

    res_classical = evaluate_detector_on_dataset(classical, annos, name="Classical OpenCV")
    res_hf = evaluate_detector_on_dataset(hf_detector, annos, name="Pretrained HF YOLOv8")

    print("\n" + "="*70)
    print(f"{'Metric':<25} | {'Classical OpenCV':<20} | {'Pretrained HF YOLOv8':<20}")
    print("="*70)
    for k in ["precision", "recall", "f1_score", "detection_rate", "mean_iou", "avg_latency_ms", "fps"]:
        print(f"{k:<25} | {res_classical[k]:<20} | {res_hf[k]:<20}")
    print("="*70)
