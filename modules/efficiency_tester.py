"""
Efficiency & Real-Time Benchmarking Module.
Tests Video Frame-Skipping Policies:
1. Every Frame (k=1, 100% processing)
2. Every 2nd Frame (k=2, 50% inference load)
3. Every 3rd Frame (k=3, 33.3% inference load)

Measures:
- Total Elapsed Time (s)
- Effective System FPS
- Average Latency per Frame (ms)
- Mean IoU against Ground Truth
- Detection Rate Retention
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.detector_huggingface import HuggingFacePlateDetector
from modules.detector_classical import ClassicalPlateDetector
from modules.evaluate_detection import compute_iou

def run_frame_skipping_benchmark(annotations, detector=None, skip_policies=[1, 2, 3]):
    """
    Simulates real-time sequential video processing with frame skipping.
    
    For independent images (non-video), we only evaluate detection on frames
    where inference actually runs, and report coverage (% of frames processed).
    This prevents misleading metrics from applying one image's boxes to another.
    """
    if detector is None:
        detector = HuggingFacePlateDetector()

    results = []

    for k in skip_policies:
        print(f"[*] Testing frame skipping policy: k={k} (Inference every {k} frame{'s' if k > 1 else ''})...")
        
        total_frames = len(annotations)
        inference_calls = 0
        all_ious = []
        tp_count = 0
        gt_on_inferred = 0  # GT boxes only on frames where inference ran

        t_start = time.perf_counter()

        for idx, item in enumerate(annotations):
            img_path = item["filepath"]
            gt_boxes = item.get("bboxes", [])

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Policy check: Only run and evaluate on inferred frames
            if idx % k == 0:
                inference_calls += 1
                gt_on_inferred += len(gt_boxes)
                res = detector.detect(img)
                pred_boxes = res[0] if isinstance(res, tuple) else res
                
                # Evaluate only on this inferred frame
                matched_gt = set()
                for p_box in pred_boxes:
                    best_iou = 0.0
                    best_g_idx = -1
                    for g_idx, g_box in enumerate(gt_boxes):
                        if g_idx in matched_gt:
                            continue
                        iou = compute_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_g_idx = g_idx
                    
                    if best_iou >= 0.5 and best_g_idx >= 0:
                        tp_count += 1
                        matched_gt.add(best_g_idx)
                        all_ious.append(best_iou)

        t_elapsed = time.perf_counter() - t_start
        fps = total_frames / (t_elapsed + 1e-6)
        avg_latency_ms = (t_elapsed * 1000.0) / total_frames
        mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
        detection_rate = tp_count / (gt_on_inferred + 1e-6)  # Rate on inferred frames only
        coverage = inference_calls / total_frames  # Fraction of frames actually processed

        results.append({
            "Policy": f"Every {k}{'st' if k==1 else ('nd' if k==2 else 'rd')} Frame (k={k})",
            "Inference Duty Cycle (%)": round(100.0 / k, 1),
            "Total Frames": total_frames,
            "Inference Calls": inference_calls,
            "Total Time (s)": round(float(t_elapsed), 3),
            "Effective FPS": round(float(fps), 1),
            "Latency/Frame (ms)": round(float(avg_latency_ms), 2),
            "Mean IoU": round(float(mean_iou), 4),
            "Detection Rate": round(float(detection_rate), 4),
            "Frame Coverage (%)": round(float(coverage * 100), 1)
        })

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    anno_path = "data/annotations_test.json"
    if not os.path.exists(anno_path):
        print("Run data/download_dataset.py first.")
        exit(1)

    with open(anno_path) as f:
        annos = json.load(f)[:40]

    detector = HuggingFacePlateDetector()
    df = run_frame_skipping_benchmark(annos, detector=detector)

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/efficiency_frame_skipping.csv", index=False)
    print("\n[OK] Results saved to 'results/efficiency_frame_skipping.csv'.")
    print("\n" + df.to_string(index=False))
