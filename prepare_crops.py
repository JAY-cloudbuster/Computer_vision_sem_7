import csv
import json
import os
import cv2
import time
import numpy as np
from modules.detector_classical import ClassicalPlateDetector
from modules.evaluate_detection import compute_iou


def prepare_crops(num_images=600, out_path="data/crop_dataset_224.npz"):
    """
    Builds the 224x224 crop dataset for plate vs non-plate binary classification.
    Uses combined_dataset/ if available, otherwise falls back to data/annotations_train.json.
    """
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    t0 = time.time()
    crops = []
    labels = []
    detector = ClassicalPlateDetector()

    combined_csv = "combined_dataset/labels.csv"
    combined_img_dir = "combined_dataset/images"

    if os.path.exists(combined_csv) and os.path.exists(combined_img_dir):
        print(f"[*] Extracting crops from '{combined_csv}' (max {num_images} images)...")
        with open(combined_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        
        if num_images:
            reader = reader[:num_images]

        for row in reader:
            fn = row["filename"]
            fp = os.path.join(combined_img_dir, fn)
            img = cv2.imread(fp)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            try:
                xmin = max(0, min(int(float(row["xmin"])), w_img - 1))
                ymin = max(0, min(int(float(row["ymin"])), h_img - 1))
                xmax = max(xmin + 5, min(int(float(row["xmax"])), w_img))
                ymax = max(ymin + 5, min(int(float(row["ymax"])), h_img))
            except Exception:
                continue

            # 1. Positive crop (True License Plate)
            pos_crop = img[ymin:ymax, xmin:xmax]
            if pos_crop.size > 0 and pos_crop.shape[0] >= 5 and pos_crop.shape[1] >= 5:
                crops.append(cv2.resize(pos_crop, (224, 224)))
                labels.append(1)

            # 2. Negative crops (Non-plate candidates from Classical detector)
            gt_box = [xmin, ymin, xmax - xmin, ymax - ymin]
            c_boxes = detector.detect(img)
            neg_found = False
            for cand in c_boxes:
                iou = compute_iou(cand, gt_box)
                if iou < 0.10:
                    cx, cy, cw, ch = [int(v) for v in cand]
                    cx = max(0, min(cx, w_img - 1))
                    cy = max(0, min(cy, h_img - 1))
                    cw = max(5, min(cw, w_img - cx))
                    ch = max(5, min(ch, h_img - cy))
                    neg_crop = img[cy:cy+ch, cx:cx+cw]
                    if neg_crop.size > 0:
                        crops.append(cv2.resize(neg_crop, (224, 224)))
                        labels.append(0)
                        neg_found = True
                        break  # 1 negative per image for balance

            # Fallback negative patch (e.g. car hood/bumper region above or below plate)
            if not neg_found:
                bw = xmax - xmin
                bh = ymax - ymin
                # Sample background region
                sy = max(0, ymin - bh - 10) if ymin > bh + 10 else min(h_img - bh, ymax + 10)
                sx = max(0, min(xmin, w_img - bw))
                neg_patch = img[sy:sy+bh, sx:sx+bw]
                if neg_patch.size > 0 and neg_patch.shape[0] >= 5 and neg_patch.shape[1] >= 5:
                    crops.append(cv2.resize(neg_patch, (224, 224)))
                    labels.append(0)

    print(f"[+] Extracted {len(crops)} crops ({sum(labels)} positive plates, {len(labels)-sum(labels)} negative non-plates).")
    print("[*] Saving compressed cache to:", out_path)
    resized_arr = np.array(crops, dtype=np.uint8)
    labels_arr = np.array(labels, dtype=np.int64)
    np.savez_compressed(out_path, images=resized_arr, labels=labels_arr)
    file_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[OK] Saved {len(resized_arr)} crops to '{out_path}' in {time.time()-t0:.2f}s (Cache size: {file_mb:.2f} MB).")
    return out_path


if __name__ == "__main__":
    prepare_crops()

