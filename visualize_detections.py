"""
Side-by-Side Detection Visualization for 5 Sample Images.
Shows: Original + GT | Classical OpenCV | Pretrained HF YOLO
for 5 diverse samples from the test set.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from modules.detector_classical import ClassicalPlateDetector
from modules.detector_huggingface import HuggingFacePlateDetector
from modules.evaluate_detection import compute_iou

def draw_gt_boxes(image, gt_boxes, color=(0, 255, 0), thickness=2):
    """Draw ground truth boxes in green."""
    annotated = image.copy()
    for (x, y, w, h) in gt_boxes:
        cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
        cv2.putText(annotated, "GT", (int(x), max(15, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated

def draw_classical_boxes(image, boxes, gt_boxes, iou_thresh=0.5, thickness=2):
    """Draw classical detector boxes: green=TP, red=FP."""
    annotated = image.copy()
    matched_gt = set()
    
    for box in boxes:
        best_iou = 0.0
        best_g = -1
        for gi, g in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(box, g)
            if iou > best_iou:
                best_iou = iou
                best_g = gi
        
        x, y, w, h = [int(v) for v in box]
        if best_iou >= iou_thresh and best_g >= 0:
            color = (0, 200, 0)  # Green = TP
            matched_gt.add(best_g)
            label = f"TP IoU:{best_iou:.2f}"
        else:
            color = (0, 0, 220)  # Red = FP
            label = "FP"
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(annotated, label, (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return annotated

def draw_hf_boxes(image, boxes, confs, gt_boxes, iou_thresh=0.5, thickness=2):
    """Draw HF YOLO boxes: green=TP, red=FP, with confidence."""
    annotated = image.copy()
    matched_gt = set()
    
    for i, box in enumerate(boxes):
        best_iou = 0.0
        best_g = -1
        for gi, g in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(box, g)
            if iou > best_iou:
                best_iou = iou
                best_g = gi
        
        x, y, w, h = [int(v) for v in box]
        conf = confs[i] if i < len(confs) else 0.0
        if best_iou >= iou_thresh and best_g >= 0:
            color = (0, 200, 0)  # Green = TP
            matched_gt.add(best_g)
            label = f"TP {conf:.2f} IoU:{best_iou:.2f}"
        else:
            color = (0, 0, 220)  # Red = FP
            label = f"FP {conf:.2f}"
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(annotated, label, (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return annotated

def add_title_bar(image, title, bar_height=32, bg_color=(40, 40, 40), text_color=(255, 255, 255)):
    """Add a dark title bar above the image."""
    h, w = image.shape[:2]
    bar = np.full((bar_height, w, 3), bg_color, dtype=np.uint8)
    font_scale = 0.55
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (bar_height + text_size[1]) // 2
    cv2.putText(bar, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
    return np.vstack([bar, image])

def generate_side_by_side(annotations, output_dir="results/figures", num_samples=5):
    """Generate num_samples side-by-side detection comparison images."""
    os.makedirs(output_dir, exist_ok=True)
    
    classical = ClassicalPlateDetector()
    hf_detector = HuggingFacePlateDetector()
    
    # Select diverse samples: pick images with varying GT box counts and positions
    valid_annos = [a for a in annotations if os.path.exists(a["filepath"])]
    
    # Pick evenly spaced samples for diversity
    step = max(1, len(valid_annos) // num_samples)
    selected = [valid_annos[i * step] for i in range(min(num_samples, len(valid_annos)))]
    
    print(f"[*] Generating {len(selected)} side-by-side detection visualizations...")
    
    individual_paths = []
    
    for sample_idx, item in enumerate(selected):
        img = cv2.imread(item["filepath"])
        if img is None:
            continue
        
        gt_boxes = item.get("bboxes", [])
        
        # Detect with both models
        c_boxes = classical.detect(img)
        hf_boxes, hf_confs = hf_detector.detect(img)
        
        # Create 3 panels
        target_h = 280
        aspect = img.shape[1] / img.shape[0]
        target_w = int(target_h * aspect)
        
        img_resized = cv2.resize(img, (target_w, target_h))
        
        # Scale GT boxes for resized image
        sx = target_w / img.shape[1]
        sy = target_h / img.shape[0]
        gt_scaled = [[x*sx, y*sy, w*sx, h*sy] for x, y, w, h in gt_boxes]
        c_scaled = [[x*sx, y*sy, w*sx, h*sy] for x, y, w, h in c_boxes]
        hf_scaled = [[x*sx, y*sy, w*sx, h*sy] for x, y, w, h in hf_boxes]
        
        # Panel 1: Original + GT (green)
        panel_gt = draw_gt_boxes(img_resized, gt_scaled)
        panel_gt = add_title_bar(panel_gt, f"Ground Truth ({len(gt_boxes)} plate{'s' if len(gt_boxes)!=1 else ''})")
        
        # Panel 2: Classical OpenCV (green=TP, red=FP)
        panel_classical = draw_classical_boxes(img_resized, c_scaled, gt_scaled)
        n_c_tp = sum(1 for b in c_boxes for gi, g in enumerate(gt_boxes) if compute_iou(b, g) >= 0.5)
        panel_classical = add_title_bar(panel_classical, f"Classical OpenCV ({len(c_boxes)} det, green=TP, red=FP)")
        
        # Panel 3: HF YOLO (green=TP, red=FP, with conf)
        panel_hf = draw_hf_boxes(img_resized, hf_scaled, hf_confs, gt_scaled)
        panel_hf = add_title_bar(panel_hf, f"HF YOLOv8 ({len(hf_boxes)} det, green=TP, red=FP)")
        
        # Ensure all panels same height
        max_h = max(p.shape[0] for p in [panel_gt, panel_classical, panel_hf])
        panels = []
        for p in [panel_gt, panel_classical, panel_hf]:
            if p.shape[0] < max_h:
                pad = np.full((max_h - p.shape[0], p.shape[1], 3), 30, dtype=np.uint8)
                p = np.vstack([p, pad])
            panels.append(p)
        
        # Add thin separator between panels
        sep = np.full((max_h, 3, 3), 100, dtype=np.uint8)
        combined = np.hstack([panels[0], sep, panels[1], sep, panels[2]])
        
        # Add sample header
        header_h = 36
        header = np.full((header_h, combined.shape[1], 3), (25, 25, 25), dtype=np.uint8)
        title = f"Sample {sample_idx + 1}: {item['filename']} ({item.get('width', '?')}x{item.get('height', '?')})"
        cv2.putText(header, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        combined = np.vstack([header, combined])
        
        out_path = os.path.join(output_dir, f"side_by_side_sample_{sample_idx + 1}.jpg")
        cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
        individual_paths.append(out_path)
        print(f"  [OK] Saved {out_path}")
    
    # Also create a single stacked overview of all 5
    if len(individual_paths) > 0:
        all_imgs = [cv2.imread(p) for p in individual_paths if os.path.exists(p)]
        if len(all_imgs) > 0:
            # Normalize widths
            max_w = max(im.shape[1] for im in all_imgs)
            padded = []
            for im in all_imgs:
                if im.shape[1] < max_w:
                    pad = np.full((im.shape[0], max_w - im.shape[1], 3), 30, dtype=np.uint8)
                    im = np.hstack([im, pad])
                padded.append(im)
                # Thin separator between samples
                padded.append(np.full((2, max_w, 3), 80, dtype=np.uint8))
            
            stacked = np.vstack(padded[:-1])  # Remove last separator
            overview_path = os.path.join(output_dir, "side_by_side_all_5_samples.jpg")
            cv2.imwrite(overview_path, stacked, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  [OK] Saved combined overview: {overview_path}")
    
    return individual_paths

if __name__ == "__main__":
    anno_path = "data/annotations_test.json"
    if not os.path.exists(anno_path):
        print("Run data/download_dataset.py first.")
        exit(1)
    
    with open(anno_path) as f:
        annos = json.load(f)
    
    generate_side_by_side(annos, num_samples=5)
    print("\n[OK] All side-by-side visualizations generated.")
