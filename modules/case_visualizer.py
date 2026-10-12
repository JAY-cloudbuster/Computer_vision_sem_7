"""
Shared Diagnostic Test Case Visualizer for CNN Architectures.
Provides:
- Model loading from checkpoint
- Multi-panel side-by-side rendering (Ground Truth, Classical Proposals, Model Verified Output)
- Bottom crop inset cards with confidence scores
- Output organization into results/<model_name>/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.cnn_individual_runner import build_model
from modules.detector_classical import ClassicalPlateDetector
from modules.evaluate_detection import compute_iou

def load_trained_model(arch_name, ckpt_path, device="cpu"):
    """Loads a trained model checkpoint for inference."""
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(arch_name, num_classes=2, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model

def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_length=8):
    """Draws a dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)

def add_header_banner(img, title, subtitle="", height=42, bg_color=(28, 30, 36)):
    """Creates a dark sleek banner above image."""
    w = img.shape[1]
    banner = np.full((height, w, 3), bg_color, dtype=np.uint8)
    cv2.putText(banner, title, (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(banner, subtitle, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 190, 205), 1, cv2.LINE_AA)
    return np.vstack([banner, img])

def build_crop_card(crop_img, pred_class, prob, is_tp, arch_name="CNN", card_w=240, card_h=105):
    """Builds an informative inset card for an individual crop."""
    card = np.full((card_h, card_w, 3), (32, 34, 42), dtype=np.uint8)
    thumb_size = 76
    h_c, w_c = crop_img.shape[:2]
    scale = min(thumb_size / w_c, thumb_size / h_c)
    nw, nh = max(1, int(w_c * scale)), max(1, int(h_c * scale))
    resized_thumb = cv2.resize(crop_img, (nw, nh))
    
    bx = 10 + (thumb_size - nw) // 2
    by = 10 + (thumb_size - nh) // 2
    border_color = (40, 200, 80) if pred_class == "PLATE" else (70, 70, 210)
    cv2.rectangle(card, (8, 8), (8 + thumb_size + 4, 8 + thumb_size + 4), border_color, 1)
    card[by:by+nh, bx:bx+nw] = resized_thumb

    tx = 8 + thumb_size + 14
    class_color = (80, 230, 110) if pred_class == "PLATE" else (100, 130, 255)
    status_color = (60, 220, 90) if is_tp else (120, 140, 240)
    
    cv2.putText(card, f"{arch_name}: {pred_class}", (tx, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, class_color, 1, cv2.LINE_AA)
    cv2.putText(card, f"Confidence: {prob*100:.1f}%", (tx, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
    status_label = "TP [KEPT]" if is_tp else "REJECTED FP"
    cv2.putText(card, f"Result: {status_label}", (tx, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)
    
    action = ">> ACCEPTED" if pred_class == "PLATE" else ">> FILTERED"
    badge_color = (40, 160, 60) if pred_class == "PLATE" else (60, 60, 160)
    cv2.putText(card, action, (tx, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.36, badge_color, 1, cv2.LINE_AA)
    
    return card

def render_case_panel(case_info, model, transform, arch_name, output_dir):
    """Renders one complete side-by-side test case demonstration."""
    img_path = case_info["filepath"]
    img = cv2.imread(img_path)
    if img is None:
        return None

    h_orig, w_orig = img.shape[:2]
    gt_boxes = case_info.get("bboxes", [])
    
    detector = ClassicalPlateDetector()
    c_boxes = detector.detect(img)

    candidates_eval = []
    for box in c_boxes:
        bx, by, bw, bh = [int(v) for v in box]
        bx, by = max(0, bx), max(0, by)
        bw, bh = min(bw, w_orig - bx), min(bh, h_orig - by)
        if bw < 5 or bh < 5:
            continue
        crop = img[by:by+bh, bx:bx+bw]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(cv2.resize(crop_rgb, (224, 224)))
        inp = transform(crop_pil).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(inp), dim=1).numpy()[0]
        
        prob_plate = float(probs[1])
        pred_label = "PLATE" if prob_plate >= 0.5 else "NON-PLATE"
        max_iou = max([compute_iou(box, g) for g in gt_boxes], default=0.0)
        is_tp = (pred_label == "PLATE") and (max_iou >= 0.45)
        
        candidates_eval.append({
            "box": box,
            "crop": crop,
            "prob_plate": prob_plate,
            "pred_label": pred_label,
            "max_iou": max_iou,
            "is_tp": is_tp
        })

    target_h = 320
    aspect = w_orig / h_orig
    target_w = int(target_h * aspect)
    sx = target_w / w_orig
    sy = target_h / h_orig

    img_resized = cv2.resize(img, (target_w, target_h))

    # --- PANEL 1: Ground Truth ---
    panel1 = img_resized.copy()
    for (gx, gy, gw, gh) in gt_boxes:
        cv2.rectangle(panel1, (int(gx*sx), int(gy*sy)), (int((gx+gw)*sx), int((gy+gh)*sy)), (40, 220, 60), 2)
        cv2.putText(panel1, "GT PLATE", (int(gx*sx), max(18, int(gy*sy) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 220, 60), 2, cv2.LINE_AA)
    panel1 = add_header_banner(panel1, "1. Ground Truth Target", f"{len(gt_boxes)} True License Plate", height=42, bg_color=(20, 35, 25))

    # --- PANEL 2: Classical Proposals ---
    panel2 = img_resized.copy()
    for cand in candidates_eval:
        cx, cy, cw, ch = [int(v) for v in cand["box"]]
        x1, y1 = int(cx*sx), int(cy*sy)
        x2, y2 = int((cx+cw)*sx), int((cy+ch)*sy)
        color = (50, 210, 80) if cand["max_iou"] >= 0.45 else (50, 50, 230)
        label = f"Prop (IoU:{cand['max_iou']:.2f})" if cand["max_iou"] >= 0.45 else "False Alarm"
        cv2.rectangle(panel2, (x1, y1), (x2, y2), color, 2)
        cv2.putText(panel2, label, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    panel2 = add_header_banner(panel2, "2. Classical Proposals (Raw)", f"{len(candidates_eval)} Candidates Generated", height=42, bg_color=(45, 25, 25))

    # --- PANEL 3: Neural Network Verified Output ---
    panel3 = img_resized.copy()
    n_kept = sum(1 for c in candidates_eval if c["pred_label"] == "PLATE")
    n_rej = sum(1 for c in candidates_eval if c["pred_label"] == "NON-PLATE")
    
    for cand in candidates_eval:
        cx, cy, cw, ch = [int(v) for v in cand["box"]]
        x1, y1 = int(cx*sx), int(cy*sy)
        x2, y2 = int((cx+cw)*sx), int((cy+ch)*sy)
        
        if cand["pred_label"] == "PLATE":
            cv2.rectangle(panel3, (x1, y1), (x2, y2), (30, 240, 70), 3)
            txt = f"{arch_name}: PLATE ({cand['prob_plate']*100:.1f}%) [KEPT]"
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
            cv2.rectangle(panel3, (x1, max(0, y1 - 22)), (x1 + tw + 8, y1), (20, 120, 40), -1)
            cv2.putText(panel3, txt, (x1 + 4, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            draw_dashed_rect(panel3, (x1, y1), (x2, y2), (60, 60, 210), thickness=2, dash_length=6)
            txt = f"Non-Plate ({(1-cand['prob_plate'])*100:.0f}%) [FILTERED]"
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
            cv2.rectangle(panel3, (x1, max(0, y1 - 18)), (x1 + tw + 6, y1), (40, 40, 80), -1)
            cv2.putText(panel3, txt, (x1 + 3, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 240), 1, cv2.LINE_AA)
                        
    panel3 = add_header_banner(panel3, f"3. {arch_name} Verified Output", f"{n_kept} Kept | {n_rej} Filtered Out", height=42, bg_color=(20, 40, 55))

    divider = np.full((panel1.shape[0], 3, 3), (80, 85, 95), dtype=np.uint8)
    top_row = np.hstack([panel1, divider, panel2, divider, panel3])

    cards = []
    for cand in candidates_eval[:6]:
        card = build_crop_card(
            cand["crop"],
            cand["pred_label"],
            cand["prob_plate"] if cand["pred_label"] == "PLATE" else (1.0 - cand["prob_plate"]),
            cand["is_tp"],
            arch_name=arch_name,
            card_w=240, card_h=105
        )
        cards.append(card)

    card_sep = np.full((105, 10, 3), (22, 24, 28), dtype=np.uint8)
    cards_row_items = []
    for i, c in enumerate(cards):
        cards_row_items.append(c)
        if i < len(cards) - 1:
            cards_row_items.append(card_sep)
            
    if cards_row_items:
        cards_strip = np.hstack(cards_row_items)
        if cards_strip.shape[1] < top_row.shape[1]:
            diff = top_row.shape[1] - cards_strip.shape[1]
            pad_l = np.full((cards_strip.shape[0], diff // 2, 3), (22, 24, 28), dtype=np.uint8)
            pad_r = np.full((cards_strip.shape[0], diff - diff // 2, 3), (22, 24, 28), dtype=np.uint8)
            cards_strip = np.hstack([pad_l, cards_strip, pad_r])
        elif cards_strip.shape[1] > top_row.shape[1]:
            cards_strip = cv2.resize(cards_strip, (top_row.shape[1], 105))
    else:
        cards_strip = np.full((105, top_row.shape[1], 3), (22, 24, 28), dtype=np.uint8)

    strip_title = np.full((32, top_row.shape[1], 3), (28, 30, 36), dtype=np.uint8)
    cv2.putText(strip_title, f"INDIVIDUAL PROPOSAL CROPS EVALUATED BY {arch_name.upper()} CLASSIFIER HEAD",
                (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 210, 255), 1, cv2.LINE_AA)

    summary_bar = np.full((36, top_row.shape[1], 3), (18, 20, 24), dtype=np.uint8)
    reduc_rate = (n_rej / len(candidates_eval) * 100) if candidates_eval else 0.0
    summary_txt = f"Test Case Metrics: Raw Candidates = {len(candidates_eval)} | Kept Plates = {n_kept} | Rejected False Alarms = {n_rej} | False Alarm Reduction = {reduc_rate:.1f}%"
    cv2.putText(summary_bar, summary_txt, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

    master_header = np.full((54, top_row.shape[1], 3), (16, 18, 22), dtype=np.uint8)
    cv2.putText(master_header, case_info["title"], (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(master_header, case_info["desc"], (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 180, 195), 1, cv2.LINE_AA)

    case_canvas = np.vstack([
        master_header,
        top_row,
        strip_title,
        cards_strip,
        summary_bar
    ])

    out_path = os.path.join(output_dir, case_info["save_name"])
    cv2.imwrite(out_path, case_canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [OK] Generated: {out_path}")
    return out_path, case_canvas

def run_case_demonstrations(arch_name, ckpt_path=None, output_dir=None, device="cpu"):
    """Main entrypoint for generating all 5 test cases for a given architecture."""
    if ckpt_path is None:
        ckpt_path = f"models/{arch_name.lower()}_plate_classifier.pth"
    if output_dir is None:
        output_dir = f"results/{arch_name.lower()}"
        
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print(f"\n{'='*75}")
        print(f"  [!] CHECKPOINT NOT FOUND: '{ckpt_path}'")
        print(f"  Please train {arch_name} first by running:")
        print(f"      python run_cnn_{arch_name.lower()}.py --epochs 3 --device {device}")
        print(f"{'='*75}\n")
        return False

    print(f"\n[*] Loading trained {arch_name} model from '{ckpt_path}' on {device.upper()}...")
    model = load_trained_model(arch_name, ckpt_path, device=device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    anno_path = "data/annotations_test.json"
    with open(anno_path) as f:
        test_ann = json.load(f)

    test_cases = [
        {
            "idx": 1,
            "title": f"CASE 1: Daytime Frontal View ({arch_name} Plate Verification)",
            "desc": f"Vehicle front view. Classical detector hits plate and bumper edge. {arch_name} verifies plate and suppresses bumper seam.",
            "save_name": f"{arch_name.lower()}_sample_case_1.jpg"
        },
        {
            "idx": 5,
            "title": f"CASE 2: Radiator Grille False-Alarm Elimination",
            "desc": f"Repetitive horizontal grille slits deceive edge detection into multiple candidates. {arch_name} suppresses non-plate grilles.",
            "save_name": f"{arch_name.lower()}_sample_case_2.jpg"
        },
        {
            "idx": 20,
            "title": f"CASE 3: Dense Multi-Candidate Clutter Filtration",
            "desc": f"Five raw proposals triggered across headlights and body seams. {arch_name} eliminates false positives, isolating legitimate plate.",
            "save_name": f"{arch_name.lower()}_sample_case_3.jpg"
        },
        {
            "idx": 43,
            "title": f"CASE 4: Wide Street Scene & Environmental Distractor Suppression",
            "desc": f"Candidates generated on road texture and car hood. {arch_name} suppresses environmental distractors.",
            "save_name": f"{arch_name.lower()}_sample_case_4.jpg"
        },
        {
            "idx": 38,
            "title": f"CASE 5: Rear Bumper Shadow & Tail-Light Clutter Suppression",
            "desc": f"Rear perspective with dark bumper recess and tail-lamp reflections. {arch_name} filters out reflections and confirms true plate.",
            "save_name": f"{arch_name.lower()}_sample_case_5.jpg"
        }
    ]

    generated_paths = []
    generated_canvases = []

    print(f"[*] Rendering 5 Test Cases for {arch_name} into '{output_dir}/'...")
    for c in test_cases:
        ann_item = test_ann[c["idx"]]
        c["filepath"] = ann_item["filepath"]
        c["bboxes"] = ann_item.get("bboxes", [])
        c["filename"] = ann_item["filename"]
        out_p, canvas = render_case_panel(c, model, transform, arch_name, output_dir)
        generated_paths.append(out_p)
        generated_canvases.append(canvas)

    if len(generated_canvases) == 5:
        max_w = max(c.shape[1] for c in generated_canvases)
        stacked_items = []
        for i, c in enumerate(generated_canvases):
            if c.shape[1] < max_w:
                pad = np.full((c.shape[0], max_w - c.shape[1], 3), (16, 18, 22), dtype=np.uint8)
                c = np.hstack([c, pad])
            stacked_items.append(c)
            if i < 4:
                sep = np.full((6, max_w, 3), (60, 70, 90), dtype=np.uint8)
                stacked_items.append(sep)
                
        overview = np.vstack(stacked_items)
        overview_path = os.path.join(output_dir, f"{arch_name.lower()}_all_5_cases_overview.jpg")
        cv2.imwrite(overview_path, overview, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  [OK] Generated Master Overview: {overview_path}")

    print(f"\n[SUCCESS] All 5 {arch_name} test cases saved to '{output_dir}/'!")
    return True
