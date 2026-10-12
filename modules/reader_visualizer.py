"""
=============================================================================
  DIAGNOSTIC TEST CASE VISUALIZER FOR ALPR (DETECTION-TO-READER)
=============================================================================
Produces presentation-ready diagnostic panels displaying:
1. Full vehicle scene with candidate detection and CNN verification badge
2. High-res raw plate crop vs. CLAHE/Bilateral preprocessed plate crop
3. Comprehensive Reader Result Card:
   - Ground Truth vs. Predicted text
   - Character-level alignment with visual match highlighting
   - Levenshtein NED & Character Accuracy
   - End-to-end latency breakdown (CNN Gate ms + OCR ms)
4. Master 5-case composite overview graphic
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.case_visualizer import load_trained_model
from modules.reader_evaluator import (
    preprocess_plate_for_ocr,
    read_license_plate,
    compute_alpr_metrics,
    clean_and_format_plate_text,
    get_easyocr_reader
)

# 5 Defined Benchmark Diagnostic Test Cases
READER_TEST_CASES = [
    {
        "case_id": 1,
        "name": "case_1_karnataka_frontal",
        "title": "Case 1: Standard Frontal View (Clean Plate Reading)",
        "subtitle": "High contrast registration plate with standard font typography",
        "filename": "image_1.jpeg",
        "gt_text": "KA19TR02",
        "bbox": [140, 210, 339, 260] # xmin, ymin, xmax, ymax
    },
    {
        "case_id": 2,
        "name": "case_2_maharashtra_sedan",
        "title": "Case 2: Sedan Lower Bumper Perspective",
        "subtitle": "Standard 10-character Indian plate format (State-RTO-Series-Number)",
        "filename": "image_4.jpeg",
        "gt_text": "MH20CS9817",
        "bbox": [185, 290, 374, 339]
    },
    {
        "case_id": 3,
        "name": "case_3_kerala_compact",
        "title": "Case 3: Compact Hatchback Front Angle",
        "subtitle": "Tight bounding box with shadow gradients across plate surface",
        "filename": "image_5.jpeg",
        "gt_text": "KL05AK3300",
        "bbox": [335, 313, 453, 347]
    },
    {
        "case_id": 4,
        "name": "case_4_maharashtra_urban",
        "title": "Case 4: Urban Traffic Natural Environment",
        "subtitle": "Direct natural lighting with surrounding bumper clutter",
        "filename": "image_9.jpeg",
        "gt_text": "MH12JC2813",
        "bbox": [158, 204, 365, 254]
    },
    {
        "case_id": 5,
        "name": "case_5_uttarpradesh_highres",
        "title": "Case 5: High Resolution SUV Radiator View",
        "subtitle": "High-definition plate test under challenging grille proximity",
        "filename": "image_11.jpg",
        "gt_text": "UP50AS4535",
        "bbox": [398, 427, 617, 487]
    }
]

def add_header_banner(img, title, subtitle="", height=44, bg_color=(24, 26, 32)):
    """Creates a dark sleek banner above image."""
    w = img.shape[1]
    banner = np.full((height, w, 3), bg_color, dtype=np.uint8)
    cv2.putText(banner, title, (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(banner, subtitle, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 180, 195), 1, cv2.LINE_AA)
    return np.vstack([banner, img])

def build_reader_card(
    arch_name,
    cnn_prob,
    gt_text,
    pred_text,
    metrics,
    t_cnn_ms,
    t_ocr_ms,
    card_w=420,
    card_h=340
):
    """
    Builds a presentation-quality information card summarizing OCR recognition.
    """
    card = np.full((card_h, card_w, 3), (28, 30, 38), dtype=np.uint8)

    # 1. Header ribbon
    is_exact = bool(metrics["exact_match"])
    header_color = (36, 140, 60) if is_exact else ((20, 100, 180) if metrics["ned"] >= 0.7 else (50, 50, 160))
    cv2.rectangle(card, (0, 0), (card_w, 34), header_color, -1)
    status_str = "EXACT MATCH - 100%" if is_exact else (f"HIGH SIMILARITY ({metrics['ned']*100:.0f}%)" if metrics["ned"] >= 0.7 else "PARTIAL MATCH")
    cv2.putText(card, f"ALPR RESULT: {status_str}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Gate Verification Details
    y = 58
    cv2.putText(card, f"CNN Gate Backbone : {arch_name}", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1, cv2.LINE_AA)
    y += 22
    gate_status = "ACCEPTED [PLATE]" if cnn_prob >= 0.5 else "REJECTED [NON-PLATE]"
    gate_col = (70, 215, 100) if cnn_prob >= 0.5 else (80, 80, 220)
    cv2.putText(card, f"Gate Confidence  : {cnn_prob*100:.1f}% ({gate_status})", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, gate_col, 1, cv2.LINE_AA)

    # Divider
    y += 14
    cv2.line(card, (14, y), (card_w - 14, y), (50, 55, 68), 1)

    # 3. Ground Truth vs Predicted Text Blocks
    y += 26
    cv2.putText(card, "Ground Truth :", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 185), 1, cv2.LINE_AA)
    cv2.putText(card, gt_text, (140, y), cv2.FONT_HERSHEY_DUPLEX, 0.58, (80, 220, 240), 1, cv2.LINE_AA)

    y += 30
    cv2.putText(card, "Predicted OCR :", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 185), 1, cv2.LINE_AA)
    pred_display = pred_text if pred_text else "[NO TEXT]"
    pred_col = (80, 240, 110) if is_exact else (255, 255, 255)
    cv2.putText(card, pred_display, (140, y), cv2.FONT_HERSHEY_DUPLEX, 0.58, pred_col, 1, cv2.LINE_AA)

    # Divider
    y += 18
    cv2.line(card, (14, y), (card_w - 14, y), (50, 55, 68), 1)

    # 4. Character Alignment / Visual Diff Bar
    y += 24
    cv2.putText(card, "Character Alignment Diff:", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 190, 205), 1, cv2.LINE_AA)
    y += 22

    # Render character comparison tiles
    start_x = 16
    char_w = 26
    char_h = 24
    max_chars = min(12, max(len(gt_text), len(pred_text)))
    for c_i in range(max_chars):
        c_gt = gt_text[c_i] if c_i < len(gt_text) else "-"
        c_pred = pred_text[c_i] if c_i < len(pred_text) else "-"
        match = (c_gt == c_pred)

        box_col = (35, 130, 55) if match else (40, 45, 140)
        bx = start_x + c_i * (char_w + 6)
        cv2.rectangle(card, (bx, y), (bx + char_w, y + char_h), box_col, -1)
        cv2.rectangle(card, (bx, y), (bx + char_w, y + char_h), (80, 90, 110), 1)

        # Character symbol
        c_disp = c_pred if c_pred != "-" else c_gt
        cv2.putText(card, c_disp, (bx + 7, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Performance Metrics
    y += 38
    cv2.putText(card, f"Levenshtein Distance : {metrics['edit_distance']}", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 210, 225), 1, cv2.LINE_AA)
    cv2.putText(card, f"Normalized Similarity : {metrics['ned']*100:.1f}%", (210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 220, 240), 1, cv2.LINE_AA)

    y += 22
    cv2.putText(card, f"Character Accuracy   : {metrics['char_accuracy']*100:.1f}%", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 240, 110), 1, cv2.LINE_AA)
    total_ms = t_cnn_ms + t_ocr_ms
    cv2.putText(card, f"Latency: Gate {t_cnn_ms:.0f}ms | OCR {t_ocr_ms:.0f}ms | Total {total_ms:.0f}ms", (14, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 170, 185), 1, cv2.LINE_AA)

    return card

def render_reader_case_panel(
    case_info,
    model,
    ocr_reader,
    arch_name,
    cnn_transform,
    device="cpu",
    target_height=380
):
    """
    Renders one full diagnostic test case panel:
    [ Scene with Box ] + [ Crop & Preprocessed Crop ] + [ Reader Result Card ]
    """
    img_path = os.path.join("combined_dataset", "images", case_info["filename"])
    img = cv2.imread(img_path)
    if img is None:
        return None

    h_orig, w_orig = img.shape[:2]
    xmin, ymin, xmax, ymax = case_info["bbox"]

    # Crop
    crop = img[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        return None

    # 1. Run CNN Gate Inference
    t0 = time.perf_counter()
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)
    tensor_crop = cnn_transform(pil_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_crop)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        cnn_prob = float(probs[1])
    t_cnn_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Run OCR Inference
    raw_pred, clean_pred, ocr_conf, t_ocr_ms = read_license_plate(
        crop, reader=ocr_reader, device=device
    )

    # 3. Compute Metrics
    gt_clean = clean_and_format_plate_text(case_info["gt_text"])
    metrics = compute_alpr_metrics(clean_pred, gt_clean)

    # ----------------------------------------------------
    # Panel 1: Scene Image with Detection & Verification Box
    # ----------------------------------------------------
    scene_disp = img.copy()
    box_color = (40, 210, 80) if cnn_prob >= 0.5 else (60, 60, 220)
    cv2.rectangle(scene_disp, (xmin, ymin), (xmax, ymax), box_color, 2)

    # Tag badge
    badge_label = f"{arch_name}: PLATE {cnn_prob*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tag_ymin = max(0, ymin - th - 8)
    cv2.rectangle(scene_disp, (xmin, tag_ymin), (xmin + tw + 10, tag_ymin + th + 6), (20, 22, 28), -1)
    cv2.putText(scene_disp, badge_label, (xmin + 5, tag_ymin + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    # Scale scene to target height
    scale_scene = target_height / float(h_orig)
    nw_scene = max(1, int(w_orig * scale_scene))
    scene_disp = cv2.resize(scene_disp, (nw_scene, target_height))
    scene_panel = add_header_banner(
        scene_disp,
        "1. Vehicle Scene & CNN Candidate Gate",
        f"Verified Box: ({xmin},{ymin}) to ({xmax},{ymax})"
    )

    # ----------------------------------------------------
    # Panel 2: Inset Column (Raw Plate Crop + CLAHE Preprocessed Crop)
    # ----------------------------------------------------
    crop_col_w = 260
    sub_h = target_height // 2

    # Raw crop
    h_c, w_c = crop.shape[:2]
    scale_c = min((crop_col_w - 20) / float(w_c), (sub_h - 40) / float(h_c))
    nw_c, nh_c = max(1, int(w_c * scale_c)), max(1, int(h_c * scale_c))
    resized_crop = cv2.resize(crop, (nw_c, nh_c))

    raw_panel = np.full((sub_h, crop_col_w, 3), (32, 34, 42), dtype=np.uint8)
    ox = (crop_col_w - nw_c) // 2
    oy = (sub_h - nh_c) // 2
    raw_panel[oy:oy+nh_c, ox:ox+nw_c] = resized_crop
    cv2.rectangle(raw_panel, (ox-1, oy-1), (ox+nw_c+1, oy+nh_c+1), (40, 200, 80), 1)
    cv2.putText(raw_panel, "Raw Plate Candidate", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 220, 235), 1, cv2.LINE_AA)

    # Preprocessed crop (CLAHE + Bilateral)
    prep_gray, _ = preprocess_plate_for_ocr(crop)
    if prep_gray is not None:
        prep_bgr = cv2.cvtColor(prep_gray, cv2.COLOR_GRAY2BGR)
        h_p, w_p = prep_bgr.shape[:2]
        scale_p = min((crop_col_w - 20) / float(w_p), (sub_h - 40) / float(h_p))
        nw_p, nh_p = max(1, int(w_p * scale_p)), max(1, int(h_p * scale_p))
        resized_prep = cv2.resize(prep_bgr, (nw_p, nh_p))
        prep_panel = np.full((sub_h, crop_col_w, 3), (26, 28, 35), dtype=np.uint8)
        px = (crop_col_w - nw_p) // 2
        py = (sub_h - nh_p) // 2
        prep_panel[py:py+nh_p, px:px+nw_p] = resized_prep
        cv2.rectangle(prep_panel, (px-1, py-1), (px+nw_p+1, py+nh_p+1), (80, 210, 240), 1)
        cv2.putText(prep_panel, "CLAHE & Bilateral Filter", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 220, 240), 1, cv2.LINE_AA)
    else:
        prep_panel = raw_panel.copy()

    crops_col = np.vstack([raw_panel, prep_panel])
    crops_panel = add_header_banner(
        crops_col,
        "2. Plate Preprocessing",
        f"Resolution: {w_c}x{h_c} px"
    )

    # ----------------------------------------------------
    # Panel 3: Reader Result Card
    # ----------------------------------------------------
    card_disp = build_reader_card(
        arch_name=arch_name,
        cnn_prob=cnn_prob,
        gt_text=gt_clean,
        pred_text=clean_pred,
        metrics=metrics,
        t_cnn_ms=t_cnn_ms,
        t_ocr_ms=t_ocr_ms,
        card_w=430,
        card_h=target_height
    )
    result_panel = add_header_banner(
        card_disp,
        "3. OCR Reader Performance",
        f"Status: {gt_clean} -> {clean_pred}"
    )

    # Combine 3 panels horizontally with thin borders
    border = np.full((scene_panel.shape[0], 4, 3), (18, 20, 24), dtype=np.uint8)
    combined = np.hstack([scene_panel, border, crops_panel, border, result_panel])

    # Top Master Banner for this Test Case
    title_banner = np.full((48, combined.shape[1], 3), (16, 18, 22), dtype=np.uint8)
    cv2.putText(title_banner, f"ALPR BENCHMARK [{arch_name.upper()}] - {case_info['title']}", (18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(title_banner, case_info['subtitle'], (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 175, 195), 1, cv2.LINE_AA)

    final_panel = np.vstack([title_banner, combined])
    return final_panel

def run_reader_case_demonstrations(
    arch_name,
    ckpt_path,
    output_dir=None,
    device="cpu"
):
    """
    Generates all 5 side-by-side diagnostic test cases + 1 master composite graphic.
    """
    if output_dir is None:
        output_dir = f"results/{arch_name.lower()}_reader"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"  GENERATING 5 ALPR DIAGNOSTIC TEST CASES: {arch_name.upper()}")
    print(f"{'='*75}")
    print(f"  Model Checkpoint : {ckpt_path}")
    print(f"  Output Directory : {output_dir}")

    # 1. Load trained CNN model
    model = load_trained_model(arch_name, ckpt_path, device=device)
    if model is None:
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'! Please train {arch_name} first.")

    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load EasyOCR reader
    ocr_reader = get_easyocr_reader(device=device)

    rendered_panels = []

    for case in READER_TEST_CASES:
        print(f"[*] Rendering {case['name']} ({case['gt_text']})...", flush=True)
        panel = render_reader_case_panel(
            case_info=case,
            model=model,
            ocr_reader=ocr_reader,
            arch_name=arch_name,
            cnn_transform=cnn_transform,
            device=device,
            target_height=360
        )
        if panel is not None:
            out_file = os.path.join(output_dir, f"{case['name']}.jpg")
            cv2.imwrite(out_file, panel, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"  [OK] Saved -> {out_file}", flush=True)
            rendered_panels.append(panel)

    # 3. Build Master Composite Graphic (all 5 cases stacked)
    if rendered_panels:
        print("[*] Creating 5-case master overview graphic...", flush=True)
        target_w = 1280
        resized_panels = []
        for p in rendered_panels:
            h, w = p.shape[:2]
            scale = target_w / float(w)
            resized_p = cv2.resize(p, (target_w, int(h * scale)))
            divider = np.full((10, target_w, 3), (12, 14, 18), dtype=np.uint8)
            resized_panels.extend([resized_p, divider])

        overview_img = np.vstack(resized_panels[:-1])

        # Master Overview Header
        master_hdr = np.full((60, target_w, 3), (12, 14, 18), dtype=np.uint8)
        cv2.putText(master_hdr, f"COMPREHENSIVE ALPR BENCHMARK REPORT: {arch_name.upper()} BACKBONE", (24, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(master_hdr, "All 5 Diagnostic Vehicle Test Cases | Detection Bounding Box -> CNN Quality Gate -> Preprocessing -> EasyOCR Reader", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 195, 215), 1, cv2.LINE_AA)

        master_overview = np.vstack([master_hdr, overview_img])
        overview_path = os.path.join(output_dir, f"{arch_name.lower()}_all_5_reader_cases_overview.jpg")
        cv2.imwrite(overview_path, master_overview, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"[SUCCESS] Master 5-case overview saved to '{overview_path}'.")

    print(f"\n{'='*75}")
    print(f"  [COMPLETED] All test cases generated successfully in '{output_dir}'.")
    print(f"{'='*75}")
