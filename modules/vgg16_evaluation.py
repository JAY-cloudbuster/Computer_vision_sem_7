"""VGG16-only detection/OCR evaluation and truthful diagnostic figures."""
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from PIL import Image

from modules.case_visualizer import load_trained_model
from modules.detector_classical import ClassicalPlateDetector
from modules.evaluate_detection import compute_iou, evaluate_detector_on_dataset
from modules.reader_evaluator import get_easyocr_reader, read_license_plate, preprocess_plate_for_ocr, levenshtein_distance
from modules.reader_visualizer import READER_TEST_CASES
from modules.vgg16_training import NORMALIZE, OUT, CHECKPOINT, sync, update_only_vgg_row, write_json

READER_OUT = Path("results/vgg16_reader")


def normalize_text(text):
    """Normalize presentation only; never alter the ground truth's characters."""
    text = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    return text[3:] if text.startswith("IND") and len(text) > 7 else text


def clean_prediction(text):
    clean = normalize_text(text)
    # Preserve already-valid short/long series and numeric suffixes.
    if re.fullmatch(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}", clean):
        return clean
    # Apply position correction only to the unambiguous ten-character layout.
    if len(clean) == 10:
        digit_to_letter = str.maketrans("0124568", "OIZASGB")
        letter_to_digit = str.maketrans("ODQILT ZASGB".replace(" ", ""), "00011124568")
        candidate = clean[:2].translate(digit_to_letter) + clean[2:4].translate(letter_to_digit)
        candidate += clean[4:6].translate(digit_to_letter) + clean[6:].translate(letter_to_digit)
        if re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}", candidate):
            return candidate
    return clean


def text_metrics(prediction, truth):
    pred, gt = normalize_text(prediction), normalize_text(truth)
    if not gt:
        raise ValueError("Empty ground truth cannot be evaluated")
    distance = levenshtein_distance(pred, gt)
    return dict(exact_match=int(pred == gt), edit_distance=distance,
                ned=max(0., 1-distance/max(len(pred), len(gt), 1)),
                char_accuracy=max(0., 1-distance/len(gt)))


def gate(model, crop, device="cpu"):
    if crop is None or crop.size == 0:
        raise ValueError("Empty crop")
    sync(device)
    start = time.perf_counter()
    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
    x = NORMALIZE(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.inference_mode():
        p = float(model(x).softmax(1)[0, 1].cpu())
    sync(device)
    return p, (time.perf_counter()-start)*1000


def stack_images(paths, output):
    images = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        images.append(im.resize((1600, round(im.height*1600/im.width))))
    overview = Image.new("RGB", (1600, sum(i.height for i in images)), "white")
    y = 0
    for im in images:
        overview.paste(im, (0,y))
        y += im.height
    overview.save(output, quality=94)


def annotated(image, boxes, labels, colors):
    result = image.copy()
    for (x,y,w,h), label, color in zip(boxes, labels, colors):
        x,y,w,h = map(int, (x,y,w,h))
        cv2.rectangle(result, (x,y), (x+w,y+h), color, 2)
        cv2.putText(result, label, (x, max(14,y-5)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def detection_cases():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = load_trained_model("VGG16", str(CHECKPOINT), "cpu")
    if model is None:
        raise FileNotFoundError(CHECKPOINT)
    annotations = json.loads(Path("data/annotations_test.json").read_text())
    detector = ClassicalPlateDetector()
    diagnostic_records, paths = [], []
    for number, index in enumerate([1,5,20,43,38], 1):
        item = annotations[index]
        image = cv2.imread(item["filepath"])
        if image is None:
            raise FileNotFoundError(item["filepath"])
        gt = item["bboxes"]
        candidates = []
        for box in detector.detect(image):
            x,y,w,h = map(int, box)
            crop = image[y:y+h,x:x+w]
            if min(crop.shape[:2]) < 5:
                continue
            p, ms = gate(model, crop)
            overlap = max((compute_iou(box,g) for g in gt), default=0.)
            outcome = ("TP" if overlap >= .5 else "FP") if p >= .5 else ("FN" if overlap >= .5 else "TN")
            candidates.append(dict(box=box, plate_probability=p, iou=overlap, outcome=outcome, latency_ms=ms))
        fig = plt.figure(figsize=(18,8), layout="constrained")
        grid = fig.add_gridspec(2,6, height_ratios=[3,1.25])
        axes = [fig.add_subplot(grid[0,i*2:i*2+2]) for i in range(3)]
        axes[0].imshow(annotated(image, gt, ["Ground truth"]*len(gt), [(0,200,0)]*len(gt)))
        boxes = [c["box"] for c in candidates]
        axes[1].imshow(annotated(image, boxes, [f"Proposal {i+1}" for i in range(len(boxes))], [(0,150,255)]*len(boxes)))
        axes[2].imshow(annotated(image, boxes, [f"{c['outcome']} {c['plate_probability']:.2f}" for c in candidates],
                                [(0,190,0) if c['plate_probability']>=.5 else (0,0,220) for c in candidates]))
        for ax,title in zip(axes,["1. Ground truth", "2. Classical proposals", "3. VGG16 decisions (threshold 0.50)"]):
            ax.set_title(title); ax.axis("off")
        for i,c in enumerate(candidates[:6]):
            ax = fig.add_subplot(grid[1,i]); x,y,w,h=map(int,c["box"])
            ax.imshow(cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2RGB))
            ax.set_title(f"{c['outcome']} | P(plate)={c['plate_probability']:.3f}\nIoU={c['iou']:.3f}", fontsize=10)
            ax.axis("off")
        kept = sum(c["plate_probability"]>=.5 for c in candidates)
        fig.suptitle(f"VGG16 case {number}: {item['filename']} | {len(candidates)} proposals, {kept} accepted\n"
                     "TP/FP: accepted true/false plate; FN/TN: rejected true/non-plate. IoU threshold = 0.50.", fontsize=14)
        path=OUT/f"vgg16_sample_case_{number}.jpg"
        fig.savefig(path,dpi=140); plt.close(fig); paths.append(path)
        diagnostic_records.append(dict(case=number,filename=item['filename'],candidates=candidates))
    stack_images(paths,OUT/"vgg16_all_5_cases_overview.jpg")
    write_json(OUT/"detection_case_predictions.json",diagnostic_records)

    class VerifiedDetector:
        def detect(self,image):
            accepted=[]
            for box in detector.detect(image):
                x,y,w,h=map(int,box)
                crop=image[y:y+h,x:x+w]
                if min(crop.shape[:2])>=5 and gate(model,crop)[0]>=.5:
                    accepted.append(box)
            return accepted
    results=[evaluate_detector_on_dataset(detector,annotations,name="Classical proposals"),
             evaluate_detector_on_dataset(VerifiedDetector(),annotations,name="Classical + VGG16")]
    pd.DataFrame(results).to_csv(OUT/"detection_pipeline_results.csv",index=False)
    print("Detection diagnostics and all-50-image detection benchmark complete.",flush=True)


def summarize_reader(frame):
    n=len(frame)
    if n==0:
        raise ValueError("No reader records")
    return {"Architecture":"VGG16", "Total Evaluated":n, "Verified by Gate":int(frame.accepted.sum()),
            "Gate Pass Rate (%)":round(frame.accepted.mean()*100,2),
            "Exact Match (%)":round(frame.exact_match.mean()*100,2),
            "Avg NED (Similarity)":round(frame.ned.mean(),4),
            "Character Acc (%)":round(frame.char_accuracy.mean()*100,2),
            "CNN Latency (ms)":round(frame.cnn_ms.mean(),2),
            "OCR Latency (ms)":round(frame.loc[frame.accepted==1,'ocr_ms'].mean(),2) if frame.accepted.sum() else 0.,
            "End-to-End Latency (ms)":round(frame.total_ms.mean(),2),
            "End-to-End FPS":round(1000/frame.total_ms.mean(),2)}


def reader(samples=0):
    READER_OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    model=load_trained_model("VGG16",str(CHECKPOINT),"cpu")
    if model is None:
        raise FileNotFoundError(CHECKPOINT)
    ocr=get_easyocr_reader("cpu")
    labels=pd.read_csv("combined_dataset/labels.csv",keep_default_na=False)
    if samples:
        labels=labels.iloc[:samples]
    signature=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    signature+=hashlib.sha256(Path("combined_dataset/labels.csv").read_bytes()).hexdigest()
    signature+=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature_path=READER_OUT/"run_signature.txt"
    record_path=READER_OUT/"reader_predictions.jsonl"
    if signature_path.exists() and signature_path.read_text()!=signature:
        archive=READER_OUT/"previous_runs"/str(time.time_ns())
        archive.mkdir(parents=True)
        for name in ["run_signature.txt","reader_predictions.jsonl","reader_predictions.csv","reader_metrics.json"]:
            source=READER_OUT/name
            if source.exists():
                shutil.move(str(source),str(archive/name))
        print(f"Checkpoint, labels, or evaluation code changed; previous reader results preserved in {archive}",flush=True)
    signature_path.write_text(signature)
    existing=[json.loads(line) for line in record_path.read_text().splitlines()] if record_path.exists() else []
    if len(existing)>len(labels):
        existing=existing[:len(labels)]
    start=time.perf_counter()
    with record_path.open("a") as stream:
        for index in range(len(existing),len(labels)):
            row=labels.iloc[index]
            image=cv2.imread(str(Path("combined_dataset/images")/str(row.filename)))
            if image is None:
                raise FileNotFoundError(str(row.filename))
            h,w=image.shape[:2]
            x1,y1,x2,y2=map(int,[row.xmin,row.ymin,row.xmax,row.ymax])
            if not (0<=x1<x2<=w and 0<=y1<y2<=h):
                raise ValueError(f"Invalid annotated box at row {index}: {row.filename}")
            crop=image[y1:y2,x1:x2]
            p,cnn_ms=gate(model,crop)
            accepted=p>=.5
            raw,pred,confidence,ocr_ms="","",0.,0.
            if accepted:
                raw,_,confidence,ocr_ms=read_license_plate(crop,reader=ocr,device="cpu")
                pred=clean_prediction(raw)
            record=dict(index=index,filename=str(row.filename),gt_text=normalize_text(row.plate_text),
                        raw_prediction=raw,prediction=pred,plate_probability=p,accepted=int(accepted),
                        ocr_confidence=confidence,cnn_ms=cnn_ms,ocr_ms=ocr_ms,total_ms=cnn_ms+ocr_ms,
                        **text_metrics(pred,row.plate_text))
            stream.write(json.dumps(record)+"\n"); stream.flush(); existing.append(record)
            if (index+1)%40==0 or index==len(labels)-1:
                print(f"Reader {index+1}/{len(labels)}; exact={np.mean([r['exact_match'] for r in existing])*100:.1f}%; elapsed={time.perf_counter()-start:.1f}s",flush=True)
    frame=pd.DataFrame(existing)
    frame.to_csv(READER_OUT/"reader_predictions.csv",index=False)
    full=summarize_reader(frame)
    primary=summarize_reader(frame.iloc[:40])
    write_json(READER_OUT/"reader_metrics.json",{"first_40":primary,"full_dataset":full,
               "protocol":"Ground-truth crops, CPU VGG16 threshold .5, shared EasyOCR preprocessing; ground truth preserved; all samples in metric denominators."})
    update_only_vgg_row("results/reader_comparison_results.csv",primary)
    print(json.dumps(full,indent=2),flush=True)
    return frame


def align_characters(truth, prediction):
    """Return a minimum-edit alignment, including insertions and deletions."""
    m,n=len(truth),len(prediction)
    dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0]=i
    for j in range(n+1):
        dp[0][j]=j
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,
                         dp[i-1][j-1]+(truth[i-1]!=prediction[j-1]))
    pairs=[]
    i,j=m,n
    while i or j:
        if i and j and dp[i][j]==dp[i-1][j-1]+(truth[i-1]!=prediction[j-1]):
            pairs.append((truth[i-1],prediction[j-1]));i-=1;j-=1
        elif i and dp[i][j]==dp[i-1][j]+1:
            pairs.append((truth[i-1],"-"));i-=1
        else:
            pairs.append(("-",prediction[j-1]));j-=1
    return pairs[::-1]


def reader_cases():
    frame=pd.read_csv(READER_OUT/"reader_predictions.csv",keep_default_na=False)
    paths=[]
    for case in READER_TEST_CASES:
        found=frame[frame.filename==case['filename']]
        if len(found)!=1:
            raise ValueError(f"Missing/nonunique evaluated reader case: {case['filename']}")
        r=found.iloc[0]
        image=cv2.imread(str(Path("combined_dataset/images")/case['filename']))
        # Use the dataset annotation that was actually evaluated, never substitute canned text/boxes.
        labels=pd.read_csv("combined_dataset/labels.csv",keep_default_na=False)
        label=labels[labels.filename==case['filename']].iloc[0]
        x1,y1,x2,y2=map(int,[label.xmin,label.ymin,label.xmax,label.ymax])
        crop=image[y1:y2,x1:x2]
        gray,_=preprocess_plate_for_ocr(crop)
        fig=plt.figure(figsize=(18,6),layout="constrained")
        grid=fig.add_gridspec(2,3,width_ratios=[1.3,1,1.2])
        ax=fig.add_subplot(grid[:,0])
        color=(0,180,0) if r.accepted else (0,0,220)
        ax.imshow(annotated(image,[[x1,y1,x2-x1,y2-y1]],[f"P(plate)={r.plate_probability:.3f}"],[color]))
        ax.set_title("1. Annotated plate + VGG16 gate");ax.axis('off')
        raw_title="2a. Color OCR input" if r.accepted else "2a. Plate crop (OCR skipped)"
        enhanced_title="2b. Enhanced grayscale fallback" if r.accepted else "2b. Enhancement preview only"
        for pos,img,title in [(0,cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),raw_title),(1,gray,enhanced_title)]:
            ax=fig.add_subplot(grid[pos,1]);ax.imshow(img,cmap="gray");ax.set_title(title);ax.axis('off')
        ax=fig.add_subplot(grid[:,2]);ax.axis('off')
        status="EXACT MATCH" if r.exact_match else "OCR MISMATCH" if r.accepted else "REJECTED BY VGG16"
        detail=(f"{status}\n\nGround truth: {r.gt_text}\nPrediction: {r.prediction or '[empty]'}\n"
                f"\nPlate probability: {r.plate_probability:.3f}\nEdit distance: {int(r.edit_distance)}\n"
                f"Similarity: {r.ned*100:.1f}%\nCharacter accuracy: {r.char_accuracy*100:.1f}%\n"
                f"\nVGG16: {r.cnn_ms:.1f} ms\nOCR: {r.ocr_ms:.1f} ms\nTotal: {r.total_ms:.1f} ms")
        ax.text(.02,.98,detail,va="top",fontsize=11.5,linespacing=1.45,family="monospace")
        pairs=align_characters(str(r.gt_text),str(r.prediction))
        ax.text(.02,.25,"Character alignment: green = match, red = error",fontsize=9)
        ax.text(.02,.17,"GT",fontsize=9,va="center")
        ax.text(.02,.07,"OCR",fontsize=9,va="center")
        step=.84/max(1,len(pairs))
        for i,(gt_char,pred_char) in enumerate(pairs):
            color="#d7efdd" if gt_char==pred_char else "#f5cccc"
            for y,char in [(.135,gt_char),(.035,pred_char)]:
                x=.14+i*step
                ax.add_patch(Rectangle((x,y),step*.9,.07,facecolor=color,edgecolor="white",transform=ax.transAxes))
                ax.text(x+step*.45,y+.035,char,ha="center",va="center",fontsize=10,family="monospace",transform=ax.transAxes)
        fig.suptitle(f"VGG16 reader case {case['case_id']}: {case['filename']}\nGround-truth crop evaluation; detector localization is not measured here.",fontsize=15)
        path=READER_OUT/f"{case['name']}.jpg";fig.savefig(path,dpi=140);plt.close(fig);paths.append(path)
    stack_images(paths,READER_OUT/"vgg16_all_5_reader_cases_overview.jpg")
    print("All five reader panels and overview saved.",flush=True)
