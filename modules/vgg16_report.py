"""Create an evidence-based VGG16 report and presentation figures."""
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def report():
    out=Path("results/vgg16")
    info=json.loads((out/"training_details.json").read_text())
    readers=json.loads(Path("results/vgg16_reader/reader_metrics.json").read_text())
    detection=pd.read_csv(out/"detection_pipeline_results.csv")
    history=pd.DataFrame(info['history'])
    matrix=np.array(info['test_metrics']['confusion_matrix'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    axes[0].plot(history.epoch,history['F1-Score'],label="Validation F1")
    axes[0].plot(history.epoch,history.Recall,label="Validation plate recall")
    axes[0].axvline(info['best_epoch'],color="gray",ls="--",label="Selected epoch")
    axes[0].set(xlabel="Training epoch",ylabel="Score",ylim=(0,1.03),title="Model selected using validation images only")
    axes[0].legend()
    axes[1].imshow(matrix,cmap="Blues")
    axes[1].set(xticks=[0,1],yticks=[0,1],xticklabels=['Non-plate','Plate'],yticklabels=['Non-plate','Plate'],
                xlabel="Predicted",ylabel="Ground truth",title="Independent test crops: confusion matrix")
    for i in range(2):
        for j in range(2):
            axes[1].text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=22,
                         color="white" if matrix[i,j]>matrix.max()/2 else "black")
    fig.savefig(out/"training_and_test_results.png",dpi=180);plt.close(fig)
    full,primary=readers['full_dataset'],readers['first_40']
    metrics=['Gate Pass Rate (%)','Exact Match (%)','Character Acc (%)']
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
    x=np.arange(3)
    ax.bar(x-.18,[primary[m] for m in metrics],.36,label=f"Required first {primary['Total Evaluated']}")
    ax.bar(x+.18,[full[m] for m in metrics],.36,label=f"Full dataset ({full['Total Evaluated']})")
    ax.set(xticks=x,xticklabels=['Gate pass','Exact plate match','Character accuracy'],ylim=(0,110),ylabel="Percent",
           title="VGG16 + EasyOCR on annotated plate crops")
    ax.legend();fig.savefig(out/"reader_results.png",dpi=180);plt.close(fig)
    m=info['test_metrics']; c=info['crop_counts']; counts=info['image_counts']
    cnn=info['csv_metrics']
    table="| Metric | First 40 | Full dataset |\n|---|---:|---:|\n"
    for key in ['Total Evaluated','Verified by Gate','Gate Pass Rate (%)','Exact Match (%)','Avg NED (Similarity)',
                'Character Acc (%)','End-to-End Latency (ms)','End-to-End FPS']:
        table+=f"| {key} | {primary[key]} | {full[key]} |\n"
    det_table="| Method | Precision | Recall | F1 | CPU FPS |\n|---|---:|---:|---:|---:|\n"
    for _,r in detection.iterrows():
        det_table+=f"| {r['method']} | {r.precision} | {r.recall} | {r.f1_score} | {r.fps} |\n"
    records=pd.read_csv("results/vgg16_reader/reader_predictions.csv",keep_default_na=False)
    cases="| Image | Ground truth | OCR prediction | Gate | Exact match |\n|---|---|---|---|---|\n"
    for filename in ["image_1.jpeg","image_4.jpeg","image_5.jpeg","image_9.jpeg","image_11.jpg"]:
        r=records[records.filename==filename].iloc[0]
        decision="Accepted" if r.accepted else "Rejected; OCR skipped"
        cases+=f"| {filename} | {r.gt_text} | {r.prediction or '[empty]'} | {decision} | {'Yes' if r.exact_match else 'No'} |\n"
    text=f"""# VGG16 contribution: implementation and measured results

## What VGG16 does
For each candidate crop, VGG16 predicts **plate** or **non-plate**. The classical OpenCV detector proposes candidate boxes. VGG16 can reject false alarms, but cannot recover a plate that the detector never proposed. EasyOCR reads accepted plate crops.

VGG16 has 13 convolutional layers and three fully connected layers. The original ImageNet model's last 1,000-class layer is replaced by a 4,096-to-2 layer. Total parameters: **{cnn['Total Params (M)']} million**. Trainable parameters: **{info['trainable_parameters']:,}**. Estimated float32 weight size: **{cnn['Size (MB)']} MiB**. The existing team's checkpoint loader can load the saved model unchanged.

## Training protocol
- Started with all 6,176 annotated training images: {counts['train']} training and {counts['validation']} validation images; seed 42. Removed {len(info['excluded_test_duplicates'])} exact test-image duplicates from the training pool, if any.
- Split occurs by decoded image content before crop creation. Exact duplicate images stay within one partition. No decoded image hashes overlap with the separate {counts['test']}-image test set. Numeric source IDs repeat across the supplied splits and are not treated as globally unique.
- Crops: {c['train']} training, {c['validation']} validation, {c['test']} test. Labels are 0=non-plate, 1=plate.
- Positives are ground-truth boxes; negatives are classical detector proposals with IoU below 0.15. Crops resize to 224×224 and use ImageNet normalization.
- Pretrained VGG16 layers are frozen and evaluated deterministically. Their 4,096-dimensional features are cached after full forward passes. Normal and horizontally flipped training crops are both processed; each training example uses its flipped representation with probability 0.2 per epoch. Validation/test crops are never augmented.
- Only the final layer learns, with Adam (learning rate 0.001), batches of 32, and cross-entropy weighted using training class counts. Caching removes repeated computation of fixed features; it does not replace VGG16 or reduce the number of training examples.
- Maximum {info['max_epochs']} epochs, stopping after six consecutive epochs without validation improvement. Completed {info['completed_epochs']} epochs; selected epoch {info['best_epoch']} by validation F1, then validation loss. Test data is excluded from model selection.
- Full-network extraction ran on {info['training_device']}; measured classification timings below use CPU, batch size one, after warm-up. Head fitting runs on CPU.

## Independent classification test
Accuracy: **{m['Accuracy']*100:.2f}%**; precision: **{m['Precision']*100:.2f}%**; plate recall: **{m['Recall']*100:.2f}%**; F1: **{m['F1-Score']:.4f}**.

Confusion matrix (rows truth / columns prediction, non-plate then plate): `{matrix.tolist()}`.
Full VGG16 CPU latency: **{cnn['Latency (ms)']} ms/crop**, **{cnn['FPS']} crops/s**. This is a classification timing and excludes loading/preprocessing/detection/OCR. Training preparation + feature extraction + head fitting took {info['train_seconds']:.1f} seconds.

![Training and test results](training_and_test_results.png)

## Actual detection on all 50 test scenes
{det_table}
These results measure localization and verification together at IoU 0.50. They are separate from crop classification accuracy. A missed proposal remains a missed plate even with a strong verifier.

## Reader benchmark
{table}
The shared comparison CSV contains the required first-40 result; the full evaluation and every sample prediction are retained separately.

**Scope:** these are ground-truth plate crops from the separate combined dataset. Detector localization is not part of this reader benchmark. The first-40 subset is included in the full dataset; these are not independent experiments. All rejected crops count as recognition failures, and average total time includes rejected crops too. OCR runs on CPU using the team's EasyOCR settings. It first tries the resized padded color crop; enhanced CLAHE/bilateral grayscale is a fallback only when no text is detected.

Ground-truth text receives only capitalization, separator removal, and a leading IND badge removal. It is never truncated or character-corrected. OCR predictions preserve valid short formats and use position correction only for an unambiguous ten-character layout. NED here means **1 − edit distance / maximum string length**, so higher is better. Character accuracy is clipped **1 − edit distance / ground-truth length**. It can differ from exact match because one wrong character makes a whole plate mismatch.

![Reader results](reader_results.png)

## Five reader demonstration cases
{cases}
The figure cards include a minimum-edit character alignment: green for matching characters and red for substitutions, insertions, or deletions. A rejected crop shows no OCR prediction and zero OCR time. Its enhancement image is labelled as a preview only.

![Five reader cases](../vgg16_reader/vgg16_all_5_reader_cases_overview.jpg)

![Five detection cases](vgg16_all_5_cases_overview.jpg)

## Problems found and changes confined to VGG16
1. Training/test images referenced by the supplied JSON files were absent. Restored them from the original source while checking image dimensions and bounding boxes and preserving the existing JSON files.
2. The old VGG16 result had zero recall. Added class weighting, validation selection, an independent test, a collapse check, saved predictions, and a confusion matrix.
3. The original random crop split could put crops from one scene on both sides. The VGG16 workflow splits source images first.
4. Existing OCR cleanup could alter valid ground truth (for example KA19TR02). Fixed only in the VGG16 evaluation path, with regression tests.
5. Existing reader figures could show OCR even when the gate rejected a plate. New VGG16 figures use the actual gated evaluation records.
6. Existing diagnostic descriptions assumed successful filtering. VGG16 panels display measured TP/FP/TN/FN decisions, including failures.

Original model-specific wrappers and shared modules remain unchanged. The new entry point is `run_vgg16_complete.py`. Only VGG16 rows in the two shared comparison tables are updated; teammates' rows and classical benchmark results are preserved.

## Limits to explain during evaluation
- This is transfer learning of a binary verification head, not training VGG16 from scratch or using VGG16 for character recognition.
- Classification accuracy does not imply equal full-scene detection accuracy or OCR accuracy.
- Existing teammate figures use different hardware, sample sizes, and split protocols. Do not present this as a controlled head-to-head architecture ranking.
- Exact image duplicates are checked by decoded content hashes; near-duplicate views of the same vehicle have not been audited.
- Ground-truth-crop OCR is a component benchmark. Arbitrary new traffic scenes and video tracking are not validated by it.
- Failed OCR predictions are retained honestly; no claim of perfect recognition is made.

## Reproduce
From the repository root, activate `.venv`, then:

```bash
python run_vgg16_complete.py --stage train --device auto --epochs 30 --batch-size 8
python run_vgg16_complete.py --stage detection
python run_vgg16_complete.py --stage reader --samples 0
python run_vgg16_complete.py --stage report
python -m pytest tests/test_vgg16_workflow.py -q
```

The training checkpoint is `models/vgg16_plate_classifier.pth`. Diagnostic images are in `results/vgg16/` and `results/vgg16_reader/`. The reader saves progress after every sample and resumes only when checkpoint and labels signatures match. Python: {platform.python_version()}, PyTorch: {torch.__version__}, platform: {platform.platform()}. Exact package versions are in `requirements-vgg16-lock.txt`.
"""
    (out/"VGG16_REPORT.md").write_text(text)
    print("Report and result charts written to results/vgg16/VGG16_REPORT.md",flush=True)
