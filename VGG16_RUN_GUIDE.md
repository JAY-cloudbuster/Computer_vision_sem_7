# VGG16 contribution: how to run and present it

Use the new `run_vgg16_complete.py` entry point for this contribution. It uses the team's VGG16 architecture, checkpoint format, classical detector, and EasyOCR components. Original source files and teammate wrappers remain untouched.

## Run the completed workflow

From this repository directory:

```bash
source .venv/bin/activate
python run_vgg16_complete.py
```

This reuses a saved VGG16 checkpoint, regenerates detection diagnostics, resumes the full reader evaluation using saved per-image records, generates reader diagnostics, and refreshes the VGG16 report. It does not train the other models. Reader records are reused only when the checkpoint, labels, and evaluator code match.

For a fresh training run:

```bash
python run_vgg16_complete.py --stage train --device auto --epochs 30 --batch-size 8
python run_vgg16_complete.py --stage detection
python run_vgg16_complete.py --stage reader --samples 0
python run_vgg16_complete.py --stage report
python -m pytest tests/test_vgg16_workflow.py -q
```

`--samples 0` evaluates every OCR row. The first 40 results are also summarized to meet the team guide. `--device auto` selects CUDA, Apple MPS, or CPU according to availability. All published inference timings use CPU. Up to 30 head-training epochs run, with early stopping after six epochs without validation improvement; the selected checkpoint uses the best validation F1.

## Files for the evaluation

- `results/vgg16/VGG16_REPORT.md`: measured results, protocol, improvements, limitations, and explanation for your presentation.
- `results/vgg16/training_and_test_results.png`: validation curve and independent-test confusion matrix.
- `results/vgg16/reader_results.png`: first-40 and full-dataset reader metrics.
- `results/vgg16/vgg16_all_5_cases_overview.jpg`: five detection diagnostics; individual panels are in the same folder.
- `results/vgg16_reader/vgg16_all_5_reader_cases_overview.jpg`: five reader diagnostics; individual panels are in the same folder.
- `models/vgg16_plate_classifier.pth`: the trained model.
- `results/vgg16/training_details.json`, `training_history.csv`, `test_predictions.csv`: training and test evidence.
- `results/vgg16/detection_pipeline_results.csv`: localization plus VGG16 verification on all 50 test scenes.
- `results/vgg16_reader/reader_predictions.csv`: every OCR answer, confidence, decision, error score, and latency.
- `results/cnn_comparison_results.csv` and `results/reader_comparison_results.csv`: only VGG16's records are updated.

## Explain the model accurately

VGG16 classifies candidate crops as plate or non-plate. Its pretrained feature layers are frozen; the final two-class layer is trained. EasyOCR performs character recognition. A strong crop classifier does not guarantee perfect scene detection or perfect plate transcription.

The reader benchmark starts from ground-truth plate crops. The separate detection benchmark measures whether the classical detector and VGG16 locate plates in full scenes. Keep these two results distinct during the presentation.

Training uses all supplied training annotations, a source-image/content-separated validation split, class weighting, and validation checkpoint selection. Fixed VGG16 features are cached so the same frozen computation is not repeated each epoch. Both normal and flipped training crops are processed; full VGG16 is still used for inference and latency measurements.

## Environment and inputs

The prepared environment is `.venv` (Python 3.11). Exact installed versions are saved in `requirements-vgg16-lock.txt`. On another machine, create a compatible environment and install those dependencies; platform-specific packages may need equivalent versions.

The supplied training/test image files were initially missing. `restore_vgg16_inputs.py` restores them from the original Hugging Face parquet exports after checking that image dimensions and bounding boxes match the existing annotations. The prepared workspace already has all required images. Cached activations under `data/vgg16_cache/` can be regenerated.

The VGG16 checkpoint is approximately 512 MiB, exceeding GitHub's normal 100 MB file limit. Keep the checkpoint locally for the evaluation; a later repository upload needs Git LFS or external artifact storage. This workflow does not push to GitHub.
