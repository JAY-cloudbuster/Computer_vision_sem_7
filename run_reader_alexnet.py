"""
=============================================================================
  ALEXNET DETECTION-TO-READER (ALPR) BENCHMARK RUNNER
  Assigned Module: AlexNet (Classic 8-Layer CNN Backbone)
=============================================================================
Evaluates the end-to-end ALPR pipeline:
1. Candidate Bounding Box Verification using trained AlexNet backbone
2. Noise/Grille Candidate Filtering via AlexNet Quality Gate
3. Plate Crop Enhancement (CLAHE + Bilateral Filtering)
4. Optical Character Recognition using EasyOCR
5. Syntax Normalization & Levenshtein ALPR Metrics
6. Non-destructively records metrics into results/reader_comparison_results.csv

Usage:
  python run_reader_alexnet.py
  python run_reader_alexnet.py --samples 40 --device cpu
  python run_reader_alexnet.py --device cuda
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from modules.reader_evaluator import evaluate_reader_with_cnn

def main():
    parser = argparse.ArgumentParser(description="Run AlexNet Detection-to-Reader Benchmark")
    parser.add_argument("--samples", type=int, default=40, help="Number of test plate images to evaluate (default: 40)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--threshold", type=float, default=0.5, help="CNN Gate confidence threshold (default: 0.5)")
    parser.add_argument("--ckpt", type=str, default="models/alexnet_plate_classifier.pth", help="Checkpoint path")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    if not os.path.exists(args.ckpt):
        print("\n" + "="*80)
        print(f"  [!] Checkpoint '{args.ckpt}' not found!")
        print("  Please train AlexNet first by running:")
        print("    python run_cnn_alexnet.py --epochs 3 --device cpu")
        print("="*80 + "\n")
        sys.exit(1)

    print("\n" + "="*80)
    print("  PROJECT TASK: ALEXNET DETECTION-TO-READER (ALPR) PIPELINE")
    print("="*80)

    metrics, df_records = evaluate_reader_with_cnn(
        arch_name="AlexNet",
        ckpt_path=args.ckpt,
        labels_csv_path="combined_dataset/labels.csv",
        images_dir="combined_dataset/images",
        max_samples=args.samples,
        gate_threshold=args.threshold,
        device=device,
        save_csv=True,
        csv_out="results/reader_comparison_results.csv"
    )

    print("\n" + "="*80)
    print("  [SUCCESS] AlexNet ALPR benchmark completed!")
    print(f"  Exact Match Accuracy: {metrics['Exact Match (%)']:.1f}%")
    print(f"  Character Accuracy  : {metrics['Character Acc (%)']:.1f}%")
    print(f"  Normalized Distance : {metrics['Avg NED (Similarity)']:.4f}")
    print(f"  End-to-End Latency  : {metrics['End-to-End Latency (ms)']:.1f} ms ({metrics['End-to-End FPS']:.1f} FPS)")
    print("  Saved to: results/reader_comparison_results.csv")
    print("="*80)

if __name__ == "__main__":
    main()
