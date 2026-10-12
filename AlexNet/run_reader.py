"""
Standalone ALPR Reader Pipeline Runner for AlexNet.
Allows Teammate 1 to run the detection-to-reader pipeline independently from within the AlexNet/ folder.
"""

import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from modules.reader_evaluator import evaluate_reader_with_cnn


def main():
    parser = argparse.ArgumentParser(description="Run AlexNet Detection-to-Reader Benchmark independently")
    parser.add_argument("--samples", type=int, default=40, help="Number of test plate images to evaluate (default: 40)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--threshold", type=float, default=0.5, help="CNN Gate confidence threshold (default: 0.5)")
    parser.add_argument("--ckpt", type=str, default="models/alexnet_plate_classifier.pth", help="Checkpoint path")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("  TEAMMATE 1 INDEPENDENT WORKSPACE: ALEXNET ALPR READER PIPELINE")
    print("=" * 75)

    metrics, df_records = evaluate_reader_with_cnn(
        arch_name="AlexNet",
        ckpt_path=args.ckpt,
        labels_csv_path="combined_dataset/labels.csv",
        images_dir="combined_dataset/images",
        max_samples=args.samples,
        gate_threshold=args.threshold,
        device=args.device,
        save_csv=True,
        csv_out="results/reader_comparison_results.csv"
    )

    print("\n" + "=" * 75)
    print("  [SUCCESS] AlexNet ALPR Reader Benchmark Completed!")
    print(f"  Exact Match:          {metrics['Exact Match (%)']:.1f}%")
    print(f"  Character Accuracy:   {metrics['Character Acc (%)']:.1f}%")
    print(f"  Normalized Distance:  {metrics['Avg NED (Similarity)']:.4f}")
    print(f"  End-to-End Latency:   {metrics['End-to-End Latency (ms)']:.1f} ms ({metrics['End-to-End FPS']:.1f} FPS)")
    print("=" * 75)


if __name__ == "__main__":
    main()
