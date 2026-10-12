"""
=============================================================================
  GOOGLENET (INCEPTION V1) INDIVIDUAL CNN BENCHMARK RUNNER
  Assigned Module: GoogLeNet (Multi-Scale Inception Modules)
=============================================================================
Architecture Details:
- 22 Layers deep featuring 1x1, 3x3, and 5x5 multi-scale Inception modules.
- Replaced giant dense FC layers with Global Average Pooling (GAP).
- Parameter Count: ~5.6 Million (21.4 MB, highest parameter efficiency).
- Role in Project: Efficient multi-scale spatial feature learning with low compute.

Usage:
  python run_cnn_googlenet.py
  python run_cnn_googlenet.py --epochs 3 --device cpu
  python run_cnn_googlenet.py --device cuda
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from modules.cnn_individual_runner import train_and_benchmark

def main():
    parser = argparse.ArgumentParser(description="Run GoogLeNet Plate vs Non-Plate Benchmark")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--no_pretrained", action="store_true", help="Train from scratch without ImageNet weights")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    print("\n" + "="*80)
    print("  PROJECT TASK: GOOGLENET VERIFICATION CLASSIFIER")
    print("="*80)

    metrics, df = train_and_benchmark(
        arch_name="GoogLeNet",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        pretrained=not args.no_pretrained,
        save_model=True
    )

    print("\n" + "="*80)
    print("  [SUCCESS] GoogLeNet benchmark completed!")
    print("  Model checkpoint: models/googlenet_plate_classifier.pth")
    print("  Results saved to: results/cnn_comparison_results.csv")
    print("="*80)

if __name__ == "__main__":
    main()
