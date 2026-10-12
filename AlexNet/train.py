"""
Standalone Training Script for AlexNet License Plate Classifier.
Allows Teammate 1 to run AlexNet independently from within the AlexNet/ folder.
"""

import os
import sys
import argparse

# Add repository root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from modules.cnn_individual_runner import train_and_benchmark


def main():
    parser = argparse.ArgumentParser(description="Train AlexNet Plate Classifier independently")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--no_pretrained", action="store_true", help="Train from scratch without ImageNet weights")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("  TEAMMATE 1 INDEPENDENT WORKSPACE: ALEXNET TRAINING & BENCHMARK")
    print("=" * 75)

    metrics, df = train_and_benchmark(
        arch_name="AlexNet",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        pretrained=not args.no_pretrained,
        save_model=True
    )

    print("\n" + "=" * 75)
    print("  [SUCCESS] AlexNet Training Completed!")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
    print(f"  Throughput: {metrics['FPS']} FPS")
    print("=" * 75)


if __name__ == "__main__":
    main()
