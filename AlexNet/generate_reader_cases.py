"""
Standalone ALPR Reader Diagnostic Case Generator for AlexNet.
Produces 5 diagnostic cases + master overview in results/alexnet_reader/.
"""

import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from modules.reader_visualizer import run_reader_case_demonstrations


def main():
    parser = argparse.ArgumentParser(description="Generate 5 ALPR Diagnostic Cases for AlexNet independently")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--ckpt", type=str, default="models/alexnet_plate_classifier.pth", help="Checkpoint path")
    parser.add_argument("--out_dir", type=str, default="results/alexnet_reader", help="Output directory")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("  TEAMMATE 1 INDEPENDENT WORKSPACE: ALEXNET ALPR DIAGNOSTIC CASES")
    print("=" * 75)

    run_reader_case_demonstrations(
        arch_name="AlexNet",
        ckpt_path=args.ckpt,
        output_dir=args.out_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
