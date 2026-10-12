"""
Standalone Evaluation Script for AlexNet Diagnostic Visualizations.
Produces the 5 side-by-side diagnostic test cases and master overview in results/alexnet/.
"""

import os
import sys
import argparse

# Add repository root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from modules.case_visualizer import run_case_demonstrations


def main():
    parser = argparse.ArgumentParser(description="Generate 5 Diagnostic Test Cases for AlexNet")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--ckpt", type=str, default="models/alexnet_plate_classifier.pth", help="Checkpoint path")
    parser.add_argument("--out_dir", type=str, default="results/alexnet", help="Output directory")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("  TEAMMATE 1 INDEPENDENT WORKSPACE: ALEXNET EVALUATION & TEST CASES")
    print("=" * 75)

    run_case_demonstrations(
        arch_name="AlexNet",
        ckpt_path=args.ckpt,
        output_dir=args.out_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
