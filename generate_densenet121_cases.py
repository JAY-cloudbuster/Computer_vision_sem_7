"""
=============================================================================
  DENSENET-121 DIAGNOSTIC TEST CASE DEMONSTRATION GENERATOR
  Generates 5 High-Resolution Side-by-Side Test Cases for DenseNet-121 Alone:
  1. Case 1: Daytime Vehicle Frontal View (High-Confidence Plate Verification)
  2. Case 2: Radiator Grille False-Alarm Elimination
  3. Case 3: Dense Multi-Candidate Background Clutter Filtration
  4. Case 4: Complex Wide Street & Environmental Distractor Suppression
  5. Case 5: Rear Bumper Shadow & Tail-Light Clutter Suppression

  Outputs are saved to: results/densenet121/
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from modules.case_visualizer import run_case_demonstrations

def main():
    parser = argparse.ArgumentParser(description="Generate 5 Diagnostic Test Cases for DenseNet-121")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--ckpt", type=str, default="models/densenet121_plate_classifier.pth", help="Checkpoint path")
    parser.add_argument("--out_dir", type=str, default="results/densenet121", help="Output directory")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("\n" + "="*75)
    print("  GENERATING DENSENET-121 SIDE-BY-SIDE DIAGNOSTIC TEST CASES")
    print("="*75)

    run_case_demonstrations(
        arch_name="DenseNet121",
        ckpt_path=args.ckpt,
        output_dir=args.out_dir,
        device=device
    )

if __name__ == "__main__":
    main()
