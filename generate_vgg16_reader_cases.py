"""
=============================================================================
  VGG-16 ALPR DIAGNOSTIC TEST CASE GENERATOR
  Assigned Module: VGG-16 (Deep Homogeneous 3x3 Convolutions)
=============================================================================
Generates 5 High-Resolution Diagnostic Side-by-Side Test Cases:
1. Case 1: Standard Frontal View (KA19TR02 - Clean Plate Reading)
2. Case 2: Sedan Lower Bumper Perspective (MH20CS9817 - Standard Format)
3. Case 3: Compact Hatchback Front Angle (KL05AK3300 - Gradient Lighting)
4. Case 4: Urban Traffic Natural Environment (MH12JC2813 - Street Lighting)
5. Case 5: High Resolution SUV Radiator View (UP50AS4535 - Grille Proximity)
Plus 1 Master Composite 5-Case Overview graphic.

Outputs saved to: results/vgg16_reader/
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from modules.reader_visualizer import run_reader_case_demonstrations

def main():
    parser = argparse.ArgumentParser(description="Generate 5 ALPR Diagnostic Cases for VGG-16")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--ckpt", type=str, default="models/vgg16_plate_classifier.pth", help="Checkpoint path")
    parser.add_argument("--out_dir", type=str, default="results/vgg16_reader", help="Output directory")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if not os.path.exists(args.ckpt):
        print("\n" + "="*80)
        print(f"  [!] Checkpoint '{args.ckpt}' not found!")
        print("  Please train VGG-16 first by running:")
        print("    python run_cnn_vgg16.py --epochs 3 --device cpu")
        print("="*80 + "\n")
        sys.exit(1)

    print("\n" + "="*75)
    print("  GENERATING VGG-16 ALPR DIAGNOSTIC TEST CASES")
    print("="*75)

    run_reader_case_demonstrations(
        arch_name="VGG16",
        ckpt_path=args.ckpt,
        output_dir=args.out_dir,
        device=device
    )

if __name__ == "__main__":
    main()
