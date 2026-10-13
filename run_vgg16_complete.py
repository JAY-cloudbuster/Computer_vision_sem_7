"""Run only the VGG16 contribution. Original teammate scripts stay intact."""
import argparse
import os
from pathlib import Path


def main():
    os.chdir(Path(__file__).resolve().parent)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["all","train","detection","reader","reader-cases","report"],default="all")
    parser.add_argument("--epochs",type=int,default=30,help="Maximum head-training epochs; validation early stopping after six stale epochs")
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--device",choices=["auto","cpu","mps","cuda"],default="auto")
    parser.add_argument("--samples",type=int,default=0,help="OCR samples; 0 evaluates every row (first 40 also reported)")
    parser.add_argument("--retrain",action="store_true",help="With --stage all, replace the existing VGG16 checkpoint by retraining")
    args=parser.parse_args()
    if args.epochs<3 or args.batch_size<1 or args.samples<0:
        parser.error("Use at least three epochs, positive batch size, and nonnegative samples")
    if args.stage in ("all","train"):
        from modules.vgg16_training import train
        if args.stage=="all" and Path("models/vgg16_plate_classifier.pth").exists() and not args.retrain:
            print("Using the existing VGG16 checkpoint. Use --retrain to train again.",flush=True)
        else:
            train(epochs=args.epochs,batch_size=args.batch_size,device=args.device)
    if args.stage in ("all","detection"):
        from modules.vgg16_evaluation import detection_cases
        detection_cases()
    if args.stage in ("all","reader"):
        from modules.vgg16_evaluation import reader,reader_cases
        reader(samples=args.samples)
        reader_cases()
    if args.stage=="reader-cases":
        from modules.vgg16_evaluation import reader_cases
        reader_cases()
    if args.stage in ("all","report"):
        from modules.vgg16_report import report
        report()


if __name__=="__main__":
    main()
