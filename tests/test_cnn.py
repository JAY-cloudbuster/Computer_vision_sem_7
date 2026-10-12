import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from modules.cnn_models import build_model

for arch in ["AlexNet", "VGG16", "GoogLeNet", "ResNet50", "DenseNet121"]:
    print(f"Testing {arch}...", flush=True)
    m = build_model(arch, num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print(f"  {arch} success: out shape = {out.shape}", flush=True)
print("ALL 5 PASSED!", flush=True)
