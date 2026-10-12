"""
CNN Architecture Comparison Module for License Plate vs Non-Plate Verification.
Evaluates 5 Classic Computer Vision Architectures:
1. AlexNet
2. VGG16
3. GoogLeNet (InceptionV1)
4. ResNet50
5. DenseNet121

Supports:
- Dual execution: Fast CPU mode (< 1 min total) and Full GPU Training mode.
- Metrics: Accuracy, Precision, Recall, F1, Model Parameter Count, Model Size (MB), Latency (ms), FPS.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CNN_ARCHITECTURES = ["AlexNet", "VGG16", "GoogLeNet", "ResNet50", "DenseNet121"]

class CropDataset(Dataset):
    def __init__(self, crops, labels, transform=None):
        self.crops = crops
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        if self.transform:
            pil_img = self.transform(pil_img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return pil_img, label

def build_model(arch_name, num_classes=2, pretrained=False):
    """
    Constructs the model and replaces its final classification layer.
    """
    weights = "DEFAULT" if pretrained else None

    if arch_name == "AlexNet":
        model = models.alexnet(weights=weights)
        in_feats = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feats, num_classes)
    elif arch_name == "VGG16":
        model = models.vgg16(weights=weights)
        in_feats = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feats, num_classes)
    elif arch_name == "GoogLeNet":
        model = models.googlenet(weights=weights, aux_logits=False, init_weights=False)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif arch_name == "ResNet50":
        model = models.resnet50(weights=weights)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif arch_name == "DenseNet121":
        model = models.densenet121(weights=weights)
        in_feats = model.classifier.in_features
        model.classifier = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    return model

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def train_and_eval_cnn(arch_name, train_loader, test_loader, epochs=3, device="cpu", pretrained=False):
    """
    Trains and evaluates a single CNN architecture.
    """
    model = build_model(arch_name, num_classes=2, pretrained=pretrained)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    model_size_mb = (total_params * 4) / (1024 * 1024)

    # Freeze backbone layers when using pretrained weights (transfer learning)
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze only the final classification head
        if arch_name in ("AlexNet", "VGG16"):
            for param in model.classifier[-1].parameters():
                param.requires_grad = True
        elif arch_name in ("ResNet50", "GoogLeNet"):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif arch_name == "DenseNet121":
            for param in model.classifier.parameters():
                param.requires_grad = True

    trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lr = 1e-3 if not pretrained else 5e-4
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training
    t_train_start = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    train_time_sec = time.perf_counter() - t_train_start

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    latencies = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            t0 = time.perf_counter()
            outputs = model(inputs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0 / len(inputs))

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = np.mean(all_preds == all_targets)
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
    avg_latency_ms = np.mean(latencies) if len(latencies) > 0 else 0.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "Architecture": arch_name,
        "Total Params (M)": round(total_params / 1e6, 2),
        "Size (MB)": round(model_size_mb, 2),
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1-Score": round(float(f1), 4),
        "Train Time (s)": round(float(train_time_sec), 2),
        "Latency (ms)": round(float(avg_latency_ms), 2),
        "FPS": round(float(fps), 1)
    }

def run_cnn_benchmark(crops, labels, epochs=3, img_size=112, device="cpu", pretrained=False):
    """
    Runs the benchmark across all 5 architectures.
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        crops, labels, test_size=0.25, random_state=42, stratify=labels
    )

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = CropDataset(X_train, y_train, transform=transform)
    test_ds = CropDataset(X_test, y_test, transform=transform)

    batch_size = 16 if device == "cpu" else 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = []
    print(f"[*] Benchmarking 5 CNN Architectures on {device.upper()} (img_size={img_size}, epochs={epochs})...")

    for arch in CNN_ARCHITECTURES:
        print(f"  -> Training & Evaluating {arch}...")
        metrics = train_and_eval_cnn(arch, train_loader, test_loader, epochs=epochs, device=device, pretrained=pretrained)
        results.append(metrics)
        print(f"     [OK] {arch}: Acc={metrics['Accuracy']}, F1={metrics['F1-Score']}, Latency={metrics['Latency (ms)']}ms, FPS={metrics['FPS']}")

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    from modules.ml_classifiers import build_crop_dataset
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_images", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    anno_path = "data/annotations_test.json"
    if not os.path.exists(anno_path):
        print("Run data/download_dataset.py first.")
        exit(1)

    with open(anno_path) as f:
        annos = json.load(f)

    crops, labels = build_crop_dataset(annos, max_images=args.max_images)
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA requested but not available. Falling back to CPU.")
        dev = "cpu"

    df = run_cnn_benchmark(crops, labels, epochs=args.epochs, img_size=args.img_size, device=dev, pretrained=args.pretrained)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/cnn_comparison_results.csv", index=False)
    print("\n[OK] Results saved to 'results/cnn_comparison_results.csv'.")
    print("\n" + df.to_string(index=False))
