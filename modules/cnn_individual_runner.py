"""
Core Runner for Individual CNN Architectures.
Provides:
- Dataset loading and caching (Plate vs Non-Plate binary classification)
- Model construction with transfer learning (frozen backbone, trainable head)
- Training, validation, latency and throughput benchmarking
- Checkpoint saving to models/
- Non-destructive updates to results/cnn_comparison_results.csv
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

# Supported Architectures
ARCHITECTURES = {
    "AlexNet": {
        "builder": lambda weights: models.alexnet(weights=weights),
        "head_attr": "classifier",
        "head_idx": 6,
    },
    "VGG16": {
        "builder": lambda weights: models.vgg16(weights=weights),
        "head_attr": "classifier",
        "head_idx": 6,
    },
    "GoogLeNet": {
        "builder": lambda weights: models.googlenet(weights=weights, aux_logits=False, init_weights=False),
        "head_attr": "fc",
        "head_idx": None,
    },
    "ResNet50": {
        "builder": lambda weights: models.resnet50(weights=weights),
        "head_attr": "fc",
        "head_idx": None,
    },
    "DenseNet121": {
        "builder": lambda weights: models.densenet121(weights=weights),
        "head_attr": "classifier",
        "head_idx": None,
    }
}

class FastCropDataset(Dataset):
    """Memory-efficient Dataset for pre-cached crops."""
    def __init__(self, images, labels, transform=None):
        self.images = images  # (N, H, W, 3) uint8 BGR
        self.labels = labels  # (N,) int64
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(self.images[idx], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        if self.transform:
            pil_img = self.transform(pil_img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return pil_img, label

def load_or_create_crop_data(cache_path="data/crop_dataset_224.npz"):
    """Loads crop dataset from cache, or creates it if not present."""
    if not os.path.exists(cache_path):
        from prepare_crops import prepare_crops
        prepare_crops(out_path=cache_path)

    data = np.load(cache_path)
    images = data["images"]
    labels = data["labels"]
    print(f"[+] Loaded {len(images)} crops from cache '{cache_path}': {np.sum(labels == 1)} Plates, {np.sum(labels == 0)} Non-Plates.")
    return images, labels

def build_model(arch_name, num_classes=2, pretrained=True):
    """Builds model, configures transfer learning (frozen backbone, new head)."""
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch_name}'. Supported: {list(ARCHITECTURES.keys())}")

    cfg = ARCHITECTURES[arch_name]
    weights = "DEFAULT" if pretrained else None
    model = cfg["builder"](weights)

    # Freeze backbone weights
    if pretrained:
        for p in model.parameters():
            p.requires_grad = False

    # Replace classification head
    if cfg["head_attr"] == "classifier":
        if cfg["head_idx"] is not None:
            in_features = model.classifier[cfg["head_idx"]].in_features
            model.classifier[cfg["head_idx"]] = nn.Linear(in_features, num_classes)
            for p in model.classifier[cfg["head_idx"]].parameters():
                p.requires_grad = True
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            for p in model.classifier.parameters():
                p.requires_grad = True
    elif cfg["head_attr"] == "fc":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        for p in model.fc.parameters():
            p.requires_grad = True

    return model

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def update_cnn_results_csv(metrics, csv_path="results/cnn_comparison_results.csv"):
    """
    Non-destructively updates the CNN comparison CSV.
    Updates the existing row for this architecture or appends it.
    """
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    arch = metrics["Architecture"]
    
    if not df.empty and "Architecture" in df.columns:
        if arch in df["Architecture"].values:
            # Update existing row
            idx = df.index[df["Architecture"] == arch].tolist()[0]
            for col, val in metrics.items():
                df.at[idx, col] = val
        else:
            # Append row
            new_row = pd.DataFrame([metrics])
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.DataFrame([metrics])

    # Reorder columns standardly if possible
    preferred_order = [
        "Architecture", "Total Params (M)", "Size (MB)",
        "Accuracy", "Precision", "Recall", "F1-Score",
        "Train Time (s)", "Latency (ms)", "FPS"
    ]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df[cols]
    
    df.to_csv(csv_path, index=False)
    print(f"[OK] Updated results table saved to '{csv_path}'.")
    return df

def train_and_benchmark(
    arch_name,
    epochs=3,
    batch_size=32,
    lr=1e-3,
    device="cpu",
    pretrained=True,
    test_split=0.25,
    save_model=True
):
    """Executes full training and benchmarking lifecycle for a single CNN architecture."""
    print("="*75)
    print(f"  RUNNING BENCHMARK FOR: {arch_name.upper()}")
    print(f"  Configuration: epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}, pretrained={pretrained}")
    print("="*75)

    # 1. Load Data
    images, labels = load_or_create_crop_data()
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_split, random_state=42, stratify=labels
    )
    print(f"[+] Data split: {len(X_train)} training crops, {len(X_test)} evaluation crops.")

    # 2. Data Transforms & Loaders
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = FastCropDataset(X_train, y_train, transform=transform_train)
    test_ds = FastCropDataset(X_test, y_test, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Model Setup
    print(f"[*] Building {arch_name} (pretrained={pretrained})...")
    model = build_model(arch_name, num_classes=2, pretrained=pretrained)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    print(f"[+] Total Parameters: {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"[+] Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f} M)")
    print(f"[+] Model Weights Size: {model_size_mb:.2f} MB")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    print(f"\n[*] Starting Training ({epochs} epochs)...")
    t_train_start = time.perf_counter()
    model.train()
    for ep in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        ep_t0 = time.perf_counter()
        for batch_imgs, batch_lbls in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_lbls = batch_lbls.to(device)

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_lbls)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_lbls).sum().item()
            total += len(batch_lbls)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        ep_time = time.perf_counter() - ep_t0
        print(f"  Epoch {ep}/{epochs} - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc*100:.2f}% ({ep_time:.1f}s)")

    train_time_sec = time.perf_counter() - t_train_start
    print(f"[OK] Training complete in {train_time_sec:.2f}s.")

    # 5. Evaluation Loop & Latency Benchmarking
    print("\n[*] Evaluating on Test Set & Benchmarking Latency...")
    model.eval()
    all_preds = []
    all_targets = []
    latencies_per_sample = []

    # Warmup
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    with torch.no_grad():
        for batch_imgs, batch_lbls in test_loader:
            batch_imgs = batch_imgs.to(device)
            t0 = time.perf_counter()
            outputs = model(batch_imgs)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            lat_ms = (t1 - t0) * 1000.0 / len(batch_imgs)
            latencies_per_sample.extend([lat_ms] * len(batch_imgs))

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_lbls.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = np.mean(all_preds == all_targets)
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
    avg_latency_ms = np.mean(latencies_per_sample) if latencies_per_sample else 0.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    metrics = {
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

    print("\n" + "="*50)
    print(f"  RESULTS FOR {arch_name.upper()}")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print("="*50)

    # 6. Save Model Checkpoint
    if save_model:
        os.makedirs("models", exist_ok=True)
        ckpt_path = f"models/{arch_name.lower()}_plate_classifier.pth"
        torch.save({
            "architecture": arch_name,
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "classes": ["Non-Plate", "Plate"]
        }, ckpt_path)
        print(f"[+] Saved trained weights to '{ckpt_path}'.")

    # 7. Update CSV
    df = update_cnn_results_csv(metrics)
    return metrics, df
