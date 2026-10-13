"""VGG16-only transfer learning with source-image separation and auditable results.

The architecture/checkpoint format is identical to the team's individual runner.
Frozen, deterministic activations are cached; every crop still passes through all
VGG16 convolutional and penultimate dense layers. Timings use the full network.
"""
import hashlib
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from modules.cnn_individual_runner import build_model
from modules.detector_classical import ClassicalPlateDetector
from modules.evaluate_detection import compute_iou

OUT = Path("results/vgg16")
CACHE = Path("data/vgg16_cache")
CHECKPOINT = Path("models/vgg16_plate_classifier.pth")
NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def choose_device(requested="auto"):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return requested


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def update_only_vgg_row(path, metrics):
    """Fail on malformed tables; preserve every non-VGG16 record."""
    path = Path(path)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=metrics)
    if "Architecture" not in df.columns:
        raise ValueError(f"Malformed comparison table: {path}")
    other = df[df.Architecture != "VGG16"].copy()
    updated = pd.concat([other, pd.DataFrame([metrics])], ignore_index=True)
    tmp = path.with_suffix(".tmp")
    updated.to_csv(tmp, index=False)
    check = pd.read_csv(tmp)
    pd.testing.assert_frame_equal(other.reset_index(drop=True),
                                  check[check.Architecture != "VGG16"].reset_index(drop=True),
                                  check_dtype=False)
    os.replace(tmp, path)


def crop_records(annotations, split):
    detector = ClassicalPlateDetector()
    records = []
    for i, item in enumerate(annotations):
        image = cv2.imread(item["filepath"])
        if image is None:
            raise FileNotFoundError(item["filepath"])
        h, w = image.shape[:2]
        boxes = [(b, 1) for b in item["bboxes"]]
        boxes += [(b, 0) for b in detector.detect(image)
                  if max((compute_iou(b, g) for g in item["bboxes"]), default=0) < .15]
        for box, label in boxes:
            x, y, bw, bh = map(int, box)
            x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
            bw, bh = min(bw, w - x), min(bh, h - y)
            if min(bw, bh) < 5:
                continue
            records.append(dict(filepath=item["filepath"], image_id=item["image_id"], source_group=item["source_group"],
                                box=[x, y, bw, bh], label=label, split=split))
        if (i + 1) % 500 == 0 or i == len(annotations) - 1:
            print(f"{split}: extracted {len(records)} crops from {i+1}/{len(annotations)} images", flush=True)
    if set(r["label"] for r in records) != {0, 1}:
        raise ValueError(f"{split} requires both classes")
    return records


class Crops(Dataset):
    def __init__(self, records, flip=False):
        self.records, self.flip = records, flip

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        image = cv2.imread(r["filepath"])
        if image is None:
            raise FileNotFoundError(r["filepath"])
        x, y, w, h = r["box"]
        crop = cv2.resize(image[y:y+h, x:x+w], (224, 224))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if self.flip:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        return NORMALIZE(Image.fromarray(rgb)), r["label"]


def extract_features(model, records, device, batch_size, name, flip=False):
    signature = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    target = CACHE / f"{name}_{signature[:12]}.npy"
    if target.exists():
        feature = np.load(target)
        if feature.shape == (len(records), 4096) and np.isfinite(feature).all():
            print(f"Reusing verified feature cache: {name}", flush=True)
            return torch.from_numpy(feature)
        raise ValueError(f"Invalid cache {target}")
    loader = DataLoader(Crops(records, flip), batch_size=batch_size, shuffle=False)
    feature = np.empty((len(records), 4096), dtype=np.float32)
    model.eval()
    start, offset = time.perf_counter(), 0
    with torch.inference_mode():
        for step, (x, _) in enumerate(loader):
            x = model.features(x.to(device))
            x = model.avgpool(x).flatten(1)
            x = model.classifier[:6](x).cpu().numpy()
            feature[offset:offset+len(x)] = x
            offset += len(x)
            if step % 100 == 0 or offset == len(records):
                print(f"VGG16 {name}: {offset}/{len(records)} crops; {time.perf_counter()-start:.1f}s", flush=True)
    if not np.isfinite(feature).all():
        raise ValueError("Non-finite VGG16 features")
    with target.with_suffix(".tmp").open("wb") as f:
        np.save(f, feature)
    os.replace(target.with_suffix(".tmp"), target)
    return torch.from_numpy(feature)


def classification_metrics(targets, probs):
    preds = np.asarray(probs) >= .5
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
    return {"Accuracy": float(accuracy_score(targets, preds)), "Precision": float(precision),
            "Recall": float(recall), "F1-Score": float(f1),
            "confusion_matrix": confusion_matrix(targets, preds, labels=[0, 1]).tolist()}


def full_inference(model, records, device, batch_size=8):
    loader = DataLoader(Crops(records), batch_size=batch_size, shuffle=False)
    probs, latencies = [], []
    model.eval()
    with torch.inference_mode():
        sample = next(iter(loader))[0][:1].to(device)
        for _ in range(3):
            model(sample)
        sync(device)
        for x, _ in loader:
            x = x.to(device)
            sync(device)
            start = time.perf_counter()
            p = model(x).softmax(1)[:, 1]
            sync(device)
            latencies.extend([(time.perf_counter() - start) * 1000 / len(x)] * len(x))
            probs.extend(p.cpu().tolist())
    return np.array(probs), float(np.mean(latencies))


def train(epochs=30, batch_size=8, lr=1e-3, device="auto", **unused):
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / ".gitignore").write_text("*\n!.gitignore\n")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = choose_device(device)
    started = time.perf_counter()
    training = json.loads(Path("data/annotations_train.json").read_text())
    testing = json.loads(Path("data/annotations_test.json").read_text())
    # Numeric IDs are local to each source split. Use decoded image content to
    # prevent both same-scene crops and exact duplicate images crossing splits.
    for item in training + testing:
        image = cv2.imread(item["filepath"])
        if image is None:
            raise FileNotFoundError(item["filepath"])
        item["source_group"] = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
    test_groups = {a["source_group"] for a in testing}
    excluded = [a for a in training if a["source_group"] in test_groups]
    training = [a for a in training if a["source_group"] not in test_groups]
    groups = sorted({a["source_group"] for a in training})
    train_groups, val_groups = train_test_split(groups, test_size=.2, random_state=42)
    train_groups, val_groups = set(train_groups), set(val_groups)
    train_ann = [a for a in training if a["source_group"] in train_groups]
    val_ann = [a for a in training if a["source_group"] in val_groups]
    if train_groups & val_groups or (train_groups | val_groups) & test_groups:
        raise ValueError("Image content overlap across splits")
    manifest_path = OUT / "crop_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["source_sha256"] == hashlib.sha256(Path("data/annotations_train.json").read_bytes() + Path("data/annotations_test.json").read_bytes()).hexdigest()
        records = manifest["records"]
    else:
        records = {name: crop_records(a, name) for name, a in [("train", train_ann), ("validation", val_ann), ("test", testing)]}
        manifest = {"source_sha256": hashlib.sha256(Path("data/annotations_train.json").read_bytes() + Path("data/annotations_test.json").read_bytes()).hexdigest(),
                    "image_counts": dict(train=len(train_ann), validation=len(val_ann), test=len(testing)),
                    "excluded_test_duplicates": [a["filepath"] for a in excluded], "records": records}
        write_json(manifest_path, manifest)
    print(f"Training on {device}; image counts {manifest['image_counts']}", flush=True)
    model = build_model("VGG16", pretrained=True).to(device).eval()
    features = {s: extract_features(model, records[s], device, batch_size, s) for s in ("train", "validation")}
    flipped = extract_features(model, records["train"], device, batch_size, "train_flip", flip=True)
    feature_seconds = time.perf_counter() - started
    labels = {s: torch.tensor([r["label"] for r in records[s]]) for s in records}
    head = model.classifier[6].cpu()
    counts = torch.bincount(labels["train"], minlength=2)
    weights = len(labels["train"]) / (2 * counts.float())
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    history, best_state, best_score, stale = [], None, (-1., -float("inf")), 0
    head_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.
        for indices in torch.randperm(len(labels["train"])).split(32):
            x = features["train"][indices].clone()
            mask = torch.rand(len(indices)) < .2
            x[mask] = flipped[indices[mask]]
            optimizer.zero_grad()
            loss = criterion(head(x), labels["train"][indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        head.eval()
        with torch.no_grad():
            logits = head(features["validation"])
            val_loss = torch.nn.functional.cross_entropy(logits, labels["validation"]).item()
            metrics = classification_metrics(labels["validation"].numpy(), logits.softmax(1)[:, 1].numpy())
        entry = dict(epoch=epoch, train_loss=total_loss/len(labels["train"]), validation_loss=val_loss, **metrics)
        history.append(entry)
        print(f"Epoch {epoch}: loss={entry['train_loss']:.4f}; validation F1={metrics['F1-Score']:.4f}, recall={metrics['Recall']:.4f}", flush=True)
        score = (metrics["F1-Score"], -val_loss)
        if score > best_score:
            best_score, stale = score, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
        else:
            stale += 1
        if epoch >= 3 and stale >= 6:
            print("Validation did not improve for six epochs; restoring best checkpoint.", flush=True)
            break
    head.load_state_dict(best_state)
    model.to(device).eval()
    train_seconds = feature_seconds + time.perf_counter() - head_start
    test_probs, accelerated_ms = full_inference(model, records["test"], device, batch_size)
    metrics = classification_metrics(labels["test"].numpy(), test_probs)
    if metrics["Recall"] == 0 or metrics["Precision"] == 0 or len(np.unique(test_probs >= .5)) < 2:
        raise RuntimeError(f"VGG16 collapsed to a useless classifier: {metrics}")
    # Publish CPU timings to match the existing CSV's CPU convention.
    model.cpu()
    cpu_probs, cpu_ms = full_inference(model, records["test"], "cpu", batch_size=1)
    if not np.allclose(cpu_probs, test_probs, atol=2e-3):
        raise RuntimeError("CPU and acceleration predictions disagree beyond tolerance")
    if not np.array_equal(cpu_probs >= .5, test_probs >= .5):
        raise RuntimeError("CPU and acceleration classification decisions differ")
    metrics = classification_metrics(labels["test"].numpy(), cpu_probs)
    params = sum(p.numel() for p in model.parameters())
    row = {"Architecture": "VGG16", "Total Params (M)": round(params/1e6, 2),
           "Size (MB)": round(params*4/1024**2, 2),
           **{k: round(metrics[k], 4) for k in ("Accuracy", "Precision", "Recall", "F1-Score")},
           "Train Time (s)": round(train_seconds, 2), "Latency (ms)": round(cpu_ms, 2), "FPS": round(1000/cpu_ms, 1)}
    metadata = dict(seed=42, training_device=device, inference_device="cpu", batch_size=batch_size,
                    max_epochs=epochs, completed_epochs=len(history), best_epoch=best_epoch,
                    image_counts=manifest["image_counts"], excluded_test_duplicates=manifest["excluded_test_duplicates"],
                    crop_counts={s: len(v) for s,v in records.items()},
                    class_counts={s: torch.bincount(labels[s], minlength=2).tolist() for s in labels},
                    feature_extraction_seconds=feature_seconds, train_seconds=train_seconds,
                    accelerated_batch_latency_ms=accelerated_ms, cpu_batch_one_latency_ms=cpu_ms,
                    trainable_parameters=sum(p.numel() for p in head.parameters()),
                    test_metrics=metrics, csv_metrics=row, history=history)
    CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save({"architecture": "VGG16", "state_dict": model.state_dict(), "metrics": row,
                "classes": ["Non-Plate", "Plate"], "training_metadata": metadata}, CHECKPOINT)
    write_json(OUT / "training_details.json", metadata)
    predictions = pd.DataFrame(records["test"])
    predictions["plate_probability"] = cpu_probs
    predictions["prediction"] = (cpu_probs >= .5).astype(int)
    predictions.to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame(history).drop(columns="confusion_matrix").to_csv(OUT / "training_history.csv", index=False)
    update_only_vgg_row("results/cnn_comparison_results.csv", row)
    print(json.dumps(metadata["test_metrics"], indent=2), flush=True)
    return row, metadata
