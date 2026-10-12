"""
Dataset Downloader and Loader for License Plate Object Detection.
Source: keremberke/license-plate-object-detection (Hugging Face)
"""

import os
import io
import json
import argparse
import urllib.request
import pyarrow.parquet as pq
from PIL import Image

PARQUET_URLS = {
    "mini_train": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/refs%2Fconvert%2Fparquet/mini/train/0000.parquet",
    "mini_test": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/refs%2Fconvert%2Fparquet/mini/test/0000.parquet",
    "mini_validation": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/refs%2Fconvert%2Fparquet/mini/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/refs%2Fconvert%2Fparquet/full/test/0000.parquet",
    "validation": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/refs%2Fconvert%2Fparquet/full/validation/0000.parquet",
    "train": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/refs%2Fconvert%2Fparquet/full/train/0000.parquet",
}

def download_split(split_name="test", max_samples=None, output_dir="data"):
    """
    Downloads and extracts images and ground-truth bounding boxes for a given split.
    """
    if split_name not in PARQUET_URLS:
        raise ValueError(f"Unknown split: {split_name}. Available: {list(PARQUET_URLS.keys())}")

    url = PARQUET_URLS[split_name]
    img_dir = os.path.join(output_dir, "images", split_name)
    os.makedirs(img_dir, exist_ok=True)
    anno_file = os.path.join(output_dir, f"annotations_{split_name}.json")

    print(f"[*] Downloading parquet for split: '{split_name}' from Hugging Face...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        parquet_bytes = resp.read()
    
    print(f"[+] Download complete ({len(parquet_bytes) / (1024*1024):.2f} MB). Parsing parquet table...")
    table = pq.read_table(io.BytesIO(parquet_bytes))
    rows = table.to_pylist()

    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
        print(f"[*] Limiting to {max_samples} samples as requested.")

    annotations = []
    print(f"[*] Extracting {len(rows)} images to '{img_dir}'...")

    for idx, row in enumerate(rows):
        image_id = row.get("image_id", idx)
        width = row.get("width", 0)
        height = row.get("height", 0)
        objects = row.get("objects", {})

        # Save image
        img_filename = f"img_{image_id}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        
        raw_img_bytes = row["image"]["bytes"]
        img = Image.open(io.BytesIO(raw_img_bytes)).convert("RGB")
        actual_w, actual_h = img.size
        img.save(img_path, format="JPEG", quality=95)

        # Parse bounding boxes: [[x, y, w, h], ...] in absolute pixels
        bboxes = objects.get("bbox", [])
        categories = objects.get("category", [0] * len(bboxes))

        annotations.append({
            "image_id": image_id,
            "filename": img_filename,
            "filepath": os.path.relpath(img_path, start=".").replace("\\", "/"),
            "width": actual_w,
            "height": actual_h,
            "bboxes": bboxes,  # [x, y, w, h] format
            "categories": categories
        })

    with open(anno_file, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"[OK] Successfully saved {len(annotations)} images and metadata to '{anno_file}'.")
    return annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download license plate dataset from Hugging Face")
    parser.add_argument("--split", type=str, default="test", choices=list(PARQUET_URLS.keys()),
                        help="Split to download (default: test)")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum samples to extract (default: 100 for fast CPU testing, pass 0 or omit for all)")
    args = parser.parse_args()

    samples = None if args.max_samples <= 0 else args.max_samples
    download_split(split_name=args.split, max_samples=samples)
