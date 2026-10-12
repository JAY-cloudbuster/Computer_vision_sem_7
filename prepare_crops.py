import json
import os
import cv2
import time
import numpy as np
from modules.ml_classifiers import build_crop_dataset

def prepare_crops(num_train_images=500, out_path="data/crop_dataset_224.npz"):
    print(f"[*] Extracting crops from first {num_train_images} train images...")
    t0 = time.time()
    
    with open("data/annotations_train.json") as f:
        train_ann = json.load(f)[:num_train_images]
    
    crops, labels = build_crop_dataset(train_ann)
    print(f"[+] Extracted {len(crops)} crops ({sum(labels)} positive, {len(labels)-sum(labels)} negative).")
    
    print("[*] Resizing to 224x224 and saving compressed cache...")
    resized = np.zeros((len(crops), 224, 224, 3), dtype=np.uint8)
    for i, c in enumerate(crops):
        resized[i] = cv2.resize(c, (224, 224))
    
    labels_arr = np.array(labels, dtype=np.int64)
    np.savez_compressed(out_path, images=resized, labels=labels_arr)
    file_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[OK] Saved {len(resized)} crops to '{out_path}' in {time.time()-t0:.2f}s (Cache size: {file_mb:.2f} MB).")
    return out_path

if __name__ == "__main__":
    prepare_crops()
