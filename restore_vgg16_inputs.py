"""Restore only missing detection images; preserve existing annotations and images."""
import io
import json
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image


def restore(split, parquet_path):
    annotations = json.loads(Path(f"data/annotations_{split}.json").read_text())
    wanted = {int(a["image_id"]): a for a in annotations}
    count = 0
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            annotation = wanted.get(int(row["image_id"]))
            if annotation is None:
                continue
            path = Path(annotation["filepath"])
            if path.exists():
                continue
            image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            assert image.size == (annotation["width"], annotation["height"])
            assert row["objects"]["bbox"] == annotation["bboxes"], "Source annotations changed"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, quality=95)
            count += 1
    missing = [a["filepath"] for a in annotations if not Path(a["filepath"]).is_file()]
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} images remain missing")
    print(f"Restored {count} {split} images; all {len(annotations)} annotation paths exist.", flush=True)


if __name__ == "__main__":
    restore("train", "/tmp/vgg16_train.parquet")
    restore("test", "/tmp/vgg16_test.parquet")
