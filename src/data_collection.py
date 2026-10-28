"""Stage 1 — load Olivetti Faces from sklearn and write raw PNG images to disk."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image

from src.utils import (
    ROOT,
    class_names_olivetti,
    ensure_dir,
    load_params,
    save_json,
    set_seed,
)

NUM_CLASSES = 40


def _save_split(images: np.ndarray, labels: np.ndarray, split_dir: Path, names: list[str]) -> int:
    """Save each image as a PNG under split_dir/<class_name>/."""
    count = 0
    for idx, (image, label) in enumerate(zip(images, labels)):
        class_name = names[int(label)]
        class_dir = ensure_dir(split_dir / class_name)
        out_path = class_dir / f"{class_name}_{idx:05d}.png"
        Image.fromarray(image.astype(np.uint8)).save(out_path)
        count += 1
    return count


def _write_manifest(raw_dir: Path, names: list[str]) -> None:
    rows: list[dict[str, str]] = []
    for split in ("train", "test"):
        split_dir = raw_dir / split
        if not split_dir.exists():
            continue
        for class_name in names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.glob("*.png")):
                rows.append(
                    {
                        "filepath": str(image_path.relative_to(raw_dir)),
                        "split": split,
                        "label": class_name,
                    }
                )
    with (raw_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filepath", "split", "label"])
        writer.writeheader()
        writer.writerows(rows)


def collect_olivetti(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all 400 Olivetti Faces images and do a stratified 80/20 split."""
    from sklearn.datasets import fetch_olivetti_faces

    dataset = fetch_olivetti_faces(shuffle=True, random_state=seed)
    # images are float32 [0,1] at 64x64 — convert to uint8 [0,255] for PNG saving
    x_raw = (dataset.images * 255.0).astype(np.uint8)   # (400, 64, 64)
    y_raw = dataset.target.astype(np.int64)              # (400,)

    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in range(NUM_CLASSES):
        idx = np.where(y_raw == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * 0.20)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    return (
        x_raw[np.array(train_idx)],
        y_raw[np.array(train_idx)],
        x_raw[np.array(test_idx)],
        y_raw[np.array(test_idx)],
    )


def collect_synthetic(
    n_train: int, n_test: int, names: list[str], image_size: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: random grayscale patches, one class per subject label."""
    rng = np.random.default_rng(42)
    n_classes = len(names)

    def make_image(label: int) -> np.ndarray:
        canvas = np.full((image_size, image_size), fill_value=int(label * 6) % 220 + 20, dtype=np.uint8)
        noise = rng.integers(0, 30, size=(image_size, image_size), dtype=np.uint8)
        return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    y_train = rng.integers(0, n_classes, size=n_train)
    y_test = rng.integers(0, n_classes, size=n_test)
    x_train = np.stack([make_image(int(l)) for l in y_train])
    x_test = np.stack([make_image(int(l)) for l in y_test])
    return x_train, y_train.astype(np.int64), x_test, y_test.astype(np.int64)


def main() -> None:
    params = load_params()
    seed = int(params["base"]["random_seed"])
    set_seed(seed)

    cfg = params["data_collection"]
    names = class_names_olivetti()
    raw_dir = ensure_dir(ROOT / cfg["output_dir"])
    dataset_name = str(cfg.get("dataset_name", "olivetti")).lower()

    try:
        if dataset_name == "synthetic":
            raise RuntimeError("synthetic dataset requested")
        x_train, y_train, x_test, y_test = collect_olivetti(seed)
        source = "olivetti_faces"
    except Exception as exc:
        print(f"Olivetti Faces unavailable ({exc}). Falling back to synthetic dataset.")
        n_train = int(cfg.get("max_samples", 320))
        n_test = max(n_train // 5, 80)
        x_train, y_train, x_test, y_test = collect_synthetic(n_train, n_test, names)
        source = "synthetic"

    train_count = _save_split(x_train, y_train, ensure_dir(raw_dir / "train"), names)
    test_count = _save_split(x_test, y_test, ensure_dir(raw_dir / "test"), names)
    _write_manifest(raw_dir, names)

    save_json(
        {
            "source": source,
            "train_images": train_count,
            "test_images": test_count,
            "classes": names,
            "num_classes": NUM_CLASSES,
        },
        raw_dir / "collection_meta.json",
    )
    print(f"Saved {train_count} train and {test_count} test images to {raw_dir}")


if __name__ == "__main__":
    main()
