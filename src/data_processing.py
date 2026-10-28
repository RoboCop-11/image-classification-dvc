"""Stage 2 — resize, normalize, split, and augment Olivetti Faces images."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from src.utils import ROOT, class_names_olivetti, ensure_dir, load_params, save_json, set_seed


def _load_split(split_dir: Path, names: list[str], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load PNG images from disk into arrays, resizing to image_size x image_size."""
    images: list[np.ndarray] = []
    labels: list[int] = []
    for class_idx, class_name in enumerate(names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.glob("*.png")):
            img = Image.open(image_path).convert("L")   # grayscale
            img = img.resize((image_size, image_size), Image.BILINEAR)
            images.append(np.asarray(img, dtype=np.uint8))
            labels.append(class_idx)
    if not images:
        raise FileNotFoundError(f"No images found under {split_dir}")
    return np.stack(images), np.array(labels, dtype=np.int64)


def _augment_one(image: np.ndarray, aug: dict, rng: np.random.Generator) -> np.ndarray:
    pil_img = Image.fromarray(image, mode="L")

    rotation = float(aug.get("rotation_range", 0))
    if rotation:
        angle = float(rng.uniform(-rotation, rotation))
        pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    if bool(aug.get("horizontal_flip", False)) and rng.random() < 0.5:
        pil_img = ImageOps.mirror(pil_img)

    zoom = float(aug.get("zoom_range", 0))
    if zoom:
        factor = 1.0 + float(rng.uniform(-zoom, zoom))
        w, h = pil_img.size
        nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
        zoomed = pil_img.resize((nw, nh), Image.BILINEAR)
        if factor >= 1:
            left = (nw - w) // 2
            top = (nh - h) // 2
            pil_img = zoomed.crop((left, top, left + w, top + h))
        else:
            canvas = Image.new("L", (w, h), color=0)
            canvas.paste(zoomed, ((w - nw) // 2, (h - nh) // 2))
            pil_img = canvas

    width_shift = float(aug.get("width_shift_range", 0))
    height_shift = float(aug.get("height_shift_range", 0))
    if width_shift or height_shift:
        dx = int(width_shift * pil_img.size[0] * float(rng.uniform(-1, 1)))
        dy = int(height_shift * pil_img.size[1] * float(rng.uniform(-1, 1)))
        pil_img = pil_img.transform(
            pil_img.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            resample=Image.BILINEAR,
            fillcolor=0,
        )

    brightness = float(aug.get("brightness_delta", 0))
    if brightness:
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.0 + float(rng.uniform(-brightness, brightness)))

    return np.asarray(pil_img, dtype=np.uint8)


def _augment_set(
    images: np.ndarray, labels: np.ndarray, aug: dict, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    extra = np.stack([_augment_one(img, aug, rng) for img in images])
    return np.concatenate([images, extra], axis=0), np.concatenate([labels, labels], axis=0)


def main() -> None:
    params = load_params()
    seed = int(params["base"]["random_seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)

    cfg = params["data_processing"]
    names = class_names_olivetti()
    image_size = int(cfg["image_size"])
    raw_dir = ROOT / params["data_collection"]["output_dir"]
    out_dir = ensure_dir(ROOT / "data" / "processed")

    x_train, y_train = _load_split(raw_dir / "train", names, image_size)
    x_test, y_test = _load_split(raw_dir / "test", names, image_size)

    # Train / validation split
    val_split = float(cfg["validation_split"])
    indices = rng.permutation(len(x_train))
    val_n = max(1, int(len(indices) * val_split))
    val_idx, train_idx = indices[:val_n], indices[val_n:]
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    # Optional augmentation
    aug = cfg.get("augmentation", {})
    if bool(aug.get("enabled", False)):
        x_train, y_train = _augment_set(x_train, y_train, aug, rng)

    # Normalize to [0, 1]
    if bool(cfg.get("normalize", True)):
        x_train_f = x_train.astype(np.float32) / 255.0
        x_val_f   = x_val.astype(np.float32)   / 255.0
        x_test_f  = x_test.astype(np.float32)  / 255.0
    else:
        x_train_f = x_train.astype(np.float32)
        x_val_f   = x_val.astype(np.float32)
        x_test_f  = x_test.astype(np.float32)

    np.savez_compressed(out_dir / "train.npz", images=x_train_f, labels=y_train)
    np.savez_compressed(out_dir / "val.npz",   images=x_val_f,   labels=y_val)
    np.savez_compressed(out_dir / "test.npz",  images=x_test_f,  labels=y_test)

    save_json(
        {
            "image_size": image_size,
            "channels": 1,
            "train_samples": int(len(y_train)),
            "val_samples":   int(len(y_val)),
            "test_samples":  int(len(y_test)),
            "normalized": bool(cfg.get("normalize", True)),
            "augmented":  bool(aug.get("enabled", False)),
            "classes": names,
        },
        out_dir / "processing_meta.json",
    )
    print(
        f"Processed splits written to {out_dir}  "
        f"(train={len(y_train)}, val={len(y_val)}, test={len(y_test)})"
    )


if __name__ == "__main__":
    main()
