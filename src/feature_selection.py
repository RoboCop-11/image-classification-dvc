"""Stage 3 — reduce flattened image features before training.

Supports three methods controlled by params.yaml:
  variance   — VarianceThreshold (drop near-constant pixels)
  pca        — PCA dimensionality reduction
  selectkbest — VarianceThreshold pre-filter + SelectKBest(f_classif)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)

from src.utils import ROOT, ensure_dir, load_params, save_json, set_seed

SCORE_FUNCS = {
    "f_classif":   f_classif,
    "mutual_info": mutual_info_classif,
}


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return payload["images"], payload["labels"]


def _fit_selector(x_flat: np.ndarray, y: np.ndarray, cfg: dict):
    method = str(cfg["method"]).lower()

    if method == "variance":
        selector = VarianceThreshold(threshold=float(cfg["variance_threshold"]))
        selector.fit(x_flat)
        return selector, "variance"

    if method == "pca":
        n_components = min(
            int(cfg["pca_components"]),
            x_flat.shape[1],
            x_flat.shape[0],
        )
        selector = PCA(n_components=n_components, random_state=42)
        selector.fit(x_flat)
        return selector, "pca"

    # selectkbest (default)
    variance = VarianceThreshold(threshold=float(cfg["variance_threshold"]))
    x_var = variance.fit_transform(x_flat)
    k = min(int(cfg["k_best"]), x_var.shape[1])
    score_func = SCORE_FUNCS.get(str(cfg.get("score_func", "f_classif")), f_classif)
    kbest = SelectKBest(score_func=score_func, k=k)
    kbest.fit(x_var, y)
    return ("pipeline", variance, kbest), "selectkbest"


def _transform(selector, x_flat: np.ndarray) -> np.ndarray:
    if isinstance(selector, tuple) and selector[0] == "pipeline":
        _, variance, kbest = selector
        return kbest.transform(variance.transform(x_flat))
    return selector.transform(x_flat)


def main() -> None:
    params = load_params()
    set_seed(int(params["base"]["random_seed"]))
    cfg = params["feature_selection"]

    processed_dir = ROOT / "data" / "processed"
    out_dir = ensure_dir(ROOT / "data" / "features")

    x_train, y_train = _load_npz(processed_dir / "train.npz")
    x_val,   y_val   = _load_npz(processed_dir / "val.npz")
    x_test,  y_test  = _load_npz(processed_dir / "test.npz")

    # Flatten spatial dims (height x width → pixels)
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_val_flat   = x_val.reshape(len(x_val),   -1)
    x_test_flat  = x_test.reshape(len(x_test),  -1)

    selector, method = _fit_selector(x_train_flat, y_train, cfg)
    x_train_sel = _transform(selector, x_train_flat)
    x_val_sel   = _transform(selector, x_val_flat)
    x_test_sel  = _transform(selector, x_test_flat)

    np.savez_compressed(out_dir / "train.npz", images=x_train, labels=y_train, features=x_train_sel)
    np.savez_compressed(out_dir / "val.npz",   images=x_val,   labels=y_val,   features=x_val_sel)
    np.savez_compressed(out_dir / "test.npz",  images=x_test,  labels=y_test,  features=x_test_sel)
    joblib.dump(selector, out_dir / "selector.joblib")

    save_json(
        {
            "method": method,
            "original_feature_dim": int(x_train_flat.shape[1]),
            "selected_feature_dim": int(x_train_sel.shape[1]),
            "train_samples": int(len(y_train)),
        },
        out_dir / "feature_meta.json",
    )
    print(
        f"Selected {x_train_sel.shape[1]} / {x_train_flat.shape[1]} features "
        f"using {method} → {out_dir}"
    )


if __name__ == "__main__":
    main()
