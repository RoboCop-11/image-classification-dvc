"""Stage 5 — evaluate the trained MLP and write reports."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import ROOT, class_names_olivetti, ensure_dir, load_json, load_params, save_json, set_seed


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _load_model(history: dict):
    """Load the numpy MLP weights from disk and return a predict callable."""
    model_path = ROOT / "models" / history.get("model_path", "classifier/model.npz")
    data = np.load(model_path)
    w1, b1 = data["w1"], data["b1"]
    w2, b2 = data["w2"], data["b2"]
    w3, b3 = data["w3"], data["b3"]

    def predict_proba(x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(x @ w1 + b1, 0.0)
        h2 = np.maximum(h1 @ w2 + b2, 0.0)
        logits = h2 @ w3 + b3
        return softmax(logits)

    return predict_proba


def _save_confusion_plot(matrix: np.ndarray, names: list[str], path: Path) -> None:
    # For 40 classes, only show every 4th tick to keep the plot readable
    tick_step = 4
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(matrix, cmap="Blues")
    ticks = list(range(0, len(names), tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([names[i] for i in ticks], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels([names[i] for i in ticks], fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Olivetti Faces")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    params = load_params()
    set_seed(int(params["base"]["random_seed"]))
    eval_cfg = params["model_evaluation"]
    names = class_names_olivetti()
    average = str(eval_cfg.get("average", "macro"))

    feature_dir = ROOT / "data" / "features"
    reports_dir = ensure_dir(ROOT / "reports")
    history = load_json(ROOT / "models" / "training_history.json")

    test = np.load(feature_dir / "test.npz")
    y_true = test["labels"].astype(np.int64)
    x_features = test["features"].astype(np.float32)

    predict_proba = _load_model(history)
    proba = predict_proba(x_features)
    y_pred = proba.argmax(axis=1)

    # Only score on labels that actually appear in the test set
    present_labels = sorted(set(int(l) for l in y_true))

    metrics = {
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "precision":  float(precision_score(y_true, y_pred, average=average, zero_division=0, labels=present_labels)),
        "recall":     float(recall_score(y_true, y_pred, average=average, zero_division=0, labels=present_labels)),
        "f1":         float(f1_score(y_true, y_pred, average=average, zero_division=0, labels=present_labels)),
        "architecture": history.get("architecture", "mlp"),
        "backend":      history.get("backend", "numpy"),
        "test_samples": int(len(y_true)),
        "average": average,
    }
    save_json(metrics, reports_dir / "metrics.json")

    present_names = [names[i] for i in present_labels]
    report_text = classification_report(
        y_true, y_pred, labels=present_labels, target_names=present_names, zero_division=0
    )
    (reports_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, labels=present_labels)
    with (reports_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["actual", "predicted"])
        for actual_idx, predicted_idx in zip(y_true, y_pred):
            writer.writerow([names[int(actual_idx)], names[int(predicted_idx)]])

    if bool(eval_cfg.get("save_plots", True)):
        _save_confusion_plot(matrix, present_names, reports_dir / "confusion_matrix.png")

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Reports written to {reports_dir}")


if __name__ == "__main__":
    main()
