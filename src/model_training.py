"""Stage 4 — train a pure-NumPy two-hidden-layer MLP on selected features."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.utils import ROOT, load_params, save_json, set_seed


# ── activations ────────────────────────────────────────────────────────────

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# ── model ──────────────────────────────────────────────────────────────────

class MiniMLP:
    """Two-hidden-layer MLP (pure NumPy) for multi-class face recognition."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / n_in)
        scale2 = np.sqrt(2.0 / n_hidden)
        self.w1 = rng.normal(0.0, scale1, (n_in,     n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.w2 = rng.normal(0.0, scale2, (n_hidden, n_hidden)).astype(np.float32)
        self.b2 = np.zeros(n_hidden, dtype=np.float32)
        self.w3 = rng.normal(0.0, scale2, (n_hidden, n_out)).astype(np.float32)
        self.b3 = np.zeros(n_out, dtype=np.float32)
        self.opt: dict = {}

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, x: np.ndarray, dropout: float, rng: np.random.Generator, train: bool):
        h1 = relu(x @ self.w1 + self.b1)
        m1 = None
        if train and dropout > 0:
            m1 = (rng.random(h1.shape) >= dropout).astype(np.float32)
            h1 = h1 * m1 / max(1.0 - dropout, 1e-6)
        h2 = relu(h1 @ self.w2 + self.b2)
        m2 = None
        if train and dropout > 0:
            m2 = (rng.random(h2.shape) >= dropout).astype(np.float32)
            h2 = h2 * m2 / max(1.0 - dropout, 1e-6)
        logits = h2 @ self.w3 + self.b3
        return h1, h2, m1, m2, logits

    # ── Adam / SGD step ──────────────────────────────────────────────────
    def _step(self, name: str, grad: np.ndarray, lr: float, optimizer: str, t: int) -> np.ndarray:
        if optimizer == "sgd":
            return -lr * grad
        for k in (f"{name}_m", f"{name}_v"):
            if k not in self.opt:
                self.opt[k] = np.zeros_like(grad)
        b1_adam, b2_adam, eps = 0.9, 0.999, 1e-8
        self.opt[f"{name}_m"] = b1_adam * self.opt[f"{name}_m"] + (1 - b1_adam) * grad
        self.opt[f"{name}_v"] = b2_adam * self.opt[f"{name}_v"] + (1 - b2_adam) * grad ** 2
        m_hat = self.opt[f"{name}_m"] / (1 - b1_adam ** t)
        v_hat = self.opt[f"{name}_v"] / (1 - b2_adam ** t)
        return -lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── single mini-batch update ─────────────────────────────────────────
    def update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dropout: float,
        lr: float,
        optimizer: str,
        t: int,
        rng: np.random.Generator,
        l2: float = 0.0,
    ) -> float:
        n = x.shape[0]
        h1, h2, m1, m2, logits = self.forward(x, dropout, rng, True)
        probs = softmax(logits)
        oh = np.zeros_like(probs)
        oh[np.arange(n), y] = 1.0
        loss = float(-(oh * np.log(probs + 1e-8)).sum() / n)

        dl = (probs - oh) / n
        dw3 = h2.T @ dl + l2 * self.w3
        db3 = dl.sum(0)
        dh2 = dl @ self.w3.T
        if m2 is not None:
            dh2 = dh2 * m2 / max(1.0 - dropout, 1e-6)
        dh2 = dh2 * (h2 > 0)
        dw2 = h1.T @ dh2 + l2 * self.w2
        db2 = dh2.sum(0)
        dh1 = dh2 @ self.w2.T
        if m1 is not None:
            dh1 = dh1 * m1 / max(1.0 - dropout, 1e-6)
        dh1 = dh1 * (h1 > 0)
        dw1 = x.T @ dh1 + l2 * self.w1
        db1 = dh1.sum(0)

        for g in (dw1, db1, dw2, db2, dw3, db3):
            np.clip(g, -5.0, 5.0, out=g)

        self.w1 += self._step("w1", dw1, lr, optimizer, t).astype(np.float32)
        self.b1 += self._step("b1", db1, lr, optimizer, t).astype(np.float32)
        self.w2 += self._step("w2", dw2, lr, optimizer, t).astype(np.float32)
        self.b2 += self._step("b2", db2, lr, optimizer, t).astype(np.float32)
        self.w3 += self._step("w3", dw3, lr, optimizer, t).astype(np.float32)
        self.b3 += self._step("b3", db3, lr, optimizer, t).astype(np.float32)
        return loss

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _, _, _, _, logits = self.forward(x, 0.0, np.random.default_rng(0), False)
        return softmax(logits)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            w3=self.w3, b3=self.b3,
        )


# ── training loop ──────────────────────────────────────────────────────────

def main() -> None:
    params = load_params()
    cfg = params["model_training"]
    seed = int(cfg.get("seed", params["base"]["random_seed"]))
    set_seed(seed)
    rng = np.random.default_rng(seed)

    num_classes = int(params["base"]["num_classes"])
    feature_dir = ROOT / "data" / "features"
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.load(feature_dir / "train.npz")
    val_data   = np.load(feature_dir / "val.npz")

    x_train = train_data["features"].astype(np.float32)
    y_train = train_data["labels"].astype(np.int64)
    x_val   = val_data["features"].astype(np.float32)
    y_val   = val_data["labels"].astype(np.int64)

    hidden     = int(cfg["dense_units"])
    batch_size = int(cfg["batch_size"])
    epochs     = int(cfg["epochs"])
    lr         = float(cfg["learning_rate"])
    dropout    = float(cfg["dropout_rate"])
    optimizer  = str(cfg["optimizer"]).lower()
    l2         = float(cfg.get("l2_regularization", 0.001))
    patience   = int(cfg.get("early_stopping_patience", 5))

    model = MiniMLP(x_train.shape[1], hidden, num_classes, seed)
    step_t = 0
    history: list[float] = []
    val_history: list[float] = []
    best_val_acc = -1.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        batch_losses: list[float] = []
        for start in range(0, len(x_train), batch_size):
            idx = order[start: start + batch_size]
            step_t += 1
            batch_losses.append(
                model.update(x_train[idx], y_train[idx], dropout, lr, optimizer, step_t, rng, l2)
            )
        train_loss = float(np.mean(batch_losses))
        history.append(train_loss)

        val_proba = model.predict_proba(x_val)
        val_acc = float((val_proba.argmax(axis=1) == y_val).mean())
        val_history.append(val_acc)

        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            model.save(models_dir / "classifier" / "model.npz")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    # Persist training history
    save_json(
        {
            "architecture": "mlp",
            "backend": "numpy",
            "loss":        history,
            "val_accuracy": val_history,
            "epochs_ran":  len(history),
            "best_val_accuracy": best_val_acc,
            "model_path": "classifier/model.npz",
            "optimizer": optimizer,
            "dropout_rate": dropout,
        },
        models_dir / "training_history.json",
    )
    # Write format marker so model_evaluation can load correctly
    save_json(
        {"backend": "numpy", "model_file": "model.npz"},
        models_dir / "classifier" / "format.json",
    )
    print(f"Saved training artifacts to {models_dir}  best_val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
