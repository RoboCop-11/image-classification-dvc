# Olivetti Faces — Modular Image Classification with DVC

End-to-end MLOps pipeline for face recognition using the **Olivetti Faces** dataset. Each stage is a separate Python module, all tunables live in `params.yaml`, and `dvc.yaml` wires the stages into one reproducible run. No TensorFlow required — the model is a pure-NumPy MLP.

## Pipeline

```
data_collection → data_processing → feature_selection → model_training → model_evaluation
```

| Stage | Module | What it does |
|---|---|---|
| Data collection | `src/data_collection.py` | Loads Olivetti Faces from sklearn, stratified 80/20 split, saves labeled PNGs |
| Data processing | `src/data_processing.py` | Resizes to 64×64 grayscale, normalizes, train/val split, optional augmentation |
| Feature selection | `src/feature_selection.py` | PCA (default, 80 components) or SelectKBest / VarianceThreshold |
| Model training | `src/model_training.py` | Trains a two-hidden-layer NumPy MLP with Adam and early stopping |
| Model evaluation | `src/model_evaluation.py` | Writes accuracy/precision/recall/F1, classification report, confusion matrix |

## Dataset

**Olivetti Faces** (AT&T Laboratories Cambridge) — 400 grayscale face photographs: 40 subjects × 10 images each, 64×64 pixels. Loaded via `sklearn.datasets.fetch_olivetti_faces()` — no download or API key needed.

If sklearn is unavailable in CI, the collector automatically writes a synthetic grayscale dataset so the rest of the pipeline still runs.

## Model

Pure-NumPy two-hidden-layer MLP:
- Input: PCA-reduced features (default 80 eigenface components)
- 2 × hidden layers (512 units) with ReLU + dropout
- Output: 40 softmax units (one per subject)
- Optimizer: Adam (default) or SGD
- Early stopping on validation accuracy

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Initialize git + DVC:

```powershell
git init
python -m dvc init
git add .
git commit -m "Initial commit"
```

## Run the pipeline

```powershell
python -m dvc repro
python -m dvc metrics show
```

## Outputs

| Artifact | Description |
|---|---|
| `data/raw/` | PNG images per subject (train / test) |
| `data/processed/` | Normalized grayscale arrays (train / val / test NPZ) |
| `data/features/` | PCA-reduced feature arrays + selector.joblib |
| `models/classifier/model.npz` | Trained MLP weights |
| `models/training_history.json` | Loss / val_accuracy per epoch |
| `reports/metrics.json` | DVC-tracked metrics |
| `reports/classification_report.txt` | Per-class precision / recall / F1 |
| `reports/confusion_matrix.png` | 40×40 confusion matrix heatmap |

## Params (`params.yaml`)

| Param | Default | Description |
|---|---|---|
| `base.num_classes` | 40 | Olivetti subjects |
| `feature_selection.pca_components` | 80 | Eigenface dimensions |
| `model_training.learning_rate` | 0.005 | Adam/SGD step size |
| `model_training.batch_size` | 32 | Mini-batch size |
| `model_training.epochs` | 60 | Max training epochs |
| `model_training.dense_units` | 512 | Units per hidden layer |
| `model_training.dropout_rate` | 0.30 | Dropout rate |
| `model_training.optimizer` | adam | `adam` or `sgd` |
| `model_training.l2_regularization` | 0.001 | L2 weight decay |
| `model_training.early_stopping_patience` | 8 | Epochs without val improvement before stopping |

## Run tests

```powershell
python test.py
```

The integration test runs a fast synthetic end-to-end pipeline (200 samples, 3 epochs) and verifies `reports/metrics.json` is produced with a valid accuracy value.
