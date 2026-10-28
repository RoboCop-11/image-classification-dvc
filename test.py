"""Automated tests for the Olivetti Faces classification pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_collection import _save_split, _write_manifest, collect_synthetic
from src.utils import (
    PARAMS_PATH,
    class_names_olivetti,
    ensure_dir,
    load_json,
    load_params,
    save_json,
    set_seed,
)


class UtilsTests(unittest.TestCase):
    def test_load_params_has_pipeline_sections(self) -> None:
        params = load_params()
        for key in (
            "base",
            "data_collection",
            "data_processing",
            "feature_selection",
            "model_training",
            "model_evaluation",
        ):
            self.assertIn(key, params)

    def test_class_names_olivetti(self) -> None:
        names = class_names_olivetti()
        self.assertEqual(len(names), 40)
        self.assertEqual(len(set(names)), 40)
        self.assertEqual(names[0], "s00")
        self.assertEqual(names[39], "s39")

    def test_set_seed_is_idempotent(self) -> None:
        set_seed(7)
        set_seed(7)

    def test_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            payload = {"accuracy": 0.91, "classes": ["s00", "s01"]}
            save_json(payload, path)
            self.assertEqual(load_json(path), payload)


class DataCollectionTests(unittest.TestCase):
    def test_collect_synthetic_shapes(self) -> None:
        names = class_names_olivetti()
        x_train, y_train, x_test, y_test = collect_synthetic(80, 20, names, image_size=64)
        self.assertEqual(x_train.shape, (80, 64, 64))
        self.assertEqual(x_test.shape[0], 20)
        self.assertTrue(np.all(y_train >= 0) and np.all(y_train < len(names)))
        self.assertTrue(np.all(y_test >= 0) and np.all(y_test < len(names)))

    def test_save_split_writes_png_files(self) -> None:
        names = class_names_olivetti()
        x_train, y_train, _, _ = collect_synthetic(40, 10, names, image_size=64)
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp) / "train"
            count = _save_split(x_train, y_train, split_dir, names)
            self.assertEqual(count, 40)
            png_files = list(split_dir.rglob("*.png"))
            self.assertEqual(len(png_files), 40)


class PipelineIntegrationTests(unittest.TestCase):
    """Run a fast synthetic end-to-end pipeline (used by GitHub Actions CI)."""

    def test_full_pipeline_produces_metrics(self) -> None:
        original_params = PARAMS_PATH.read_text(encoding="utf-8")
        backup_path = PARAMS_PATH.with_suffix(".yaml.bak")
        shutil.copy2(PARAMS_PATH, backup_path)
        try:
            params = yaml.safe_load(original_params)
            params["data_collection"]["dataset_name"] = "synthetic"
            params["data_collection"]["max_samples"] = 200
            params["model_training"]["epochs"] = 3
            params["model_training"]["batch_size"] = 16
            PARAMS_PATH.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")

            stages = [
                [sys.executable, "-m", "src.data_collection"],
                [sys.executable, "-m", "src.data_processing"],
                [sys.executable, "-m", "src.feature_selection"],
                [sys.executable, "-m", "src.model_training"],
                [sys.executable, "-m", "src.model_evaluation"],
            ]
            for cmd in stages:
                completed = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"Stage failed: {' '.join(cmd)}\n{completed.stdout}\n{completed.stderr}",
                )

            metrics_path = ROOT / "reports" / "metrics.json"
            self.assertTrue(metrics_path.is_file(), "reports/metrics.json was not created")
            metrics = load_json(metrics_path)
            self.assertIn("accuracy", metrics)
            self.assertGreaterEqual(float(metrics["accuracy"]), 0.0)
            self.assertLessEqual(float(metrics["accuracy"]), 1.0)

            meta_path = ROOT / "data" / "raw" / "collection_meta.json"
            self.assertTrue(meta_path.is_file())
            meta = load_json(meta_path)
            self.assertEqual(meta["source"], "synthetic")
            self.assertEqual(meta["num_classes"], 40)
        finally:
            PARAMS_PATH.write_text(original_params, encoding="utf-8")
            backup_path.unlink(missing_ok=True)


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(UtilsTests))
    suite.addTests(loader.loadTestsFromTestCase(DataCollectionTests))
    suite.addTests(loader.loadTestsFromTestCase(PipelineIntegrationTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
