#!/usr/bin/env python3
"""Build a clean assignment submission ZIP (no .dvc/cache)."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "image-classification-dvc-submission.zip"

EXCLUDE_DIRS = {
    ".git",
    ".dvc/cache",
    ".dvc/tmp",
    "__pycache__",
    ".venv",
    "venv",
    "data/raw",
    "data/processed",
    "data/features",
    "models/classifier",
}

EXCLUDE_SUFFIXES = {".pyc", ".zip", ".yaml.bak"}


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    parts = rel.split("/")
    if parts[0] in EXCLUDE_DIRS:
        return False
    for ex in EXCLUDE_DIRS:
        if rel == ex or rel.startswith(ex + "/"):
            return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if path.name in {".DS_Store"}:
        return False
    return True


def main() -> None:
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and should_include(p))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arc = f"image-classification-dvc/{path.relative_to(ROOT).as_posix()}"
            zf.write(path, arc)
    print(f"Created: {OUTPUT}")
    print(f"Files: {len(files)}")
    print(f"Size KB: {OUTPUT.stat().st_size / 1024:.1f}")


if __name__ == "__main__":
    main()
