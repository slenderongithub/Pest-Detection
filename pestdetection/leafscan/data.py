"""Reproducible, leakage-safe data layer.

A deterministic stratified train/val/test split (with a per-class floor so the smallest
class stays evaluable) is written to a committed CSV. Because the split is committed and
the dataset is hashed, the reported metrics are verifiable on a clean clone even though
the images themselves stay gitignored.
"""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_classes(data_dir: str | Path) -> list[str]:
    """Class names in the canonical (sorted) order torchvision's ImageFolder uses."""
    d = Path(data_dir)
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def list_images(data_dir: str | Path) -> list[tuple[str, str]]:
    """Return ``(relative_path, class_name)`` for every image, sorted for determinism."""
    d = Path(data_dir)
    records: list[tuple[str, str]] = []
    for cls in list_classes(d):
        for f in sorted((d / cls).iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                records.append((str(f.relative_to(d)), cls))
    return records


def data_sha256(data_dir: str | Path) -> str:
    """Stable fingerprint over the sorted ``(relpath, size)`` inventory.

    Detects added/removed/renamed/resized files. It is an inventory fingerprint, not a full
    content hash, so an in-place edit that preserves a file's byte size is not caught —
    which is an acceptable, cheap provenance signal for a gitignored image dataset.
    """
    d = Path(data_dir)
    h = hashlib.sha256()
    for relpath, _cls in list_images(d):
        size = (d / relpath).stat().st_size
        h.update(f"{relpath}:{size}\n".encode())
    return h.hexdigest()


def build_split(
    data_dir: str | Path,
    split_csv: str | Path,
    *,
    seed: int = 1337,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    min_eval_per_class: int = 5,
) -> dict[str, int]:
    """Write a deterministic per-class stratified split CSV. Returns per-split counts.

    Each class is shuffled with the seeded RNG and partitioned so val and test each get at
    least ``min_eval_per_class`` samples (capacity permitting) and train keeps at least one.
    """
    d = Path(data_dir)
    rng = random.Random(seed)
    train_r, val_r, test_r = ratios
    rows: list[tuple[str, str, str]] = []
    for cls in list_classes(d):
        files = sorted(str(Path(cls) / f.name) for f in (d / cls).iterdir()
                       if f.suffix.lower() in IMAGE_EXTS)
        rng.shuffle(files)
        n = len(files)
        n_test = min(max(int(round(n * test_r)), min_eval_per_class), max(n - 2, 0))
        n_val = min(max(int(round(n * val_r)), min_eval_per_class), max(n - n_test - 1, 0))
        n_train = n - n_val - n_test
        assigned = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
        for relpath, split in zip(files, assigned, strict=True):
            rows.append((relpath, cls, split))

    out = Path(split_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "label", "split"])
        w.writerows(sorted(rows))

    counts = {"train": 0, "val": 0, "test": 0}
    for _p, _c, s in rows:
        counts[s] += 1
    return counts


def load_split(split_csv: str | Path) -> tuple[list[str], dict[str, list[tuple[str, int]]]]:
    """Read the split CSV. Returns ``(class_names, {split: [(relpath, label_idx)]})``."""
    rows: list[tuple[str, str, str]] = []
    with open(split_csv, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append((r["filename"], r["label"], r["split"]))
    class_names = sorted({label for _f, label, _s in rows})
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    splits: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    for relpath, label, split in rows:
        splits.setdefault(split, []).append((relpath, class_to_idx[label]))
    return class_names, splits


def class_weights(train_records: list[tuple[str, int]], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights (mean ~1) for a weighted CrossEntropy."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _relpath, label in train_records:
        counts[label] += 1
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


@dataclass
class PestDataset(Dataset):
    data_dir: Path
    records: list[tuple[str, int]]
    transform: object

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        relpath, label = self.records[idx]
        with Image.open(self.data_dir / relpath) as im:
            image = im.convert("RGB")
        tensor = self.transform(image) if self.transform else image
        return tensor, label


def seed_worker(_worker_id: int) -> None:
    """Deterministic per-worker seeding for DataLoader reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    import numpy as np

    np.random.seed(worker_seed)
