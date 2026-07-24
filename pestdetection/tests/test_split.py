"""The split must be deterministic and leakage-free."""

from __future__ import annotations

import csv

from leafscan.data import build_split, load_split


def _make_dataset(root, per_class):
    from PIL import Image

    for cls, n in per_class.items():
        (root / cls).mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (8, 8)).save(root / cls / f"{i}.jpg")


def test_split_is_deterministic(tmp_path):
    data = tmp_path / "data"
    _make_dataset(data, {"a": 30, "b": 20, "c": 40})
    csv1, csv2 = tmp_path / "s1.csv", tmp_path / "s2.csv"
    build_split(data, csv1, seed=7)
    build_split(data, csv2, seed=7)
    assert csv1.read_text() == csv2.read_text()


def test_no_leakage_and_floor(tmp_path):
    data = tmp_path / "data"
    _make_dataset(data, {"a": 30, "b": 12, "c": 40})
    split_csv = tmp_path / "s.csv"
    build_split(data, split_csv, seed=1, min_eval_per_class=3)

    rows = list(csv.DictReader(open(split_csv)))
    by_file: dict[str, set] = {}
    for r in rows:
        by_file.setdefault(r["filename"], set()).add(r["split"])
    # every file appears in exactly one split
    assert all(len(s) == 1 for s in by_file.values())

    class_names, splits = load_split(split_csv)
    assert class_names == ["a", "b", "c"]
    # per-class eval floor respected for the small class
    import collections

    for s in ("val", "test"):
        counts = collections.Counter(lbl for _f, lbl in splits[s])
        assert min(counts.values()) >= 3
