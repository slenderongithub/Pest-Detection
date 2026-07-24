"""Generate paper/report figures from REAL committed artifacts — no fabricated numbers.

Every value is sourced from:
  - data/splits/pest_v1.csv   (committed deterministic split → class counts)
  - reports/metrics.json      (real held-out test metrics from `leafscan train/evaluate`)

Run: python scripts/generate_figures.py
Output: docs/paper_figures/*.png
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "docs" / "paper_figures"
SPLIT_CSV = REPO_ROOT / "data" / "splits" / "pest_v1.csv"
METRICS = REPO_ROOT / "reports" / "metrics.json"


def load_metrics() -> dict:
    if not METRICS.exists():
        sys.exit(f"error: {METRICS} not found. Run `leafscan train` first.")
    return json.loads(METRICS.read_text())


def fig_class_distribution() -> None:
    if not SPLIT_CSV.exists():
        print("skip class distribution: split CSV missing")
        return
    counts: Counter = Counter()
    with open(SPLIT_CSV) as fh:
        for row in csv.DictReader(fh):
            counts[row["label"]] += 1
    classes = sorted(counts, key=lambda c: counts[c])
    values = [counts[c] for c in classes]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(classes, values, color="#4f46e5", edgecolor="black")
    for b, v in zip(bars, values, strict=True):
        plt.text(v + 10, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9)
    plt.xlabel("Number of images", weight="bold")
    plt.title("Dataset class distribution (real counts, 9 pest classes)", weight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "Fig1_Class_Distribution.png", dpi=200)
    plt.close()
    print("wrote Fig1_Class_Distribution.png")


def fig_per_class_f1(metrics: dict) -> None:
    per = metrics["per_class"]
    classes = list(per)
    f1 = [per[c]["f1"] for c in classes]
    support = [per[c]["support"] for c in classes]
    order = np.argsort(f1)
    classes = [classes[i] for i in order]
    f1 = [f1[i] for i in order]
    support = [support[i] for i in order]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(classes, f1, color="#45f0a3", edgecolor="black")
    for b, v, n in zip(bars, f1, support, strict=True):
        plt.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f} (n={n})", va="center", fontsize=8)
    plt.axvline(metrics["macro_f1"], color="#ff6b6b", ls="--", label=f"macro-F1 {metrics['macro_f1']:.3f}")
    plt.xlim(0, 1.05)
    plt.xlabel("F1 (held-out test)", weight="bold")
    plt.title("Per-class F1 — real test metrics", weight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "Fig2_PerClass_F1.png", dpi=200)
    plt.close()
    print("wrote Fig2_PerClass_F1.png")


def fig_confusion(metrics: dict) -> None:
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    classes = metrics["class_names"]
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (accuracy {metrics['accuracy']:.3f})", weight="bold")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm_norm[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "Fig3_Confusion_Matrix.png", dpi=200)
    plt.close(fig)
    print("wrote Fig3_Confusion_Matrix.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    fig_class_distribution()
    fig_per_class_f1(metrics)
    fig_confusion(metrics)
    print(f"Figures written to {OUT} (sourced from real metrics — nothing fabricated).")


if __name__ == "__main__":
    main()
