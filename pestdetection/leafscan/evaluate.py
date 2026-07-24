"""Evaluation: honest metrics for an imbalanced multi-class problem.

Headline is macro-F1 + per-class precision/recall/F1 (not bare accuracy — with aphids at
~51% of the data, accuracy is a vanity metric). Also emits a confusion matrix and a
reliability diagram (pre/post temperature scaling), and writes a machine-readable
``metrics.json`` that the README and figures MUST source from (no hand-transcribed numbers).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .calibration import expected_calibration_error, reliability_curve, softmax_np


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    is_mps = getattr(device, "type", device) == "mps"
    all_logits, all_labels = [], []
    for i, (images, labels) in enumerate(loader):
        logits = model(images.to(device))
        all_logits.append(logits.cpu().numpy())
        all_labels.append(np.asarray(labels))
        if is_mps and (i + 1) % 8 == 0:
            torch.mps.empty_cache()
    if is_mps:
        torch.mps.empty_cache()
    return np.concatenate(all_logits), np.concatenate(all_labels)


def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(logits, axis=1)[:, -k:]
    hits = [labels[i] in topk[i] for i in range(len(labels))]
    return float(np.mean(hits))


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    temperature: float = 1.0,
) -> dict:
    """Full metric bundle. ECE is reported both before (T=1) and after (fitted T)."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    preds = logits.argmax(axis=1)
    probs_before = softmax_np(logits, 1.0)
    probs_after = softmax_np(logits, temperature)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }
    metrics = {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(labels, preds, average="weighted", zero_division=0)), 4
        ),
        "top3_accuracy": round(top_k_accuracy(logits, labels, 3), 4),
        "temperature": round(float(temperature), 4),
        "ece_before": round(expected_calibration_error(probs_before, labels), 4),
        "ece_after": round(expected_calibration_error(probs_after, labels), 4),
        "num_samples": int(len(labels)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            labels, preds, labels=list(range(len(class_names)))
        ).tolist(),
        "class_names": list(class_names),
    }
    return metrics


def save_reports(
    reports_dir: str | Path,
    metrics: dict,
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    temperature: float,
) -> None:
    """Write metrics.json + confusion_matrix.png + reliability.png."""
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Confusion matrix
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (row-normalized) — macro-F1 {metrics['macro_f1']:.3f}")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if cm_norm[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Reliability diagram (before/after temperature scaling)
    fig, ax = plt.subplots(figsize=(6, 6))
    for temp, name, color in [(1.0, "before (T=1)", "#ff6b6b"), (temperature, f"after (T={temperature:.2f})", "#45f0a3")]:
        probs = softmax_np(logits, temp)
        centers, accs, _confs, counts = reliability_curve(probs, labels)
        valid = counts > 0
        ax.plot(centers[valid], accs[valid], "o-", color=color, label=name)
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability — ECE {metrics['ece_before']:.3f} → {metrics['ece_after']:.3f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)
