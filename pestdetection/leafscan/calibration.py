"""Confidence calibration via temperature scaling + ECE.

A single scalar temperature ``T`` is fit by minimising NLL on a HELD-OUT val split
(never train/test — fitting on test would leak and fake a good ECE). ``T`` is stored in
the checkpoint manifest and divides the logits at serve time so displayed probabilities
are honest rather than the systematically-overconfident raw softmax.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
    """Fit temperature ``T`` minimising cross-entropy of ``logits / T`` on ``labels``."""
    logits = logits.detach().float()
    labels = labels.detach().long()
    log_T = torch.zeros(1, requires_grad=True)  # optimise log T to keep T > 0
    optimizer = torch.optim.LBFGS([log_T], lr=0.05, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T.exp().item())


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Standard top-label ECE over ``n_bins`` equal-width confidence bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Return ``(bin_centers, bin_acc, bin_conf, bin_count)`` for a reliability diagram."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        centers.append((lo + hi) / 2)
        counts.append(int(mask.sum()))
        accs.append(float(accuracies[mask].mean()) if mask.any() else np.nan)
        confs.append(float(confidences[mask].mean()) if mask.any() else np.nan)
    return np.array(centers), np.array(accs), np.array(confs), np.array(counts)
