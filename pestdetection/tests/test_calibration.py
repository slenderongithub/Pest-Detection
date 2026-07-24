"""Temperature scaling must minimise val NLL and ECE must be a valid probability mass."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from leafscan.calibration import (
    expected_calibration_error,
    fit_temperature,
    softmax_np,
)


def _overconfident_logits(n=400, k=5, scale=4.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    logits = rng.normal(0, 1, size=(n, k))
    # Make the true class strongly favoured but not always correct -> overconfident.
    for i, y in enumerate(labels):
        logits[i, y] += scale
    # Flip some labels so accuracy < confidence.
    flip = rng.random(n) < 0.35
    labels[flip] = rng.integers(0, k, size=flip.sum())
    return logits.astype(np.float32), labels


def test_fit_temperature_positive_and_reduces_nll():
    logits, labels = _overconfident_logits()
    t = fit_temperature(torch.tensor(logits), torch.tensor(labels))
    assert t > 0
    nll1 = F.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()
    nllT = F.cross_entropy(torch.tensor(logits) / t, torch.tensor(labels)).item()
    assert nllT <= nll1 + 1e-4  # scaling never increases the optimised NLL


def test_temperature_scaling_reduces_ece():
    logits, labels = _overconfident_logits()
    t = fit_temperature(torch.tensor(logits), torch.tensor(labels))
    ece_before = expected_calibration_error(softmax_np(logits, 1.0), labels)
    ece_after = expected_calibration_error(softmax_np(logits, t), labels)
    assert 0.0 <= ece_after <= 1.0
    assert ece_after <= ece_before + 1e-6


def test_ece_zero_for_perfectly_calibrated():
    # Confidence == accuracy in every bin -> ECE 0.
    probs = np.array([[0.5, 0.5]] * 100)
    labels = np.array([0, 1] * 50)
    assert expected_calibration_error(probs, labels) < 0.05
