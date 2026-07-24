"""Predictor contract + the honest no-model behaviour."""

from __future__ import annotations

import json

import pytest

from leafscan.inference import ModelNotAvailableError, Predictor


def test_predict_structure_and_json_serializable(bundle_dir, sample_image):
    pred = Predictor(model_dir=bundle_dir).predict(sample_image, want_gradcam=True)
    assert pred.label in pred.class_names
    assert 0.0 <= pred.confidence <= 100.0
    assert len(pred.top_predictions) == 3
    # API dict must be JSON-serializable and carry the calibrated fields.
    api = pred.to_api_dict()
    assert "overlay" not in api
    assert api["class_count"] == len(pred.class_names)
    json.dumps(api)  # raises if anything is non-serializable


def test_abstention_gate(bundle_dir, sample_image):
    # A threshold above any possible confidence forces abstention and suppresses boxes.
    p = Predictor(model_dir=bundle_dir)
    p.bundle.abstain_threshold = 1.01
    pred = p.predict(sample_image, want_gradcam=True)
    assert pred.abstained is True
    assert pred.boxes == []
    assert pred.overlay is None


def test_boxes_are_within_image_bounds(bundle_dir):
    # Non-square, sub-224 image: boxes must be in ORIGINAL image coordinates (regression
    # for the CAM-in-224-space bug that put boxes in the top-left / out of bounds).
    import numpy as np
    from PIL import Image

    p = Predictor(model_dir=bundle_dir)
    p.bundle.abstain_threshold = 0.0  # force Grad-CAM to run
    img = Image.fromarray((np.random.default_rng(1).random((161, 312, 3)) * 255).astype("uint8"))
    pred = p.predict(img, want_gradcam=True)
    w, h = img.size  # (312, 161)
    for b in pred.boxes:
        assert 0 <= b["x1"] <= b["x2"] <= w - 1, b
        assert 0 <= b["y1"] <= b["y2"] <= h - 1, b


def test_no_model_raises(tmp_path):
    predictor = Predictor(model_dir=tmp_path / "does_not_exist")
    assert predictor.is_available is False
    with pytest.raises(ModelNotAvailableError):
        predictor.predict(None)
