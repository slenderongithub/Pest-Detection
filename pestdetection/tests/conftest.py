"""Shared pytest fixtures.

A tiny random (untrained) resnet18 bundle stands in for a real checkpoint so the whole
inference/serving stack is testable in milliseconds without a training run or a download.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

CLASS_NAMES = [
    "Adristyrannus",
    "Aleurocanthus spiniferus",
    "Ampelophaga",
    "Aphis citricola Vander Goot",
    "Apolygus lucorum",
    "alfalfa plant bug",
    "alfalfa seed chalcid",
    "alfalfa weevil",
    "aphids",
]


@pytest.fixture
def class_names():
    return list(CLASS_NAMES)


@pytest.fixture
def load_server():
    """Return a loader for app/server.py BY PATH.

    The repo-root ``app.py`` (Streamlit) shadows the ``app/`` package, so a plain
    ``import app.server`` would execute the Streamlit script. Set ``LEAFSCAN_MODEL_DIR``
    before calling — the module builds its Predictor at import time.
    """
    import importlib.util
    from pathlib import Path

    server_path = Path(__file__).resolve().parent.parent / "app" / "server.py"

    def _load():
        spec = importlib.util.spec_from_file_location("leafscan_server_under_test", server_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return _load


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory):
    """Build and save a random 9-class resnet18 bundle; return its directory."""
    from leafscan.checkpoint import ModelBundle
    from leafscan.models import build_model

    out = tmp_path_factory.mktemp("model")
    model, gradcam_layer = build_model("resnet18", len(CLASS_NAMES), pretrained=False)
    bundle = ModelBundle(
        arch="resnet18",
        class_names=CLASS_NAMES,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        input_size=224,
        gradcam_layer=gradcam_layer,
        temperature=1.25,
        abstain_threshold=0.40,
        metrics={"name": "test_fixture", "macro_f1": 0.0},
    )
    bundle.save(out, model)
    return out


@pytest.fixture
def sample_image():
    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((180, 240, 3)) * 255).astype("uint8"))
