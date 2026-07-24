"""A checkpoint must be self-describing and load with weights_only=True (no pickle path)."""

from __future__ import annotations

import torch

from leafscan.checkpoint import ModelBundle, has_bundle


def test_bundle_round_trip_carries_metadata(bundle_dir, class_names):
    assert has_bundle(bundle_dir)
    loaded = ModelBundle.load(bundle_dir)
    assert loaded.class_names == class_names
    assert loaded.temperature == 1.25
    assert loaded.mean == [0.485, 0.456, 0.406]
    assert loaded.gradcam_layer == "layer4"
    assert loaded.weights_sha256 and len(loaded.weights_sha256) == 64
    assert loaded.model is not None
    assert loaded.model.fc.out_features == len(class_names)


def test_weights_load_without_pickle(bundle_dir):
    # Loading the raw weights file with weights_only=True must succeed — proving there is
    # no pickled object graph and therefore no RCE surface.
    state = torch.load(bundle_dir / "weights.pt", map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    assert all(isinstance(v, torch.Tensor) for v in state.values())


def test_metadata_only_load_skips_build(bundle_dir, class_names):
    meta = ModelBundle.load(bundle_dir, build=False)
    assert meta.model is None
    assert meta.num_classes == len(class_names)
