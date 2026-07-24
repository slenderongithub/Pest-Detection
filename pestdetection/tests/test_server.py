"""Starlette API integration tests via TestClient, against a fixture bundle."""

from __future__ import annotations

import io

import pytest
from PIL import Image
from starlette.testclient import TestClient


@pytest.fixture
def client(bundle_dir, monkeypatch, load_server):
    monkeypatch.setenv("LEAFSCAN_MODEL_DIR", str(bundle_dir))
    with TestClient(load_server().app) as c:
        yield c


def _jpeg_bytes(size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_reports_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_available"] is True
    assert body["demo_mode"] is False


def test_analyze_happy_path(client):
    r = client.post("/analyze", files={"file": ("p.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in body["class_names"]
    assert 0 <= body["confidence"] <= 100
    assert len(body["top_predictions"]) == 3
    assert "overlay_data_url" in body


def test_analyze_frame_skips_gradcam(client):
    r = client.post("/analyze-frame", files={"file": ("f.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    # Live frames skip Grad-CAM, so no overlay/boxes are produced.
    assert r.json()["overlay_data_url"] is None
    assert r.json()["boxes"] == []


def test_missing_file_is_400(client):
    assert client.post("/analyze").status_code == 400


def test_corrupt_image_is_400(client):
    r = client.post("/analyze", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_image"


def test_oversized_upload_413(bundle_dir, monkeypatch, load_server):
    monkeypatch.setenv("LEAFSCAN_MODEL_DIR", str(bundle_dir))
    monkeypatch.setenv("LEAFSCAN_MAX_UPLOAD_BYTES", "10")  # tiny cap
    with TestClient(load_server().app) as c:
        r = c.post("/analyze", files={"file": ("p.jpg", _jpeg_bytes(), "image/jpeg")})
        assert r.status_code == 413
        assert r.json()["error"] == "file_too_large"
