"""Regression guard: with no model installed, the API must NEVER emit a prediction —
only an explicit 503 demo state. (The old build silently fabricated tomato predictions.)"""

from __future__ import annotations

from starlette.testclient import TestClient


def test_no_model_returns_503_never_a_prediction(tmp_path, monkeypatch, load_server):
    empty = tmp_path / "no_model_here"
    empty.mkdir()
    monkeypatch.setenv("LEAFSCAN_MODEL_DIR", str(empty))
    server = load_server()
    with TestClient(server.app) as c:
        health = c.get("/health").json()
        assert health["model_available"] is False
        assert health["demo_mode"] is True

        r = c.post("/analyze", files={"file": ("p.jpg", b"whatever", "image/jpeg")})
        assert r.status_code == 503
        body = r.json()
        assert body["demo_mode"] is True
        # Crucially: no fabricated prediction fields.
        assert "prediction" not in body
        assert "label" not in body
        assert "confidence" not in body
