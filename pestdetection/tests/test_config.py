"""Regression: CORS env parsing must not crash the server (the documented CSV form
previously raised SettingsError before startup)."""

from __future__ import annotations

from leafscan.config import Settings


def test_cors_csv(monkeypatch):
    monkeypatch.setenv("LEAFSCAN_CORS_ORIGINS", "https://a.com,https://b.com")
    assert Settings().cors_origin_list == ["https://a.com", "https://b.com"]


def test_cors_single_origin(monkeypatch):
    monkeypatch.setenv("LEAFSCAN_CORS_ORIGINS", "https://only.example")
    assert Settings().cors_origin_list == ["https://only.example"]


def test_cors_json_list(monkeypatch):
    monkeypatch.setenv("LEAFSCAN_CORS_ORIGINS", '["https://a.com", "https://b.com"]')
    assert Settings().cors_origin_list == ["https://a.com", "https://b.com"]


def test_cors_default_is_local():
    assert "http://localhost:8080" in Settings().cors_origin_list
