"""Runtime configuration for the serving surfaces.

Torch-free by design so it can be imported in any context (tests, CLI, the Streamlit
app, the Starlette app) without pulling in the ML stack. All values are overridable via
``LEAFSCAN_*`` environment variables, replacing the hardcoded host/port/paths/URLs that
previously lived as literals inside two separate app files.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = two levels up from this file (leafscan/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Serving/runtime settings. Override any field with e.g. ``LEAFSCAN_PORT=9000``."""

    model_config = SettingsConfigDict(
        env_prefix="LEAFSCAN_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),  # allow the `model_dir` field name
    )

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8080

    # --- Model location ---
    # Directory containing a self-describing bundle (manifest.json + weights.pt).
    # Defaults to the ResNet50 flagship (best macro-F1; see configs/pest_resnet50.yaml).
    # Override with LEAFSCAN_MODEL_DIR to serve resnet18 or any other bundle.
    model_dir: Path = REPO_ROOT / "models" / "pest_resnet50"

    # --- Serving safety ---
    # CORS allowlist as a raw string. Stored as `str` (not `list[str]`) on purpose:
    # pydantic-settings JSON-decodes complex env values before validators run, so a
    # `list[str]` field would crash on a comma-separated LEAFSCAN_CORS_ORIGINS. Parse via
    # the `cors_origin_list` property instead. Accepts CSV ("a,b") or JSON ('["a","b"]').
    # Use "*" only if you truly mean an open allowlist.
    cors_origins: str = "http://localhost:8080,http://127.0.0.1:8080"
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB hard cap on request body
    max_image_pixels: int = 40_000_000  # decompression-bomb guard for PIL

    # --- Inference behaviour ---
    # Torch intra-op threads. 0 = leave torch's default untouched.
    num_threads: int = 0
    # Abstain when the top calibrated probability is below this (honest "uncertain",
    # not a fake out-of-distribution detector — there are no non-pest negatives to fit one).
    abstain_threshold: float = 0.40
    # Compute Grad-CAM overlays on the single-image upload path, but never on live
    # camera frames (a backward pass per 1.2s frame is wasted latency).
    gradcam_on_upload: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse ``cors_origins`` (CSV or JSON list) into a list of origins."""
        raw = (self.cors_origins or "").strip()
        if not raw:
            return []
        if raw.startswith("["):
            return list(json.loads(raw))
        return [o.strip() for o in raw.split(",") if o.strip()]


def get_settings() -> Settings:
    """Construct settings from the environment. Cheap enough to call per request."""
    return Settings()
