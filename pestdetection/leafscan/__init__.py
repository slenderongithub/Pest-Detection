"""LeafScan: one shared, tested core for pest classification.

Both front-ends (Streamlit ``app.py`` and Starlette ``app/server.py``) import from
this package so the model taxonomy, preprocessing, Grad-CAM, localization, overlay,
severity/knowledge, and calibration exist in exactly one place.

The public API is exposed lazily via ``__getattr__`` so that ``import leafscan`` and
``leafscan.config`` stay cheap and torch-free — only the attributes that actually need
torch (``Predictor``, ``ModelBundle``, ``build_model``) import it, and only on first use.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__", "Predictor", "ModelBundle", "build_model", "load_bundle"]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name in {"Predictor"}:
        from .inference import Predictor

        return Predictor
    if name in {"ModelBundle", "load_bundle"}:
        from . import checkpoint

        return getattr(checkpoint, name)
    if name == "build_model":
        from .models import build_model

        return build_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
