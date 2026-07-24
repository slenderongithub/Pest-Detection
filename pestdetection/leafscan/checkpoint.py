"""Self-describing model checkpoints.

A checkpoint is a directory containing:
  - ``weights.pt``   : a PLAIN ``state_dict`` (loadable with ``weights_only=True`` — no pickle RCE)
  - ``manifest.json``: everything needed to reconstruct and correctly serve the model
                       (arch, class names, normalization, input size, Grad-CAM layer,
                        calibration temperature, data hash, git sha, metrics, versions)

Because the taxonomy and normalization travel INSIDE the checkpoint, the apps read their
class list from the model rather than a hardcoded constant — structurally eliminating the
train/serve mismatch that this project had before.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .models import build_model

MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.pt"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lib_versions() -> dict[str, str]:
    import numpy
    import torchvision

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
    }


@dataclass
class ModelBundle:
    """Metadata for a trained model plus (once loaded) the live ``model``."""

    arch: str
    class_names: list[str]
    mean: list[float]
    std: list[float]
    input_size: int
    gradcam_layer: str
    temperature: float = 1.0
    abstain_threshold: float = 0.40
    data_sha256: str | None = None
    git_sha: str | None = None
    metrics: dict[str, Any] | None = None
    created_at: str | None = None
    lib_versions: dict[str, str] | None = None
    weights_sha256: str | None = None
    # Populated on load(); never serialized.
    model: Any = field(default=None, repr=False, compare=False)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def manifest_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("model", None)
        return d

    def save(self, out_dir: str | Path, model: torch.nn.Module) -> Path:
        """Write ``weights.pt`` (plain state_dict) + ``manifest.json`` into ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        weights_path = out / WEIGHTS_NAME
        # Move to CPU so the checkpoint is portable regardless of training device.
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, weights_path)
        self.weights_sha256 = _sha256_file(weights_path)
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.lib_versions is None:
            self.lib_versions = lib_versions()
        with open(out / MANIFEST_NAME, "w", encoding="utf-8") as fh:
            json.dump(self.manifest_dict(), fh, indent=2)
        return out

    @classmethod
    def load(
        cls, model_dir: str | Path, *, map_location: str = "cpu", build: bool = True
    ) -> ModelBundle:
        """Load a bundle from ``model_dir``. If ``build``, reconstruct and attach ``model``."""
        d = Path(model_dir)
        manifest_path = d / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No {MANIFEST_NAME} in {d}")
        with open(manifest_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        meta.pop("model", None)
        bundle = cls(**meta)
        if build:
            weights_path = d / WEIGHTS_NAME
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Manifest present but {WEIGHTS_NAME} missing in {d}. "
                    "Fetch the weights (see scripts/fetch_model.py)."
                )
            model, _ = build_model(bundle.arch, bundle.num_classes, pretrained=False)
            # weights_only=True: the weights file is a pure tensor dict, no pickle code path.
            state = torch.load(weights_path, map_location=map_location, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            bundle.model = model
        return bundle


def has_bundle(model_dir: str | Path) -> bool:
    """True if a usable checkpoint (manifest + weights) exists in ``model_dir``."""
    d = Path(model_dir)
    return (d / MANIFEST_NAME).exists() and (d / WEIGHTS_NAME).exists()


def load_bundle(model_dir: str | Path, **kwargs) -> ModelBundle:
    """Convenience wrapper matching the public API (``leafscan.load_bundle``)."""
    return ModelBundle.load(model_dir, **kwargs)
