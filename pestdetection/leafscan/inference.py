"""Predictor: the single inference entry point shared by both front-ends.

Loads a self-describing checkpoint (taxonomy + normalization + temperature all come from
the manifest), runs a temperature-calibrated forward pass, applies an honest abstention
gate, and — on the upload path only — attaches a Grad-CAM overlay. If no checkpoint is
present it raises ``ModelNotAvailableError`` so the caller can show an explicit no-model /
demo state instead of ever fabricating a prediction.
"""

from __future__ import annotations

import base64
import threading
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from .checkpoint import ModelBundle, has_bundle
from .config import Settings, get_settings
from .gradcam import compute_gradcam
from .knowledge import disclaimer, get_pest_info, pretty_label
from .postprocess import boxes_from_cam, overlay_image, project_cam_to_image
from .preprocess import preprocess_image


class ModelNotAvailableError(RuntimeError):
    """Raised when inference is requested but no checkpoint is installed."""


def image_to_data_url(image: Image.Image, quality: int = 90) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@dataclass
class Prediction:
    model_name: str
    label: str
    display_label: str
    confidence: float           # calibrated top probability, 0-100
    severity: str | None
    abstained: bool
    abstain_threshold: float    # 0-100
    temperature: float          # calibration temperature applied to the logits
    top_predictions: list[dict[str, Any]]
    boxes: list[dict[str, int]]
    class_names: list[str]
    pest_info: dict[str, Any] | None
    image_size: dict[str, int]
    disclaimer: str = ""
    overlay: Any = field(default=None, repr=False, compare=False)  # PIL image, not serialized

    def to_api_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("overlay", None)
        d["class_count"] = len(self.class_names)
        d["overlay_data_url"] = image_to_data_url(self.overlay) if self.overlay is not None else None
        return d


def _pest_info_dict(info) -> dict[str, Any] | None:
    if info is None:
        return None
    return {
        "common_name": info.common_name,
        "pest_type": info.pest_type,
        "severity": info.severity,
        "description": info.description,
        "pesticides": [{"name": p.name, "dose": p.dose} for p in info.pesticides],
        "ipm": info.ipm,
    }


class Predictor:
    """Process-cached predictor over a self-describing checkpoint directory."""

    def __init__(self, model_dir: str | Path | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model_dir = Path(model_dir) if model_dir else Path(self.settings.model_dir)
        self._bundle: ModelBundle | None = None
        # Serializes the Grad-CAM backward pass: it registers hooks on the SHARED model
        # layer, so concurrent upload requests (via run_in_threadpool) would otherwise
        # cross-contaminate each other's activations/gradients.
        self._gradcam_lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        return has_bundle(self.model_dir)

    @property
    def bundle(self) -> ModelBundle:
        if self._bundle is None:
            if not self.is_available:
                raise ModelNotAvailableError(
                    f"No checkpoint in {self.model_dir}. Train one "
                    "(`leafscan train`) or fetch weights (`scripts/fetch_model.py`)."
                )
            if self.settings.num_threads > 0:
                torch.set_num_threads(self.settings.num_threads)
            self._bundle = ModelBundle.load(self.model_dir)
        return self._bundle

    @property
    def model_name(self) -> str:
        return self.bundle.metrics.get("name") if self.bundle.metrics else self.bundle.arch

    def warmup(self) -> None:
        """Force the model to load (and JIT-warm) at startup instead of first request."""
        _ = self.bundle
        dummy = Image.new("RGB", (self.bundle.input_size, self.bundle.input_size))
        self.predict(dummy, want_gradcam=False)

    def predict(self, image: Image.Image, *, want_gradcam: bool = True, topk: int = 3) -> Prediction:
        bundle = self.bundle
        model = bundle.model
        image = image.convert("RGB")
        tensor = preprocess_image(image, bundle.input_size, bundle.mean, bundle.std)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits / max(bundle.temperature, 1e-6), dim=1)[0]

        probs_np = probs.cpu().numpy()
        order = probs_np.argsort()[::-1]
        pred_index = int(order[0])
        label = bundle.class_names[pred_index]
        confidence = float(probs_np[pred_index] * 100.0)
        threshold = bundle.abstain_threshold * 100.0
        abstained = confidence < threshold

        top_predictions = [
            {
                "label": bundle.class_names[i],
                "display_label": pretty_label(bundle.class_names[i]),
                "probability": round(float(probs_np[i] * 100.0), 2),
            }
            for i in order[:topk]
        ]

        info = get_pest_info(label)
        severity = info.severity if info else None

        boxes: list[dict[str, int]] = []
        overlay = None
        do_cam = (
            want_gradcam
            and self.settings.gradcam_on_upload
            and not abstained
        )
        if do_cam:
            with self._gradcam_lock:
                cam = compute_gradcam(model, tensor, pred_index, bundle.gradcam_layer)
            # Project the model-input-space CAM back to original image coordinates so the
            # heatmap AND the boxes register to real pixels (not the 224x224 top-left corner).
            cam = project_cam_to_image(cam, image.size, bundle.input_size)
            boxes = boxes_from_cam(cam)
            if boxes:
                overlay = overlay_image(image, cam, boxes)

        return Prediction(
            model_name=self.model_name,
            label=label,
            display_label=pretty_label(label),
            confidence=round(confidence, 2),
            severity=severity,
            abstained=abstained,
            abstain_threshold=round(threshold, 2),
            temperature=round(float(bundle.temperature), 4),
            top_predictions=top_predictions,
            boxes=boxes,
            class_names=list(bundle.class_names),
            pest_info=_pest_info_dict(info),
            image_size={"width": image.size[0], "height": image.size[1]},
            disclaimer=disclaimer(),
            overlay=overlay,
        )
