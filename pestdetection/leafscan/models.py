"""Model factory.

One place that knows how to build a backbone and where its Grad-CAM target layer is.
The Grad-CAM layer name is returned alongside the model and stored in the checkpoint
manifest, so swapping backbones can never silently point Grad-CAM at the wrong layer.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm

# arch -> (builder, weights-enum, grad-cam target module name)
_ARCHES = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights, "layer4"),
    "resnet34": (tvm.resnet34, tvm.ResNet34_Weights, "layer4"),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights, "layer4"),
}

SUPPORTED_ARCHES = tuple(_ARCHES)


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> tuple[nn.Module, str]:
    """Build ``arch`` with a fresh classification head of ``num_classes`` outputs.

    Returns ``(model, gradcam_layer_name)``. ``pretrained=True`` loads ImageNet weights
    for the backbone (downloaded/cached by torchvision); ``pretrained=False`` gives a
    randomly-initialised backbone (used for tests and for reconstructing a model whose
    fine-tuned weights are about to be loaded from a checkpoint).
    """
    key = arch.lower()
    if key not in _ARCHES:
        raise ValueError(f"Unsupported arch {arch!r}; choose from {SUPPORTED_ARCHES}")
    builder, weights_enum, gradcam_layer = _ARCHES[key]
    weights = weights_enum.DEFAULT if pretrained else None
    model = builder(weights=weights)
    # All supported backbones expose a single Linear head named `fc`.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, gradcam_layer
