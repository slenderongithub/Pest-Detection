"""Arch-agnostic Grad-CAM.

The target convolutional layer is looked up BY NAME (from the checkpoint manifest), not
hardcoded to ``layer4[-1]``, so a backbone swap can't silently target the wrong layer.
Grad-CAM needs a backward pass, so this must run OUTSIDE ``torch.inference_mode``; the
caller is responsible for only invoking it on the on-demand (upload) path, never on live
camera frames.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _resolve_layer(model: nn.Module, layer_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if layer_name in modules:
        return modules[layer_name]
    # Fallbacks keep Grad-CAM working for models without the named layer.
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model, "features") and len(model.features) > 0:  # type: ignore[arg-type]
        return model.features[-1]
    raise RuntimeError(f"Could not resolve Grad-CAM layer {layer_name!r}.")


def compute_gradcam(
    model: nn.Module, image_tensor: Tensor, class_index: int, layer_name: str
) -> np.ndarray:
    """Return a ``[0, 1]``-normalized CAM sized to the input's spatial dims (H, W)."""
    activations: list[Tensor] = []
    gradients: list[Tensor] = []
    layer = _resolve_layer(model, layer_name)

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_inputs, grad_outputs):
        gradients.append(grad_outputs[0])

    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)

    try:
        tensor = image_tensor.clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        score = logits[:, class_index].sum()
        score.backward()

        feature_maps = activations[0][0]          # (C, h, w)
        gradient_maps = gradients[0][0]           # (C, h, w)
        weights = gradient_maps.mean(dim=(1, 2))  # (C,)
        cam = torch.relu((weights[:, None, None] * feature_maps).sum(dim=0))
        h, w = image_tensor.shape[-2:]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam
    finally:
        forward_handle.remove()
        backward_handle.remove()
