"""Image preprocessing.

Transforms are built FROM normalization + input size (never hardcoded), so training and
serving share one definition and a checkpoint's own normalization is always honoured.
"""

from __future__ import annotations

from PIL import Image
from torch import Tensor
from torchvision import transforms

# Ratio used by the classic 256->224 eval pipeline; generalised to any input size.
_RESIZE_RATIO = 256 / 224


def eval_transform(input_size: int, mean: list[float], std: list[float]) -> transforms.Compose:
    resize = int(round(input_size * _RESIZE_RATIO))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def train_transform(
    input_size: int,
    mean: list[float],
    std: list[float],
    *,
    randaugment_ops: int = 2,
    randaugment_magnitude: int = 7,
    horizontal_flip: bool = True,
) -> transforms.Compose:
    ops: list = [transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0))]
    if horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip())
    if randaugment_ops > 0:
        # RandAugment runs on PIL uint8 images, so it must precede ToTensor.
        ops.append(
            transforms.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_magnitude)
        )
    ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(ops)


def preprocess_image(
    image: Image.Image, input_size: int, mean: list[float], std: list[float]
) -> Tensor:
    """PIL image -> normalized ``(1, C, H, W)`` tensor ready for a forward pass."""
    tensor = eval_transform(input_size, mean, std)(image.convert("RGB"))
    return tensor.unsqueeze(0)
