"""Lesion localization + overlay compositing.

The single canonical implementation. The two apps previously carried drifted copies
(tint ``(255,86,48)`` gamma ``1.5`` alpha ``190`` vs ``(255,82,48)`` gamma ``1.4`` alpha
``180``); both are replaced by the one canonical constant below.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

# Canonical overlay style (resolves the app.py / server.py drift).
_TINT_RGB = (255, 84, 48)
_TINT_GAMMA = 1.45
_TINT_ALPHA_MAX = 185
_BOX_OUTLINE = (255, 226, 130, 255)
_BOX_WIDTH = 5
_LABEL_BG = (13, 20, 34, 220)
_LABEL_FG = (255, 241, 196, 255)


def connected_component_boxes(
    mask: np.ndarray, min_pixels: int = 40, max_boxes: int = 8, pad_frac: float = 0.012
) -> list[dict[str, int]]:
    """BFS flood-fill a boolean mask into MULTIPLE tight bounding boxes ``{x1,y1,x2,y2}``.

    Each distinct connected region (a separate pest / lesion patch) becomes its own box, so
    spread-out infection is covered by several precise boxes rather than one large box.
    Components below ``min_pixels`` are dropped as noise; the ``max_boxes`` largest survive,
    sorted by area. There is intentionally NO whole-region fallback (see ``boxes_from_cam``).
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []  # (area, min_x, min_y, max_x, max_y)
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys, xs, strict=False):
        if visited[start_y, start_x]:
            continue
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        pixel_count = 0
        while queue:
            y, x = queue.popleft()
            pixel_count += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if pixel_count >= min_pixels:
            components.append((pixel_count, min_x, min_y, max_x, max_y))

    components.sort(key=lambda c: c[0], reverse=True)
    pad = max(3, int(pad_frac * max(height, width)))
    boxes: list[dict[str, int]] = []
    for _area, min_x, min_y, max_x, max_y in components[:max_boxes]:
        boxes.append(
            {
                "x1": int(max(0, min_x - pad)),
                "y1": int(max(0, min_y - pad)),
                "x2": int(min(width - 1, max_x + pad)),
                "y2": int(min(height - 1, max_y + pad)),
            }
        )
    return boxes


def project_cam_to_image(
    cam: np.ndarray, image_size: tuple[int, int], input_size: int, resize_ratio: float = 256 / 224
) -> np.ndarray:
    """Map a model-input-space CAM back to ORIGINAL image coordinates.

    The model sees ``Resize(shorter→input_size*resize_ratio)`` then ``CenterCrop(input_size)``,
    so the raw ``input_size×input_size`` CAM only covers the center-cropped field of view.
    Without this inverse mapping, boxes derived from the CAM live in 224-space and land in
    the top-left corner of (or out of bounds for) any non-224 image. This places the CAM
    into the correct crop window of the resized canvas (zeros elsewhere), then resizes to
    the original image size — so both the heatmap and the boxes register to real pixels.
    """
    width, height = image_size
    resize_short = int(round(input_size * resize_ratio))
    scale = resize_short / min(width, height)
    resized_w = max(input_size, int(round(width * scale)))
    resized_h = max(input_size, int(round(height * scale)))
    left = max(0, (resized_w - input_size) // 2)
    top = max(0, (resized_h - input_size) // 2)
    canvas = np.zeros((resized_h, resized_w), dtype=np.float32)
    canvas[top : top + input_size, left : left + input_size] = cam
    full = Image.fromarray((canvas * 255.0).astype(np.uint8), mode="L").resize(
        (width, height), Image.BILINEAR
    )
    return np.asarray(full, dtype=np.float32) / 255.0


def boxes_from_cam(
    cam: np.ndarray,
    quantile: float = 0.85,
    floor: float = 0.5,
    min_area_frac: float = 0.002,
    max_boxes: int = 8,
) -> list[dict[str, int]]:
    """Turn a CAM into several PRECISE boxes over the hottest (pest) regions.

    A relatively high threshold (``max(floor, quantile(cam))``) fragments the activation
    into its distinct peaks, so multiple spread-out infection spots each get their own tight
    box. ``min_area_frac`` (fraction of image pixels) filters noise while keeping small real
    spots. If nothing clears the area filter, a SMALL box is drawn around the single hottest
    peak — never a box over the whole image.
    """
    height, width = cam.shape
    threshold = max(floor, float(np.quantile(cam, quantile)))
    min_pixels = max(24, int(min_area_frac * height * width))
    boxes = connected_component_boxes(cam >= threshold, min_pixels=min_pixels, max_boxes=max_boxes)
    if boxes:
        return boxes

    if float(cam.max()) <= 0.0:
        return []
    peak_y, peak_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    side = max(8, int(0.12 * min(height, width)))
    return [
        {
            "x1": int(max(0, peak_x - side)),
            "y1": int(max(0, peak_y - side)),
            "x2": int(min(width - 1, peak_x + side)),
            "y2": int(min(height - 1, peak_y + side)),
        }
    ]


def overlay_image(image: Image.Image, cam: np.ndarray, boxes: list[dict[str, int]]) -> Image.Image:
    """Composite a heatmap tint + labelled bounding boxes onto ``image`` (RGB out)."""
    base = image.convert("RGBA")
    heat = Image.fromarray((cam * 255).astype(np.uint8), mode="L").resize(base.size, Image.BILINEAR)
    heat_arr = np.asarray(heat, dtype=np.float32) / 255.0

    tint = Image.new("RGBA", base.size, (*_TINT_RGB, 0))
    alpha = np.clip((heat_arr**_TINT_GAMMA) * _TINT_ALPHA_MAX, 0, _TINT_ALPHA_MAX).astype(np.uint8)
    tint.putalpha(Image.fromarray(alpha, mode="L"))

    result = Image.alpha_composite(base, tint)
    result = ImageEnhance.Contrast(result).enhance(1.08)
    draw = ImageDraw.Draw(result)
    for index, box in enumerate(boxes, start=1):
        rect = (box["x1"], box["y1"], box["x2"], box["y2"])
        draw.rounded_rectangle(rect, radius=12, outline=_BOX_OUTLINE, width=_BOX_WIDTH)
        label_y = max(8, box["y1"] - 26)
        draw.rounded_rectangle(
            (box["x1"], label_y, box["x1"] + 150, label_y + 22), radius=8, fill=_LABEL_BG
        )
        draw.text((box["x1"] + 10, label_y + 3), f"Region {index}", fill=_LABEL_FG)
    return result.convert("RGB")
