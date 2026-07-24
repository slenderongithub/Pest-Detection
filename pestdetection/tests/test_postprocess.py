from __future__ import annotations

import numpy as np
from PIL import Image

from leafscan.postprocess import boxes_from_cam, connected_component_boxes, overlay_image


def test_connected_components_single_block():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True  # 100 px block
    boxes = connected_component_boxes(mask, min_pixels=50)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["x1"] <= 10 and b["y1"] <= 10 and b["x2"] >= 19 and b["y2"] >= 19


def test_two_disjoint_components():
    mask = np.zeros((60, 60), dtype=bool)
    mask[5:18, 5:18] = True
    mask[40:55, 40:55] = True
    boxes = connected_component_boxes(mask, min_pixels=50)
    assert len(boxes) == 2


def test_empty_mask_yields_no_boxes():
    assert connected_component_boxes(np.zeros((30, 30), dtype=bool)) == []


def test_boxes_from_cam_and_overlay():
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[20:44, 20:44] = 1.0
    boxes = boxes_from_cam(cam)
    assert boxes
    img = Image.new("RGB", (64, 64))
    out = overlay_image(img, cam, boxes)
    assert out.size == (64, 64) and out.mode == "RGB"


def test_spread_infection_yields_multiple_tight_boxes():
    # Three separated hot spots -> three distinct boxes, none covering the whole image.
    cam = np.zeros((120, 120), dtype=np.float32)
    cam[10:26, 10:26] = 1.0
    cam[90:106, 90:106] = 0.95
    cam[12:26, 92:106] = 0.9
    boxes = boxes_from_cam(cam)
    assert len(boxes) >= 3, boxes
    for b in boxes:
        area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        assert area < 0.4 * 120 * 120, ("box too large / merged", b)


def test_no_whole_image_fallback_box():
    # A single tiny sub-threshold speck must NOT produce a giant whole-image box.
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[50, 50] = 1.0  # one pixel peak
    boxes = boxes_from_cam(cam)
    assert len(boxes) == 1
    b = boxes[0]
    assert (b["x2"] - b["x1"]) < 60 and (b["y2"] - b["y1"]) < 60  # small peak box, not whole image
