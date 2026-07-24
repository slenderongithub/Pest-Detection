"""Render a REAL sample output (was previously a fabricated mock-up).

Runs the actual trained model on a real image and saves the genuine Grad-CAM overlay
plus the model's genuine top-3 prediction — nothing is hand-typed.

Run: python scripts/generate_fig7.py [image_path]
Output: docs/paper_figures/Fig_Sample_Output.png
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from leafscan.inference import ModelNotAvailableError, Predictor  # noqa: E402

OUT = REPO_ROOT / "docs" / "paper_figures"


def pick_image() -> str | None:
    if len(sys.argv) > 1:
        return sys.argv[1]
    for pattern in ("data/Pest_Dataset/aphids/*.jpg", "docs/images/*.jpg"):
        found = sorted(glob.glob(str(REPO_ROOT / pattern)))
        if found:
            return found[0]
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(model_dir=REPO_ROOT / "models" / "pest_resnet50")
    image_path = pick_image()
    if image_path is None:
        sys.exit("error: no sample image found (pass one as an argument).")
    try:
        pred = predictor.predict(Image.open(image_path), want_gradcam=True)
    except ModelNotAvailableError:
        sys.exit("error: no trained model. Run `leafscan train` first.")

    overlay = pred.overlay if pred.overlay is not None else Image.open(image_path).convert("RGB")
    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax_img.imshow(overlay)
    ax_img.axis("off")
    ax_img.set_title("Real Grad-CAM overlay", weight="bold")

    ax_txt.axis("off")
    lines = [
        f"Prediction: {pred.display_label}",
        f"Calibrated confidence: {pred.confidence:.1f}%"
        + ("  (ABSTAINED)" if pred.abstained else ""),
        f"Severity: {pred.severity or '—'}",
        "",
        "Top-3:",
    ]
    for tp in pred.top_predictions:
        lines.append(f"  • {tp['display_label']}: {tp['probability']:.1f}%")
    if pred.pest_info and pred.pest_info["pesticides"]:
        lines.append("")
        lines.append("Reference treatment (per L water):")
        for p in pred.pest_info["pesticides"]:
            lines.append(f"  • {p['name']}: {p['dose']}")
    ax_txt.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    fig.suptitle(f"Sample pipeline output — {Path(image_path).name} (real inference)", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "Fig_Sample_Output.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'Fig_Sample_Output.png'} (real inference on {image_path})")


if __name__ == "__main__":
    main()
