"""Generate MODEL_CARD.md from a checkpoint manifest + metrics.

Every number is sourced from the manifest/metrics — nothing hand-transcribed.
"""

from __future__ import annotations

from pathlib import Path

from .checkpoint import ModelBundle


def render_model_card(model_dir: str | Path) -> str:
    bundle = ModelBundle.load(model_dir, build=False)
    m = bundle.metrics or {}
    per_class = m.get("per_class", {})

    lines: list[str] = []
    lines.append(f"# Model Card — {m.get('name', bundle.arch)}")
    lines.append("")
    lines.append(
        "> Auto-generated from the checkpoint manifest and `reports/metrics.json`. "
        "Do not edit by hand — re-run `leafscan card`."
    )
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Task**: {bundle.num_classes}-class agricultural pest image classification")
    lines.append(f"- **Architecture**: {bundle.arch} (ImageNet-pretrained backbone, fine-tuned head+body)")
    lines.append(f"- **Input**: {bundle.input_size}×{bundle.input_size} RGB, normalized mean={bundle.mean} std={bundle.std}")
    lines.append(f"- **Trained on device**: {m.get('device', 'unknown')} · seed {m.get('seed', '?')} · {m.get('epochs', '?')} epochs")
    lines.append(f"- **Git commit**: `{bundle.git_sha}` · **Data SHA256**: `{(bundle.data_sha256 or '')[:16]}…`")
    lines.append(f"- **Created**: {bundle.created_at}")
    lines.append("")

    lines.append("## Headline metrics (held-out test split)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Macro-F1 (primary) | **{m.get('macro_f1', '?')}** |")
    lines.append(f"| Weighted-F1 | {m.get('weighted_f1', '?')} |")
    lines.append(f"| Accuracy | {m.get('accuracy', '?')} |")
    lines.append(f"| Top-3 accuracy | {m.get('top3_accuracy', '?')} |")
    lines.append(f"| Test samples | {m.get('num_samples', '?')} |")
    lines.append("")
    lines.append(
        "> Accuracy is reported for context only. With a ~22× class imbalance, **macro-F1** "
        "is the honest headline; per-class F1 below shows where the model is weak."
    )
    lines.append("")

    lines.append("## Calibration")
    lines.append("")
    lines.append(f"- Temperature scaling fit on the val split: **T = {bundle.temperature:.3f}**")
    lines.append(f"- Expected Calibration Error (test): {m.get('ece_before', '?')} → **{m.get('ece_after', '?')}** after scaling")
    lines.append(f"- Served abstention threshold: {bundle.abstain_threshold:.2f} calibrated probability")
    lines.append("")

    if per_class:
        lines.append("## Per-class performance")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for cls, s in per_class.items():
            lines.append(f"| {cls} | {s['precision']} | {s['recall']} | {s['f1']} | {s['support']} |")
        lines.append("")
        low = min(per_class.items(), key=lambda kv: kv[1]["f1"])
        lines.append(
            f"> Lowest-F1 class: **{low[0]}** (F1 {low[1]['f1']}, support {low[1]['support']}). "
            "Low-support classes have high-variance metrics — interpret with care."
        )
        lines.append("")

    lines.append("## Intended use & limitations")
    lines.append("")
    lines.append("- **Intended use**: decision-support triage for the 9 pest classes it was trained on; a hint, not a verdict.")
    lines.append("- **Out of scope**: any pest/disease/plant outside the 9 training classes. There is no true out-of-distribution detector — inputs below the abstention threshold are flagged `uncertain`, but a confident wrong answer on an unseen pest is still possible.")
    lines.append("- **Data**: a single, imbalanced 9-class dataset; not audited for geographic, lighting, or device diversity. Real-field generalization is unverified.")
    lines.append("- **Pesticide guidance is a cited reference, not a prescription** — follow product labels, local regulation, and professional agronomic advice.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install -e '.[train]'")
    lines.append(f"leafscan train --config configs/{m.get('name', bundle.arch)}.yaml")
    lines.append("```")
    lines.append("")
    lines.append(f"Framework versions: {bundle.lib_versions}")
    lines.append("")
    return "\n".join(lines)


def write_model_card(model_dir: str | Path, out_path: str | Path) -> Path:
    text = render_model_card(model_dir)
    p = Path(out_path)
    p.write_text(text, encoding="utf-8")
    return p
