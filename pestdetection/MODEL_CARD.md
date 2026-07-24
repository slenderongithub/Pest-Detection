# Model Card — pest_resnet50

> Auto-generated from the checkpoint manifest and `reports/metrics.json`. Do not edit by hand — re-run `leafscan card`.

## Overview

- **Task**: 9-class agricultural pest image classification
- **Architecture**: resnet50 (ImageNet-pretrained backbone, fine-tuned head+body)
- **Input**: 224×224 RGB, normalized mean=[0.485, 0.456, 0.406] std=[0.229, 0.224, 0.225]
- **Trained on device**: cpu · seed 1337 · 8 epochs
- **Git commit**: `28aa666` · **Data SHA256**: `a42cf713e5685a15…`
- **Created**: 2026-07-12T12:13:00.414052+00:00

## Headline metrics (held-out test split)

| Metric | Value |
|---|---|
| Macro-F1 (primary) | **0.7731** |
| Weighted-F1 | 0.828 |
| Accuracy | 0.8142 |
| Top-3 accuracy | 0.9623 |
| Test samples | 716 |

> Accuracy is reported for context only. With a ~22× class imbalance, **macro-F1** is the honest headline; per-class F1 below shows where the model is weak.

## Calibration

- Temperature scaling fit on the val split: **T = 0.874**
- Expected Calibration Error (test): 0.0905 → **0.0591** after scaling
- Served abstention threshold: 0.40 calibrated probability

## Per-class performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Adristyrannus | 0.8276 | 0.8571 | 0.8421 | 28 |
| Aleurocanthus spiniferus | 0.8824 | 0.9677 | 0.9231 | 62 |
| Ampelophaga | 0.9265 | 0.913 | 0.9197 | 69 |
| Aphis citricola Vander Goot | 0.3506 | 0.8438 | 0.4954 | 32 |
| Apolygus lucorum | 0.5435 | 0.7353 | 0.625 | 34 |
| alfalfa plant bug | 0.7963 | 0.7288 | 0.7611 | 59 |
| alfalfa seed chalcid | 0.6522 | 0.8824 | 0.75 | 17 |
| alfalfa weevil | 0.7358 | 0.8298 | 0.78 | 47 |
| aphids | 0.9631 | 0.7799 | 0.8619 | 368 |

> Lowest-F1 class: **Aphis citricola Vander Goot** (F1 0.4954, support 32). Low-support classes have high-variance metrics — interpret with care.

## Intended use & limitations

- **Intended use**: decision-support triage for the 9 pest classes it was trained on; a hint, not a verdict.
- **Out of scope**: any pest/disease/plant outside the 9 training classes. There is no true out-of-distribution detector — inputs below the abstention threshold are flagged `uncertain`, but a confident wrong answer on an unseen pest is still possible.
- **Data**: a single, imbalanced 9-class dataset; not audited for geographic, lighting, or device diversity. Real-field generalization is unverified.
- **Pesticide guidance is a cited reference, not a prescription** — follow product labels, local regulation, and professional agronomic advice.

## Reproduce

```bash
pip install -e '.[train]'
leafscan train --config configs/pest_resnet50.yaml
```

Framework versions: {'python': '3.12.7', 'torch': '2.2.0', 'torchvision': '0.17.0', 'numpy': '1.26.4'}
