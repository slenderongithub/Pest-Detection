# Machine Learning System (highest-priority analysis)

## Current models in use

| Backend kind | Architecture | Format | Trigger condition |
|---|---|---|---|
| `torch-resnet50` | TorchVision ResNet50, FC swapped to 39 classes | Raw `state_dict` in a `.pth` | `export_resnet50_model.pth` exists |
| `fastai-resnet50` | Same ResNet50, wrapped in a FastAI v1 `Learner` | `.pkl` (FastAI export) | `.pth` absent, `.pkl` ResNet50 export exists |
| `fastai-resnet34` | ResNet34, FastAI v1 `Learner` | `.pkl` (legacy, auto-downloaded) | Neither ResNet50 file exists |
| heuristic (`missing`, Starlette only) | None — pure color-space rules | n/a | No model file found at all |

**Why ResNet was chosen**: not documented anywhere in the repo. ResNet50/34
are reasonable, well-understood defaults for transfer learning on a
leaf-image classification task (moderate depth, ImageNet-pretrained
weights readily available via `torchvision.models.ResNet50_Weights
.DEFAULT`), but no ablation, comparison, or rationale is recorded — the
`notebook/` directory's 9 architecture experiments (VGG16/19,
DenseNet121, custom CNN, etc.) suggest ResNet50 "won" empirically at
some point, but no results/metrics comparing them are committed.

## Training pipeline (as it actually exists — fragmented, not unified)

- **`notebooks/Pest_Detection_Training.ipynb`** — the only notebook with
  concrete, reported results. Trains a **custom CNN** (3×[Conv2D+ReLU+
  MaxPool] → Flatten → Dense(128)+Dropout(0.5) → Softmax(9)) in
  TensorFlow/Keras on a **9-class pest photo dataset**
  (`data/Pest_Dataset/`, ~4,770 images, manually split 70/15/15 via a
  hardcoded script in cell 0). 15 epochs, Adam optimizer,
  categorical cross-entropy. **Final validation accuracy ≈ 63.87%**
  (from the recorded per-epoch history, also reproduced verbatim in
  `scripts/generate_figures.py`'s Fig 5 as a hardcoded list rather than
  loaded from a saved history object — meaning Fig 5 is only as
  trustworthy as whoever transcribed those 15×4 numbers correctly).
- **`notebook/*.ipynb`** (9 files) — independent architecture
  experiments (PyTorch generic ResNet fine-tuning, FastAI v1 on
  PlantVillage via Colab, DenseNet121, VGG16/19, TensorFlow, a
  generic "plant_disease_detector"). Spot-checked two: the FastAI
  notebook pulls `PlantVillage.tar.gz` from a personal Google Drive via
  Colab-mounted drive; the ResNet50-titled notebook is largely Colab
  environment setup boilerplate (`pip install`s, `apt-get`s, Google
  Drive auth via `ocamlfuse`) rather than a clean, portable training
  script. **None of these notebooks is demonstrated end-to-end to
  produce `export_resnet50_model.pth`** as consumed by the shipped apps
  — the link between "a notebook trained something" and "this exact
  file is what's in `app/models/`" is not documented or reproducible
  from this repo alone.
- **No shared training config, no experiment tracker, no seed control.**
  Each notebook independently `pip install`s packages inline, some
  version-pinned to ancient releases (e.g. `imgaug==0.2.5`,
  `numpy==1.16.0`) inconsistent with each other and with the shipped
  `requirements.txt`.

## Inference pipeline (fully documented in `docs/Data-Flow.md`)

Preprocess (`Resize(256)→CenterCrop(224)→ToTensor→ImageNet-normalize`) →
forward pass → softmax → Grad-CAM on `layer4[-1]` → percentile threshold
→ BFS connected components → bounding boxes → overlay compositing →
severity/treatment lookup.

## Confidence scoring & output interpretation

- Confidence = softmax probability of the argmax class × 100, displayed
  to 2 decimal places. **This is raw softmax confidence, uncalibrated**
  — no temperature scaling, no Platt scaling, no conformal prediction,
  no reported ECE (expected calibration error). Deep classifiers are
  well known to be overconfident post-softmax; there is no evidence this
  was corrected for.
- Severity is derived from a **hardcoded disease→severity mapping**
  (19/38 classes) or, for unmapped classes, from raw confidence bands
  (`≥85→High, ≥60→Medium, else Low`) — conflating "how sure the model is"
  with "how dangerous the disease is," which are not the same axis. A
  model could be 95% confident about a mild, cosmetic issue and this
  logic would still report "High" severity.
- Grad-CAM boxes are gated by an **arbitrary confidence floor of 55%**
  and a fixed CAM threshold (`max(0.55, 82nd percentile)`) — untuned
  against any labeled localization ground truth (there is no bounding-box
  annotation anywhere in this dataset).

## Evaluation metrics

Only one number is recorded anywhere in the repo: **~63.87% validation
accuracy**, for a **9-class pest classifier that is not the model the
shipped apps actually run** (which classify against a 38-class disease
taxonomy). There is no accuracy, precision/recall, F1, confusion matrix,
or per-class breakdown for the actual ResNet50/34 PlantVillage model in
production. This is the single largest ML-credibility gap in the
project — **there is currently no verifiable claim about how well the
shipped model performs.**

## Performance bottlenecks

- Grad-CAM's backward pass runs synchronously in the request handler for
  every single inference (including every 1.2s camera tick) — a full
  extra backward pass per request, not batched, not cached, not
  optional.
- The Starlette `/analyze`/`/analyze-frame` handler is `async def` but
  contains fully synchronous, CPU-bound PyTorch calls with no `await`
  inside them — this blocks the single-threaded asyncio event loop for
  the duration of every inference, meaning concurrent camera clients (or
  even the upload path while a camera scan is running) would queue up
  behind each other rather than running in parallel.
- No batching, no ONNX/TensorRT export, no quantization, no GPU-check
  beyond a naive `torch.cuda.is_available()` — CPU inference on a full
  ResNet50 plus a Grad-CAM backward pass, every ~1.2s during live
  scanning, will be noticeably slow on modest hardware.

## Current limitations, scalability, and accuracy

- **Single-model, single-process, no batching, no autoscaling** — see
  `docs/Architecture.md`'s Scalability section.
- **Accuracy is unverified for the deployed model** (see Evaluation
  Metrics above).
- **No confidence calibration, no out-of-distribution detection** — a
  photo of literally anything other than a leaf (a hand, a wall, a cat)
  will still receive a confident-looking top-1 class from the softmax,
  with no "this doesn't look like a leaf" guard rail.
- **The heuristic fallback actively fabricates plausible-looking
  results** (`docs/Known-Issues.md` #10) with no UI signal that it's not
  a real model — the single biggest trust risk in the current system.

## Security considerations (ML-specific)

- Untrusted image bytes are decoded directly by PIL and fed into a
  torch model with no size/dimension caps beyond what PIL/PyTorch
  enforce implicitly — a maliciously huge image could be a resource
  exhaustion vector (no request size limit configured).
- `torch.load(..., weights_only=False)` is explicitly re-enabled by a
  monkey-patch in both apps specifically to load old pickle-based
  checkpoints. `weights_only=False` means **arbitrary code execution is
  possible if a malicious `.pth`/`.pkl` file is ever loaded** — this is
  fine for a locally-placed, developer-trusted file, but would be a
  serious vulnerability if model files were ever accepted from an
  untrusted upload path (they currently are not — models only load from
  `app/models/` and one hardcoded Drive URL — but this constraint isn't
  enforced anywhere beyond "nobody wrote that code path yet").

## Explainability, fairness, reproducibility

- **Explainability**: Grad-CAM only; no SHAP/LIME/integrated-gradients
  cross-check, no quantitative localization evaluation (no ground-truth
  lesion boxes exist to evaluate against).
- **Fairness**: no analysis of per-class accuracy disparities, no
  documented awareness of class imbalance (PlantVillage is known to be
  imbalanced across the 38 classes; the pest dataset here ranges from
  111 to 2,456 images per class — a >20x imbalance with **no
  class-weighting, oversampling, or focal loss** visible anywhere in the
  training notebooks).
- **Reproducibility**: low. No fixed random seeds found in the training
  notebook, no environment lockfile matching what was actually used to
  train, no saved model card, no dataset version pin, hardcoded
  machine-specific paths.

---

## Upgrade brainstorm — making ML the flagship feature

Each idea is rated **Impact / Difficulty / User Value / Complexity**
(High/Medium/Low). This is ideation only — no implementation performed.

### Tier 1 — foundational fixes (do these before anything fancier)

1. **Establish real evaluation on the actual deployed model** (accuracy,
   per-class F1, confusion matrix on a held-out PlantVillage split).
   Impact: High · Difficulty: Low · User value: High (trust) · Complexity: Low.
2. **Unify training pipeline**: one scripted, config-driven training run
   (not a notebook) that reproducibly produces the exact checkpoint the
   apps consume, with a fixed seed and a saved model card (architecture,
   data version, metrics, date).
   Impact: High · Difficulty: Medium · User value: Medium · Complexity: Medium.
3. **Confidence calibration** (temperature scaling is a ~20-line addition
   post-hoc on a validation set). Impact: Medium-High · Difficulty: Low ·
   User value: High (users currently see falsely precise-looking
   percentages) · Complexity: Low.
4. **Out-of-distribution / "is this even a leaf" gate** (a cheap
   binary classifier or a simple embedding-distance check before running
   the full disease classifier) to stop confidently misclassifying
   non-leaf photos. Impact: High · Difficulty: Medium · User value: High ·
   Complexity: Medium.

### Tier 2 — architecture upgrades

5. **Modern CNN backbones** (ConvNeXt, EfficientNetV2) as a drop-in
   ResNet50 replacement — likely accuracy gains for similar or lower
   inference cost. Impact: Medium · Difficulty: Low-Medium · User value:
   Medium · Complexity: Low.
6. **Vision Transformer / hybrid approaches** (ViT-B/16, or a
   CNN+attention hybrid) for potentially better fine-grained
   disease-vs-cosmetic-damage discrimination, at the cost of needing
   more training data/augmentation to avoid overfitting on a modest
   dataset. Impact: Medium · Difficulty: Medium-High · User value:
   Medium · Complexity: High.
7. **Ensemble of 2–3 architectures** (e.g. ResNet50 + EfficientNet +
   ViT, majority/averaged vote) for both accuracy and a natural
   uncertainty signal (disagreement = low confidence flag).
   Impact: Medium-High · Difficulty: Medium · User value: Medium ·
   Complexity: Medium.
8. **Multi-task head**: joint disease classification + severity
   regression + lesion segmentation, trained end-to-end instead of the
   current bolt-on heuristics for severity and Grad-CAM-derived boxes.
   Impact: High · Difficulty: High · User value: High · Complexity: High.

### Tier 3 — data & feature engineering

9. **Real lesion segmentation dataset/labels** — even a modest hand- or
   semi-automatically-labeled segmentation set would let boxes come from
   a trained segmentation head instead of a Grad-CAM proxy, dramatically
   improving localization trustworthiness.
   Impact: High · Difficulty: High · User value: High · Complexity: High.
10. **Data augmentation modernization** (RandAugment/AutoAugment,
    MixUp/CutMix) — cheap to add to any retraining run, directly
    addresses the severe class imbalance (111–2,456 images/class) seen
    in the pest dataset. Impact: Medium · Difficulty: Low · User value:
    Low (indirect) · Complexity: Low.
11. **Class-balanced loss (focal loss / class-weighted CE)** to address
    the imbalance directly rather than only through data augmentation.
    Impact: Medium · Difficulty: Low · User value: Low (indirect) ·
    Complexity: Low.
12. **Larger, better-provenance training set** — commission or curate a
    disease-labeled dataset matching the actual 38-class PlantVillage
    taxonomy served in production (closing the current train/serve class
    mismatch entirely). Impact: High · Difficulty: High (data
    collection) · User value: High · Complexity: Medium.

### Tier 4 — serving & explainability

13. **Model export to ONNX/TorchScript + quantization (int8/fp16)** for
    materially faster CPU inference — directly benefits the live-camera
    mode's per-frame latency. Impact: Medium-High · Difficulty: Medium ·
    User value: High (perceived responsiveness) · Complexity: Medium.
14. **Edge deployment** (TFLite/CoreML/ONNX Runtime Mobile) to enable a
    true offline mobile field-scanning app, which fits this project's
    inferred rural/smallholder use case far better than a webserver.
    Impact: High · Difficulty: High · User value: High · Complexity: High.
15. **Batched/async inference queue** (replace synchronous in-handler
    forward passes with a worker pool or a small task queue) to unblock
    the Starlette event loop under concurrent load.
    Impact: Medium · Difficulty: Medium · User value: Medium (mostly
    invisible until under load) · Complexity: Medium.
16. **Confidence-visualization upgrade**: show calibrated confidence with
    a visual uncertainty band (not just a bare percentage), and surface
    ensemble disagreement (if Tier 2 idea 7 is adopted) as an explicit
    "the model is unsure" state instead of a falsely confident number.
    Impact: Medium · Difficulty: Low · User value: High · Complexity: Low.
17. **Human-in-the-loop feedback loop**: let a user flag "this was
    wrong," store the image + correction (with consent), and use it as a
    continuous fine-tuning/active-learning signal. Impact: High (long
    term) · Difficulty: Medium (needs the database work in
    `docs/Database.md`) · User value: Medium · Complexity: Medium.
18. **Explainability upgrade beyond Grad-CAM**: add Grad-CAM++ or
    Score-CAM for sharper localization, and/or a textual explanation
    layer (e.g. mapping activated regions to human-readable symptom
    descriptions) to make the "why" more actionable than a heatmap alone.
    Impact: Medium · Difficulty: Medium · User value: Medium ·
    Complexity: Medium.

### Tier 5 — research directions (longer horizon)

19. **Foundation-model fine-tuning** (e.g. fine-tuning a
    vision-language model to produce natural-language diagnosis +
    treatment text directly, instead of a fixed 38-class taxonomy +
    static dict) — would generalize far beyond PlantVillage's crop set.
    Impact: High · Difficulty: High · User value: High · Complexity: High.
20. **Multi-modal input** (leaf photo + simple metadata like crop type,
    region, recent weather) to disambiguate visually similar
    diseases across different plant species. Impact: Medium-High ·
    Difficulty: High (needs new data collection) · User value: Medium ·
    Complexity: High.
21. **Federated/on-device learning** across field users to improve the
    model over time without centralizing sensitive farm imagery.
    Impact: Medium (long-term) · Difficulty: Very High · User value:
    Low (near-term) · Complexity: Very High.

See `docs/Upgrade-Ideas.md` for how these map onto the overall
prioritized roadmap alongside non-ML work.
