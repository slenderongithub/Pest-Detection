# LeafScan — Agricultural Pest Detection

LeafScan identifies agricultural pests from a photo, shows a **calibrated** confidence,
highlights the region the model focused on (Grad-CAM), abstains when it is unsure, and
surfaces reference treatment guidance. One tested Python package (`leafscan`) powers both
a Streamlit UI and a Starlette API — there is no duplicated ML logic.

> **Scope (honest):** the flagship model (**ResNet50**) classifies the **9 pest classes** in the only
> real dataset present (`data/Pest_Dataset/`). It is decision-support triage, not a
> verdict, and does not detect pests/diseases outside those 9 classes.

## Headline metrics — ResNet50 flagship (held-out test split, real)

| Metric | Value |
|---|---|
| **Macro-F1** (primary) | **0.773** |
| Weighted-F1 | 0.828 |
| Accuracy | 0.814 |
| Top-3 accuracy | 0.962 |
| ECE (before → after temperature scaling) | 0.091 → 0.059 |
| Test samples | 716 |

Macro-F1 is the honest headline because the dataset is ~22× imbalanced (aphids ≈ 51%).
Full per-class numbers, calibration, and limitations are in
[`MODEL_CARD.md`](MODEL_CARD.md); the machine-readable source is
[`reports/metrics.json`](reports/metrics.json). Every figure and number in this repo is
generated from that file — nothing is hand-transcribed or mocked.

## Quickstart

```bash
# 1. Install (editable, with serve + train + dev extras).
#    On a CPU-only host, install torch first from the CPU wheel index:
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[serve,train,dev]'

# 2. Get a model — either train one (needs data/Pest_Dataset/) ...
leafscan train --config configs/pest_resnet50.yaml   # flagship (use pest_resnet18.yaml for a lighter model)
#    ... or fetch a released checkpoint (hash-verified):
python scripts/fetch_model.py

# 3. Run either front-end
python app/server.py serve          # Starlette API + live-camera SPA on :8080
streamlit run app.py                # Streamlit UI
```

If **no** model is installed, both surfaces show an explicit **demo / no-model state** —
they never fabricate a prediction.

## How it works

```
image → preprocess (from bundle) → resnet50 → logits
      → softmax(logits / T)  [temperature-calibrated]
      → top-1 + abstain gate (uncertain if below threshold)
      → Grad-CAM (upload path only) → focused-region overlay
      → pest knowledge (severity + reference treatment)
```

**Self-describing checkpoints.** A checkpoint is a directory with `weights.pt` (a plain
tensor `state_dict`, loaded with `weights_only=True` — no pickle RCE) and `manifest.json`
carrying the class names, normalization, input size, Grad-CAM layer, calibration
temperature, dataset hash, git sha, and metrics. **The apps read their taxonomy from the
model**, so the train/serve class mismatch is impossible by construction.

## CLI

```bash
leafscan split      --config configs/pest_resnet50.yaml   # build the deterministic split
leafscan train      --config configs/pest_resnet50.yaml   # train → calibrate → evaluate → bundle
leafscan evaluate   --model-dir models/pest_resnet50      # metrics on the test split
leafscan predict    path/to/image.jpg                     # single-image inference
leafscan card       --model-dir models/pest_resnet50      # regenerate MODEL_CARD.md
```

## Development

```bash
make lint     # ruff
make test     # pytest (unit + Starlette integration)
make smoke    # fast training smoke (frozen backbone, tiny subset)
make docker   # build the CPU serving image
```

CI (`.github/workflows/ci.yml`) runs ruff + the test suite + import smoke + a tiny
training smoke + a `/health` boot check on every push.

## Project structure

```
leafscan/                 # the one shared package (all ML logic)
  config.py               # env-driven settings (host/port/model dir/CORS/abstain…)
  models.py preprocess.py # backbone factory + transforms (built from the bundle)
  checkpoint.py registry.py  # self-describing bundle + JSON model registry
  data.py                 # deterministic stratified split, class weights, dataset
  train.py evaluate.py calibration.py  # training, metrics, temperature scaling
  gradcam.py postprocess.py            # Grad-CAM + localization/overlay (one copy)
  inference.py            # Predictor (shared by both apps)
  knowledge.py            # pest severity + reference treatment (from data/knowledge)
  cli.py modelcard.py
app.py                    # thin Streamlit shell → leafscan
app/server.py             # thin Starlette shell → leafscan (run: python app/server.py serve)
app/static/ app/view/     # SPA assets
configs/                  # training configs (YAML)
data/knowledge/pests.yaml # pest knowledge base (committed)
data/splits/pest_v1.csv   # committed deterministic split (images stay gitignored)
models/pest_resnet50/     # deployed flagship: manifest.json (committed) + weights.pt (release asset)
models/pest_resnet18/     # lighter alternative model
models/registry.json      # which model is deployed and how good it is
reports/                  # metrics.json + confusion_matrix.png + reliability.png
scripts/                  # figure generation (from real metrics) + fetch_model.py
tests/                    # pytest suite
```

## Safety & honesty features

- **Calibrated confidence** (temperature scaling fit on a disjoint val split).
- **Abstention gate** — low-confidence inputs are flagged `uncertain` rather than
  confidently misclassified (an honest gate, not a true out-of-distribution detector).
- **No fabrication** — no model ⇒ explicit demo state / HTTP 503, never a fake prediction.
- **Safe checkpoint loading** (`weights_only=True`); the old `torch.load` monkey-patch and
  the pickle-based fastai path are gone.
- **Serving hardening** — upload size + decompression-bomb caps, verify-then-reopen image
  validation, structured JSON errors, `run_in_threadpool` so inference never blocks the
  event loop, Grad-CAM off for live camera frames, config-driven CORS allowlist.

## Disclaimer

Pesticide names and dosages are a **cited reference, not a prescription**. Always follow
product labels, local regulations, and qualified agronomic advice.

## License

MIT — see [LICENSE](LICENSE).
