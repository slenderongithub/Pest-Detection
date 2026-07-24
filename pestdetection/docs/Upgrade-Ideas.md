# Upgrade Master Plan

Consolidated, prioritized roadmap combining `docs/Machine-Learning.md`'s
ML-specific ideas and `docs/Future-Ideas.md`'s product/UX/platform
ideas. Grouped per the discovery-session brief. Nothing here has been
implemented — this is planning input for the next (implementation)
session.

## Quick Wins (low difficulty, do first)

| Idea | Why it matters | Impact | Difficulty | Dependencies |
|---|---|---|---|---|
| Fix `download_model.py`'s broken import (`export_file_url`→`LEGACY_MODEL_URL`) | Currently 100% broken on run | Low-Medium | Low | None |
| Remove the dead first `Starlette()`/`load_backend()` duplicate definitions in `app/server.py` | Removes a maintenance trap, no behavior change | Low | Low | None |
| Compress `app/static/logo.png` (4MB→<200KB) and delete the 3 unused screenshot PNGs in `app/view/` | Faster load, smaller repo | Low | Low | None |
| Update `README.md` to match the current flattened structure | Stops onboarding confusion | Medium | Low | None |
| Correct or remove the confidence-gated pesticide-xlsx claim in the README | Docs currently describe a feature that doesn't exist | Medium | Low | None |
| Add basic `try/except` around image decode + inference in `/analyze` | Prevents raw 500s/tracebacks reaching users | Medium | Low | None |

## High Impact Improvements

| Idea | Why it matters | Impact | Difficulty | Dependencies |
|---|---|---|---|---|
| Extract shared `leafscan_core` module (class list, disease dict, Grad-CAM, boxes, overlay, model loading) used by both `app.py` and `app/server.py` | Root cause of most drift/duplication bugs today | High | Medium | None — do this before other ML changes |
| Fix the model-provisioning story: publish/host the actual ResNet50 checkpoint (or a documented, reproducible retraining path) instead of relying on a personal Google Drive link to the wrong (ResNet34) model | The flagship "ResNet50" experience is currently unreproducible from a clean clone | High | Medium | None |
| Establish real evaluation metrics for the deployed model (accuracy/F1/confusion matrix on a held-out PlantVillage split) | No verifiable performance claim exists today | High | Low | Requires a labeled eval set |
| Add a visible "demo/no-model" state instead of the Starlette heuristic silently faking tomato predictions | Current behavior is actively misleading | High | Low | None |
| Pin a genuinely installable `requirements.txt` (or migrate to `uv`/`poetry` with a lockfile) | Nothing can be reliably reproduced or CI'd until this works | High | Medium | None |

## ML Evolution Roadmap (see `docs/Machine-Learning.md` for full detail)

1. **Tier 1 — foundational**: real evaluation, unified reproducible
   training pipeline, confidence calibration, OOD "is this a leaf" gate.
2. **Tier 2 — architecture**: modern CNN backbones (ConvNeXt/
   EfficientNetV2), ViT/hybrid exploration, small ensembles, multi-task
   head (classification + severity + segmentation).
3. **Tier 3 — data**: real lesion-segmentation labels, modern
   augmentation (RandAugment/MixUp/CutMix), class-balanced loss, a
   larger/better-provenance dataset matching the served taxonomy.
4. **Tier 4 — serving/explainability**: ONNX/TorchScript + quantization,
   edge/mobile deployment, async/batched inference, calibrated
   confidence visualization, human-in-the-loop feedback capture,
   Grad-CAM++/Score-CAM.
5. **Tier 5 — research**: fine-tuned vision-language diagnosis model,
   multi-modal input (photo + crop/region/weather metadata), federated/
   on-device learning.

## Architecture Improvements

- Deduplicate the two front-end/back-end implementations (see High
  Impact table above) — this is the architectural prerequisite for
  almost everything else moving faster.
- Decide on **one** canonical serving surface (Streamlit for rapid
  iteration/internal demo, or Starlette+SPA for a real product) rather
  than maintaining both indefinitely; if both are kept intentionally,
  document *why* explicitly so it reads as a decision, not drift.
- Introduce a config layer (env vars or a settings file) for host, port,
  model directory, and the legacy model URL — currently all hardcoded
  literals.
- Move inference off the request-handling event loop (worker pool or
  task queue) so the Starlette app doesn't block concurrent requests
  during a synchronous forward+backward pass.

## UI/UX Improvements

See `docs/UI-UX-Review.md` and `docs/Future-Ideas.md` for full detail:
unify design tokens across both surfaces, add an onboarding explainer
for Grad-CAM/confidence, accessibility pass (ARIA live regions, focus
states, icon-redundant severity), mobile-first camera capture.

## Infrastructure Improvements

- Add a `Dockerfile` (or one per app) and basic CI (lint + import smoke
  test + `/health` check).
- Add a model registry/manifest (checkpoint hash, data version, metrics,
  date) so "which model is running and how good is it" has one answer.
- Add persistence (see `docs/Database.md`) for scan history and the
  disease/pesticide knowledge base, retiring the orphaned `.xlsx` and
  the two hardcoded dict copies in favor of one queryable source.

## Production Readiness

- Authentication/authorization (currently none).
- CORS allowlist instead of `allow_origins=["*"]`.
- Rate limiting and request size limits on `/analyze`/`/analyze-frame`.
- Structured error responses and basic exception monitoring (e.g.
  Sentry) instead of raw tracebacks.
- A real test suite (unit tests for the pure functions, integration
  tests for the two HTTP routes) as a safety net for all of the above.

## Stretch Goals

- Native mobile app / PWA with fully offline on-device inference
  (TFLite/CoreML/ONNX Runtime Mobile) — best fit for the inferred
  rural/field use case.
- Vision-language foundation-model fine-tuning for open-vocabulary,
  natural-language diagnosis beyond the fixed 38-class taxonomy.
- Federated/continual learning from opt-in field corrections, closing
  the loop between real-world usage and model improvement.

## How to sequence this

Recommended order for a future implementation session, given
everything above: **(1) Quick Wins → (2) shared-module extraction →
(3) real evaluation + model provenance fix → (4) pick-one-surface
decision → (5) ML Tier 1 → (6) infra/production-readiness → (7)
everything else, prioritized by what the user actually wants this
project to become** (portfolio piece vs. real field tool vs. research
artifact — see the open question in `ai-context/NEXT_SESSION.md`).
