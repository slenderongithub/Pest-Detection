# Glossary

- **Backend (in this codebase's vocabulary)** — not a server backend;
  a `dict` describing which model is loaded (`kind`, `model`, `classes`,
  `preprocess`). Returned by `load_backend()`. Overloaded term — don't
  confuse with "the Starlette backend" as a whole.
- **CAM / Grad-CAM (Class Activation Mapping / Gradient-weighted CAM)** —
  a technique for visualizing which spatial regions of a CNN's last
  convolutional feature map most influenced the predicted class. Here,
  computed manually via forward/backward hooks on `layer4[-1]` of a
  ResNet, weighted by the mean gradient per channel, ReLU'd, and
  upsampled to 224×224.
- **`CLASS_NAMES` / `CLASSES`** — the 39-entry list (38 PlantVillage
  disease labels + `background`) both apps use as the classifier's
  output vocabulary. Duplicated verbatim between `app.py` and
  `app/server.py`.
- **Connected-component boxes** — this project's hand-rolled
  bounding-box extractor: BFS flood-fill over a boolean mask
  (thresholded Grad-CAM or heuristic lesion mask), grouping adjacent
  `True` pixels into components, discarding components under a minimum
  pixel count, then emitting an axis-aligned bounding box (with padding)
  per surviving component. Not related to OpenCV's or scipy's
  `connectedComponents`/`label` — no such library is used.
- **`DISEASE_INFO`** — hardcoded Python dict of ~19 keyword → {severity,
  cause, tip} entries. Matched via substring search against the
  predicted label (`get_disease_info`). Only ~19 of the 38 disease
  classes have an entry; the rest fall back to a generic message.
- **Heuristic fallback / `heuristic_classify`** — a non-ML, pure
  color-space classifier used by the Starlette app **only** when no
  model file is found on disk. Buckets pixels into brown/yellow/white/dark
  masks by RGB/saturation thresholds and maps the dominant mask to a
  **hardcoded tomato-family label**, regardless of what's actually in
  the photo. Exists so the app never crashes with zero weights, at the
  cost of being actively misleading.
- **Learner (FastAI)** — FastAI v1's bundled object (`model` +
  `data.classes` + training metadata) produced by `Learner.export()` and
  reloaded via `load_learner()`. Two legacy `.pkl` exports are referenced:
  a ResNet50 learner and a ResNet34 learner.
- **PlantVillage** — the public 38-class (+background) leaf disease
  image dataset that the shipped apps' `CLASS_NAMES` vocabulary is
  modeled on. Not present in this repo; the local `data/Pest_Dataset/`
  is a *different*, 9-class pest dataset used only by the notebooks.
- **Severity** — one of `Healthy | Low | Medium | High | Critical`,
  either looked up from `DISEASE_INFO` or derived from confidence bands
  (`≥85→High, ≥60→Medium, else Low`) when the disease isn't in the
  dictionary. Drives both a colored "pill" badge and the overlay tint.
- **Starlette "app" (the module-level variable)** — instantiated
  **twice** in `app/server.py` (lines 185 and 645); the first instance
  is fully configured (middleware + static mount) and then discarded
  when the name is rebound at line 645. See `docs/Known-Issues.md` for
  why this is dead code rather than a functional bug.
- **Torch monkey-patch (`torch.load` override)** — both apps override
  `torch.load` at import time to default `weights_only=False`, because
  modern PyTorch defaults `weights_only=True` and would otherwise reject
  the old pickle-based FastAI/ResNet checkpoints this project loads.
