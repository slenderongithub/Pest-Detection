# Purpose

## Problem it solves

Smallholder and hobbyist growers often cannot visually distinguish early
plant disease/pest symptoms from cosmetic leaf damage, and don't have
easy access to an agronomist. This project's stated goal (consistent
across the README, CONTEXT.md, and the paper figures) is to let someone
**point a camera or upload a photo of a leaf and get back**:

1. A disease/pest identification,
2. A visual pointer to *where* on the leaf the problem is (Grad-CAM +
   bounding boxes),
3. A severity read (Healthy → Critical),
4. A plain-language treatment/pesticide suggestion.

## Target users (inferred, not stated anywhere explicitly)

- **Primary**: hobbyist/home gardeners and small-scale farmers without
  access to a plant pathologist, using a phone or webcam.
- **Secondary**: the author themself, as a portfolio/academic
  deliverable — the paper-figure generator and dosage spreadsheet exist
  to produce evidence for a report/thesis, not because end users need a
  pie chart of the training set.
- There is **no evidence of any real user testing, user accounts, or
  telemetry** — this is a single-user local/demo tool, not a deployed
  multi-tenant product.

## Primary use cases

1. **One-shot photo diagnosis** (both apps): upload a JPEG/PNG of a leaf,
   get a classification + overlay + tip in a few seconds.
2. **Continuous camera monitoring** (Starlette app only): point a webcam
   at a plant and get a live-updating diagnosis every 1.2 seconds — the
   closest thing to a "field scanning" workflow.
3. **Research figure generation** (author-only workflow): regenerate the
   7 paper figures from `scripts/`, most of which are static or
   hand-entered data, not derived from live training runs.

## Core workflow / user journey

```
Open app → (upload photo | start camera)
        → model runs inference (ResNet50/34, or heuristic if no weights)
        → Grad-CAM computed on last conv layer
        → CAM thresholded → connected components → bounding boxes
        → overlay image composited (tint + boxes + labels)
        → severity derived from disease-name lookup or confidence bands
        → pesticide/treatment tip shown if disease is known
        → result appended to "last 5 scans" history
```

There is no login, no persistence across sessions/restarts, no way to
export a result, and no multi-image/batch mode.

## Business value

As shipped, this is not a business — no payments, no user accounts, no
analytics, no SaaS packaging. Its value today is:
- A demonstrable ML + full-stack skill portfolio piece (two different
  serving stacks, custom Grad-CAM, a from-scratch connected-component
  algorithm, dark-glassmorphism UI design).
- A latent utility tool that *could* become an actual product for
  smallholder agriculture if the model quality, dataset provenance, and
  reliability issues in `docs/Known-Issues.md` were addressed.

## Current strengths

- **UI polish is genuinely good** for a demo project — consistent dark
  glassmorphism design system, CSS custom properties, real typography
  choices (Inter/Space Grotesk), animated "scan ring" affordance on the
  camera view.
- **Grad-CAM and bounding-box localization are hand-implemented**
  end-to-end (forward/backward hooks, manual BFS connected components) —
  shows genuine understanding of the underlying math rather than an
  opaque library call.
- **Model-loading fallback chain** (ResNet50 checkpoint → ResNet50
  FastAI export → legacy ResNet34 → heuristic) means the app never hard
  fails to start even with zero committed weights.
- **Two serving paradigms explored** (Streamlit for rapid iteration,
  Starlette for a real API + custom frontend) shows range.

## Current weaknesses

- **No model weights ship with the repo** (by design — `.gitignore`
  excludes `*.pth`/`*.pkl`) and the only auto-download path targets an
  unauthenticated Google Drive URL for a *different, older* ResNet34
  model — a fresh clone cannot reproduce the "real" ResNet50 experience
  at all.
- **The two front-ends are fully divergent, hand-duplicated
  implementations** of the same ~12 functions (class list, disease info,
  Grad-CAM, box-finding, overlay compositing, severity logic). A model
  change or bug fix must be applied twice.
- **Training data ≠ inference classes.** The notebooks train a 9-class
  pest-only classifier; the shipped apps classify against a 38-class
  PlantVillage disease taxonomy. There is no evidence any notebook here
  actually produced `export_resnet50_model.pth`/`.pkl` — the shipped
  model's provenance is undocumented.
- **The "confidence ≥ 85% → pesticide recommendation" gate** described in
  the README does not exist in the shipped code (`app.py`/`server.py`
  show disease tips at any confidence, gated only by `DISEASE_INFO`
  dict membership, not a confidence threshold and not the xlsx file).
- **`requirements.txt` cannot be installed as-is** on the same Python
  version that runs Streamlit ≥1.32 (see `docs/Tech-Stack.md`).

## What makes this project distinctive

Most "plant disease detector" tutorial clones (and this looks like it
began as one — `Plant_Disease_Detection/` in the stale README, FastAI
ResNet34, AWS/GCP deployment guide are hallmarks of a well-known public
template) stop at classification. This project's differentiators are:
(1) the hand-rolled Grad-CAM + lesion localization overlay, (2) the live
webcam scanning mode, and (3) the severity/treatment layer bolted on top
of raw classification output.
