# Project Overview

## What this repository is

**LeafScan / Pest Detection** is a deep-learning plant-disease and pest
identification system. It is a **portfolio / academic research project**
(evidence: a 7-figure "paper_figures" set, a pesticide dosage spreadsheet
framed as a "recommendation module," and an evaluation write-up style
consistent with a course or thesis deliverable) that has since been
partially re-skinned into a more polished consumer-facing demo called
**LeafScan**.

The repository actually contains **four semi-independent things** bolted
together under one root:

1. **A Streamlit single-page app** (`app.py`, root) — the primary, most
   polished surface. Branded "LeafScan." Upload a leaf photo, get a
   disease class, a Grad-CAM-derived lesion overlay with bounding boxes,
   a severity badge, and a treatment tip.
2. **A Starlette/Uvicorn web API + hand-written HTML/CSS/JS SPA**
   (`app/`) — a second, independent implementation of almost the exact
   same product, but adding **live webcam scanning** (auto-capture every
   1.2s). Titled "Plant Disease Detector" in its `<title>` tag.
3. **A research/paper artifact generator** (`scripts/generate_figures.py`,
   `scripts/generate_fig7.py`) — produces 7 publication-style PNGs
   (class distribution, pipeline diagram, CNN architecture diagram,
   flowchart, accuracy/loss curves, pie chart, and a **simulated/fabricated
   "sample output" screenshot** — see `docs/Known-Issues.md`).
4. **A collection of training notebooks** (`notebooks/` and `notebook/`,
   two different directories) exploring 9 different model
   architectures/frameworks (custom CNN in Keras/TensorFlow, PyTorch,
   FastAI, ResNet34/50, DenseNet121, VGG16/19) against two different
   datasets (a 9-class pest photo dataset and the 38-class PlantVillage
   disease dataset).

These four pieces are **not unified**: the Streamlit app and the
Starlette app duplicate ~90% of their ML logic independently (see
`docs/Technical-Debt.md`), the notebooks train on a 9-class pest dataset
while the shipped apps run inference against a 38-class PlantVillage
disease taxonomy, and the "paper" figures describe a pipeline
(image → custom CNN → pesticide lookup) that is materially different
from what either running app actually does (image → ResNet50/34 →
Grad-CAM → severity heuristic).

## Provenance signal

- The training notebook (`notebooks/Pest_Detection_Training.ipynb`)
  contains a hardcoded Windows path
  `C:\Users\ephra\OneDrive\Desktop\Application of AI project\Pest_Dataset`
  — this strongly suggests the notebook (and possibly the original
  project skeleton) originated from a different author/machine and was
  imported into this repo rather than authored here from scratch.
- The root `README.md` still documents an older repository layout
  (`Plant_Disease_Detection/` subfolder with a Flask app and an AWS/GCP
  `deployment_guide/`) that **no longer exists** — the most recent commit
  (`28aa666`, "Flatten Pest Detection directory structure and remove
  boilerplate") flattened that structure, but the README was not updated
  to match. Treat `README.md` as stale; `CONTEXT.md` (repo root) is the
  accurate, current description.

## Repository root layout (current, verified)

```
pestdetection/
├── app.py                  # Streamlit app — "LeafScan" (954 lines)
├── download_model.py       # Broken helper script (ImportError on run)
├── requirements.txt        # Outdated/conflicting pins
├── CONTEXT.md              # Pre-existing, accurate session-context file
├── README.md               # STALE — describes a pre-flatten layout
├── LICENSE
├── app/
│   ├── server.py           # Starlette API + duplicated ML logic (681 lines)
│   ├── models/models.md    # Placeholder; no weights committed (gitignored)
│   ├── static/             # client.js, style.css, logo.png (4MB), leaf.png
│   └── view/               # index.html + 3 stray dev screenshots
├── data/Pest_Dataset/       # 9 pest-class folders, gitignored, ~4,770 imgs
├── docs/                   # Pesticide xlsx + paper_figures/ + this docs set
├── scripts/                # generate_figures.py, generate_fig7.py
├── notebooks/              # Pest_Detection_Training.ipynb (Keras/TF)
├── notebook/               # 9 other architecture-exploration notebooks
└── .venv39/                # Local virtualenv (not committed logic)
```

## Two runnable products, one purpose

| | Streamlit (`app.py`) | Starlette (`app/server.py`) |
|---|---|---|
| Run | `streamlit run app.py` | `python app/server.py serve` |
| Port | Streamlit default (8501) | 8080 |
| Input | File upload only | File upload **and** live camera (1.2s interval) |
| Output | Rendered in-page | JSON API + hand-rolled JS renderer |
| Fallback if no model | Crashes (FastAI download path) | Color-heuristic "fake" classifier |
| State | `st.session_state` (last 5 scans) | In-memory JS array (last 5 scans) |

Both exist in parallel; neither has been deprecated in favor of the
other. See `docs/Architecture.md` for why this matters.
