# Tech Stack

## Runtime

| Component | Version (as pinned/observed) | Notes |
|---|---|---|
| Python | 3.9+ (venv named `.venv39`) | `str.removeprefix` (3.9+) used in `clean_state_dict` |
| PyTorch | pinned `1.4.0` in requirements.txt (2020-era) | Incompatible with Python 3.9+/modern numpy; actual venv almost certainly has a newer torch installed manually — pin is aspirational/stale |
| TorchVision | pinned `0.5.0` | Paired with torch 1.4; `ResNet50_Weights.DEFAULT` enum used in code requires torchvision ≥0.13 — **directly contradicts this pin** |
| FastAI | pinned `1.0.60` (legacy v1 API: `fastai.vision.load_learner`, `open_image`) | FastAI v1 is EOL; incompatible with torch ≥1.6 in places |
| NumPy | pinned `1.16.3` (2019) | Old |
| Streamlit | `>=1.32.0` | Modern; needs modern numpy — conflicts with the 1.16.3 pin above |
| Starlette | pinned `0.12.0` (2019) | Old API; still compatible with the small surface used here |
| Uvicorn | pinned `0.7.1` | Very old |
| aiohttp | pinned `3.5.4` | Used only for the (never-called-correctly) model downloader |
| python-multipart | `0.0.5` | Required by Starlette for `request.form()` |

**This pin set cannot be installed together.** Streamlit ≥1.32 pulls in
a numpy/pyarrow/pandas stack that is incompatible with `numpy==1.16.3`
and `torch==1.4.0`. The repository only runs today because whatever is
actually installed in `.venv39` diverges from `requirements.txt` — the
lockfile is decorative, not functional. See `docs/Known-Issues.md`.

## ML / CV libraries

- **PyTorch + TorchVision** — primary inference path (`torch-resnet50` /
  `torch-resnet34` backend kind).
- **FastAI v1** — legacy path for `.pkl` learner exports
  (`fastai-resnet50` / `fastai-resnet34` backend kind). Both apps import
  `fastai.vision.load_learner` and `open_image` lazily (only inside the
  branch that needs them) so a FastAI-less environment still boots if a
  `.pth` file is present.
- **Pillow (PIL)** — image decode, overlay compositing
  (`ImageDraw`, `ImageEnhance`, `alpha_composite`).
- **NumPy** — heatmap thresholding, connected-component masks, heuristic
  color-space math.
- No `scipy`, `opencv-python`, `pytorch-grad-cam`, or `scikit-learn` in
  the shipped app dependency list — Grad-CAM, connected components, and
  bounding boxes are all hand-rolled (see `docs/Machine-Learning.md`).
  (OpenCV *is* used inside `notebook/` experimentation notebooks, but
  never in the shipped apps.)

## Frontend

- **Streamlit** (`app.py`): server-rendered Python UI, styled via a
  ~210-line injected `<style>` block (`CUSTOM_STYLE`) rather than
  Streamlit's native theming API — CSS targets private Streamlit
  `data-testid` attributes (`stFileUploader`, `stMetric`, ...), which are
  **not a public/stable API** and can break on a Streamlit version bump.
- **Vanilla JS + hand-written HTML/CSS** (`app/static/`, `app/view/`):
  no framework, no bundler, no build step — a single IIFE (`client.js`,
  286 lines) directly manipulates the DOM via `getElementById`.
- **Fonts**: Google Fonts CDN (`Inter`, `Space Grotesk`) — loaded via
  `@import` in Streamlit CSS and `<link>` tags in `index.html`. Both
  apps depend on an external network fetch for typography; no local
  font fallback bundling.

## Design system

Both UIs share one dark "glassmorphism" visual language, independently
re-implemented in CSS twice:
- Palette: `--accent:#45f0a3` (green), `--accent-2:#7dd3fc` (blue),
  `--warn:#ffb86b`, `--danger:#ff6b6b`/`#ff5d5d`.
- Cards: translucent panels (`rgba(8-15,15-27,26-31,0.72-0.95)`), 1px
  `rgba(148,163,184,0.10-0.16)` borders, 16–28px border radius.
- Severity color-coding is duplicated as a Python dict
  (`SEVERITY_CONFIG` in `app.py`) and as CSS classes
  (`.severity-critical/high/medium/low/healthy` in `style.css`) — two
  independent sources of truth for the same 5-way color mapping.

## Data / storage

- **No database of any kind.** "Data" is: (a) a gitignored image folder
  (`data/Pest_Dataset/`) used only by notebooks, (b) an Excel workbook
  (`docs/Pesticides_With_Agri_Guideline_Dosage.xlsx`) referenced in
  README as the pesticide source of truth but **never read by any
  running code** — both apps use a hardcoded Python dict
  (`DISEASE_INFO`) instead.
- **No persistence layer** — "scan history" lives only in
  `st.session_state` (Streamlit, cleared on server restart) or a JS
  array in browser memory (Starlette client, cleared on page reload).

## Build / dev tooling

- No bundler, no transpiler, no `package.json` — the JS/CSS are served
  as static files by Starlette's `StaticFiles` mount.
- No linter/formatter config (no `pyproject.toml`, `ruff.toml`,
  `.flake8`, `.eslintrc`) anywhere in the repo.
- No test suite, no CI configuration (no `.github/workflows`).
- No `Dockerfile`/`docker-compose.yml` — deployment is "run the process
  directly" only (see `docs/Deployment.md`).

## External services

- Google Fonts CDN (typography, both apps).
- Google Drive direct-download URL (`LEGACY_MODEL_URL`) — the only
  automated model-weight fetch mechanism, and it targets the *wrong*
  (legacy ResNet34) checkpoint even for the "preferred" ResNet50 path.
- `models.ResNet50_Weights.DEFAULT` — pulls ImageNet-pretrained
  TorchVision weights from PyTorch's own CDN on first run if no local
  checkpoint exists yet (used only for building the *architecture*, not
  as the final classifier — see `docs/Machine-Learning.md`).
