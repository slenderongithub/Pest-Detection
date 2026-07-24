# Known Issues

Verified directly against source on 2026-07-11. Where a pre-existing
note in root `CONTEXT.md` turned out to be imprecise, this doc states
the corrected, verified finding and flags the discrepancy.

## 🔴 Critical

1. **No model weights ship with the repo, and the only auto-download
   path fetches the wrong (legacy) model.** `.gitignore` excludes
   `*.pkl`/`*.pth`/`*.h5`; `app/models/` contains only a placeholder
   `models.md`. The single automated fetch
   (`download_legacy_model_if_needed`) targets a Google Drive URL for
   the **ResNet34** FastAI export, not the "preferred" ResNet50 path.
   A fresh clone of the Streamlit app either downloads an old model or
   crashes if that link is dead/rate-limited; a fresh clone of the
   Starlette app silently serves fabricated tomato-only predictions via
   the heuristic fallback with no visible warning to the user. **Impact:
   the flagship "ResNet50" experience is not reproducible from a clean
   checkout.**

2. **`download_model.py` is broken** — `from app.server import
   download_file, export_file_url` raises `ImportError` immediately;
   `app/server.py` defines `LEGACY_MODEL_URL`, not `export_file_url`.
   Running this script as documented/implied fails on line 2, always.

3. **Massive, drifting code duplication between `app.py` and
   `app/server.py`.** Both files independently define `CLASS_NAMES`/
   `CLASSES`, `DISEASE_INFO`, `build_resnet50`, `compute_gradcam`,
   `component_boxes`/`connected_component_boxes`, `make_overlay`/
   `overlay_image`, `load_backend`, `preprocess_pil`/`preprocess_image`,
   `pretty_label`, `get_disease_info`, `get_severity`,
   `clean_state_dict`. They have **already drifted**: overlay tint is
   `(255,86,48)`/`alpha=(heat**1.5)*190` in `app.py` vs.
   `(255,82,48)`/`alpha=(heat**1.4)*180` in `app/server.py`. Any future
   fix applied to one and not the other silently deepens the
   divergence.

4. **`requirements.txt` cannot be installed as a coherent set.**
   `torch==1.4.0` + `numpy==1.16.3` (2019–2020 era) directly conflict
   with `streamlit>=1.32.0`'s modern dependency tree, and
   `ResNet50_Weights.DEFAULT` (used in `build_resnet50`) requires a
   torchvision version far newer than the pinned `0.5.0`. The installed
   `.venv39` almost certainly diverges from this file already — the
   lockfile does not describe a working environment.

5. **`scripts/generate_fig7.py` fabricates a "sample pipeline output"
   figure rather than capturing a real one.** It draws a matplotlib mock
   camera frame with hand-typed overlay text
   (`"aphids | Imidacloprid 17.8% SL... Confidence: 94.2%"`) — not a
   screenshot or programmatic render of any actual model output. If
   this figure is used in a paper/report as evidence of system
   behavior, that is a reproducibility/integrity concern, independent
   of code quality. **Recommend either clearly labeling it as an
   illustrative mock-up, or replacing it with a real captured
   inference.**

## 🟡 Medium (corrected from prior notes)

6. **`app = Starlette()` is instantiated twice (lines 185, 645) — this
   is dead/redundant code, *not* a live CORS/static-file break.**
   Root `CONTEXT.md` previously claimed the second instantiation "loses"
   the CORS middleware and `/static` mount from the first. **Verified
   against the actual file: this is incorrect.** The second
   instantiation block (lines 645–647) *also* calls
   `app.add_middleware(...)` and `app.mount("/static", ...)`
   immediately after re-creating `app` — so the running server does
   have working CORS and static files. The real issue is that the
   **first** `Starlette()` instance and its setup are pure dead code,
   discarded the moment `app` is rebound — wasted work and a trap for
   anyone who edits the first block expecting it to take effect.
   *(Corrected finding — update any prior notes/memory that cite the
   original claim.)*

7. **`load_backend()` is defined twice in `app/server.py` (lines 258 and
   589), identically.** Python's "last definition wins" means the
   258-line copy is unreachable dead code; the 589 copy is what actually
   runs. No functional bug (bodies are identical) but a stark
   maintenance hazard.

8. **The README's pesticide-recommendation feature description does not
   match the shipped code.** README states pesticide/dosage data comes
   from `docs/Pesticides_With_Agri_Guideline_Dosage.xlsx` and is
   "returned when model confidence ≥ 85%." Neither app reads that
   `.xlsx` file at all, and neither app gates its treatment tip on any
   confidence threshold — the tip comes unconditionally from the
   hardcoded `DISEASE_INFO` dict whenever the predicted label matches
   one of its 19 keys.

9. **`README.md` describes a repository layout that no longer exists.**
   It documents a `Plant_Disease_Detection/` subfolder with a Flask app
   and an AWS/GCP `deployment_guide/` — both removed by the most recent
   commit (`28aa666`, "Flatten Pest Detection directory structure and
   remove boilerplate"). The README was not updated to match. Treat
   `CONTEXT.md` (repo root) as the accurate current description.

10. **`app/server.py`'s `heuristic_classify` fallback always predicts a
    tomato-family class** based purely on brightness/hue/saturation
    buckets, regardless of the actual plant in the photo, whenever no
    model file exists. This activates silently — the UI has no
    "demo mode" / "no model loaded, showing illustrative results"
    banner, so a user could easily mistake fabricated output for a real
    diagnosis.

11. **Training data ≠ served classes, and no notebook here demonstrably
    produced the shipped model.** `notebooks/Pest_Detection_Training.ipynb`
    trains a 9-class **pest** classifier (custom Keras CNN, ~63.9% val
    accuracy); the running apps classify against a **38-class
    PlantVillage disease** taxonomy via a ResNet. The provenance of
    `export_resnet50_model.pth`/`.pkl` is undocumented — no notebook in
    this repo visibly trains a 39-class ResNet50 on PlantVillage data.

12. **Untrusted/incomplete error handling around model inference.**
    `/analyze` has no try/except around image decoding or the forward
    pass; a corrupt upload or an unexpected model state surfaces as an
    unhandled 500. The `torch-resnet50` branch in `classify_image` will
    raise `RuntimeError("No model is available...")` for an
    unrecognized `backend["kind"]`, which is unreachable given current
    `load_backend()` logic but is a latent trap if a new backend kind is
    ever added without updating this dispatcher too.

## 🟢 Low

13. **`app/static/logo.png` is 4MB** — unreasonably large for a web
    asset served on every page load.

14. **`app/view/` contains 3 unused screenshot files** (`1 (3).PNG`,
    `2 (3).PNG`, `4 (3).PNG`, ~1–1.5MB each) not referenced by any
    HTML/CSS/JS.

15. **Two separate, ambiguous notebook directories** (`notebook/` vs.
    `notebooks/`) with different, overlapping purposes and no README
    pointing between them.

16. **No linter/formatter/type-checker configuration** anywhere
    (no `pyproject.toml`, `ruff.toml`, `.flake8`, `mypy.ini`).

17. **No automated tests, no CI.** Zero `test_*.py`/`*_test.py` files
    found in the repository.

18. **`notebooks/Pest_Detection_Training.ipynb` hardcodes an absolute
    Windows path from a different machine/user**
    (`C:\Users\ephra\OneDrive\Desktop\...`), making the notebook
    non-portable without manual edits, and signaling the notebook's
    likely external origin.

19. **Google Fonts CDN dependency with no local fallback** in both UIs —
    an offline/air-gapped deployment (plausible for a rural-agriculture
    use case) silently loses its intended typography.
