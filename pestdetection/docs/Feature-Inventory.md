# Feature Inventory

For each feature: purpose, implementation, files, completion status,
improvement/bug notes, priority for the next upgrade session.

---

### 1. Image upload → classification (both apps)
- **Purpose**: core product loop — identify disease/pest from a photo.
- **Implementation**: PIL decode → `Resize(256)→CenterCrop(224)→ToTensor→
  ImageNet-normalize` → model forward pass → softmax → argmax.
- **Files**: `app.py:311-320,482-508`; `app/server.py:305-314,551-558`.
- **Status**: Complete, but only functions correctly if a model file is
  present; otherwise Streamlit crashes and Starlette silently returns a
  fake tomato-family prediction.
- **Bugs/improvements**: unify into one shared module; surface a clear
  "no model loaded" UI state instead of crashing/faking.
- **Priority**: High (correctness/trust issue).

### 2. Grad-CAM lesion heatmap
- **Purpose**: explainability — show *where* the model is looking.
- **Implementation**: manual forward/backward hooks on `layer4[-1]`,
  gradient-weighted channel sum, ReLU, bilinear upsample to 224×224,
  min-max normalized to [0,1].
- **Files**: `app.py:331-367`; `app/server.py:325-357` (identical logic).
- **Status**: Complete and functionally correct for ResNet-family models
  (`layer4`/`features` duck-typing). Will silently break for any
  non-ResNet, non-`features`-attribute architecture.
- **Priority**: Medium (works today, fragile to model swaps).

### 3. Bounding-box lesion localization
- **Purpose**: turn a diffuse heatmap into discrete "infected area" boxes.
- **Implementation**: threshold CAM at `max(0.55, 82nd percentile)` →
  boolean mask → hand-rolled BFS connected components → min-pixel filter
  → padded bounding box per component; falls back to one box over the
  whole masked region if no component clears the pixel threshold.
- **Files**: `app.py:370-416`; `app/server.py:360-406`.
- **Status**: Complete. No IoU/NMS merging of overlapping boxes; O(n)
  BFS is fine at 224×224 but would not scale to higher resolutions
  without optimization.
- **Priority**: Low (works; revisit only if resolution increases).

### 4. Overlay image compositing
- **Purpose**: visual, single-image deliverable combining heatmap tint +
  boxes + labels.
- **Implementation**: alpha-blended color tint from the CAM, rounded-
  rectangle boxes drawn with PIL, small label chips ("Infected area N").
- **Files**: `app.py:419-443`; `app/server.py:409-429`.
- **Status**: Complete; cosmetic-only differences between the two apps
  (tint RGB/alpha curve differs slightly — `255,86,48`/pow 1.5 vs
  `255,82,48`/pow 1.4 — an example of duplication drifting out of sync).
- **Priority**: Low.

### 5. Severity scoring + treatment tip
- **Purpose**: translate a raw class label into an actionable severity +
  human-readable guidance.
- **Implementation**: `get_severity()` + `get_disease_info()` substring
  match against `DISEASE_INFO` (19 entries); confidence-band fallback for
  unmapped classes.
- **Files**: `app.py:213-231`; `app/server.py:194-212` (identical).
- **Status**: Partially complete — only 19 of 38 non-healthy classes have
  a real entry; the rest get a generic fallback with no cause listed.
- **Priority**: Medium (directly affects perceived product quality).

### 6. Live camera scanning (Starlette app only)
- **Purpose**: continuous, near-real-time field scanning instead of a
  single upload.
- **Implementation**: `getUserMedia` → draw video frame to canvas
  (mirrored horizontally to undo the front-camera flip) → `toBlob` JPEG
  → POST to `/analyze-frame` every 1.2s via `setInterval`; a `busy` flag
  drops overlapping requests.
- **Files**: `app/static/client.js:168-219,131-166`.
- **Status**: Complete, no Streamlit equivalent. No back-pressure beyond
  the `busy` flag — if inference takes >1.2s consistently, frames queue
  up in perception (each tick just no-ops while busy) rather than
  adapting the interval.
- **Priority**: Medium (works, but no visible "processing" affordance
  between frames — see `docs/UI-UX-Review.md`).

### 7. Scan history (last 5)
- **Purpose**: let a user glance back at recent results in-session.
- **Implementation**: Streamlit — list in `st.session_state.history`,
  truncated to 5, rendered as styled cards. Starlette — JS array in
  `client.js` `state.history`, same 5-item cap, rendered client-side.
- **Files**: `app.py:915-944`; `app/static/client.js:87-108`.
- **Status**: Complete but fully ephemeral (lost on reload/restart); two
  independent implementations of the same feature.
- **Priority**: Low.

### 8. Model auto-provisioning / fallback chain
- **Purpose**: let the app boot even without a manually placed model
  file.
- **Implementation**: priority order `.pth` (torch ResNet50) →
  `.pkl` (FastAI ResNet50 learner) → `.pkl` (FastAI ResNet34 legacy,
  auto-downloaded from a public Google Drive link) → (Starlette only)
  color heuristic.
- **Files**: `app.py:263-308`; `app/server.py:258-302,589-633` (defined
  **twice**, see `docs/Known-Issues.md`).
- **Status**: Functionally works for the Starlette app (always has the
  heuristic floor). **Broken** for the Streamlit app if no model exists
  anywhere — it falls all the way to the FastAI ResNet34 legacy download,
  and if that URL is rate-limited/expired, the whole app fails hard with
  no user-facing error state.
- **Priority**: High.

### 9. `download_model.py` standalone downloader
- **Purpose**: pre-fetch the legacy model outside of app startup.
- **Implementation**: imports `download_file` and `export_file_url` from
  `app.server`.
- **Files**: `download_model.py`.
- **Status**: **Broken.** `export_file_url` does not exist in
  `app/server.py` (the actual name is `LEGACY_MODEL_URL`). Raises
  `ImportError` immediately on run.
- **Priority**: High if this script is meant to be part of any
  onboarding/deploy flow; Low if it's dead/unused tooling — confirm
  intent with the user before fixing or removing.

### 10. Pesticide/dosage reference spreadsheet
- **Purpose**: per README, "match detected pest against
  `Pesticides_With_Agri_Guideline_Dosage.xlsx`... only when confidence
  ≥ 85%."
- **Implementation**: the `.xlsx` file exists in `docs/` but **no code
  in `app.py` or `app/server.py` reads it**. The actual treatment tip
  comes entirely from the hardcoded `DISEASE_INFO` dict, with no
  confidence gate at all.
- **Files**: `docs/Pesticides_With_Agri_Guideline_Dosage.xlsx` (unused by
  runtime code); described in `README.md`.
- **Status**: Documented-but-not-implemented / stale — either the
  feature was replaced by `DISEASE_INFO` and the README wasn't updated,
  or the xlsx integration was never finished.
- **Priority**: Medium — decide and document which one is canonical.

### 11. Research figure generation (`scripts/`)
- **Purpose**: produce the 7 paper/report figures.
- **Implementation**: matplotlib, `Agg` backend, writes to
  `docs/paper_figures/` (gitignored — regenerate on demand).
- **Files**: `scripts/generate_figures.py`, `scripts/generate_fig7.py`.
- **Status**: Complete but **Fig 5 (accuracy/loss curves) uses hardcoded
  literal number lists**, not data read from an actual training log, and
  **Fig 7 ("sample output") is an entirely synthetic mock-up** — a
  matplotlib drawing of a fake camera feed with hand-typed prediction
  text ("aphids | Imidacloprid 17.8% SL... Confidence: 94.2%"), not a
  real captured inference. Anyone citing Fig 7 as a real system output
  would be citing fabricated evidence.
- **Priority**: High if this repo feeds an actual academic submission —
  this is a reproducibility/integrity issue, not just a code smell.

### 12. Training notebooks (9 architectures × 2 datasets)
- **Purpose**: model architecture exploration.
- **Implementation**: independent notebooks per architecture
  (custom CNN, PyTorch ResNet, FastAI, DenseNet121, VGG16/19, Keras,
  TensorFlow generic).
- **Files**: `notebooks/Pest_Detection_Training.ipynb`,
  `notebook/*.ipynb` (9 files).
- **Status**: Exploratory/non-productionized. The one with concrete,
  reported results (`Pest_Detection_Training.ipynb`) trains a 9-class
  **pest** classifier (not the 38-class disease taxonomy the apps
  serve), hits **~63.9% validation accuracy after 15 epochs**, uses a
  hardcoded absolute path from another machine, and its final cells zip
  the entire dataset+model into local archives rather than producing a
  deployable artifact.
- **Priority**: Medium — no single notebook currently produces the
  actual `export_resnet50_model.pth` the apps expect; that gap should be
  closed before claiming the ResNet50 path is "the model."
