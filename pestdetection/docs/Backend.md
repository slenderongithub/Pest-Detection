# Backend

There are two backends, doing the same job with independent code.

## Streamlit backend (inline in `app.py`)

Not a network service — the "backend" is just the top-level function
calls in the same Python process as the UI. Key functions:

| Function | Line | Role |
|---|---|---|
| `load_backend()` | 264 | Model discovery/loading chain, `@st.cache_resource` |
| `build_resnet50()` | 253 | Constructs a ResNet50 with a swapped 39-class FC head |
| `clean_state_dict()` | 246 | Strips `module.` DataParallel prefix from checkpoint keys |
| `preprocess_pil()` | 311 | `Resize(256)→CenterCrop(224)→ToTensor→Normalize` |
| `compute_gradcam()` | 331 | Manual forward/backward-hook Grad-CAM |
| `component_boxes()` | 370 | Hand-rolled BFS connected components → boxes |
| `make_overlay()` | 419 | PIL compositing of heatmap tint + boxes + labels |
| `analyze_fastai()` / `analyze_torch()` | 451 / 482 | Backend-specific inference paths |
| `analyze_image()` | 511 | Dispatches to the above by `backend["preprocess"]` |

`@st.cache_resource(show_spinner=False)` on `load_backend()` means the
model is loaded once per Streamlit server process and reused across
reruns/sessions — this is correct and intentional caching.

## Starlette backend (`app/server.py`)

A genuine ASGI service, `uvicorn.run(app, host="0.0.0.0", port=8080)`
when invoked as `python app/server.py serve`.

### Routes

See `docs/API.md` for full request/response detail. Summary:

| Route | Method | Handler | Purpose |
|---|---|---|---|
| `/` | GET | `homepage` | Serves `app/view/index.html` verbatim |
| `/health` | GET | `health` | `{ok, model, kind}` status JSON |
| `/analyze` | POST | `analyze` | Single-image classification |
| `/analyze-frame` | POST | `analyze` | **Identical handler** — camera frames reuse `/analyze`'s logic |

### Middleware

- `CORSMiddleware(allow_origins=["*"], allow_headers=["*"])` — fully
  open CORS, no origin allowlist. Acceptable for a local dev demo, a
  liability if deployed publicly without tightening.
- `StaticFiles` mounted at `/static` serving `app/static/`.

### Duplicate initialization (verified, and *partially* miscorrected in
prior notes — see `docs/Known-Issues.md` #2 for the precise, verified
behavior)

`app = Starlette()` is instantiated **twice** — once at line 185 (with
middleware + static mount attached immediately after), and again at line
645 (again with middleware + static mount attached immediately after,
lines 646–647). The **first** `Starlette` instance, along with everything
built on it, is simply discarded garbage once `app` is rebound at line
645 — no routes were ever added to it. Because the second instantiation
independently re-applies the same middleware and static mount, **the
running app is not actually missing CORS or `/static` at runtime** — the
bug is redundant/dead initialization work and a maintenance trap (a
future edit to the first block's middleware config would silently do
nothing), not a live functional break.

### `load_backend()` defined twice (lines 258 and 589)

Byte-for-byte identical bodies. Python's normal "last definition wins"
rule means the call at line 642 (`backend = load_backend()`) invokes the
second (589) definition. The first (258) definition is unreachable dead
code. No functional bug (identical behavior either way), but a stark
maintenance hazard — anyone editing the first copy will observe no
effect and be confused.

### Inference functions

Same shape as the Streamlit file, prefixed without `preprocess_pil`
naming (`preprocess_image`, `target_layer`, `compute_gradcam`,
`connected_component_boxes`, `overlay_image`) — logically identical,
textually independent, already drifted slightly (tint color/alpha
curve — see `docs/Feature-Inventory.md` §4).

### Heuristic fallback (`heuristic_classify`, unique to this backend)

`app/server.py:439-506`. Pure NumPy color-space thresholds
(brightness/saturation/RGB ratios) bucket the image into
brown/yellow/white/dark lesion masks, and the **dominant mask always
maps to a tomato-family class** — e.g. any "white-dominant" image
becomes `Squash___Powdery_mildew`, any "yellow-dominant" image becomes
`Tomato___Tomato_Yellow_Leaf_Curl_Virus` — regardless of the actual
plant species in the photo. Only triggers when `backend["kind"] ==
"missing"` (no model files at all found).

### `download_file` / `download_legacy_model_if_needed` (async)

`app/server.py:219-226, 636-639`. Streams a URL to disk via `aiohttp` if
the destination doesn't already exist. Correct implementation; only
reachable from the `load_backend()` legacy-ResNet34 branch, and from the
separately-broken `download_model.py` script (see
`docs/Known-Issues.md` #3).
