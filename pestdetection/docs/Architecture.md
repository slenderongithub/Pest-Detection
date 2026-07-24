# Architecture

## High-level shape

This is **not one architecture** — it's two parallel monolithic scripts
that each embed the full stack (UI + inference + explainability +
presentation) in a single file, plus a separate, disconnected
figure-generation tool and a pile of notebooks. There is no shared
library, no package boundary, and no dependency injection: both `app.py`
and `app/server.py` independently define the same ~15 functions and
constants.

```
                     ┌───────────────────────────┐
                     │        User (browser)      │
                     └───────────┬───────────────┘
              ┌──────────────────┼──────────────────┐
              ▼                                      ▼
   ┌─────────────────────┐               ┌─────────────────────────┐
   │  Streamlit process   │               │  Uvicorn/Starlette      │
   │  app.py               │               │  process app/server.py │
   │  (server-rendered UI  │               │  (JSON API + static    │
   │   + inference inline) │               │   HTML/JS/CSS SPA)     │
   └──────────┬───────────┘               └───────────┬─────────────┘
              │  st.cache_resource                     │  module-level
              ▼  load_backend()                        ▼  load_backend()
   ┌─────────────────────────────────────────────────────────────────┐
   │        Model loading chain (duplicated, near-identical)          │
   │  .pth ResNet50 → .pkl ResNet50 (FastAI) → .pkl ResNet34 (legacy) │
   │  → [Starlette-only] color-heuristic fallback                     │
   └─────────────────────────────────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Inference: preprocess → forward pass → softmax → Grad-CAM        │
   │  → threshold → connected components → overlay compositing         │
   └─────────────────────────────────────────────────────────────────┘
```

Neither process talks to the other; neither shares a model cache;
running both simultaneously loads the model into memory twice.

## Frontend architecture

- **Streamlit surface**: no client-server split in the conventional
  sense — Streamlit re-executes the entire `app.py` script top-to-bottom
  on every interaction, with `@st.cache_resource` memoizing only the
  model-loading call. All "components" are just sequential
  `st.markdown(..., unsafe_allow_html=True)` calls injecting raw HTML/CSS
  strings — there is no component tree, no reusable widget abstraction.
- **Starlette surface**: a genuine client-server split. `index.html` is
  a static shell with fixed `id`-addressed elements; `client.js` is a
  single IIFE that wires up `fetch()` calls and DOM mutation by hand — no
  virtual DOM, no framework, no state management library. All UI state
  lives in one `state` object at the top of the IIFE.

## Backend architecture

- **Streamlit**: the "backend" is inline in the same process as the UI
  — there is no network boundary between "frontend" and "backend" at all.
- **Starlette**: a minimal 4-route ASGI app (`/`, `/health`, `/analyze`,
  `/analyze-frame`) with no auth, no rate limiting, no request logging
  beyond Uvicorn's default access log, and `CORSMiddleware` wide open
  (`allow_origins=["*"]`).

## API architecture

See `docs/API.md` for the full route reference. In short: one route
(`/analyze`) is aliased twice (`/analyze` and `/analyze-frame` point at
the identical `analyze()` handler) — there is no code difference between
a "single upload" and a "camera frame" request server-side; the
distinction is purely a client-side interval timer.

## Database architecture

There is none. See `docs/Database.md`.

## ML architecture

See `docs/Machine-Learning.md` for the full deep-dive. In one line: a
ResNet (50 preferred, 34 legacy) fine-tuned externally (provenance
undocumented), served via either a raw PyTorch `state_dict` or a FastAI
v1 `Learner` export, explained post-hoc via a hand-rolled Grad-CAM.

## Data flow

See `docs/Data-Flow.md` for the step-by-step trace.

## Authentication / authorization

**None exists anywhere in this repository.** Both apps are open,
anonymous, single-tenant tools. There is no session concept beyond
Streamlit's implicit per-browser-tab `session_state` and the
Starlette app's in-memory JS array. If this is ever deployed publicly,
this is the first gap to close (see `docs/Known-Issues.md`).

## Design patterns actually present

- **Fallback chain / chain-of-responsibility** (informally) in
  `load_backend()` — tries increasingly-degraded model sources.
- **Strategy-by-dict-key** in `classify_image()`/`analyze_image()` —
  branches on `backend["kind"]`/`backend["preprocess"]` rather than
  polymorphic backend classes.
- **No** repository pattern, no service layer, no DTOs/schemas (routes
  return raw dicts as JSON), no dependency injection framework.

## Scalability

- Single-process, single-model-instance, synchronous inference inline in
  the request handler (`async def analyze` awaits nothing CPU-bound —
  the actual `model(tensor)` forward pass blocks the event loop). Under
  concurrent load, Starlette/Uvicorn would serialize all inference
  through one blocked event loop; there is no worker pool, no batching,
  no queueing.
- Streamlit is inherently single-user-per-process by design (each
  browser session can spin a new Python interpreter thread but the
  cached model is process-wide).
- No horizontal scaling story, no containerization, no process manager
  config (`docs/Deployment.md`).

## Maintainability

The single biggest maintainability risk is **the ~90%-duplicated ML
core between `app.py` and `app/server.py`.** Any bug fix, class list
update, or Grad-CAM tweak must be manually mirrored across both files,
and they have already begun to drift (see the tint-alpha discrepancy in
`docs/Feature-Inventory.md` §4). Extracting a shared `leafscan_core`
module is the highest-leverage structural change available (see
`docs/Technical-Debt.md` and `docs/Upgrade-Ideas.md`).
