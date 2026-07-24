# Developer Guide

## Getting oriented

Read in this order for a new-engineer ramp-up:
1. `CONTEXT.md` (repo root) — accurate, current, concise orientation.
2. `docs/Project-Overview.md` — what's actually in this repo.
3. `docs/Architecture.md` + `docs/Data-Flow.md` — how a request moves.
4. `docs/Machine-Learning.md` — the ML system, its real limitations, and
   the upgrade roadmap (this is the flagship area per project priority).
5. `docs/Known-Issues.md` + `docs/Technical-Debt.md` — what to be
   careful of before touching anything.

**Do not trust `README.md` for structure** — it describes a pre-flatten
layout that no longer exists (see `docs/Known-Issues.md` #9).

## Local setup

```bash
cd /Users/slender/Developer/Codes/pestdetection
python3.9 -m venv .venv39            # or reuse the existing .venv39
source .venv39/bin/activate
pip install -r requirements.txt      # WARNING: likely to fail/conflict,
                                      # see docs/Tech-Stack.md — expect to
                                      # hand-resolve torch/numpy/streamlit
                                      # versions rather than installing
                                      # this file verbatim
```

## Running the two apps

```bash
# Streamlit ("LeafScan")
streamlit run app.py

# Starlette API + SPA ("Plant Disease Detector")
python app/server.py serve
```

Both expect a model file under `app/models/` for real predictions
(`export_resnet50_model.pth` preferred). Without one: Streamlit will
attempt a legacy-model download and may crash if that fails; Starlette
will silently serve heuristic (non-ML) fake predictions — see
`docs/Known-Issues.md` #1 and #10 before assuming either app "works" out
of the box.

## Conventions observed in the existing code

- **Naming**: snake_case (Python), camelCase (JS). CSS uses BEM-ish class
  names (`panel-head`, `scan-ring-alt`, `severity-critical`).
- **Design tokens**: dark glassmorphism; accent green `#45f0a3`, accent
  blue `#7dd3fc`, warn `#ffb86b`, danger `#ff5d5d`/`#ff6b6b`. Keep new UI
  work consistent with these unless deliberately redesigning (see
  `docs/UI-UX-Review.md`).
- **Model weights are always gitignored** — never commit `.pkl`, `.pth`,
  or `.h5` files.
- **`torch.load` is monkey-patched** in both app entry points to default
  `weights_only=False`, needed for the old pickle-based checkpoints this
  project loads. Be aware this re-enables arbitrary-code-execution risk
  on untrusted checkpoint files (see `docs/Machine-Learning.md` §Security).
- **Grad-CAM and connected components are hand-rolled**, intentionally,
  with no third-party CAM or `scipy.ndimage.label` dependency. If you
  add a dependency to replace these, note it explicitly — it's a
  deliberate existing choice, not an oversight, though not necessarily
  one worth preserving forever.

## Before making any change

1. **If touching ML logic (Grad-CAM, class list, severity, disease
   info, model loading)**: the change almost certainly needs to be
   applied in **both** `app.py` and `app/server.py` today, since there is
   no shared module (`docs/Technical-Debt.md` #1). Strongly consider
   doing the extraction-to-shared-module refactor *first*, as a
   standalone, behavior-preserving change, before adding new logic on
   top of the duplication.
2. **If touching the model-loading chain**: check both `load_backend()`
   definitions in `app/server.py` (lines 258 and 589) — only the second
   is actually live; don't be fooled into editing the dead first copy.
3. **If touching CORS/static file serving in `app/server.py`**: remember
   there are two `Starlette()` instantiations (185, 645); only the
   second is live. See `docs/Known-Issues.md` #6 for the verified,
   corrected explanation of what this bug actually does (dead code, not
   a live break).
4. **If touching `requirements.txt`**: treat it as broken already: don't
   assume "it works because it's pinned" — verify by actually installing
   into a clean environment before relying on any version listed there.
5. **No test suite exists.** Any refactor (especially the duplication
   extraction in item 1) should be manually smoke-tested against both
   apps' `/health` (Starlette) and a live upload (both) before and after,
   since there's no automated safety net yet. Consider writing the first
   tests as part of the same change, since none exist to build on.

## Generating research figures

```bash
cd scripts
python generate_figures.py
# → ../docs/paper_figures/ (gitignored, regenerate on demand)
```
Be aware Fig 5's data is a hardcoded literal list (not read from a real
training log) and Fig 7 is an entirely synthetic mock-up, not a real
captured inference — see `docs/Known-Issues.md` #5 before using either
in anything citing real system performance.
