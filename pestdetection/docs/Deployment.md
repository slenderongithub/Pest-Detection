# Deployment

## Current state: none automated

There is no `Dockerfile`, no `docker-compose.yml`, no
`.github/workflows/`, no `Procfile`, no `render.yaml`/`fly.toml`/etc.,
and no CI of any kind in the current (flattened) repository. The stale
`README.md` references a `deployment_guide/` (AWS/GCP) that existed in
the *prior*, pre-flatten repo layout — it does not exist today. Treat
"how do we deploy this" as an open question requiring a decision, not an
existing-but-undocumented pipeline.

## How to run today (manual, local only)

### Streamlit app
```bash
cd /Users/slender/Developer/Codes/pestdetection
streamlit run app.py
# → http://localhost:8501 (Streamlit default)
```

### Starlette server
```bash
cd /Users/slender/Developer/Codes/pestdetection
python app/server.py serve
# → http://localhost:8080
```
Note the `serve` argv gate (`app/server.py:680`) — running the file
without that argument does nothing (module import only, no server
start), which is an easy footgun for anyone running `python
app/server.py` expecting it to just work.

### Prerequisites (not automated)
- A Python 3.9+ environment (the repo assumes `.venv39`, not committed).
- `pip install -r requirements.txt` — **will likely fail or produce a
  broken environment** as pinned (see `docs/Tech-Stack.md`); expect to
  hand-resolve versions.
- A model file manually placed in `app/models/`, or patience while the
  legacy-ResNet34 Google Drive fallback downloads (Streamlit path only
  — Starlette will happily run "successfully" with zero models via the
  heuristic fallback, which is arguably worse since it looks like it's
  working).

### Generate paper figures
```bash
cd /Users/slender/Developer/Codes/pestdetection/scripts
python generate_figures.py
# Output → ../docs/paper_figures/ (gitignored)
```

## Environment variables

None are used or read anywhere in the codebase (`os.environ` does not
appear in `app.py` or `app/server.py`). All configuration (paths, URLs,
class lists, port `8080`, host `0.0.0.0`) is hardcoded as Python
literals. This means there is currently no way to point either app at a
different model directory, port, or CORS policy without editing source.

## Process/runtime notes

- Starlette binds `0.0.0.0:8080` unconditionally — fine for local
  development, but **listens on all interfaces by default**, which is
  worth being deliberate about if this is ever run on a shared or
  cloud-exposed machine without a reverse proxy in front of it.
- No process supervisor (systemd unit, pm2, supervisord config) is
  defined — either process, if used as a long-running service, would
  need one added externally.
- No health-check-driven readiness gate beyond the app-level `/health`
  route (which itself doesn't verify the model actually loaded
  successfully vs. fell back to `"missing"`/heuristic — see
  `docs/Known-Issues.md`).

## What a real deployment would need (see `docs/Upgrade-Ideas.md` for
ranked detail)

1. A pinned, installable `requirements.txt` (or migrate to `uv`/
   `poetry` with a lockfile).
2. A `Dockerfile` per app (or one multi-stage image if both are kept).
3. A model artifact strategy that isn't "download from a personal
   Google Drive link with no auth, no checksum, no version pin."
4. Basic CI (lint + a smoke test that both apps import and `/health`
   responds) before any deploy automation.
5. A decision on which of the two apps (Streamlit vs. Starlette) is the
   one going to production — running both indefinitely doubles the
   maintenance burden documented throughout these docs.
