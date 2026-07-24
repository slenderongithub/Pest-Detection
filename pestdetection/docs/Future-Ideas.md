# Future Ideas (product / UX / non-ML-core)

Brainstorm only — nothing here has been implemented. ML-specific ideas
live in `docs/Machine-Learning.md`; this file covers product, UX, and
platform ideas. See `docs/Upgrade-Ideas.md` for the consolidated,
ranked master roadmap combining both.

## Product

- **Native mobile app / PWA** — the inferred target user (a field-based
  grower) is more likely to have a phone than a desktop with a webcam;
  a PWA with offline-capable on-device inference (see edge deployment in
  `docs/Machine-Learning.md`) would fit the actual use case far better
  than the current webserver-only surfaces.
- **Multi-image / batch mode** — scan an entire row of plants in one
  session and get an aggregate health report instead of one photo at a
  time.
- **Persistent scan history with export** — replace the ephemeral
  5-item in-memory list with real storage (see `docs/Database.md`) and a
  CSV/PDF export for record-keeping (useful for farm compliance/
  insurance documentation).
- **Multi-language support** — treatment tips and UI copy are
  English-only; agriculture is a global use case and localization would
  materially widen the addressable audience.
- **Real pesticide database integration** — actually wire up
  `docs/Pesticides_With_Agri_Guideline_Dosage.xlsx` (or a proper
  successor) instead of the current hardcoded 19-entry dict, and
  restore (or explicitly retire) the confidence-gated recommendation
  logic the README describes but the code doesn't implement.
- **Regional pest/disease relevance filtering** — surface only
  diseases plausible for the user's stated crop/region, reducing
  confusion from a flat 38-class taxonomy that spans many unrelated
  crops.

## UX

- **Onboarding / "how this works" explainer** — neither app currently
  explains what Grad-CAM overlays mean or how confident the model
  actually is; a short first-run explainer would build trust.
- **Comparison view** — before/after or side-by-side of consecutive
  scans of the same plant to visualize disease progression over time
  (requires the persistence work above).
- **Unified single app** — pick one of the two current surfaces (or
  build a true successor) rather than maintaining both indefinitely; see
  `docs/Upgrade-Ideas.md` for the tradeoffs.
- **Accessibility pass** — ARIA live regions for async results, visible
  focus states, and icon-redundant severity signaling (see
  `docs/UI-UX-Review.md`).

## Developer experience / automation

- **CI pipeline**: lint + import-smoke-test + `/health` check on every
  push, blocking merges that break either app from starting.
- **Pre-commit hooks**: formatting (black/ruff) and basic secret
  scanning, given there's currently no linting at all.
- **One-command dev environment** (`make dev`, or a devcontainer) so a
  new contributor doesn't have to hand-resolve the broken
  `requirements.txt` themselves.
- **Model registry / versioning** (even a simple manifest file listing
  checkpoint hash, training data version, and reported metrics) to
  finally answer "which model is actually running, and how good is it."

## Analytics & monitoring

- **Basic usage telemetry** (opt-in): scan volume, top predicted
  classes, confidence distribution over time — would surface model
  drift and popular-but-unsupported plant types.
- **Error/exception monitoring** (e.g. Sentry) — right now an
  unhandled 500 or a Streamlit crash is invisible to anyone but the user
  experiencing it.
- **Uptime/health monitoring** for the Starlette `/health` endpoint if
  this is ever deployed as a long-running service.

## Security & production-readiness

- **Add authentication** if this ever serves real user data or is
  publicly deployed — currently fully anonymous/open.
- **Tighten CORS** from `allow_origins=["*"]` to an explicit allowlist
  before any public deployment.
- **Request size/rate limiting** on `/analyze` and `/analyze-frame` to
  prevent trivial resource-exhaustion abuse.
- **Input validation** (file type/size checks before decode) rather than
  relying on PIL to reject bad input via exception.
