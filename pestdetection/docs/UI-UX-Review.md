# UI/UX Review

Reviewed as a product audit of both live surfaces, without changing any
code.

## Visual design (both apps)

- **Consistent, above-average dark glassmorphism aesthetic**: translucent
  panels, soft shadows, a coherent green/blue accent pair, and real
  typographic choices (Inter for body, Space Grotesk for display
  numerals/headings). This is the project's strongest UX asset.
- **Inconsistency between the two surfaces**: subtly different border
  radii (Streamlit's `.glass-card` uses 24px vs. Starlette's `.panel`
  uses 28px), different shadow depths, and independently-tuned color
  literals vs. CSS variables — a user bouncing between the two apps
  would notice they're "almost but not quite" the same product.
- **4MB logo** (`app/static/logo.png`) likely causes a visible pop-in on
  first load over a slow connection — undermines the otherwise
  performance-conscious design.

## Layout

- Streamlit: clean 2-column layout (upload | diagnosis) that reflows
  reasonably at Streamlit's own responsive breakpoints (framework-
  managed, not custom).
- Starlette: custom CSS grid (`dashboard-grid`, `results-grid`) with
  two explicit breakpoints (1100px, 720px) collapsing to single-column —
  a more deliberately engineered responsive design than the Streamlit
  surface, which mostly inherits Streamlit's own column behavior.

## Typography

- Both use Inter (body) + Space Grotesk (display) at consistent weights
  — good discipline. Confidence numbers use `clamp()` for fluid sizing
  in both apps, a nice touch for the "hero number" treatment.

## Navigation

- Both are single-screen apps with no navigation at all — appropriate
  given the scope, but also means there's no way to reach, e.g., a
  "how this works" explainer, a model-version indicator beyond the name
  string, or a settings page, without editing source.

## Loading / empty / error states

- **Streamlit**: has an explicit, well-designed **empty state** ("Drop a
  leaf image to begin" / "No image uploaded yet") and a spinner
  ("Analyzing leaf texture and lesion patterns...") during inference.
  **No explicit error state** — if `load_backend()` raises (e.g. the
  legacy-model download fails), the entire app crashes with Streamlit's
  default traceback screen, which is jarring and unbranded.
- **Starlette**: has empty states for boxes ("No strong lesion region
  was detected") and a busy/mode indicator ("Scanning" → "Detected"/
  "Healthy"), but **no dedicated visual loading state on the overlay
  image itself** — between camera ticks the previous frame's overlay
  just sits there until the next result arrives, which could read as
  "stuck" if inference is slow. On error, `setStatus("Analysis failed",
  "severity-critical")` is a reasonable, low-drama failure state — better
  handled than the Streamlit crash-to-traceback path.

## Microinteractions / animation

- The Starlette camera view's "breathing" scan-ring animation
  (`@keyframes breathe`, two concentric rings offset in phase) is a
  genuinely nice touch that communicates "actively scanning" without
  needing text.
- Streamlit has essentially no motion design (consistent with
  Streamlit's rendering model — full script rerun per interaction makes
  CSS transitions less natural to rely on).
- Button hover states (`translateY(-1px)`) exist in the Starlette CSS;
  Streamlit's native buttons aren't restyled with equivalent feedback
  (the custom CSS targets file-uploader/metric widgets, not buttons,
  since the Streamlit app doesn't actually use `st.button` — everything
  is driven by the file uploader's `on_change` implicitly via rerun).

## Visual hierarchy

- Both apps correctly lead with the confidence number and disease name
  as the dominant visual element — good instinct for a diagnostic tool
  where the headline result should be unmissable.
- Severity is color + text + (Streamlit only) a small glyph
  (`!`/`~`/`+`/`OK`) — reasonably redundant signaling, though the glyphs
  are terse enough to be ambiguous on first encounter without a legend.

## Accessibility gaps (see also `docs/Frontend.md`)

- No ARIA live regions for async result updates in either app — a
  screen-reader user gets no announcement when a new diagnosis renders.
- Severity signaling leans on color + a small glyph but not on shape/
  icon variety — acceptable but not best-practice for color-vision
  deficiency.
- No visible custom focus-ring treatment for keyboard navigation on
  either app's interactive elements (upload dropzone, buttons) — relies
  entirely on browser/User-Agent defaults, which can be low-contrast
  against the dark backgrounds used here.

## Modernization opportunities (no redesign performed — ideas only)

- Merge the two surfaces' design tokens into one shared source (see
  `docs/Technical-Debt.md` #2) so future visual changes apply once.
- Add a persistent "why this result" affordance (expandable Grad-CAM
  explanation) rather than a bare overlay image, to build user trust in
  an uncalibrated-confidence system (see `docs/Machine-Learning.md`).
- Add an explicit "demo/no-model" banner state so the Starlette
  heuristic fallback (`docs/Known-Issues.md` #10) never masquerades as a
  real diagnosis.
- Consider a true mobile-first camera flow (native `capture="environment"`
  input attribute, or a PWA wrapper) — the target user (field-based
  grower) is more likely on a phone than a desktop webcam.
