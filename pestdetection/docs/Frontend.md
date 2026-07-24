# Frontend

Two entirely separate frontends exist. Neither imports/depends on the
other.

## 1. Streamlit UI (`app.py`)

- **Page config**: wide layout, collapsed sidebar, title "LeafScan -
  Plant Disease Detector", 🌿 favicon (`app.py:29-34`).
- **Styling**: a single triple-quoted `CUSTOM_STYLE` string (`app.py:538-
  749`, ~210 lines) injected via `st.markdown(..., unsafe_allow_html=
  True)`. It targets Streamlit's internal `data-testid` selectors
  (`stFileUploader`, `stMetric`, `stMetricLabel`, `stMetricValue`) to
  restyle native widgets — these are **undocumented, private DOM hooks**
  that can silently stop matching after a Streamlit version bump.
- **Layout**: hero banner → 4-column metric row (active model / class
  count / device / recent scans) → 2-column body (upload left, 1.02fr;
  diagnosis right, 1.18fr) → conditional "recent scans" section.
- **Custom "components"** are just parameterized `st.markdown` HTML
  strings — `severity_pill()`, `history_card()` (`app.py:517-535`) — not
  real Streamlit custom components.
- **State**: `st.session_state.history`, a list capped at 5, is the only
  persisted state; everything else is recomputed from the current
  `uploaded_file` on each script rerun.
- **No routing** — Streamlit apps are single-page by construction.

## 2. Starlette SPA (`app/view/index.html` + `app/static/`)

- **`index.html`** (146 lines): static shell, Google Fonts preconnect +
  stylesheet link, five `<section>` blocks (hero stats, camera+upload
  grid, results grid, predictions panel, history panel), all elements
  addressed by fixed `id`s that `client.js` looks up once at load time.
- **`client.js`** (286 lines, single IIFE, no imports/exports, no
  bundler):
  - `elements` — one-time `getElementById` lookup table.
  - `state` — `{stream, intervalId, busy, history}` — the entire client
    state model.
  - Rendering functions (`renderBoxes`, `renderTopPredictions`,
    `pushHistory`, `renderResult`) do direct `innerHTML` template-string
    assembly — **user-influenced data (model labels, box coordinates)
    is interpolated into `innerHTML` without escaping.** Since label
    text originates from the trusted server's own class list (not raw
    user input), this is low-risk today, but it is an XSS-shaped pattern
    that would become a real issue if labels ever became
    user-editable or model-name/tip strings became configurable.
  - Camera flow: `getUserMedia` → mirrored canvas draw (`scale(-1,1)`) →
    `toBlob` JPEG @ 0.9 quality → POST every 1.2s via `setInterval`.
  - A `busy` boolean is the only concurrency guard — no queue, no abort
    of in-flight requests, no adaptive interval if the server is slow.
- **`style.css`** (511 lines): CSS custom properties in `:root`
  (`--bg-0/1/2`, `--panel`, `--border`, `--accent`, `--warn`,
  `--danger`), two responsive breakpoints (1100px, 720px). This is the
  more disciplined of the two stylesheets (real custom properties vs.
  the Streamlit file's repeated literal color values).

## Design system reality check

The two stylesheets encode **the same design intent** (dark
glassmorphism, identical accent green `#45f0a3`/blue `#7dd3fc`) but are
**not shared** — Streamlit's `CUSTOM_STYLE` hardcodes colors as literals
throughout; `style.css` uses CSS variables. A palette change today
requires editing two files with two different techniques.

## Assets

- `app/static/logo.png` — **4MB**, unreasonably large for a web logo
  (see `docs/Known-Issues.md`).
- `app/static/leaf.png` — small favicon, fine.
- `app/view/1 (3).PNG`, `2 (3).PNG`, `4 (3).PNG` — ~1–1.5MB each,
  auto-generated-looking filenames (dev screenshots), **not referenced
  by any HTML/CSS/JS** — dead weight in the repo.

## Accessibility (see `docs/UI-UX-Review.md` for full audit)

- No `alt` text strategy beyond static placeholders (`Selected leaf
  preview`, `Localized disease overlay`) — fine for those, but no ARIA
  live region announces async result updates to screen readers in
  either app.
- Color-only severity signaling (text color + border color) with no
  icon/shape redundancy beyond a plain `!`/`~`/`+`/`OK` glyph in the
  Streamlit pill — borderline for colorblind users but not absent.
- No visible focus states are customized — relies on browser defaults,
  which the dark background may wash out for `:focus` outlines on some
  buttons.
