# Database

**There is no database in this project.** This file exists to document
that fact explicitly (per the discovery-session checklist) rather than
to describe a schema.

## What stands in for persistence today

| Need | What's actually used | Durability |
|---|---|---|
| Class taxonomy | Hardcoded Python list `CLASS_NAMES`/`CLASSES` (39 entries), duplicated in both apps | Source-controlled, but two copies to keep in sync |
| Disease knowledge (cause/severity/tip) | Hardcoded Python dict `DISEASE_INFO` (19 entries), duplicated in both apps | Same as above |
| Pesticide dosage reference | `docs/Pesticides_With_Agri_Guideline_Dosage.xlsx` | **Written but never read** by any running code — orphaned data asset |
| Scan history | Streamlit: `st.session_state.history` (list, cap 5); Starlette: JS `state.history` (array, cap 5) | In-memory only; gone on process restart / page reload |
| Model weights | Local files under `app/models/` (gitignored) or a Google Drive URL | Not versioned, not checksummed, no model registry |
| Training images | `data/Pest_Dataset/` (gitignored, notebook-only) | Local filesystem only, no dataset versioning (DVC, etc.) |

## Implications for an upgrade session

If this project is meant to grow into a real product, "add a database"
is a legitimate near-term recommendation, specifically for:
1. **Disease/pesticide knowledge base** — replace the hardcoded dict +
   orphaned xlsx with one queryable source of truth (even SQLite would
   remove the current two-sources-of-truth problem and make the xlsx
   data actually usable).
2. **Scan history** — persisting predictions (with image hash, model
   version, timestamp) enables analytics, model-drift monitoring, and a
   real "history" feature that survives a reload.
3. **Model registry** — tracking which checkpoint is deployed, its
   training data version, and its evaluation metrics, instead of "drop a
   `.pth` file in a folder and hope."

See `docs/Upgrade-Ideas.md` for a ranked treatment of this.
