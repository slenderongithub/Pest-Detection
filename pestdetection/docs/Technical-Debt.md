# Technical Debt

Distinct from `Known-Issues.md` (concrete bugs/risks): this file is
about structural cost that slows future development even when nothing
is actively broken.

## 1. Duplicated ML core (highest-leverage debt)

`app.py` and `app/server.py` each carry a full, independent copy of:
class taxonomy, disease knowledge dict, model-building, checkpoint
loading, preprocessing, Grad-CAM, connected components, overlay
compositing, and severity logic (~400+ lines duplicated per file). This
is the single largest source of both current bugs (the drifted overlay
tint values, see `Known-Issues.md` #3) and future maintenance cost.
**Fix**: extract a `leafscan_core` (or similarly named) package with one
copy of each function; have both `app.py` and `app/server.py` import
from it. This is a pure refactor with no behavior change and should be
the first thing tackled in an upgrade session, before adding any new ML
capability, so new features aren't written twice as well.

## 2. Two parallel front-end stacks with no shared design tokens

The dark-glassmorphism visual language is reimplemented twice: once as
literal hex values inside a Python triple-quoted string
(`app.py`'s `CUSTOM_STYLE`), once as CSS custom properties
(`app/static/style.css`). A palette or spacing change requires editing
both, by hand, with two different syntaxes. No design-token file
(JSON/YAML or shared CSS) exists to be the single source of truth.

## 3. `requirements.txt` is decorative, not functional

Pins are 5–7 years old and mutually incompatible with the
`streamlit>=1.32.0` requirement in the same file (see
`docs/Tech-Stack.md`). Nobody can `pip install -r requirements.txt` into
a clean environment and get a working app. This blocks any CI or
reproducible-build effort until resolved.

## 4. Two notebook directories, nine explored architectures, zero
   canonical training pipeline

`notebook/` (9 architecture-exploration notebooks: PyTorch, DenseNet121,
FastAI, Keras, TensorFlow, ResNet50, VGG16, VGG19, and a generic
"plant_disease_detector") plus `notebooks/` (the one notebook with
reported metrics, training a *different* 9-class pest dataset). None of
these notebooks is wired into a repeatable "run this to produce
`export_resnet50_model.pth`" pipeline — there's no `train.py`, no config
file, no experiment tracking (MLflow/W&B/etc.), and the one notebook
with concrete results has a hardcoded path from a different machine.
Any future retraining effort starts from near-zero reproducibility.

## 5. No shared "constants" module

`CLASS_NAMES`, `DISEASE_INFO`, `SEVERITY_CONFIG` all exist as
copy-pasted literals. Beyond the duplication-across-files problem (item
1), even *within* a single file these are large inline literals mixed
into application code rather than isolated as data — makes the main
logic harder to scan and the data harder to unit test independently.

## 6. Orphaned data asset

`docs/Pesticides_With_Agri_Guideline_Dosage.xlsx` is committed, described
in the README as load-bearing, and never read by any code. Either wire
it up or remove the claim — right now it's debt in the form of
misleading documentation plus an unused artifact.

## 7. No tests at any layer

Zero unit tests for the pure-function pieces that would be trivial to
test in isolation (`pretty_label`, `get_severity`, `get_disease_info`,
`connected_component_boxes`, `clean_state_dict`) and zero integration
tests for the two HTTP routes. This makes the refactor in item 1
riskier than it needs to be — there's no safety net to confirm behavior
is preserved.

## 8. No linting/formatting/type-checking config

No `ruff`/`black`/`flake8`/`mypy` configuration exists, despite the code
using modern type hints (`from __future__ import annotations`,
`dict[str, Any]`, `list[tuple[int,int,int,int]]`) inconsistently between
files (e.g. `app/server.py`'s `connected_component_boxes` doesn't
annotate its return type where `app.py`'s `component_boxes` does).

## 9. Hardcoded configuration throughout

Host (`0.0.0.0`), port (`8080`), model directory paths, and the legacy
model URL are all Python literals with no environment-variable or config
file override. This is fine for a single-developer demo but becomes
friction the moment more than one deployment target exists.

## 10. Dev-only debris in the repo

3 unreferenced screenshot PNGs (~4MB total) in `app/view/`, a 4MB logo
that could trivially be optimized to <200KB, and a `.venv39/` directory
that (per `.gitignore`) shouldn't be tracked but whose presence in the
working tree adds noise to any full-repo scan.

## Debt NOT found (worth noting as a positive)

- No evidence of committed secrets, API keys, or credentials anywhere
  in the scanned source.
- No obviously vulnerable dependency usage (e.g. no `eval`, no
  unsanitized `subprocess` shell calls, no SQL — because there's no SQL
  at all) beyond the wide-open CORS and unauthenticated endpoints
  already called out in `docs/Known-Issues.md`.
