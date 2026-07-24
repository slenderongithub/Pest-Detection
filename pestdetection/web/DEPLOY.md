# Deploy LeafScan on Netlify (fully static, no backend)

The model runs **in the browser** via onnxruntime-web (WASM). There is no server, no
Python, no cold start, no cost. Netlify just hosts static files.

## What's in `web/`

```
web/
├── index.html          # the SPA (same UI as the Starlette app)
├── style.css           # design system
├── app.js              # in-browser engine: preprocess → ONNX → calibrate → abstain → CAM overlay
├── favicon.svg
└── model/
    ├── model.int8.onnx # quantized ResNet50 (~24 MB), served to the browser
    ├── fc.bin          # fc weights (float32) for the gradient-free CAM
    └── meta.json       # classes, normalization, temperature, abstain threshold, knowledge
```

## 1. Generate the model (once, or whenever you retrain)

The `.onnx`/`.bin` are build artifacts (gitignored) — produce them from the checkpoint:

```bash
pip install -e '.[train]'          # needs torch + onnx + onnxruntime (already deps)
python scripts/export_onnx.py --quantize
# → writes web/model/{model.int8.onnx, fc.bin, meta.json}, verifies ONNX==PyTorch parity,
#   and checks int8 top-1 agrees with fp32 (keeps fp32 automatically if it doesn't).
```
Lighter/faster download: `--model-dir models/pest_resnet18`. Skip `--quantize` for a
lossless (but ~94 MB) fp32 model.

## 2. Deploy

**Option A — CLI / drag-and-drop (recommended; model stays out of git):**
```bash
npx netlify-cli deploy --prod --dir=web
```
or drag the `web/` folder onto https://app.netlify.com/drop.

**Option B — Git-based:** un-ignore `web/model/model.int8.onnx` (see `.gitignore`), commit
`web/`, and connect the repo. `netlify.toml` already sets `publish = "web"` and no build
command.

## Notes

- **First load downloads ~24 MB** (the model), cached immutably after that. Inference is a
  few seconds in WASM on a laptop; the live-camera loop deliberately skips the CAM overlay
  for latency (hit **Capture** for a focus overlay).
- **Grad-CAM → CAM:** the browser has no autograd, so the focus map is classic CAM
  (fc-weighted layer4 activations) — equivalent to Grad-CAM for this ResNet.
- **No model / load failure** → the UI shows an honest demo state, never a fake prediction.
- onnxruntime-web + its WASM load from the jsdelivr CDN; single-threaded, so no COOP/COEP
  headers are needed.
