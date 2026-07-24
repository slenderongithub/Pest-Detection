# Data Flow

## Inference request, step by step (identical logical flow in both apps)

```
1. User action
   Streamlit: st.file_uploader → uploaded_file (bytes in memory)
   Starlette: <input type=file> or camera canvas.toBlob → FormData POST

2. Image decode
   PIL.Image.open(...).convert("RGB")

3. Preprocessing (torchvision.transforms.Compose)
   Resize(256) → CenterCrop(224) → ToTensor() →
   Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
   → tensor shape (1, 3, 224, 224)

4. Forward pass
   torch-* backend:  logits = model(tensor); probs = softmax(logits)
   fastai-* backend: pred_class, _, probs = learner.predict(fastai_image)
                     (FastAI does its own internal preprocessing —
                      NOTE: the torch preprocessing above is computed
                      *again*, separately, purely to feed Grad-CAM,
                      meaning fastai paths preprocess the image twice
                      with two different code paths)

5. Top-1 class + confidence
   pred_index = argmax(probs); confidence = probs[pred_index] * 100

6. Grad-CAM (only for torch-executable models; skipped in the
   heuristic-fallback path, which instead derives a mask directly from
   HSV-ish thresholds)
   forward/backward hooks on layer4[-1] → weighted channel sum → ReLU →
   bilinear resize to 224x224 → min-max normalize to [0,1]

7. Thresholding + localization (skipped if predicted "healthy" or
   confidence < 55%)
   mask = cam >= max(0.55, 82nd-percentile(cam))
   boxes = connected_component_boxes(mask)   # BFS, min-pixel filtered

8. Severity + treatment lookup
   severity = "Healthy" if "healthy" in label
              else DISEASE_INFO[matched_key]["severity"]
              else confidence-band fallback
   disease_info = DISEASE_INFO.get(matched_key)   # None for ~19/38 classes

9. Overlay compositing
   RGBA base + alpha-tinted heatmap layer + rounded-rect boxes + labels
   → flattened to RGB

10. Response
    Streamlit: bundle rendered directly into the page (no serialization)
    Starlette: JSON with overlay re-encoded as a base64 JPEG data: URL
               (image_to_data_url) — the ENTIRE overlay image is
               inlined into the JSON payload on every single request,
               including every 1.2s camera tick.

11. Client-side history append
    Both apps prepend the new result to a length-5 list/array.
```

## Where the two apps' data flow actually diverges

- **Streamlit** returns a `PredictionBundle` dataclass in-process — no
  serialization boundary, so the overlay is a live PIL `Image` object
  handed straight to `st.image()`.
- **Starlette** must cross an HTTP boundary, so every overlay image is
  JPEG-encoded, base64'd, and stuffed into a JSON string
  (`overlay_data_url`) — meaningfully more bytes-over-the-wire per
  request than a binary image response would be, and this happens on
  **every camera tick (every 1.2 seconds while scanning)**, not just on
  manual upload.

## Data at rest

- `data/Pest_Dataset/` — gitignored, notebook-only training images (9
  pest classes). Never read by either running app.
- `docs/Pesticides_With_Agri_Guideline_Dosage.xlsx` — never read by any
  code path; the README's description of its role (confidence-gated
  pesticide lookup) is not what's actually implemented (`DISEASE_INFO`
  dict is used instead, unconditionally).
- No database, no object storage, no on-disk cache of past
  predictions — every request is stateless end-to-end except for the
  in-memory "last 5" list, which does not survive a process
  restart/page reload.

## External network calls in the data flow

- Google Fonts CDN — blocking `@import`/`<link>` fetch on every page
  load of either UI.
- Google Drive direct-download URL — only on cold start with zero model
  files present, fetching the **legacy ResNet34** `.pkl` regardless of
  whether a ResNet50 checkpoint was actually intended.
- PyTorch's model-weights CDN — only when `build_resnet50()` needs
  ImageNet-pretrained backbone weights (its `try/except` falls back to
  `weights=None`, i.e. random init, if that fetch fails or is
  unreachable — meaning a network outage combined with a missing local
  ResNet50 checkpoint could silently degrade to an **untrained**
  ResNet50, which would then produce near-random predictions with no
  warning to the user).
