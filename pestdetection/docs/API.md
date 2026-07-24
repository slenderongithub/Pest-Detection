# API Reference

Only the Starlette app (`app/server.py`) exposes an HTTP API. Streamlit
has no API surface (its "protocol" is Streamlit's internal websocket,
not something this project defines).

Base URL (local): `http://localhost:8080`

## `GET /`

Returns the SPA shell.
- **Handler**: `homepage` (`app/server.py:650`)
- **Response**: `text/html`, the verbatim contents of
  `app/view/index.html` (read from disk on every request — no caching,
  fine at this scale).

## `GET /health`

- **Handler**: `health` (`app/server.py:655`)
- **Response** `200 application/json`:
  ```json
  { "ok": true, "model": "ResNet50", "kind": "torch-resnet50" }
  ```
  `model` is `backend["model_name"]`; `kind` is one of
  `torch-resnet50 | fastai-resnet50 | fastai-resnet34 | missing`.
- **Use**: `client.js:boot()` calls this once on page load purely to
  populate the "Model" stat card — not polled repeatedly, so a model
  hot-swap while the server is running would not be reflected until
  page reload anyway (the server doesn't reload the model on its own
  regardless).

## `POST /analyze` and `POST /analyze-frame`

- **Handler**: `analyze` for **both** routes — genuinely the same
  function object, registered twice (`app/server.py:673-676`). There is
  no server-side distinction between a manual upload and a camera tick.
- **Request**: `multipart/form-data` with a single field `file`
  (binary image). Missing field → `400 {"error": "Missing file
  upload."}`.
- **Response** `200 application/json`, shape depends on backend kind but
  is consistent across both:
  ```json
  {
    "model": "ResNet50",
    "prediction": "Tomato___Early_blight",
    "display_prediction": "Tomato - Early blight",
    "confidence": 87.42,
    "severity": "Medium",
    "disease_info": {"severity": "Medium", "cause": "...", "tip": "..."},
    "healthy": false,
    "top_predictions": [
      {"label": "Tomato___Early_blight", "probability": 87.42},
      {"label": "Tomato___Septoria_leaf_spot", "probability": 6.1},
      {"label": "Tomato___healthy", "probability": 2.3}
    ],
    "boxes": [{"x1": 40, "y1": 55, "x2": 190, "y2": 210}],
    "overlay_data_url": "data:image/jpeg;base64,/9j/4AAQ...",
    "image_size": {"width": 1280, "height": 960}
  }
  ```
  `disease_info` is `null` for the ~19/38 classes with no
  `DISEASE_INFO` entry.
- **Errors**: no `try/except` around image decode or model inference —
  a corrupt upload or an inference exception (e.g. no model and the
  heuristic path also failing) will surface as an unhandled Starlette
  500, with a raw traceback if debug mode is on, or a generic 500
  otherwise. No structured error schema beyond the one explicit 400
  above.
- **Auth**: none. **Rate limiting**: none. **Payload size limit**: none
  configured explicitly (Starlette/Uvicorn defaults apply).

## What's conspicuously absent

- No versioning (`/v1/...`), no OpenAPI/Swagger spec, no batch endpoint,
  no way to fetch past results, no webhook/async job pattern for slower
  inference, no request-id/correlation-id in responses for debugging.
