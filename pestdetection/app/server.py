"""LeafScan Starlette API — a thin shell over the shared ``leafscan`` package.

All ML logic lives in ``leafscan`` (imported, not reimplemented). This file is only:
routing, request validation, threadpool offload of the CPU-bound forward pass, and an
honest no-model/demo state.

Run with ``python app/server.py serve``. Do NOT use ``uvicorn app.server:app``: the
repo-root ``app.py`` (Streamlit) shadows the ``app/`` package, so importing ``app.server``
would execute the Streamlit script instead.
"""

from __future__ import annotations

import sys
from functools import partial
from io import BytesIO
from pathlib import Path

# Make the repo-root `leafscan` package importable even when this file is run directly
# (python app/server.py sets sys.path[0] to app/, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager  # noqa: E402

import uvicorn  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402
from starlette.middleware.cors import CORSMiddleware  # noqa: E402
from starlette.responses import HTMLResponse, JSONResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

from leafscan.config import get_settings  # noqa: E402
from leafscan.inference import ModelNotAvailableError, Predictor  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
VIEW_DIR = APP_DIR / "view"

settings = get_settings()
predictor = Predictor(settings=settings)

# Guard against decompression-bomb images.
Image.MAX_IMAGE_PIXELS = settings.max_image_pixels


def _demo_payload() -> dict:
    return {
        "error": "no_model",
        "demo_mode": True,
        "message": (
            "No trained model is installed. Train one with `leafscan train` or fetch "
            "weights with `python scripts/fetch_model.py`. This app never fabricates a "
            "prediction — it shows this honest state instead."
        ),
    }


async def homepage(request):
    return HTMLResponse((VIEW_DIR / "index.html").read_text(encoding="utf-8"))


async def health(request):
    body = {
        "ok": True,
        "model_available": predictor.is_available,
        "model": None,
        "demo_mode": not predictor.is_available,
    }
    if predictor.is_available:
        bundle = predictor.bundle
        body.update(
            {
                "model": predictor.model_name,
                "arch": bundle.arch,
                "class_count": bundle.num_classes,
                "macro_f1": (bundle.metrics or {}).get("macro_f1"),
                "input_size": bundle.input_size,
            }
        )
    return JSONResponse(body)


async def _analyze(request, *, want_gradcam: bool):
    if not predictor.is_available:
        return JSONResponse(_demo_payload(), status_code=503)

    # Reject oversized uploads on the declared Content-Length BEFORE buffering the body.
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > settings.max_upload_bytes:
                return JSONResponse(
                    {"error": "file_too_large", "max_bytes": settings.max_upload_bytes},
                    status_code=413,
                )
        except ValueError:
            return JSONResponse({"error": "invalid_content_length"}, status_code=400)

    try:
        form = await request.form()
    except Exception:
        return JSONResponse({"error": "invalid_form"}, status_code=400)

    file = form.get("file")
    if file is None or not hasattr(file, "read"):
        return JSONResponse({"error": "missing_file"}, status_code=400)

    data = await file.read()
    if not data:
        return JSONResponse({"error": "empty_file"}, status_code=400)
    if len(data) > settings.max_upload_bytes:
        return JSONResponse(
            {"error": "file_too_large", "max_bytes": settings.max_upload_bytes},
            status_code=413,
        )

    try:
        # verify() then reopen (verify invalidates the image object).
        Image.open(BytesIO(data)).verify()
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        return JSONResponse({"error": "invalid_image"}, status_code=400)

    try:
        prediction = await run_in_threadpool(
            partial(predictor.predict, image, want_gradcam=want_gradcam)
        )
    except ModelNotAvailableError:
        return JSONResponse(_demo_payload(), status_code=503)
    except Exception:  # pragma: no cover - defensive; never leak a raw traceback
        return JSONResponse({"error": "inference_failed"}, status_code=500)

    return JSONResponse(prediction.to_api_dict())


async def analyze(request):
    # Single-image upload: compute the Grad-CAM overlay.
    return await _analyze(request, want_gradcam=True)


async def analyze_frame(request):
    # Live camera frame (~1.2s cadence): skip Grad-CAM to keep latency low.
    return await _analyze(request, want_gradcam=False)


@asynccontextmanager
async def lifespan(app):
    if settings.num_threads > 0:
        import torch

        torch.set_num_threads(settings.num_threads)
    if predictor.is_available:
        # Warm the model off the event loop so the first real request is fast.
        await run_in_threadpool(predictor.warmup)
    yield


routes = [
    Route("/", homepage),
    Route("/health", health),
    Route("/analyze", analyze, methods=["POST"]),
    Route("/analyze-frame", analyze_frame, methods=["POST"]),
    Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
]

app = Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app=app, host=settings.host, port=settings.port, log_level="info")
    else:
        print("Usage: python app/server.py serve", file=sys.stderr)
        sys.exit(1)
