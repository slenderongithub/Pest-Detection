# Multi-stage CPU-only image serving the LeafScan Starlette API under uvicorn.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LEAFSCAN_HOST=0.0.0.0 \
    LEAFSCAN_PORT=8080

WORKDIR /app

# System deps for Pillow (image decode).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch/torchvision first (small wheels, no CUDA), then the project.
COPY pyproject.toml README.md ./
COPY leafscan ./leafscan
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install '.[serve]'

# App entry points + assets + a model bundle if one is baked in (usually mounted/fetched).
COPY app ./app
COPY configs ./configs
COPY data/knowledge ./data/knowledge
COPY scripts ./scripts
COPY models ./models

EXPOSE 8080

# Run BY PATH (the repo-root app.py shadows the app/ package, so `uvicorn app.server:app`
# would import the Streamlit script). server.py calls uvicorn.run internally.
CMD ["python", "app/server.py", "serve"]
