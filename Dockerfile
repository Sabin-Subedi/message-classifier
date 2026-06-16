# syntax=docker/dockerfile:1.7

# ---- Stage 1: builder ---------------------------------------------------
# Use the official uv image to get a fast resolver / installer, then build
# a self-contained virtual environment that we copy into the runtime stage.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so the layer is cached when only source changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy application source and install the project itself (no dev extras).
COPY app ./app
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Bake NLTK corpora into the image so the runtime never needs network.
ENV NLTK_DATA=/app/.nltk_data
RUN mkdir -p ${NLTK_DATA} \
 && /app/.venv/bin/python -m nltk.downloader -d ${NLTK_DATA} \
        punkt punkt_tab stopwords wordnet omw-1.4

# ---- Stage 2: runtime ---------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    NLTK_DATA=/app/.nltk_data \
    PORT=8000

# curl is needed for the container HEALTHCHECK; everything else is in the venv.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1001 app \
 && useradd  --system --uid 1001 --gid app --home /app --shell /sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv      /app/.venv
COPY --from=builder --chown=app:app /app/.nltk_data /app/.nltk_data
COPY --chown=app:app app     ./app
COPY --chown=app:app scripts ./scripts

# Models are mounted from the host (or baked separately); ensure the dir exists.
RUN mkdir -p /app/models /app/data && chown -R app:app /app/models /app/data

USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
