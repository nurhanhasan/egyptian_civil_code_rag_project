# ─── FastAPI — production multi-stage image ─────────────────────
#
# Build:        docker build -t ??
# Run API:      docker run --rm -p 8000:8000 --env-file .env.<env_name> ecc/backend:latest
# Run migrate:  docker run --rm --env-file .env.<env_name> ecc/backend:latest alembic upgrade head

ARG PYTHON_IMAGE=python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

FROM ${PYTHON_IMAGE} AS deps

WORKDIR /app

RUN useradd --create-home --uid 1000 app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.17@sha256:10787c682e4184e4f290de1171fd4703dc63de99221f10fe1c99002ce7fa9acc /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy

COPY --chown=app:app pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ── Runtime stage ─────────────────────────────────────────────────────────────

FROM ${PYTHON_IMAGE} AS runtime

ARG WORKERS=4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=${ENVIRONMENT} \
    WORKERS=${WORKERS}

WORKDIR /app

RUN useradd --create-home --uid 1000 app

COPY --from=deps --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PORT=8000

# Application source + migration tree.
COPY --chown=app:app pyproject.toml uv.lock ./
COPY --chown=app:app src/ /app/
# COPY --chown=app:app ./migrations    ./migrations
# COPY --chown=app:app alembic.ini     ./

USER app

EXPOSE 8000

STOPSIGNAL SIGTERM

# Curl-free health probe via the venv's own interpreter (no curl needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).read()"]

CMD ["sh", "-c", "exec uvicorn ragapp.entrypoints.fastapi.main:create_app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
