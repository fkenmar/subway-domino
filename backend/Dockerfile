# syntax=docker/dockerfile:1
# Subway Domino backend (ARCHITECTURE §7): FastAPI + one uvicorn worker on python:3.11-slim.
# Build context: backend/ (see .dockerignore). Run: docker run -e PORT=8080 -p 8080:8080 <image>

# ---- bundle: pick the runtime files out of the build context ---------------------------
# network.json and network_geo.json are required. The ONNX policy is optional: without
# models/nyc_subway_rcc_ppo.onnx the server runs Normal Schedule only and reports
# model_loaded=false (AI mode unavailable; never a heuristic fallback).
FROM python:3.11-slim AS bundle
WORKDIR /src
COPY . .
RUN test -s data/network.json && test -s data/network_geo.json \
    && mkdir -p models

# ---- runtime -----------------------------------------------------------------------------
FROM python:3.11-slim
WORKDIR /app
# UVICORN_WS_MAX_SIZE: uvicorn reads UVICORN_* env vars as CLI options, so the contract's CMD
# stays verbatim while inbound WebSocket messages are capped at 64 KiB instead of 16 MiB
# (client messages are < 8 KiB; the app rejects anything over MAX_MESSAGE_BYTES = 16 KiB).
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UVICORN_WS_MAX_SIZE=65536

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY --from=bundle /src/sim/ sim/
COPY --from=bundle /src/app/ app/
COPY --from=bundle /src/data/network.json /src/data/network_geo.json data/
COPY --from=bundle /src/models/ models/
RUN python -m compileall -q sim app \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin subway

USER subway
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, sys, urllib.request; port = os.environ.get('PORT', '8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=4).status == 200 else 1)"]
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
