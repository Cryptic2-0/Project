# syntax=docker/dockerfile:1.6
# Stage 1: build deps
FROM python:3.11-slim AS builder
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN uv pip install --system --no-cache ".[train]" \
    && uv pip install --system --no-cache .

# Stage 2: runtime
FROM python:3.11-slim AS runtime
RUN useradd -m -u 1000 app
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY src/ ./src/
# Bundle the static UI so /ui serves directly from the Fargate task. Costs ~30 KB.
COPY frontend/ ./frontend/
# ONNX models only (not raw HF weights or checkpoints)
COPY models/distilbert_onnx_int8/ ./models/distilbert_onnx_int8/
COPY models/distilbert_onnx/ ./models/distilbert_onnx/
# v2 multi-task ONNX is NOT bundled in the image. The serving lifespan
# downloads it from S3 (s3://${MS_DVC_BUCKET}/multitask_onnx/) on first boot
# when MS_MULTITASK_S3_PREFIX is set. Keeps image size flat at v1 levels and
# lets the v2 artefact roll independently of the container.

ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}
# The lifespan needs to write the v2 multi-task ONNX into /app/models at boot
# when bootstrapping from S3. /app/models is created by the earlier COPYs as
# root-owned; chown it (and the rest of /app) to the runtime user so the
# bootstrap can mkdir + write the new subdirectory.
RUN chown -R app:app /app
USER app
ENV PYTHONPATH=/app/src
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "moviesentiment.serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
