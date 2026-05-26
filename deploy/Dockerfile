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

ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}
USER app
ENV PYTHONPATH=/app/src
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "moviesentiment.serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
