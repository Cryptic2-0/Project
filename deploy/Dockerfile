# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS builder
ENV UV_SYSTEM_PYTHON=1
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN uv pip install --system --no-cache .

FROM python:3.11-slim AS runtime
RUN useradd -m -u 1000 app
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/src /app/src
COPY models/ /app/models/
USER app
ENV PYTHONPATH=/app/src
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "moviesentiment.serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
