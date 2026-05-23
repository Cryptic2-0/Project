# MovieSentiment — End-to-End MLOps Build Guide

> **Audience:** Claude Code (and Soumya). This document is a self-contained spec to build, ship, and demo a portfolio-grade MLOps project for an ML Engineer interview. Read this end-to-end before starting, then execute week-by-week.

---

## 0. Project Summary

**What:** A production-style sentiment analysis service for IMDb movie reviews. A scraper feeds reviews into a training pipeline that produces two models (a TF-IDF + Logistic Regression baseline and a fine-tuned DistilBERT). The best model is served behind a FastAPI inference API, packaged in Docker, deployed to a free cloud tier, instrumented with Prometheus metrics, monitored for data drift with Evidently, and continuously retrained when drift is detected.

**Why this project for an MLOps interview:**
- Demonstrates the **full lifecycle**: data → train → evaluate → register → serve → monitor → retrain.
- Reuses Soumya's existing `Imdb_review_scrapper-2026-` repo — narrative continuity.
- Sentiment is a *boring* problem on purpose: interviewers focus on **how you ship**, not novel modeling.
- Hits every keyword on an MLE/MLOps job description (MLflow, Docker, FastAPI, CI/CD, monitoring, drift, IaC-lite).

**Definition of done (interview-ready):**
1. Public GitHub repo, pinned, with a clean `README.md` and an architecture diagram.
2. A live URL the interviewer can `curl` for predictions.
3. MLflow tracking UI with at least 6 logged runs comparing baseline vs. transformer.
4. CI green on every push, building and pushing a Docker image to GHCR.
5. A monitoring dashboard (screenshots in README is fine) showing latency + drift metrics.
6. A 3-minute Loom walkthrough linked from the README.

---

## 1. Architecture

```
                    ┌─────────────────────┐
                    │  IMDb (web)         │
                    └──────────┬──────────┘
                               │ scrape (weekly cron, GH Actions)
                               ▼
                    ┌─────────────────────┐
                    │  raw/ (CSV/Parquet) │  ◄── DVC-tracked
                    └──────────┬──────────┘
                               │ clean + split
                               ▼
                    ┌─────────────────────┐
                    │  processed/         │
                    └──────────┬──────────┘
                               │ train (Make targets)
                ┌──────────────┼───────────────┐
                ▼                              ▼
       ┌────────────────┐             ┌──────────────────┐
       │ TF-IDF + LR    │             │ DistilBERT FT    │
       └────────┬───────┘             └─────────┬────────┘
                │                                │
                └────────────┬───────────────────┘
                             ▼
                  ┌────────────────────┐
                  │  MLflow Tracking   │
                  │  + Model Registry  │
                  └──────────┬─────────┘
                             │ promote best
                             ▼
                  ┌────────────────────┐
                  │ FastAPI service    │
                  │ (Dockerized, ONNX) │
                  └──────────┬─────────┘
                             │ /predict
                             ▼
                  ┌────────────────────┐
                  │ Fly.io / HF Space  │
                  └──────────┬─────────┘
                             │ /metrics (Prometheus)
                             ▼
                  ┌────────────────────┐
                  │ Grafana + Evidently│
                  │ drift dashboards   │
                  └────────────────────┘
```

---

## 2. Tech Stack & Rationale

| Layer | Choice | Why this, not the alternative |
|---|---|---|
| Language | Python 3.11 | Standard for ML; 3.11 for speed and pattern matching. |
| Env mgmt | `uv` (preferred) or `venv` + `pip-tools` | `uv` is fast and reproducible. Falls back to pip. |
| Data versioning | DVC with local + S3/GDrive remote | Industry standard, free. Git-LFS works but is less ML-aware. |
| Experiment tracking | MLflow (self-hosted, SQLite backend) | Free, runs locally, supports Model Registry. W&B is also fine but adds an account. |
| Baseline model | scikit-learn `TfidfVectorizer` + `LogisticRegression` | Trains in seconds, ~88% acc on IMDb — a credible baseline. |
| Deep model | `distilbert-base-uncased` (HuggingFace) | Small enough to fine-tune on Colab/free GPU; ~92% acc. |
| Optimization | ONNX Runtime + dynamic quantization | Cuts BERT latency ~3x — shows latency-awareness. |
| Serving | FastAPI + Uvicorn + Pydantic v2 | De-facto standard for ML APIs. |
| Container | Docker multi-stage, distroless or python-slim | Multi-stage = small image = MLOps maturity signal. |
| Registry | GitHub Container Registry (GHCR) | Free with GitHub Actions. |
| CI/CD | GitHub Actions | Free, integrates with the repo. |
| Deploy | Fly.io (preferred) or Hugging Face Spaces | Fly = real container; HF = easiest demo. Pick one. |
| Metrics | `prometheus-fastapi-instrumentator` | One-liner instrumentation. |
| Drift | Evidently AI | Free, generates HTML drift reports. |
| Dashboards | Grafana Cloud free tier OR screenshots | Free tier supports Prom scraping. |
| Orchestration | GitHub Actions cron (NOT Airflow) | Airflow is overkill for a portfolio project; using it is a red flag for over-engineering. |

---

## 3. Repository Structure

```
moviesentiment/
├── .github/
│   └── workflows/
│       ├── ci.yml                # lint, test, build
│       ├── train.yml             # weekly retrain on cron
│       └── scrape.yml            # weekly data refresh
├── data/                         # DVC-tracked, gitignored
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── architecture.png
│   └── interview_talking_points.md
├── models/                       # DVC-tracked
├── notebooks/
│   └── 01_eda.ipynb              # exactly ONE notebook, for EDA only
├── src/
│   └── moviesentiment/
│       ├── __init__.py
│       ├── config.py             # Pydantic Settings, env vars
│       ├── data/
│       │   ├── __init__.py
│       │   ├── scrape.py         # adapted from Imdb_review_scrapper-2026-
│       │   ├── clean.py
│       │   └── split.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── baseline.py       # TF-IDF + LR
│       │   ├── transformer.py    # DistilBERT FT
│       │   └── registry.py       # MLflow promote/load helpers
│       ├── eval/
│       │   ├── __init__.py
│       │   └── metrics.py        # accuracy, F1, confusion matrix, calibration
│       ├── serve/
│       │   ├── __init__.py
│       │   ├── api.py            # FastAPI app
│       │   ├── schemas.py        # Pydantic request/response
│       │   └── inference.py      # ONNX runner
│       ├── monitor/
│       │   ├── __init__.py
│       │   ├── drift.py          # Evidently report generation
│       │   └── prometheus.py     # custom metrics
│       └── cli.py                # typer CLI: train, evaluate, serve, drift
├── tests/
│   ├── conftest.py
│   ├── test_clean.py
│   ├── test_api.py
│   ├── test_inference.py
│   └── fixtures/
│       └── sample_reviews.csv
├── deploy/
│   ├── Dockerfile                # multi-stage
│   ├── docker-compose.yml        # local stack: api + prometheus + grafana
│   ├── fly.toml                  # Fly.io config
│   └── prometheus.yml
├── scripts/
│   ├── bootstrap.sh              # one-shot dev setup
│   └── load_test.py              # locust scenarios
├── .dvc/
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml
├── dvc.yaml                      # pipeline DAG
├── params.yaml                   # hyperparameters (DVC reads this)
├── pyproject.toml                # build + deps + tool config
├── Makefile                      # `make train`, `make serve`, `make test`
├── README.md
└── LICENSE                       # MIT
```

---

## 4. Prerequisites

### Local install (Windows / PowerShell)

```powershell
# 1. Python 3.11
winget install Python.Python.3.11

# 2. uv (fast package manager)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Docker Desktop
winget install Docker.DockerDesktop

# 4. Git LFS (for model artifacts if DVC remote is unavailable)
winget install GitHub.GitLFS

# 5. GitHub CLI
winget install GitHub.cli
```

### Accounts needed (all free)

- GitHub (have)
- Fly.io OR Hugging Face (pick one for deploy)
- Optional: Grafana Cloud free tier

### Secrets to configure in GitHub repo settings → Secrets and variables → Actions

| Secret | Used by | How to get |
|---|---|---|
| `FLY_API_TOKEN` | `deploy.yml` | `flyctl auth token` after `flyctl auth login` |
| `HF_TOKEN` | optional, model push | huggingface.co/settings/tokens |
| `DVC_REMOTE_*` | optional, data sync | depends on remote chosen |

---

## 5. Week-by-Week Execution Plan

> Each week ends with a tagged release (`v0.1`, `v0.2`, …) and a screenshot or log in `docs/progress/`.

### Week 1 — Data pipeline + baseline model

**Goal:** Trainable, reproducible pipeline producing a baseline model with tracked metrics.

#### Day 1 — Bootstrap
- [ ] `gh repo create moviesentiment --public --description "End-to-end MLOps for IMDb sentiment analysis"`
- [ ] Initialize structure from §3.
- [ ] `pyproject.toml` with dependencies (see §6).
- [ ] `pre-commit install` with `ruff`, `black`, `mypy`, `pytest` hooks.
- [ ] `dvc init` and configure a remote (GDrive is easiest for portfolio).
- [ ] Commit, push, tag `v0.0-scaffold`.

#### Day 2 — Port the scraper
- [ ] Copy logic from `Imdb_review_scrapper-2026-` into `src/moviesentiment/data/scrape.py`.
- [ ] Refactor into a function `scrape_reviews(movie_ids: list[str], out_path: Path) -> int`.
- [ ] Output schema: `review_id, movie_id, text, rating, scraped_at`.
- [ ] Label rule: `rating >= 7 → positive`, `rating <= 4 → negative`, drop neutrals.
- [ ] Persist as Parquet (`data/raw/reviews_YYYYMMDD.parquet`).
- [ ] Add to `dvc.yaml` as stage `scrape`.

#### Day 3 — Clean + split
- [ ] `src/moviesentiment/data/clean.py`: lowercase, strip HTML, remove URLs, drop dupes.
- [ ] `src/moviesentiment/data/split.py`: stratified train/val/test 70/15/15, fixed seed in `params.yaml`.
- [ ] Add DVC stages `clean` and `split`.
- [ ] Unit test cleaning on `tests/fixtures/sample_reviews.csv`.

#### Day 4 — Baseline model
- [ ] `src/moviesentiment/models/baseline.py`:
  - `TfidfVectorizer(ngram_range=(1,2), max_features=50000)`
  - `LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')`
  - Pipeline persisted with `joblib`.
- [ ] Log to MLflow: params, metrics (accuracy, macro-F1, ROC-AUC), confusion matrix artifact, the pipeline itself.
- [ ] Run `mlflow ui` locally; screenshot to `docs/progress/`.

#### Day 5 — Evaluation rigor
- [ ] `src/moviesentiment/eval/metrics.py`: confusion matrix, ROC, PR curve, calibration plot, per-class report.
- [ ] Slice analysis: accuracy by review length quartile (short reviews are harder).
- [ ] Document failure cases in `docs/error_analysis.md` (5 false positives + 5 false negatives, with hypotheses).

**Week 1 deliverable:** Tag `v0.1-baseline`. README has the architecture diagram, a "How to reproduce" section, and a metrics table.

---

### Week 2 — Transformer model + serving

**Goal:** A stronger model and a working API.

#### Day 6–7 — DistilBERT fine-tuning
- [ ] `src/moviesentiment/models/transformer.py` using HuggingFace `Trainer`.
- [ ] Hyperparameters in `params.yaml`: `lr=2e-5, batch=16, epochs=3, warmup=0.1, weight_decay=0.01`.
- [ ] Train on Colab if no GPU locally; download artifacts back into `models/`.
- [ ] Log to MLflow alongside baseline. Tag the run `model=distilbert`.
- [ ] Promote the better model to the `Staging` stage in MLflow Model Registry.

#### Day 8 — ONNX export + quantization
- [ ] Export to ONNX using `optimum`.
- [ ] Apply dynamic quantization (INT8). Verify accuracy degradation <1%.
- [ ] Benchmark: latency p50/p95 on CPU, before vs. after. Save to `docs/benchmarks.md`.

#### Day 9 — FastAPI service
- [ ] `src/moviesentiment/serve/schemas.py`: `PredictRequest { texts: list[str] }`, `PredictResponse { predictions: list[Prediction] }`.
- [ ] `src/moviesentiment/serve/inference.py`: loads ONNX model once at startup, batched inference.
- [ ] `src/moviesentiment/serve/api.py`:
  - `GET /healthz` — liveness
  - `GET /readyz` — model loaded check
  - `POST /predict` — main endpoint
  - `GET /metrics` — Prometheus exposure
  - `GET /version` — model + git SHA
- [ ] Pydantic validation: text length 1–5000 chars, batch size ≤ 32.
- [ ] CORS middleware permissive (it's a demo).

#### Day 10 — Dockerize
- [ ] Multi-stage `Dockerfile`:
  - Stage 1: `python:3.11-slim` + `uv` to install deps to a venv.
  - Stage 2: `python:3.11-slim`, copy venv + app, non-root user, `EXPOSE 8000`, `CMD ["uvicorn", "..."]`.
- [ ] `.dockerignore`: exclude `data/`, `notebooks/`, `models/*.bin` (download from MLflow at start).
- [ ] Target image size: <500 MB.
- [ ] `docker-compose.yml` runs api + prometheus + grafana locally.

**Week 2 deliverable:** Tag `v0.2-serving`. `docker run` locally returns predictions. Latency table in README.

---

### Week 3 — CI/CD + deployment + monitoring

**Goal:** Push button → live service.

#### Day 11 — CI
- [ ] `.github/workflows/ci.yml`:
  - Triggers: PR, push to main.
  - Jobs: `lint` (ruff + black --check), `type` (mypy), `test` (pytest with coverage), `build` (docker build, push to GHCR with tag = git SHA + `latest`).
  - Cache `uv` deps.
  - Fail if test coverage <70%.
- [ ] Add status badges to README.

#### Day 12 — Deploy
- [ ] **Option A — Fly.io (recommended):**
  - `flyctl launch --no-deploy`, edit `fly.toml`: 1 shared CPU, 512 MB.
  - `flyctl secrets set MLFLOW_TRACKING_URI=…`
  - GH Actions deploy job: pull image, `flyctl deploy --image …`.
  - Custom domain optional; the `*.fly.dev` URL is fine.
- [ ] **Option B — HF Space (easier):**
  - Create Space with Docker SDK, push Dockerfile.
  - Persistent storage for the model artifact.

#### Day 13 — Prometheus + Grafana
- [ ] Instrument with `prometheus-fastapi-instrumentator`.
- [ ] Custom metrics:
  - `prediction_confidence_histogram` (per-prediction)
  - `prediction_class_total` (counter by predicted label)
  - `model_version_info` (gauge with labels = model name, version, SHA)
- [ ] Local stack via docker-compose. Grafana dashboard JSON committed to `deploy/grafana_dashboard.json`.
- [ ] Screenshot to `docs/`.

#### Day 14 — Load testing
- [ ] `scripts/load_test.py` with Locust: ramp 1→50 users over 2 min.
- [ ] Document p50/p95/p99 latency and throughput.
- [ ] Tune `uvicorn` workers if needed (`--workers 2 --loop uvloop`).

#### Day 15 — Buffer / catch-up day. DO NOT skip.

**Week 3 deliverable:** Tag `v0.3-deployed`. Live URL in README. CI badge green.

---

### Week 4 — Drift, retraining, polish

**Goal:** Close the loop and prep the demo.

#### Day 16–17 — Drift detection
- [ ] `src/moviesentiment/monitor/drift.py`:
  - Load reference distribution (training set features).
  - Compute Evidently `DataDriftPreset` on the latest week of production inputs.
  - Generate HTML report → `docs/drift_reports/YYYY-MM-DD.html`.
- [ ] Production input capture: log every request payload + prediction to a Parquet rolling file (sample 10% if traffic gets real).
- [ ] Threshold: if `drift_share > 0.3`, set GH Actions output `should_retrain=true`.

#### Day 18 — Retraining workflow
- [ ] `.github/workflows/train.yml`:
  - Trigger: weekly cron + `workflow_dispatch`.
  - Steps: pull new data via DVC → run training → log to MLflow → run eval → if new model beats current on val by ≥0.5% F1, promote to `Production` in registry and trigger deploy.
- [ ] Deploy job downloads the `Production`-stage model at container startup.

#### Day 19 — Documentation polish
- [ ] README sections (template in §8):
  1. Hero (one-line + live demo link + curl example)
  2. Architecture diagram
  3. Quickstart (5 commands)
  4. Results table (baseline vs. transformer, latency, drift handling)
  5. Tradeoffs & what I'd do in real production
  6. Roadmap
- [ ] Architecture diagram: build in Excalidraw, export PNG.
- [ ] `docs/interview_talking_points.md` — see §9.

#### Day 20 — Demo recording
- [ ] 3-minute Loom: walk through architecture diagram → MLflow UI → live `curl` to /predict → Grafana dashboard → drift report → CI run.
- [ ] Embed link at top of README.

#### Day 21 — Final pass
- [ ] Pin the repo on your GitHub profile.
- [ ] Update LinkedIn with the live URL and Loom.
- [ ] Open a self-PR titled "v1.0 release" with the release notes — interviewers look at PR hygiene too.
- [ ] Tag `v1.0`.

---

## 6. Dependencies

### `pyproject.toml` (excerpt)

```toml
[project]
name = "moviesentiment"
version = "1.0.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "scikit-learn>=1.4",
    "pandas>=2.2",
    "pyarrow>=15",
    "joblib>=1.3",
    "transformers>=4.38",
    "torch>=2.2",
    "datasets>=2.18",
    "optimum[onnxruntime]>=1.17",
    "onnxruntime>=1.17",
    "mlflow>=2.11",
    "dvc[gdrive]>=3.48",
    "evidently>=0.4",
    "prometheus-fastapi-instrumentator>=7.0",
    "typer>=0.9",
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.3",
    "black>=24.3",
    "mypy>=1.9",
    "pytest>=8.1",
    "pytest-cov>=4.1",
    "pytest-asyncio>=0.23",
    "locust>=2.24",
    "pre-commit>=3.6",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]
```

---

## 7. Key Code Skeletons

> Full code generated in implementation phase; these are interface contracts.

### `src/moviesentiment/config.py`

```python
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    model_dir: Path = project_root / "models"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "moviesentiment"
    model_name: str = "moviesentiment-classifier"
    model_stage: str = "Production"
    max_text_length: int = 5000
    max_batch_size: int = 32

    class Config:
        env_prefix = "MS_"
        env_file = ".env"

settings = Settings()
```

### `src/moviesentiment/serve/api.py` (skeleton)

```python
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from .schemas import PredictRequest, PredictResponse, HealthResponse
from .inference import InferenceEngine
from ..config import settings

app = FastAPI(title="MovieSentiment", version="1.0.0")
engine: InferenceEngine | None = None

@app.on_event("startup")
def _load_model():
    global engine
    engine = InferenceEngine.from_registry(settings.model_name, settings.model_stage)

@app.get("/healthz", response_model=HealthResponse)
def healthz():
    return HealthResponse(status="ok")

@app.get("/readyz", response_model=HealthResponse)
def readyz():
    if engine is None or not engine.is_ready():
        raise HTTPException(503, "model not loaded")
    return HealthResponse(status="ready")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if engine is None:
        raise HTTPException(503, "model not loaded")
    return PredictResponse(predictions=engine.predict(req.texts))

Instrumentator().instrument(app).expose(app)
```

### `dvc.yaml` (pipeline DAG)

```yaml
stages:
  scrape:
    cmd: python -m moviesentiment.cli scrape --out data/raw/reviews.parquet
    outs:
      - data/raw/reviews.parquet
  clean:
    cmd: python -m moviesentiment.cli clean --in data/raw/reviews.parquet --out data/interim/clean.parquet
    deps:
      - data/raw/reviews.parquet
      - src/moviesentiment/data/clean.py
    outs:
      - data/interim/clean.parquet
  split:
    cmd: python -m moviesentiment.cli split --in data/interim/clean.parquet --out-dir data/processed
    deps:
      - data/interim/clean.parquet
    params:
      - split.test_size
      - split.val_size
      - split.seed
    outs:
      - data/processed/train.parquet
      - data/processed/val.parquet
      - data/processed/test.parquet
  train_baseline:
    cmd: python -m moviesentiment.cli train baseline
    deps:
      - data/processed/train.parquet
      - data/processed/val.parquet
      - src/moviesentiment/models/baseline.py
    params:
      - baseline
    metrics:
      - metrics/baseline.json
  train_transformer:
    cmd: python -m moviesentiment.cli train transformer
    deps:
      - data/processed/train.parquet
      - data/processed/val.parquet
      - src/moviesentiment/models/transformer.py
    params:
      - transformer
    metrics:
      - metrics/transformer.json
```

### `deploy/Dockerfile`

```dockerfile
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
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "moviesentiment.serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### `.github/workflows/ci.yml`

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install uv
        run: pip install uv
      - name: Install deps
        run: uv pip install --system -e ".[dev]"
      - run: ruff check .
      - run: black --check .
      - run: mypy src
      - run: pytest --cov=moviesentiment --cov-fail-under=70

  build-image:
    needs: lint-test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: deploy/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:${{ github.sha }}
            ghcr.io/${{ github.repository }}:latest
```

---

## 8. README Template

```markdown
# MovieSentiment

> Production-style sentiment analysis for IMDb reviews. Built end-to-end as an MLOps demo.

[![ci](https://github.com/Cryptic2-0/moviesentiment/actions/workflows/ci.yml/badge.svg)](…)
[![live demo](https://img.shields.io/badge/demo-online-brightgreen)](https://moviesentiment.fly.dev/docs)

**3-minute video walkthrough:** [Loom link]

```bash
curl -X POST https://moviesentiment.fly.dev/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["A complete masterpiece.", "Two hours I will never get back."]}'
```

## Why this exists
Most ML portfolio projects stop at a Jupyter notebook. This one ships: data pipeline, model registry, container, CI/CD, deployment, monitoring, drift detection, and an automated retraining loop.

## Architecture
![architecture](docs/architecture.png)

## Results

| Model | Accuracy | Macro F1 | p95 latency (CPU) | Size |
|---|---|---|---|---|
| TF-IDF + LR | 0.881 | 0.880 | 4 ms | 12 MB |
| DistilBERT | 0.927 | 0.926 | 78 ms (fp32) / 26 ms (ONNX-INT8) | 65 MB |

## Quickstart
```bash
git clone https://github.com/Cryptic2-0/moviesentiment
cd moviesentiment
uv pip install -e ".[dev]"
dvc pull
make train
make serve
```

## What I'd do differently in real production
- Replace the SQLite MLflow backend with a managed Postgres + S3 artifact store.
- Move from GitHub Actions cron to Airflow/Dagster once we have >5 pipelines.
- Add a feature store (Feast) once features are shared across models.
- Shadow-deploy new models for 24h before promotion instead of relying on offline metrics alone.

## Roadmap
- [ ] Multi-language support (es, fr)
- [ ] Active learning loop on low-confidence predictions
```

---

## 9. Interview Talking Points (save to `docs/interview_talking_points.md`)

For each of these, prepare a 60-second answer:

1. **"Walk me through the architecture."** — Follow the diagram top-to-bottom. End with "drift triggers retraining" to land the closed-loop story.
2. **"Why DistilBERT and not BERT-base or a larger model?"** — Latency budget, deployable on free tier, accuracy delta is small.
3. **"How do you detect drift?"** — Evidently DataDrift on input text features (length, vocabulary overlap, embedding distance); threshold 0.3 share triggers retrain.
4. **"How do you decide whether to promote a new model?"** — Offline: ≥0.5% F1 improvement on held-out test + no slice regression. Real prod would add shadow traffic.
5. **"What does your CI do?"** — Lint, type, test with coverage gate, build, push image. Mention what's *not* there: e2e API tests against a deployed staging environment — that's the next step.
6. **"What was the hardest tradeoff?"** — Latency vs. accuracy on BERT; solved with ONNX + quantization (link benchmark table). Or: choosing not to use Airflow because cron was enough — over-engineering is a real cost.
7. **"How would this scale 100x?"** — Move ONNX → Triton Inference Server, add a Redis cache for repeated requests, horizontal pod autoscaling on QPS, batch inference for the offline scoring path.
8. **"How do you handle PII / safety?"** — Right now: nothing, it's a demo. In production: drop user identifiers at ingest, add an input-toxicity filter, log only hashes.

---

## 10. Anti-Patterns to Avoid

- **Don't** include large model binaries in git — use DVC or HuggingFace Hub.
- **Don't** train inside CI on every push — it's slow and expensive. Cron + manual dispatch only.
- **Don't** put credentials in `.env.example` files — use GitHub Secrets.
- **Don't** create a separate notebook for each experiment — log to MLflow.
- **Don't** add Kubernetes "because MLOps" — Fly.io is enough for a portfolio project and explaining *why you chose not to* is itself a senior signal.
- **Don't** skip the README's "what I'd do differently" section — it's the single highest-leverage paragraph in the project.

---

## 11. Stretch Goals (only if ahead of schedule)

- A/B testing harness: route 10% of traffic to a challenger model and compare metrics.
- LLM-judge eval: use Claude/GPT to grade subjective cases the F1 metric misses.
- Active learning: surface low-confidence predictions in a tiny Streamlit app for hand-labeling, feed back into training set.
- Multi-task head: also predict review usefulness (rating count) — shows multi-output design.

---

## 12. Daily Checklist Template

Copy this into `docs/progress/day-XX.md` each day:

```markdown
# Day XX — YYYY-MM-DD

## Planned
- [ ] task 1
- [ ] task 2

## Done
- …

## Blockers
- …

## Decisions / tradeoffs
- …

## Tomorrow
- …
```

---

## 13. Handoff Notes for Claude Code

When future Claude Code sessions pick up this project:

1. Read this file first, then `README.md`, then `docs/progress/` (latest).
2. The source of truth for **what to do next** is the daily checklist in `docs/progress/`.
3. The source of truth for **how things work** is `dvc.yaml` (pipeline) and `pyproject.toml` (deps).
4. Never commit to `main` directly — open a PR even for solo work. The PR history is part of the demo.
5. Tag releases at week boundaries (`v0.1`, `v0.2`, `v0.3`, `v1.0`).
6. Run `make ci-local` before pushing — it runs the same checks as CI.
7. If something in this guide conflicts with reality (e.g. a library API changed), update the guide as part of the same PR. This doc must stay current.

---

**End of guide.** Start with §5 Day 1.
