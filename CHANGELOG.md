# Changelog

## v1.0 — 2026-05-25

### End-to-end MLOps pipeline, interview-ready.

#### Data
- IMDb scraper with HuggingFace fallback; weekly GH Actions cron (`scrape.yml`)
- Clean + stratified split (70/15/15) as DVC pipeline stages
- Data versioned in S3 via DVC (`s3://moviesentiment-dvc-soumya/dvc`)

#### Models
- **Baseline**: TF-IDF (1–2 grams, 50k features) + Logistic Regression — accuracy 0.904, macro-F1 0.904, AUC 0.967
- **Transformer**: DistilBERT fine-tuned via SageMaker (`ml.g4dn.xlarge`, PyTorch 2.2) — metrics to land after training
- **ONNX export + INT8 quantization**: 2.1× latency speedup at p50, <1% accuracy drop

#### Serving
- FastAPI (`/healthz`, `/readyz`, `/predict`, `/metrics`, `/version`)
- Pydantic v2 validation: text 1–5000 chars, batch ≤ 32
- ONNX Runtime inference engine, model loaded once at startup
- Prometheus instrumentation + custom `prediction_confidence`, `prediction_class_total`, `model_version_info` metrics

#### Infrastructure
- Multi-stage Docker image (`python:3.11-slim`, non-root user)
- docker-compose stack: api + Prometheus + Grafana
- Grafana dashboard JSON committed to `deploy/grafana_dashboard.json`
- Deployed to Fly.io (`moviesentiment.fly.dev`), 1 shared CPU / 512 MB

#### CI/CD
- `ci.yml`: lint (ruff + black), mypy, pytest ≥70% coverage, Docker build → GHCR push
- `deploy.yml`: triggers on `ci` success; `flyctl deploy` with GHCR image
- `train.yml`: weekly retrain cron; promotes model only if F1 improves ≥0.5%
- `scrape.yml`: weekly data refresh
- `drift.yml`: weekly drift detection; triggers retrain if `drift_share > 0.3`

#### Monitoring
- Evidently `DataDriftPreset` drift reports to `docs/drift_reports/`
- Production input capture at `/predict` with 10% sampling
- Grafana dashboard auto-provisioned from JSON

#### Fixed in v1.0
- `dvc remote modify --local myremote` → `s3remote` in all 4 workflows (credentials were never injected into CI)
- SageMaker source bundle: 882 MB → 95 KB by staging minimal dir (SDK has no ignore-file support)

---

## v0.3-deployed
Fly.io deploy + CI/CD pipeline + drift detection + retraining workflow.

## v0.2-serving
FastAPI service + ONNX INT8 inference + Docker multi-stage image.

## v0.1-baseline
TF-IDF + LR baseline with MLflow tracking, eval rigor (ROC, PR curve, calibration).

## v0.0-scaffold
Initial project structure, DVC pipeline, scraper.
