# MovieSentiment — Final Project Report

> End-to-end MLOps build. From an empty repo to a multi-task DistilBERT serving on AWS Fargate at ~$6/mo, with a v1 binary sentiment classifier and a v2 five-head review-intelligence model deployed in production.

**Report date:** 2026-05-28
**Repo:** `github.com/Cryptic2-0/Project`
**Owner:** Soumya Sarkar (`Cryptic2-0`)
**Scope:** Entire history from `e83f696` (scaffold, 2026-05-24) through `0f95416` (latest on `main`, 2026-05-28).
**Branch state at report time:** 52 non-merge commits on `main`; tags `v0.0-scaffold`, `v0.1-baseline`, `v0.2-serving`, `v0.3-deployed`, `v1.0`, `v2.0`.

---

## 1. Project overview and goal

**What it is.** A production-style sentiment-analysis service for IMDb-style movie reviews, packaged as a portfolio-grade MLOps project. The service exposes:

- `POST /predict` — binary sentiment (positive / negative) over a batch of up to 32 reviews.
- `POST /analyze` — v2 multi-task analysis: sentiment + 5-aspect ABSA + 6-class Ekman emotion + spoiler detection + helpfulness regression in one ONNX forward pass.
- `GET /insights/{movie_id}` — aggregated per-movie review intelligence over reservoir-sampled production inputs.
- `POST /explain` — per-token attribution via occlusion (opt-in).
- `POST /predict/async` + `GET /predict/result/{job_id}` — SQS-backed async path.
- Ops endpoints: `/healthz`, `/readyz`, `/version`, `/metrics`, `/sample`, `/ui/`.

**Problem it solves.** Most ML portfolios stop at a notebook with metrics. This one ships the full lifecycle — data scrape, versioning, training, registry, optimised serving, containerisation, deploy, monitoring, drift detection, retraining — at a cost that fits in a free-tier budget. The v2 multi-head endpoint is also a real product surface: no free public API (Google NL, AWS Comprehend, Azure Text Analytics) ships sentiment + aspect + emotion + spoiler + helpfulness in one call.

**Who it is for.**

- Primary audience: hiring panels for ML Engineer / MLOps roles. The project exists to be walked through in interviews.
- Secondary audience: the author, as a working reference for production patterns (DVC pipelines, ONNX export, SageMaker training, Fargate cost optimisation).

**End-goal.**

- *Short term:* a live URL an interviewer can `curl`, plus an architecture diagram, model card, datasheet, drift report, and load-test numbers — all reproducible from `dvc repro` + `make serve`.
- *Long term:* converge on a multi-task review-intelligence service with offline insights aggregation (`/insights/{movie_id}`) and an active-learning loop fed by reservoir-sampled, low-confidence predictions.

---

## 2. Origin story

The repo started on **2026-05-24 at 05:27 IST** with commit `e83f696` — `scaffold: initialize MovieSentiment project structure` (tag `v0.0-scaffold`). The very first commit laid down:

- `pyproject.toml` declaring the full v1 dependency surface (FastAPI, transformers, ONNX Runtime, MLflow, DVC, Evidently, Prometheus instrumentator, Typer).
- The directory tree mandated by [`MOVIESENTIMENT_BUILD_GUIDE.md`](MOVIESENTIMENT_BUILD_GUIDE.md) §3 — `src/moviesentiment/{data,models,eval,serve,monitor}/`, `tests/`, `deploy/`, `scripts/`.
- DVC initialised, `dvc.yaml` placeholder, `params.yaml` with the train/val/test split seed.
- Pre-commit (`ruff`, `black`, `mypy`, `bandit`) and the empty `.github/workflows/` directory.

The guide itself ([`MOVIESENTIMENT_BUILD_GUIDE.md`](MOVIESENTIMENT_BUILD_GUIDE.md)) was authored before the scaffold and acts as the source of truth for **what to build, when, and why**. It is a 758-line spec with a day-by-day plan, dependency list, code skeletons, README template, and interview talking-points outline. Every later phase traces back to a section in that file.

Within **the first 13 hours of calendar time** (commits `e83f696` 05:27 IST → `7ceafe8` 18:15 IST, all on 2026-05-24) the data pipeline was working end-to-end. The `day-N` tags in commit messages refer to the build-guide plan days, **not** calendar days — the 21-day plan was compressed into 5 calendar days of intense work:

- `8541b52` — ported the IMDb scraper from the author's pre-existing `Imdb_review_scrapper-2026-` repo into `src/moviesentiment/data/scrape.py` and added the `scrape` DVC stage.
- `63ee20c` — a hard pivot the same day: the live IMDb GraphQL endpoint sits behind Akamai WAF and was unreliable from the CI runner; added a HuggingFace `imdb` dataset fallback. `params.yaml::scrape.source` toggles `"hf" | "live"`. This was the first "principled fallback" decision and set the pattern for later infra choices.
- `4efc855` — first tests: `tests/test_clean.py` + `tests/test_split.py`.
- `e20d259` — [`docs/external_setup.md`](docs/external_setup.md), the manual-step checklist (accounts, secrets, DVC remote, AWS).
- `8c0b876` then `5e49092` (same day) — DVC remote configured: started on Google Drive, switched to S3 within hours when the GDrive OAuth flow proved fragile in CI.
- `7ceafe8` (v0.1-baseline, day 5) — full eval rigor: confusion matrix, ROC, PR curve, calibration plot, per-class report, [`docs/error_analysis.md`](docs/error_analysis.md) with five FPs + five FNs.

By the end of day 5 the v1 baseline (TF-IDF + Logistic Regression) trained reproducibly, the metrics were logged to MLflow, and the docs scaffold was in place.

---

## 3. Full build journey, chronological

Phases align with the git tags. Each phase ends with a tagged release. The "days" below are **plan days** from the build guide; the **calendar date** column is the actual ship date.

### Phase 0 — Scaffold and data pipeline (`v0.0-scaffold`, plan days 1–3, calendar 2026-05-24)

- `e83f696` scaffold
- `8541b52`–`63ee20c` scraper + two-source fallback (live IMDb GraphQL with HF dataset fallback)
- `4efc855` end-to-end tests for `clean_reviews` + `split_dataset`
- `e20d259` external setup doc
- `8c0b876` → `5e49092` DVC remote: GDrive → S3

### Phase 1 — Baseline + eval (`v0.1-baseline`, plan day 5, calendar 2026-05-24)

- `7ceafe8` — `src/moviesentiment/eval/metrics.py` with ROC, PR, calibration; per-class report; slice analysis by review length.
- Baseline numbers (held-out 5k test, IMDb 50K): TF-IDF (50k features, 1-2 ngrams) + LR (`C=1.0`, `class_weight=balanced`) → **Accuracy 0.904, Macro F1 0.904, ROC-AUC 0.967**.

Phases 0 + 1 — nine commits, full data pipeline + baseline + eval — all landed in a single calendar day.

### Phase 2 — Transformer + ONNX + serving (`v0.2-serving`, plan days 6–10, calendar 2026-05-24 to 2026-05-25)

- `5c0c644` — `src/moviesentiment/models/transformer.py` with HF `Trainer`; first run via Colab notebook (no local GPU available).
- `ac50361` — ONNX export + INT8 quantization pipeline (`src/moviesentiment/models/onnx_export.py`).
- `6416530` — ONNX inference engine + Prometheus metrics (`prediction_confidence`, `prediction_class_total`, `model_version_info`).
- `92eefc1` (v0.2-serving) — multi-stage Dockerfile + `deploy/docker-compose.yml` with provisioned Prometheus + Grafana.

Latency (raw ONNX, batch 2, max_length 128, n=100, CPU):

| Variant | p50 | p95 | Size |
|---|---|---|---|
| FP32 ONNX | 12.3 ms | 18.0 ms | 256 MB |
| INT8 ONNX | 6.9 ms | 22.8 ms | 64 MB |

**INT8 speedup: 1.8× at p50** with <1% accuracy drop ([`docs/benchmarks.md`](docs/benchmarks.md)).

### Phase 3 — CI/CD, deploy, monitoring (`v0.3-deployed`, plan days 11–15, calendar 2026-05-25)

- `9de74e6` — `.github/workflows/ci.yml`: `ruff` + `black` + `mypy` + `pytest` + Docker build to GHCR, `workflow_call`-reusable.
- `9350c0e` — Locust load test scenarios (`scripts/load_test.py`); reference numbers in [`docs/loadtest.md`](docs/loadtest.md).
- `d340772` — Fly.io config (`deploy/fly.toml`) + a Fly deploy workflow. *Later replaced — see §9.*
- `01956dc` (v0.3-deployed) — drift detection workflow + retraining pipeline; `.github/workflows/drift.yml` runs Evidently weekly, triggers `train.yml` when `drift_share > 0.3`.

**The SageMaker training fight.** Between `5c0c644` and `4239d95` the transformer trained on Colab; the move to SageMaker for the v1 fine-tune surfaced a chain of dependency-pinning problems that took six commits to resolve:

- `5a81a00` — exclude ONNX dirs from the source bundle (SageMaker source-upload size limit).
- `a9270f4` — `dvc pull` ONNX models before Docker build, not inside it (faster + cacheable).
- `728702d` (v1.0) — rename DVC remote `myremote → s3remote` across all workflows; slim sagemaker source bundle.
- `fd232d5` — expose pip output + inject `src/` into `sys.path` before import (SageMaker container path issue).
- `2e2c468` — `pip install --no-deps` so the user-requirements step did not clobber the SageMaker container's CUDA-built `torch`.
- `476ab37` — cap `transformers <5.0` (5.x drops `optimum-onnx` compat used by the export pipeline).
- `1f02146` — install `accelerate>=0.26.0` (HF `Trainer` runtime requirement).
- `4239d95` — first **real** SageMaker-trained DistilBERT artefacts land + DVC-tracked. Test-set Macro F1 = **0.939**.

### Phase 4 — ECR + ECS pivot, mypy/lint cleanup (between `v1.0` and `v2.0`, calendar 2026-05-25 to 2026-05-26)

- `94e4951` — *push image to ECR + force ECS deploy (replace Fly deploy)*. Fly.io free tier was running into reliability issues during early load tests (cold-starts on `auto_stop_machines=true` blew p99); migrated to AWS ECS Fargate Graviton (ARM64). The `fly.toml` is kept in `deploy/` for reference but the live deployment is ECS.
- `0f06834` — *re-export ONNX from the fine-tuned classifier, not the base MaskedLM head*. This bug went unnoticed for a day; the live `/predict` was returning low-confidence near-uniform outputs because the ONNX graph was wrapping `DistilBertForMaskedLM` instead of the fine-tuned `…ForSequenceClassification`. Caught when MLflow run F1 didn't match deployed-model accuracy on smoke tests.
- `5da7c1f`, `6bfb583` — ruff import-sort fixes on tests.
- `fbb1512` — mypy overrides for transformers + refactor inference loop for type inference.
- `d486007` — omit untested model files from coverage; drop threshold to 55%.
- `e170a9a` — `setup-python` action in `build-image` job (Ubuntu 24.04 enforces PEP 668 — system pip blocked).
- `9190a37` — lowercase repo name for GHCR (Docker registries require lowercase).
- `e2f224c` (v2.0) — ECS deploy task definition + `/version` fix + Mermaid architecture diagram in README.

### Phase 5 — v2 Review Intelligence multi-task (calendar 2026-05-26 to 2026-05-28, ~48 hours)

This is a separate mini-build inside the project. Everything from `e5a53cc` to `0f95416`.

- `e5a53cc` — workspace cleanup + the v2 plan landed as `docs/future_improvements.md` (§v2 section).
- `639ec9c` — multi-task skeleton: `src/moviesentiment/models/multitask.py` with shared DistilBERT encoder + 5 linear heads on the `[CLS]` representation.
- `271ffd6` — multitask ONNX export (`onnx_export_multitask.py`), hourly insights Lambda (`deploy/lambda/insights_aggregator.py`), DVC + CLI wiring.
- `e151d2a` — SageMaker launcher for multi-task (`scripts/sagemaker_launch_multitask.py`) + JSONL spoiler loader (Kaggle IMDB Spoiler Dataset).
- `fc6de55` — three fixes in one commit: collate dtypes; S3 bootstrap for multitask ONNX (`MultiTaskInferenceEngine.from_s3`); `Settings.model_config["extra"] = "ignore"` so `.env` can carry AWS_*/KAGGLE_* keys without Pydantic rejecting them.
- `0f17cf2` — None-safe collate (attempt 2): `_as_int`, `_as_int_list`, `_as_float` helpers guard against `Could not infer dtype of NoneType`.
- `8baf104` — collate fix (attempt 3, the one that worked): replaced `datasets.concatenate_datasets` / pyarrow concatenation with a plain Python list-of-dicts `_ListDataset`, because pyarrow quietly coerces sentinel `-100` and `NaN` to `null`, which surfaces as `NoneType` in `torch.tensor`.
- `617019c` — mypy cleanup: explicit re-export of `settings`, `Dataset[dict[str, Any]]` generic.
- `61647b8` — `# nosec` annotations on the multi-task export's `AutoTokenizer.from_pretrained` and `torch.load` (local DVC paths, not Hub identifiers).
- `29fa078` — *`chown /app to app user so S3 bootstrap can write multitask ONNX`*. The Fargate task ran as `app` (uid 1000), but `/app/models` was created root-owned by earlier `COPY` lines, so the lifespan `from_s3` could not write the v2 artefact on first boot. Added `RUN chown -R app:app /app` before `USER app`.
- `ace25b3` — pushed test coverage 76% → **86%**, raised the CI gate to 85% (`tests/test_api_loaded.py`, `tests/test_async_api.py`, `tests/test_cli.py`).
- `5a7e594` — widen tests mypy override (`arg-type`, `unused-ignore`).
- `0f95416` (HEAD) — `tests/__init__.py` so the `tests.*` mypy override pattern actually applies.

### Production verification

The v2 multi-task model trained on SageMaker on-demand (spot quota was 0 for `ml.g4dn.xlarge` in `ap-southeast-2`). **Five submissions were needed** — the first four failed on container-side issues (dependency / path / data-staging) and were fixed iteratively from the SageMaker job logs (logs since deleted to control CloudWatch cost). The fifth submission, base job name `moviesentiment-multitask-2026-05-26-22-07-16-892`, ran to `Completed` and produced the checkpoint. On-demand rate ≈ $0.526/hr × ~1.4 hr → **~$0.75 for the successful run**; the four failed runs were short (fail-fast on entrypoint errors) so total v2 SageMaker spend is in the same ballpark. Multi-output INT8 ONNX exported with the legacy TorchScript path (`dynamo=False`) because the new dynamo exporter emits external-data files that break `quantize_dynamic` with `[ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (768) vs (2)` on multi-output graphs. The exported artefact was DVC-pushed AND uploaded to `s3://moviesentiment-dvc-soumya/multitask_onnx/` so the Fargate lifespan can bootstrap it on boot.

Local artefact sizes (verified on disk):

| File | Size |
|---|---|
| `models/distilbert_multitask_onnx/model.onnx` (INT8) | **63.7 MB** |
| `models/distilbert_multitask_onnx/model_fp32.onnx` (FP32) | **253.4 MB** |
| `tokenizer.json` | 695 KB |
| `vocab.txt` | 226 KB |

Live verification across the session captured five distinct ECS task ENI IPs as Fargate restarted the task during deploys: `54.206.111.36`, `15.135.233.125`, `3.106.121.110`, `13.211.172.87`, `3.25.226.240`. Each was reachable on port 8000 with all five `/analyze` heads returning. Spoiler probability on the test phrase *"hero dies, villain wins"* was ~0.80.

---

## 4. Initial plan vs current reality

The plan in [`MOVIESENTIMENT_BUILD_GUIDE.md`](MOVIESENTIMENT_BUILD_GUIDE.md) §5 is a 21-day week-by-week schedule ending at `v1.0`. Reality landed differently:

| Area | Plan | Reality | Reason for divergence |
|---|---|---|---|
| Python version | 3.11 (`pyproject.toml >=3.11,<3.12`) | 3.10 (`requires-python = ">=3.10"`, mypy `python_version = "3.10"`) | Local interpreter on the dev machine was 3.10; the build guide's 3.11 lower bound caused install failures. Loosened to ≥3.10 rather than upgrade the local stack. |
| Env manager | `uv` preferred, pip-tools fallback | `uv` for install, `scripts/compile_requirements.sh` to compile lockfile | Plan held. |
| Data versioning | DVC with GDrive or S3 | DVC S3 (`s3remote`) | GDrive remote tried first (`8c0b876`); switched to S3 the same day (`5e49092`) because the GDrive OAuth flow is fragile in CI runners. |
| Deploy target | Fly.io preferred, HF Space fallback | **AWS ECS Fargate ARM64 Graviton** | Fly.io was set up (`d340772`) and partially working, but cold-starts under `auto_stop_machines=true` blew p99, and the migrated container ran cheaper per-month on a single 0.25 vCPU / 1 GB ARM64 Fargate task with `aws ecs update-service --force-new-deployment` rolling on every CI green main push. `deploy/fly.toml` is retained for reference. |
| Model | Just DistilBERT FT | **DistilBERT FT + v2 multi-task (5 heads)** | Added in §11 of the guide (stretch goals) and called out as "interview value 5/5". Trained for $0.75, hosted at no recurring cost beyond the existing Fargate task. |
| MLflow backend | SQLite | SQLite | Plan held. Production would use Postgres + S3 (called out in §10 README and model card). |
| Drift detection | Evidently 0.4 API | Evidently 0.7 with a `legacy` fallback path | Evidently 0.7 relocated the v0.4 API under `evidently.legacy.*`; `src/moviesentiment/monitor/drift.py` tries the current namespace first, falls back to legacy. |
| Coverage gate | 70% | **85%** (raised twice: 55% → 75% → 85%) | The first push to merge dropped the gate to 55% during the transformer integration phase (`d486007`); raised to 75% during phase 4 stabilisation; pushed to 85% on day 21 along with `tests/test_api_loaded.py`, `tests/test_async_api.py`, `tests/test_cli.py`. |
| Production input capture | 10% random sampling | **Reservoir sampling (Vitter Algorithm R, k=1000)** | Random sampling biases toward early traffic; reservoir guarantees uniform coverage at fixed memory. |
| Async path | Not in plan | `POST /predict/async` → SQS → Lambda → DynamoDB | Added during phase 4 polish for the "scale 100×" interview line. |
| Observability | Prometheus + Grafana | Prometheus + Grafana + **OpenTelemetry traces to Grafana Cloud free tier**, structlog with request-id correlation | Tracing was added to break down the 5 ms HTTP-vs-ONNX gap surfaced by the load test (see [`docs/loadtest.md`](docs/loadtest.md)). |
| Frontend | "screenshots in README is fine" | Self-contained `frontend/index.html` mounted at `/ui/` on the same Fargate task | Zero added cost; lets interviewers click instead of curl. |
| Distillation teacher (ABSA) | Phase-5 plan | Deferred; aspect head is currently random-init | Out of $0.24 v2 budget; the joint loss tolerates `-100` ignore values on the aspect labels so the head trains only via the joint-supervision passes that include ABSA labels. |

The biggest single divergence is Fly.io → ECS Fargate. That decision flowed downstream: ARM64 Graviton (cheaper than X86_64), separate `ecsTaskRole` for S3 read on the `multitask_onnx/*` prefix, `aws ecs update-service --force-new-deployment` in the CI build job, and the S3 bootstrap pattern in `MultiTaskInferenceEngine.from_s3` so the v2 artefact rolls independently of the container image.

---

## 5. Architecture and how it works

```mermaid
flowchart TB
    subgraph "Data"
        IMDB["IMDb live (Akamai WAF)<br/>or HuggingFace dataset"]
        S3[("S3 — DVC remote<br/>moviesentiment-dvc-soumya")]
        PROC["data/processed/<br/>train · val · test"]
    end

    subgraph "Training"
        BASE["TF-IDF + LogReg<br/>(local, &lt;1 min)"]
        SM["SageMaker on-demand<br/>ml.g4dn.xlarge T4 GPU"]
        MLF[("MLflow SQLite<br/>+ Model Registry")]
        ONNXEXP["ONNX export<br/>+ INT8 quantize<br/>(legacy TorchScript path)"]
    end

    subgraph "Serving (AWS ap-southeast-2)"
        ECR[("ECR repo<br/>moviesentiment")]
        ECS["ECS Fargate ARM64<br/>0.25 vCPU · 1 GB · 1 task"]
        API["FastAPI + uvicorn :8000<br/>/predict /analyze /explain<br/>/predict/async /insights"]
        SQS["SQS queue<br/>(async path)"]
        LAM["Lambda async worker<br/>+ hourly insights Lambda"]
        DDB[("DynamoDB jobs table")]
        S3MT[("S3 multitask_onnx/<br/>bootstrap on boot")]
    end

    subgraph "CI / CD"
        GHA["GitHub Actions<br/>ci.yml · train.yml · drift.yml<br/>scrape.yml · weekly_bench.yml · scorecard.yml"]
    end

    subgraph "Monitoring"
        PROM[("Prometheus<br/>scrape /metrics")]
        GRAF["Grafana Cloud<br/>+ local docker-compose"]
        EVID["Evidently 0.7<br/>drift report"]
        OTEL["OpenTelemetry OTLP<br/>→ Grafana Cloud free tier"]
    end

    IMDB -->|weekly cron| S3
    S3 -->|clean + split + validate| PROC
    PROC --> BASE
    PROC --> SM
    BASE --> MLF
    SM --> MLF
    MLF -->|promote| ONNXEXP
    ONNXEXP -->|v1 INT8 in image| ECR
    ONNXEXP -->|v2 INT8 standalone| S3MT
    GHA -->|build ARM64 + push| ECR
    ECR --> ECS
    S3MT -->|from_s3 lifespan| ECS
    ECS --> API
    API -->|/predict/async| SQS
    SQS --> LAM
    LAM --> DDB
    API -->|/predict/result| DDB
    API -->|/metrics| PROM
    PROM --> GRAF
    API -->|OTLP spans| OTEL
    OTEL --> GRAF
    API -->|reservoir sample 1000| EVID
    EVID -->|drift_share &gt; 0.3| GHA
```

### Key components

- **Data ingest** (`src/moviesentiment/data/scrape.py`): two-source — live IMDb GraphQL with HuggingFace `imdb` dataset fallback. Schema: `review_id, movie_id, text, rating, scraped_at`. Label rule: rating ≥7 → positive, ≤4 → negative, 5–6 dropped.
- **DVC pipeline** ([`dvc.yaml`](dvc.yaml)): nine stages — `scrape → clean → validate_data → split → {train_baseline, train_transformer, prep_emotion, prep_helpfulness, train_multitask} → export_multitask_onnx`. Each declares deps + outs + params slice from [`params.yaml`](params.yaml) so DVC re-runs only what changed.
- **Models**:
  - Baseline (`src/moviesentiment/models/baseline.py`): TF-IDF (50k features, 1–2 ngrams) + Logistic Regression. Trains in <1 min on CPU.
  - DistilBERT v1 (`src/moviesentiment/models/transformer.py`): `distilbert-base-uncased` fine-tuned via HF `Trainer`. SageMaker `ml.g4dn.xlarge`. Saved + ONNX-exported + INT8 dynamic-quantized.
  - DistilBERT v2 multi-task ([`src/moviesentiment/models/multitask.py`](src/moviesentiment/models/multitask.py)): shared encoder + five linear heads on the `[CLS]` representation — `sentiment` (2-class), `aspect_head` (5×3 = `acting/plot/visuals/pacing/sound` × `neg/neu/pos`), `emotion_head` (6-class Ekman: `joy/anger/fear/sadness/surprise/disgust`), `spoiler_head` (2-class), `helpfulness_head` (sigmoid regression in [0,1]). Joint loss is task-weighted sum of CE / CE / CE / CE / MSE, ignoring `-100` and `NaN` per-task so each parquet only needs to supervise the task it has labels for.
- **ONNX export** ([`src/moviesentiment/models/onnx_export_multitask.py`](src/moviesentiment/models/onnx_export_multitask.py)): wraps the dataclass-returning module in `_OnnxWrap` so `torch.onnx.export` sees a flat tuple. Five named outputs (`sentiment`, `aspect`, `emotion`, `spoiler`, `helpfulness`) with dynamic batch/seq axes. Uses **legacy TorchScript** exporter (`dynamo=False`) because the new dynamo exporter emits external-data files and breaks `quantize_dynamic` with a shape-inference error on multi-output graphs. INT8 via `onnxruntime.quantization.quantize_dynamic(weight_type=QuantType.QInt8)`.
- **Serving** ([`src/moviesentiment/serve/api.py`](src/moviesentiment/serve/api.py)):
  - FastAPI app with `lifespan` context manager that loads v1 from disk and v2 from S3 if `MS_DVC_BUCKET` + `MS_MULTITASK_S3_PREFIX` are set, else from disk.
  - Per-route `slowapi` rate limits: `/predict` 60/min, `/analyze` 30/min, `/explain` 10/min.
  - Custom request-id middleware (header round-trip for log correlation).
  - CORS configurable via `MS_CORS_ALLOW_ORIGINS`; wildcard auto-disables `allow_credentials` (browsers reject the combination).
  - Reservoir sampler (Vitter Algorithm R, k=1000) on every `/predict` and `/analyze`; periodically flushed to `data/production/recent.parquet`.
- **Async path** ([`src/moviesentiment/serve/async_api.py`](src/moviesentiment/serve/async_api.py)): `POST /predict/async` enqueues to SQS, Lambda runs inference, result lands in DynamoDB; `GET /predict/result/{job_id}` polls. Returns 503 if `MS_SQS_QUEUE_URL` + `MS_JOBS_TABLE` are not configured, so local dev works without AWS plumbing.
- **Monitoring**:
  - Prometheus FastAPI Instrumentator on `/metrics` + custom: `prediction_confidence` histogram, `prediction_class_total` counter, `model_version_info` labelled gauge.
  - Evidently drift on text-length + word-count features; 0.7 with legacy fallback ([`src/moviesentiment/monitor/drift.py`](src/moviesentiment/monitor/drift.py)).
  - NannyML CBPE for label-free F1 estimation ([`src/moviesentiment/monitor/perf_estimate.py`](src/moviesentiment/monitor/perf_estimate.py)).
  - OpenTelemetry OTLP exporter to Grafana Cloud free tier ([`src/moviesentiment/serve/tracing.py`](src/moviesentiment/serve/tracing.py)).
  - Local stack via `docker compose -f deploy/docker-compose.yml up` — Prometheus + Grafana with auto-provisioned dashboard ([`deploy/grafana_dashboard.json`](deploy/grafana_dashboard.json)).
- **Frontend** (`frontend/index.html`): self-contained single-page demo with three-mode toggle (mock keyword classifier · live-auto via public `api.json` · custom IP paste), animated SVG arch flow, live activity strip. Bundled into the same Docker image and mounted at `/ui/`. ~30 KB.

### Why DistilBERT and INT8 ONNX

- DistilBERT is 40% smaller and 60% faster than BERT-base while retaining ~97% of its task performance on sentiment. The accuracy gap to BERT-base is <1% — not worth the latency.
- ONNX Runtime on CPU is ~2× faster than PyTorch eager mode for inference.
- Dynamic INT8 quantization (weights only, FP32 activations) drops the model from ~260 MB to ~64 MB and cuts p50 latency from 12.3 ms → 6.8 ms with <1% accuracy drop. The whole stack is `transformers → optimum/ONNX export → onnxruntime.quantize_dynamic → ORT InferenceSession`.

### Latency optimisation journey (end-to-end)

| Stage | p50 | Notes |
|---|---|---|
| PyTorch eager, FP32 (no ONNX) | ~25 ms | Reference; HF `Trainer.predict` on the same machine. |
| ONNX FP32 (after `optimum.exporters.onnx`) | 12.3 ms | ORT InferenceSession on CPUExecutionProvider. |
| INT8 ONNX (dynamic quantisation, weights only) | **6.8 ms** | `quantize_dynamic(weight_type=QInt8)`. <1% accuracy drop. |
| End-to-end HTTP `/predict` | 12 ms | Adds ~5 ms HTTP + uvicorn + Pydantic + middleware (CORS, request-id, slowapi, structlog, OTel, Prom). |

Where the 5 ms HTTP-side overhead splits (from OTel spans, [`docs/loadtest.md`](docs/loadtest.md)):

- 2.5 ms — uvicorn loop + Pydantic serialisation.
- 1.5 ms — middleware stack (CORS, request-id, slowapi, structlog, OTel).
- 1.0 ms — slowapi key lookup + Prometheus histogram observe.

The multi-task v2 model adds <0.5 ms p50 over v1 (five linear heads over the same `[CLS]` representation; ~100 KB extra weights total).

### Why multi-task with a shared encoder

One forward pass over the shared `distilbert-base-uncased` encoder produces logits for all five heads. The added per-head linear layers are <100 KB combined, so additional latency is <0.5 ms p50 over the v1 binary classifier. Serving cost per request is essentially flat versus v1. No free public API ships this combination — Google NL, AWS Comprehend, and Azure Text Analytics each offer subsets, charged per call.

---

## 6. Cost-cutting measures

Every infra choice was filtered through the **~$13/mo ceiling** the author set at the start. Final monthly cost is ~$6 — well under budget.

| Decision | Saving | Notes |
|---|---|---|
| Fly.io → AWS ECS Fargate **ARM64 Graviton** | ~30% vs X86_64 Fargate | Task definition pins `runtimePlatform.cpuArchitecture=ARM64`. |
| Single Fargate task at **0.25 vCPU / 1 GB** | Cheapest paid Fargate slice (~$6/mo) | Documented headroom in `docs/loadtest.md`: saturates at ~120 RPS, fine for portfolio traffic. |
| **GitHub Actions** for scheduling (cron) | Free tier vs. Airflow / Dagster ($0 vs $30+/mo managed) | `MOVIESENTIMENT_BUILD_GUIDE.md` §10 explicitly flags this as the "right" portfolio call. |
| **SQLite** MLflow backend | $0 vs managed Postgres | Trade-off documented in README §"What I'd do differently". |
| **DVC + S3** for artefact storage | ~$0.05/mo for 50 MB labels + ONNX models | One bucket, no per-call cost. |
| **INT8 ONNX dynamic quantization** | 4× model-size reduction (256 MB → 64 MB) | Smaller image → faster ECS task spin-up → lower Fargate CPU-seconds during cold deploys. |
| **GHCR free tier** for image storage | $0 vs ECR-only | Images push to GHCR + ECR; ECR pulled by Fargate. |
| Frontend bundled into the **same** Fargate task at `/ui/` | $0 vs separate static-hosting bill | ~30 KB delta, no new resource. |
| **Lambda + EventBridge** for hourly insights aggregation | Free tier (1 invocation/hr, <5 s wall clock, <256 MB) | Lightweight TF-IDF + NMF in place of BERTopic (too heavy for Lambda layer). |
| **OpenTelemetry → Grafana Cloud free tier** | $0 vs paid APM | Documented one-time setup in [`docs/grafana_cloud_setup.md`](docs/grafana_cloud_setup.md). |
| **SageMaker on-demand for v2** (spot quota = 0) | One-time ~$0.75 instead of recurring | `ml.g4dn.xlarge` ~$0.526/hr × 1.4 hr. v1 fine-tune was also ~$0.75 one-time. |
| **Reservoir sampler** at fixed k=1000 | Bounded memory regardless of traffic | Prevents Fargate task OOM under load; cheaper than re-sizing the task. |
| **Skip Airflow / Kubernetes** | Avoid platform fixed cost ($30–100/mo cluster floor) | Interview-positive: README "what I'd do differently" calls this out as a senior signal. |
| Container `auto_stop_machines = true` on Fly.io (when active) | Stopped tasks bill at machine-hour rate of $0 | Migrated away from Fly.io because cold-starts hurt p99 — but kept the principle for the Lambda async worker (scale-to-zero by default). |

The **v2 multi-task addition added $0.75 one-time** + ~$0.01/mo recurring. No new long-running services were introduced — the artefact lives in S3 and bootstraps into the existing Fargate task on boot.

---

## 7. Hosting at minimum cost

**Production runtime path:** `git push main` → CI builds ARM64 Docker image → pushes to GHCR + ECR → `aws ecs update-service --force-new-deployment` rolls the single Fargate task.

### Components live in `ap-southeast-2`

| AWS resource | Purpose | ~Monthly cost |
|---|---|---|
| ECS Fargate task (ARM64, 0.25 vCPU, 1 GB) | Serves FastAPI :8000 | **~$6** |
| ECR repo `moviesentiment` | Docker image registry | <$0.10 (50 MB image, free tier) |
| S3 `moviesentiment-dvc-soumya` | DVC remote + v2 ONNX bootstrap + reservoir parquet | <$0.05 (~50 MB hot data) |
| CloudWatch Logs `/ecs/moviesentiment` | Container stdout/stderr | <$0.10 |
| IAM roles: `ecsTaskExecutionRole`, `ECSTaskRole-moviesentiment` | Task execution + scoped S3 read on `multitask_onnx/*` | $0 |
| Lambda async worker + insights aggregator | Async predict + hourly insights | $0 (free tier) |
| SQS queue + DynamoDB jobs table | Async path glue | $0 (free tier at portfolio QPS) |
| **Total** | | **~$6.25/mo** |

### Why this is cheap

- **No always-on database or VM.** Everything that can be event-driven is (Lambda + SQS + DynamoDB).
- **Single shared Graviton task.** No autoscaling group, no LB. The Fargate task is reachable via its public ENI IP; the IP rotates on task restart and is published via `make smoke-test` which resolves the current one via `aws ecs describe-tasks` + `aws ec2 describe-network-interfaces`.
- **DVC over S3** instead of a model-serving service (no $40+/mo SageMaker endpoint, no MLflow-server cost). The model registry is just files in S3 keyed by DVC hashes.
- **CI minutes** stay free because builds are reasonably short (most heavy steps are cached: `uv` cache, BuildKit cache, dvc pull only the small ONNX `*.dvc` files).

### AWS free-tier headroom (which limits actually bind)

| Service | Free-tier monthly allowance | Project usage at portfolio traffic | Bind risk |
|---|---|---|---|
| Lambda | 1M requests + 400k GB-seconds | Insights aggregator: 720 invocations/mo (`rate(1 hour)`), ~5 s × 256 MB ≈ 0.9k GB-s | None |
| SQS | 1M requests | Async path: ≤ 1 message per `/predict/async` call, manual smoke-tests only | None |
| DynamoDB | 25 GB storage + 25 RCU + 25 WCU on-demand | Jobs table: <1 GB, <1 RPS sustained | None |
| CloudWatch Logs | 5 GB ingest, 5 GB storage | ECS task stdout ~10 MB/day → 300 MB/mo | None |
| ECR | 500 MB | One image, ~250 MB | None |
| Fargate | NOT free-tier | 0.25 vCPU × 1 GB × 730 hr | **Only binding cost.** |
| S3 | First 5 GB stored + 20k GET + 2k PUT | ~50 MB stored, hundreds of GETs (CI + Fargate bootstrap) | None |

The only paid line is the Fargate task itself. Everything else stays well under free-tier. The project would have to do >100× current traffic before any other service started billing.

### Tradeoffs we explicitly took

- Public ENI IP rotates on task restart → not a stable URL. Mitigation: `make smoke-test` Makefile target resolves the current IP. Real production would put this behind an Application Load Balancer (~$18/mo) or API Gateway HTTP API ($1 per million requests) — both above the budget ceiling.
- Single task = single point of failure. `min_machines_running = 0` semantics on Fly.io (auto-stop) traded p99 for $0 idle; ECS rolling deploys (`--force-new-deployment`) cover image rollout but not zone failure.
- No HTTPS terminator. The endpoint is HTTP-only on port 8000. Adding TLS requires the ALB or CloudFront, again above the budget.
- v2 multi-task ONNX is bootstrapped from S3 on first boot, adding ~5 s to cold-start time. Accepted because the image stays at v1 size and the artefact can roll independently of code.

---

## 8. Scaling

The system is **designed for low-cost steady-state, with a clear next-step ladder** if traffic actually arrives. Documented in [`docs/loadtest.md`](docs/loadtest.md).

**Measured ceiling (ARM Fargate, 0.25 vCPU, 1 GB, 1 task):**

| Endpoint | p50 | p95 | p99 | Sustained RPS |
|---|---|---|---|---|
| `/predict` (single) | 12 ms | 28 ms | 41 ms | ~85 |
| `/predict` (batch-4) | 24 ms | 52 ms | 78 ms | ~22 batches/s (~88 reviews/s) |
| `/healthz` | 2 ms | 4 ms | 8 ms | n/a |

At ~120 RPS p99 climbs above 200 ms — request queueing inside the uvicorn `--workers 1` loop. The next-step ladder:

1. **Vertical scale** — bump Fargate task to 0.5 vCPU (+$3.50/mo) → ~2× headroom.
2. **Horizontal scale via ECS service** to 2 tasks (+$6/mo) → linear scale-out with the public ENI behind an ALB.
3. **Burst spikes → SQS async path** already in the code. `POST /predict/async` decouples client latency from inference throughput; Lambda scales-to-zero when idle.
4. **Bulk re-scoring of large corpora** — would move to an offline DVC stage rather than the live API.
5. **At 10×–100× live traffic**, the README §"What I'd do differently" calls out the production-grade swaps: Triton Inference Server (dynamic batching, GPU support), Redis cache for identical-text dedup, HPA on QPS-derived CloudWatch alarm, per-route auth + per-user rate limit. None of those changes are in code today — they're documented as the next step, which is itself the interview signal.

**Inference-side scaling** (what the v2 multi-head model bought us): five heads in one ONNX forward pass adds <0.5 ms p50 over v1, instead of five sequential model calls (~5 × 7 ms = 35 ms). The shared encoder is the throughput-multiplier.

**Training-side scaling:** SageMaker `ml.g4dn.xlarge` finished the v2 multi-task fine-tune in 1.4 hr on 130k examples × 2 epochs. Spot would have cut that to ~$0.16/hr but spot quota for the instance type was 0 at run time. The retraining loop reuses the same launcher and is wired into `train.yml` cron + `workflow_dispatch` + the drift-triggered fan-in.

---

## 9. Problems faced and how we overcame them

Each problem is a real commit. The full set is documented inline in commit history; this section captures the load-bearing ones.

### 9.1 Live IMDb scraping behind Akamai WAF (`63ee20c`)

The original scraper hit the live IMDb GraphQL endpoint. Akamai WAF blocked the CI runner. Solved by adding a HuggingFace `imdb` dataset fallback, gated by `params.yaml::scrape.source` (`"hf" | "live"`). Lesson: every external data source needs a reproducible fallback if it's behind a WAF / rate-limiter.

### 9.2 DVC remote churn — GDrive → S3 (`8c0b876` → `5e49092`)

Tried GDrive first because it's free and OAuth-flow-only on a workstation. CI runners hit the OAuth refresh path on every job and failed. Migrated to S3 within hours. Lesson: pick the remote your CI can authenticate against without an interactive flow.

### 9.3 SageMaker dependency hell (`5a81a00`–`1f02146`, six commits)

Bringing the v1 transformer training to SageMaker surfaced a chain:

- Source bundle too big (ONNX dirs and prod logs included) — `5a81a00` excludes them.
- SageMaker's pre-built PyTorch image has CUDA-built `torch`; pip-installing the project requirements clobbered it with a CPU build. Solved with `pip install --no-deps` plus a separate `pip install accelerate` step (`2e2c468` + `1f02146`).
- `transformers 5.x` removed `optimum-onnx` compat used by the export script (`476ab37` caps `<5.0`).
- `src/` was not on `sys.path` inside the container (`fd232d5` injects it).
- DVC remote naming mismatch — `myremote` vs `s3remote` across workflows (`728702d`).

Lesson: **Cloud training is mostly path / version plumbing**. The model code is the easy part.

### 9.4 ONNX exported the wrong head (`0f06834`)

For ~24 hours the deployed `/predict` returned low-confidence near-uniform outputs. The ONNX export wrapper had grabbed the base `DistilBertForMaskedLM` head instead of the fine-tuned `DistilBertForSequenceClassification` head. Caught when MLflow run F1 (0.939) didn't match smoke-test accuracy on the deployed model. Fix: re-export ONNX from the classifier-head model file specifically. Lesson: smoke-test the deployed endpoint, not just MLflow metrics.

### 9.5 Multi-output ONNX quantization fails on dynamo exporter (v2)

The new `torch.onnx.export(..., dynamo=True)` path emits an external-data layout that `onnxruntime.quantization.quantize_dynamic` cannot shape-infer through, raising `[ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (768) vs (2)` on the helpfulness output (which collapses to (B,) after `squeeze`). Fix: stick with the legacy TorchScript exporter (`dynamo=False`). Documented in the source file's docstring so future readers don't try to "upgrade" the call.

### 9.6 pyarrow turned `-100` and `NaN` into NoneType (v2, three commits)

The v2 trainer originally used `datasets.Dataset.from_dict` + `concatenate_datasets`. Each per-task parquet had ignore-value sentinels (`-100` for CE-skipped labels, `NaN` for the helpfulness regressor), but pyarrow's schema unification silently coerced them to `null`, which surfaces in the DataLoader collate as `Could not infer dtype of NoneType`.

- `fc6de55` — first attempt: cast in collate. Didn't catch every path.
- `0f17cf2` — second attempt: explicit `_as_int`, `_as_int_list`, `_as_float` helpers per label.
- `8baf104` — **the fix that worked**: replace the `datasets` library entirely with a plain Python list-of-dicts `_ListDataset`. Pyarrow never touches the values; Python ints stay ints, `float('nan')` stays a float.

Lesson: when sentinel values need to travel through a typed schema layer, **don't trust the schema layer**. Keep the values in pure Python until the collate function.

### 9.7 Fargate `app` user couldn't write the v2 bootstrap dir (`29fa078`)

The Dockerfile sets `USER app` (uid 1000), but `/app/models` was created during the `COPY` lines as root-owned. The lifespan's `MultiTaskInferenceEngine.from_s3` tries to `mkdir(parents=True, exist_ok=True)` a new subdirectory and got `PermissionError: '/app/models/distilbert_multitask_onnx'`. Fix: `RUN chown -R app:app /app` immediately before `USER app`. Lesson: Docker layer ordering matters — chown after every `COPY` that lands in a path the non-root user will write to.

### 9.8 ECS task definition platform mismatch

First push to ECR was X86_64, task definition was ARM64 (Graviton). ECS service silently failed to start tasks: *"image Manifest does not contain descriptor matching platform 'linux/amd64'"*. Fix: switch BuildKit to `linux/arm64` via `docker/setup-qemu-action`. The task definition rev 5 also pins `runtimePlatform.cpuArchitecture=ARM64` so the platform is asserted at registration time, not discovery time.

### 9.9 Encoding crash on ONNX export (Windows)

`torch.onnx.export` verbose-prints a checkmark (`✅`). Windows console default code page (`cp1252`) doesn't carry it → `UnicodeEncodeError`. Fix: `PYTHONIOENCODING=utf-8` set in the export script's environment. Trivial but cost ~30 minutes the first time.

### 9.10 Pydantic rejected AWS keys in `.env` (v2)

When the AWS / Kaggle keys landed in `.env` for the SageMaker submission, Pydantic raised `ValidationError: Extra inputs are not permitted` because `Settings` had no fields for them. Fix: `model_config["extra"] = "ignore"` so unknown env vars pass through harmlessly. Lesson: `.env` is shared across consumers — be permissive on unknown keys.

### 9.11 Mypy `tests.*` override never applied

The pyproject `[[tool.mypy.overrides]] module = ["tests.*", "conftest"]` block had no effect because the `tests/` directory had no `__init__.py`, so mypy didn't treat the test files as a package. Fix: add an empty `tests/__init__.py` (`0f95416`). The override now matches.

### 9.12 Pre-commit black mutated files between commits

Local `pre-commit` runs black, which sometimes reformats files between `git add` and `git commit`, then the hook fails because the index is stale. Workflow: run `pre-commit run --all-files` manually first, *then* `git add -A && git commit`. The `--no-verify` flag is reserved for cases the user explicitly approves (the author's CLAUDE.md feedback memory).

---

## 10. Why this is better than alternatives

Honest comparison against the realistic alternatives a portfolio project chooses between.

### vs. "just train a model in a notebook"

The notebook portfolio is the floor. This project covers everything past the notebook:

- Reproducible from `dvc repro` + `make serve` on any machine.
- CI builds + deploys on every push.
- Metrics + drift + retraining loop closes.
- Model + data lineage in one place.

The notebook portfolio fails the "can I `curl` it" test that interviewers reliably ask. This one passes.

### vs. Streamlit / HF Space demo

HF Space is one Docker image, end of story. This project ships:

- Multi-model serving with per-model registry hashes (`/version`, `model_version_info` gauge).
- Per-route rate limiting + CORS + request-id correlation + Prometheus + OTel + structlog.
- Async inference (SQS + Lambda + DynamoDB) decoupling client latency from inference.
- Reservoir sampling so production drift can be measured against the training distribution.
- A documented load-test ceiling and a next-step scaling ladder.

An HF Space cannot run async paths, Lambda, or cross-AWS-service plumbing. It's a UX layer; this is an infra layer.

### vs. Google Natural Language / AWS Comprehend / Azure Text Analytics

For the v2 multi-head endpoint specifically:

| Service | Sentiment | ABSA | Emotion | Spoiler | Helpfulness | Per-movie topics | $ |
|---|---|---|---|---|---|---|---|
| Google NL | ✓ | partial (entity-level only) | ✗ | ✗ | ✗ | ✗ | $0.50–$2 / 1k calls |
| AWS Comprehend | ✓ | key phrases only | ✗ | ✗ | ✗ | ✗ | per-call |
| Azure Text Analytics | ✓ | opinion mining (thin ABSA) | ✗ | ✗ | ✗ | ✗ | per-call |
| **MovieSentiment v2** | ✓ | ✓ (5 aspects × 3-class) | ✓ (6-class Ekman) | ✓ | ✓ (regression) | ✓ (hourly NMF aggregate) | ~$6/mo flat |

The headline isn't capability parity — it's the **combination in one forward pass at flat cost**. A domain-specific multi-task model can do this; a generic per-call API cannot without batching across endpoints.

### vs. paying for a managed inference endpoint

Hugging Face Inference Endpoints, SageMaker Real-time, Replicate, Modal, Beam:

- All charge $20–$80+/mo at the cheapest persistent tier.
- All hide the model-serving plumbing (rate limit, request id, drift), which is exactly the surface an MLOps interview probes.
- The cost ceiling here was self-imposed at $13/mo specifically to **force the engineering decisions** that demonstrate that surface.

This project lands at **~$6/mo with the plumbing exposed and documented**, with `docs/interview_talking_points.md` mapping every decision back to a tradeoff.

### Honest weaknesses

- No HTTPS terminator (Fargate ENI is HTTP-only on port 8000).
- IP rotates on task restart — not a stable demo URL without `make smoke-test` or a CloudFront/ALB layer.
- ABSA head in v2 is currently random-init because the distillation teacher run was deferred. Joint loss tolerates this, but ABSA outputs are unreliable until the teacher distillation lands.
- Frontend mode-toggle has a "mock keyword classifier" path; this is honestly labelled in the UI but adds confusion if a viewer doesn't read carefully.
- The retraining loop's drift threshold (`drift_share > 0.3`) and F1-improvement threshold (`>= 0.005`) are hard-coded. Production would have these in a runtime config.
- v2 multi-task model is **not** covered by the test suite end-to-end (it requires the ONNX artefact on disk; covered by `tests/test_multitask.py` for schema and 503-fallback, not the full pipeline). Listed as a coverage omit in `pyproject.toml::tool.coverage.run.omit`.

---

## 11. Current state

As of 2026-05-28 (HEAD = `0f95416`):

- **CI**: green on `main`; 85% coverage gate enforced.
- **Live API**: ECS Fargate task in `ap-southeast-2`; current ENI IP can be resolved via `make smoke-test`. `/predict`, `/analyze`, `/explain`, `/insights`, `/predict/async`, `/healthz`, `/readyz`, `/version`, `/metrics`, `/sample`, `/ui/` all wired and reachable. Author-confirmed: `/analyze` returns all five heads live (S3-bootstrap path active); async path (`/predict/async` + `/predict/result/{job_id}`) backed by a real SQS queue + DynamoDB table; hourly insights Lambda backed by a real EventBridge schedule rule.
- **v1 DistilBERT INT8 ONNX**: deployed (bundled in image, 64 MB), Macro F1 0.939, p50 6.8 ms.
- **v2 multi-task INT8 ONNX**: deployed (S3-bootstrap on lifespan, 63.7 MB), all five heads validated live.
- **MLflow**: SQLite store with v1 and v2 runs logged; `Production` stage points at the deployed model.
- **Drift + retraining**: weekly cron wired, threshold `drift_share > 0.3`; triggers `train.yml` which runs baseline + transformer retrain, compares F1, promotes if ≥0.5% improvement, redeploys.
- **OpenTelemetry traces**: flowing to Grafana Cloud free tier.
- **Documentation**: README + model card + datasheet + benchmarks + load-test + interview talking points + future improvements + drift reports archive.
- **Security**: bandit + pip-audit on every CI run; OSSF Scorecard weekly; documented residual risks in `SECURITY.md` (transformers PYSEC-2025-217, starlette PYSEC-2026-161, fastapi MAL-2026-4750 false positive, diskcache CVE-2025-69872, pyOpenSSL CVE-2026-27448/27459 — each with an "accepted because…" justification).
- **Tags**: `v0.0-scaffold`, `v0.1-baseline`, `v0.2-serving`, `v0.3-deployed`, `v1.0`, `v2.0`. Release branches `release/v1.0` and `release/v2.0` pushed.
- **Credential hygiene (confirmed rotated)**: the AWS access key (`AKIAVOXZ4QDJXESRBUDW`) and Kaggle token (`KGAT_bb2819f25d3106c009349947fe15dee4`) that were pasted into the conversation during the SageMaker submission have been rotated. Verified during this report run — `aws sts get-caller-identity` against those credentials returns `InvalidClientTokenId`. The keys remain in the local `.env` (gitignored) for record only; they no longer authenticate to AWS.

---

## 12. Things a thorough engineer would expect to see

### Security

- Threat model and mitigations enumerated in [`SECURITY.md`](SECURITY.md). Covered: untrusted input → RCE, DoS, log injection via headers, HF namespace takeover (pinned `revision`), supply-chain (pip-audit + Dependabot), CORS abuse, PII leakage in logs, container privilege escalation, TLS in DVC S3 fetch.
- Pydantic validation at every API boundary; hard caps via `settings.max_batch_size` (32) and `settings.max_text_length` (5000).
- `slowapi` per-IP rate limiting on every inference route.
- **Shared-secret API key**: `/predict` + `/analyze` honour `X-API-Key` against `MS_API_KEY`. Empty key (default) = demo mode (no auth); production sets the key + clients send the header. Mismatched/missing → 401.
- No `pickle` / `exec` / `eval` anywhere in the serving path. ONNX-only inference (no torch.load at request time).
- Pre-commit hooks: `bandit`, `detect-private-key`.
- CI: `bandit -c pyproject.toml -r src -ll` + `pip-audit --skip-editable` with documented `--ignore-vuln` entries for accepted residual CVEs.
- Container runs as `app` (uid 1000), read-only-filesystem-friendly.

### Testing

- **15 test files** under `tests/` (`test_*.py`): data (clean / split / scrape / validate), model (inference / onnx_export / multitask / explain), API (api / api_loaded / async_api / cli), monitoring (drift), reservoir, hypothesis-based property tests on clean (`tests/test_clean_hypothesis.py`).
- 85% coverage gate enforced in CI (`pytest --cov=moviesentiment --cov-fail-under=85`).
- `pytest -m "not slow"` runs the fast suite; `-m slow` runs the ONNX-export end-to-end test (also in `weekly_bench.yml`).
- Test design: stubs the engine via `_StubEngine` for API tests; uses Typer `CliRunner` for CLI; fake AWS clients for the async path. Tests run in <30 s on the dev box.

### Performance

- p50 6.8 ms raw ONNX → 12 ms full HTTP. The ~5 ms HTTP overhead is broken down in [`docs/loadtest.md`](docs/loadtest.md): 2.5 ms uvicorn + pydantic, 1.5 ms middleware stack, 1 ms slowapi key lookup + Prom histogram.
- INT8 dynamic quantization buys 1.8× speedup at p50.
- Reservoir sampling at fixed k=1000 bounds memory regardless of traffic.
- Five-head multi-task adds <0.5 ms p50 (shared encoder).

### Developer experience

- `make ci-local` runs the same lint + type + test gate as CI.
- `make smoke-test` resolves the live Fargate IP and curls `/predict` end-to-end.
- `dvc repro` rebuilds anything that changed.
- `make docs` builds `workflow.pdf` + `quickstart.pdf` from LaTeX.
- Pre-commit on every commit catches the lint / type / security issues before they hit CI.
- Structured logs (`structlog`) with `x-request-id` round-trip so a curl with `-H 'X-Request-ID: abc123'` shows up in the local container logs and OTel spans.

### Documentation

- [`README.md`](README.md) — quickstart, results table, reproduce, monitoring, CI/CD, endpoints, what'd be different in prod, roadmap.
- [`MOVIESENTIMENT_BUILD_GUIDE.md`](MOVIESENTIMENT_BUILD_GUIDE.md) — the original spec; retained because it's part of the project history.
- [`docs/model_card.md`](docs/model_card.md) — Mitchell et al. template: intended use, factors, metrics (with bootstrap CI), per-length / per-confidence / sarcasm / non-English slices, ethical considerations, environmental impact.
- [`docs/datasheet.md`](docs/datasheet.md) — Gebru et al. template.
- [`docs/benchmarks.md`](docs/benchmarks.md), [`docs/loadtest.md`](docs/loadtest.md), [`docs/error_analysis.md`](docs/error_analysis.md), [`docs/grafana_cloud_setup.md`](docs/grafana_cloud_setup.md), [`docs/interview_talking_points.md`](docs/interview_talking_points.md), [`docs/future_improvements.md`](docs/future_improvements.md), [`docs/external_setup.md`](docs/external_setup.md).
- LaTeX docs: `workflow.tex` and `quickstart.tex` — built via `scripts/build_docs.sh` / `.ps1` to PDF.
- `.gitattributes` uses `export-ignore` so GitHub Release tarballs ship the slim runtime set; clone the repo to see everything.

### Technical debt

- Aspect head (v2) is random-init pending ABSA distillation from `yangheng/deberta-v3-base-absa-v1.1`. Score outputs are not meaningful until that lands.
- Helpfulness regressor uses an IMDb-rating proxy; ground truth would come from the live IMDb scraper's vote counts.
- BERTopic was planned for `/insights` topic extraction but replaced with TF-IDF + NMF in the Lambda layer because BERTopic + UMAP + HDBSCAN exceeds the 250 MB Lambda layer size.
- Two ONNX models (`distilbert_onnx` FP32 and `distilbert_onnx_int8`) are both bundled in the image; only INT8 is loaded at serve time. FP32 is kept for `weekly_bench.yml` accuracy-regression checks but could be moved to S3-only.
- No structured config for thresholds (drift 0.3, F1 improvement 0.005) — they're hard-coded in workflow YAML. A config file would let ops tune without a code change.
- Single-task Fargate is a SPOF. Documented in §8 + README "What I'd do differently".

### Cost-control infrastructure

- `scripts/setup_aws_budget.py` (boto3) creates / updates an **AWS Budget at $13/mo** with email alerts at 80% forecast + 100% actual. Idempotent. Run once:
  ```bash
  MS_BUDGET_EMAIL=you@example.com python scripts/setup_aws_budget.py
  ```
  The script is checked-in proof that the cost ceiling exists; it survives an account audit.

### Model-quality additions (landed in this session)

| Capability | File | Status |
|---|---|---|
| Adversarial robustness tests | `tests/test_adversarial.py` | Character perturbation + synonym swap + unicode / whitespace / casing edge cases. 8 tests over a stub engine; swap to the real ONNX engine to get a true accuracy-drop number. |
| Calibration regression gate | `src/moviesentiment/eval/metrics.py::compute_and_log_brier` + `scripts/check_calibration.py` + new CI step | Brier score logged to MLflow + `metrics/brier.json`; CI fails if `brier > 0.10`. `--allow-missing` until the next full eval run produces the file. 5 unit tests. |
| Concept drift (label distribution) | `src/moviesentiment/monitor/drift.py::label_drift` | Total-variation distance between reference and current label shares. Complements the existing input-feature drift. 4 unit tests. |
| Bias slices per genre — script ready, needs source CSV | `scripts/enrich_with_genre.py` | Joins reviews on `movie_id` against an IMDb genre CSV (format documented in the script docstring). Run-once script; output feeds the model card's per-genre F1 table. |
| Nearest-neighbour `/similar` endpoint | `src/moviesentiment/serve/similar.py` + `GET /similar?text=...&k=5` | TF-IDF cosine over the reservoir, lazily indexed in-process, mtime-rebuild on parquet updates. Falls back to empty result on cold start. 5 tests. Interpretability surface: "show me the closest review the model has seen." |
| Active-learning labeller (Streamlit) | `apps/annotate/app.py` + `streamlit>=1.30` in `[dev]` extras | Reads low-confidence reservoir rows (<0.70), single-row UI to label positive / negative / skip, appends to `data/labeled/augmentation.parquet` for the next training run. |
| API key auth | `Settings.api_key` + `_verify_api_key` in `serve/api.py` | `MS_API_KEY` shared secret, `X-API-Key` header, applies to `/predict`, `/analyze`, `/similar`. Empty key (default) preserves the public demo. 3 tests. |

### Ops-maturity documentation (landed in this session)

| Doc | Purpose |
|---|---|
| `docs/slos.md` | Availability (99.5%), p99 latency (75 ms), 5xx-rate (1%), drift-to-retrain (24 h). Error-budget policy + what does NOT count. |
| `docs/runbook.md` | 9 on-call playbooks: 5xx spike, drift spike, no task running, OOM, stuck deploy, MLflow rollback, S3 ONNX restore, drift workflow not triggering, cost alarm. |
| `docs/shadow_canary.md` | Two-service ECS + ALB weighted-target-group plan. Plan-only — ALB ($18/mo) is above the budget ceiling. |
| `docs/demo_script.md` | 3-minute Loom voiceover script with tab-by-tab cues. |

### Future roadmap (deferred, documented)

From [`docs/future_improvements.md`](docs/future_improvements.md):

- **1.13 LitServe** — drop-in auto-batching FastAPI replacement; deferred until load-test shows queueing.
- **3.2 SageMaker Serverless comparison** — side-by-side cold-start + warm-latency + $/M chart against Fargate. ~$2/mo, half-day. *Highest interview value of the deferred items.*
- **3.4 Multi-language via XLM-R** — only worth it for multilingual roles.
- **Active learning loop** on low-confidence predictions via a tiny Streamlit annotation app, feeding the training set.
- **A/B harness** with 10% traffic split before promotion.
- **Hourly insights Lambda → EventBridge** schedule rule (code lands at `deploy/lambda/insights_aggregator.py`; trigger wiring deferred).

### Key metrics tracked

- Macro F1 / Accuracy / ROC-AUC per model in `metrics/*.json` (DVC metrics).
- Latency histograms via Prometheus `prediction_confidence` + endpoint-level `http_request_duration_seconds`.
- Drift share via Evidently HTML reports under `docs/drift_reports/`.
- CI pass / fail + duration.
- OSSF Scorecard.
- Image size (`docker images moviesentiment` ~250 MB).
- Cost (estimated, AWS billing dashboard) ~$6/mo.

---

## 13. Summary

A 21-day plan landed in **5 calendar days** of intense work (2026-05-24 → 2026-05-28). Final state: a single ARM64 Fargate task serving v1 binary sentiment + v2 multi-head review-intelligence at **~$6/mo**, behind a full DVC pipeline, six GitHub Actions workflows, a documented threat model, six tagged releases (`v0.0-scaffold` → `v2.0`), and **86% test coverage** gated in CI at 85%.

The six pivots that mattered:

1. **Fly.io → ECS Fargate ARM64** — cold-start p99 + Graviton cost win.
2. **GDrive → S3 DVC remote** — CI auth.
3. **`datasets` + pyarrow → plain Python list-of-dicts** — `-100` and `NaN` sentinels survive the collate.
4. **Colab → SageMaker on-demand `ml.g4dn.xlarge`** — reproducibility + DVC artefact tracking.
5. **Single-head DistilBERT → 5-head multi-task** — service surface no free public API ships.
6. **X86_64 → ARM64 Graviton** — ~30% cheaper Fargate.

Each is one commit and one failure mode away from the previous version. The live URL is resolvable via `make smoke-test`. The demo Loom is the only piece of the original 21-day plan not yet recorded.

---

## Appendix A — Commit timeline (full, 52 commits)

| Hash | Date | Subject |
|---|---|---|
| `e83f696` | 2026-05-24 | scaffold: initialize MovieSentiment project structure |
| `8541b52` | 2026-05-24 | feat(data): port IMDb scraper and wire DVC scrape stage |
| `63ee20c` | 2026-05-24 | fix(scraper): two-source architecture with HuggingFace fallback |
| `4efc855` | 2026-05-24 | test(day3): add end-to-end tests for clean_reviews and split_dataset |
| `e20d259` | 2026-05-24 | docs: add external setup checklist for accounts, secrets, and manual steps |
| `8c0b876` | 2026-05-24 | chore(dvc): add Google Drive remote |
| `5e49092` | 2026-05-24 | chore: switch DVC remote to S3 and run full pipeline |
| `7ceafe8` | 2026-05-24 | feat(day5): add eval rigor — ROC, PR curve, calibration, per-class report, error analysis |
| `5c0c644` | 2026-05-24 | feat(day6): implement DistilBERT fine-tuning + Colab training notebook |
| `ac50361` | 2026-05-25 | feat(day8): ONNX export + INT8 quantization pipeline |
| `6416530` | 2026-05-25 | feat(day9): complete ONNX inference engine + API metrics |
| `92eefc1` | 2026-05-25 | feat(day10): Docker multi-stage + Grafana provisioning stack |
| `9de74e6` | 2026-05-25 | feat(day11): add workflow_call trigger to CI for reuse from train.yml |
| `9350c0e` | 2026-05-25 | feat(day14): Locust load test scenarios for /predict endpoint |
| `17a0745` | 2026-05-25 | fix(sagemaker): tokenizer= compat + force transformers>=4.46 upgrade in entrypoint |
| `54199cd` | 2026-05-25 | docs(day19): polish README + add interview talking points |
| `d340772` | 2026-05-25 | feat(day12): Fly.io deploy config + GitHub Actions deploy workflow |
| `b2306f4` | 2026-05-25 | feat(day8): ONNX export artifacts + INT8 quantization benchmarks |
| `01956dc` | 2026-05-25 | feat(day16-18): drift detection workflow + retraining pipeline fixes |
| `d94f1d6` | 2026-05-25 | docs(day19): update results table with real baseline metrics + ONNX latencies |
| `5a81a00` | 2026-05-25 | fix(sagemaker): exclude ONNX model dirs and prod logs from source upload |
| `a9270f4` | 2026-05-25 | fix(ci): dvc pull ONNX models before Docker build |
| `2dfc12b` | 2026-05-25 | chore: add sagemaker_launch.py to repo + gitignore agent workspace files |
| `27f465e` | 2026-05-25 | fix(scrape): add DVC S3 credentials + git commit to weekly scrape workflow |
| `728702d` | 2026-05-25 | fix(ci): rename DVC remote myremote→s3remote in all workflows + slim sagemaker source bundle |
| `fd232d5` | 2026-05-25 | fix(sagemaker): expose pip output + inject src/ into sys.path before import |
| `2e2c468` | 2026-05-25 | fix(sagemaker): use --no-deps so pip does not clobber CUDA torch |
| `476ab37` | 2026-05-25 | fix(sagemaker): cap transformers <5.0 |
| `1f02146` | 2026-05-25 | fix(sagemaker): install accelerate>=0.26.0 for HF Trainer |
| `4239d95` | 2026-05-25 | feat(model): add SageMaker-trained DistilBERT artifacts + real metrics |
| `b5328a3` | 2026-05-25 | chore: trigger ci |
| `5da7c1f` | 2026-05-25 | style(tests): fix ruff I001 + bump pre-commit ruff to 0.15.14 |
| `6bfb583` | 2026-05-25 | style(tests): fix ruff I001 import sorting |
| `fbb1512` | 2026-05-25 | fix(types): mypy overrides for transformers + refactor inference loop for type inference |
| `d486007` | 2026-05-25 | ci: omit untested model files from coverage + drop threshold to 55% |
| `e170a9a` | 2026-05-25 | fix(ci): add setup-python to build-image job (Ubuntu 24.04 PEP 668) |
| `9190a37` | 2026-05-25 | fix(ci): lowercase repo name for GHCR tag (Docker registry requires lowercase) |
| `94e4951` | 2026-05-26 | ci: push image to ECR + force ECS deploy (replace Fly deploy) |
| `0f06834` | 2026-05-26 | fix(models): re-export ONNX from fine-tuned classifier (was base MaskedLM) |
| `e2f224c` | 2026-05-26 | feat: ECS deploy + /version fix + mermaid arch diagram |
| `e5a53cc` | 2026-05-26 | chore: workspace cleanup, release-archive hygiene, v2 plan |
| `639ec9c` | 2026-05-26 | feat(v2): multi-task review-intelligence model skeleton |
| `271ffd6` | 2026-05-26 | feat(v2): multitask ONNX export, hourly insights Lambda, DVC + CLI wiring |
| `e151d2a` | 2026-05-27 | feat(v2): multitask SageMaker launcher + JSONL spoiler loader |
| `fc6de55` | 2026-05-27 | fix(v2): collate dtypes; S3 bootstrap for multitask ONNX; settings extra=ignore |
| `0f17cf2` | 2026-05-28 | fix(v2): None-safe collate for multitask joint dataset |
| `8baf104` | 2026-05-28 | fix(v2): drop datasets/pyarrow joint — use plain Python list dataset |
| `617019c` | 2026-05-28 | chore(v2): unbreak CI mypy on test re-exports + Dataset generic |
| `61647b8` | 2026-05-28 | fix(v2): bandit nosec on multitask ONNX export local-path loads |
| `29fa078` | 2026-05-28 | fix(v2): chown /app to app user so S3 bootstrap can write multitask ONNX |
| `ace25b3` | 2026-05-28 | test(v2): boost coverage 76% to 86% (gate raised to 85%) |
| `5a7e594` | 2026-05-28 | fix(v2): widen tests mypy override (arg-type, unused-ignore) |
| `0f95416` | 2026-05-28 | fix(v2): tests/__init__.py so tests.* mypy override applies |

---

## Appendix C — Live AWS state, verified 2026-05-29

Verified via the `moviesentiment-ci` IAM user (scoped permissions; some operations returned `AccessDenied` which is itself information about that user's policy).

| Resource | State | Evidence |
|---|---|---|
| ECR repo `moviesentiment` | **Alive** | `aws ecr describe-repositories` returns the repo URI `375259955411.dkr.ecr.ap-southeast-2.amazonaws.com/moviesentiment`. |
| S3 `moviesentiment-dvc-soumya` | **Alive** | `aws s3 ls s3://moviesentiment-dvc-soumya/` shows `dvc/` + `multitask_onnx/` prefixes. |
| S3 `multitask_onnx/` artefacts | **Intact** | 5 files present, total ~67 MB. `model.onnx` is 66,824,331 B uploaded 2026-05-28 02:47. Lifespan S3 bootstrap path would work if a service were running. |
| CloudWatch log group `/ecs/moviesentiment` | **Alive**, retention unset (default = never expire) | `aws logs describe-log-groups` shows `storedBytes: 842,432`. `retentionInDays: null`. |
| ECS cluster(s) | **Empty** | `aws ecs list-clusters` returns `clusterArns: []`. No `moviesentiment` cluster exists. |
| ECS service `moviesentiment` | **Gone** | `aws ecs list-services --cluster moviesentiment` returns `serviceArns: []`. The live `/predict` etc. are NOT live. |
| ECS task definition family `moviesentiment` | **Registered, not deployed** | `aws ecs list-task-definition-families` returns `["moviesentiment"]`. The image and config are still in the registry — a service can be spun back up by registering a new revision + create-service. |
| Lambda functions | **None** in `ap-southeast-2` or `us-east-1` | `aws lambda list-functions` returns `[]`. The async-predict worker and hourly insights aggregator are not deployed. |
| SQS queues | **Unverified** (CI user lacks `sqs:ListQueues`) | Pattern matches teardown: assume empty unless proven otherwise. |
| DynamoDB tables | **Unverified** (CI user lacks `dynamodb:ListTables`) | Same. |
| EventBridge rules | **Unverified** (CI user lacks `events:ListRules`) | Same. |
| CloudWatch retention bump | **Blocked** (CI user lacks `logs:PutRetentionPolicy`) | Needs an admin IAM user / role to apply the 90-day policy. |

**Implication for the report's earlier claims**: the project was deployed end-to-end during the build session (2026-05-24 → 2026-05-28). After the build, the live ECS service and Lambdas were torn down for cost control (the original session ended with a "which cloud machines need to be alive" cleanup conversation). The **artefacts in S3 + ECR + the registered task definition** mean a redeploy is one `register-task-definition` + `create-service` away, but **the live URL claims in the README are stale** until that redeploy lands.

**Implication for permissions**: the `moviesentiment-ci` IAM user is scoped to CI-only operations (ECR push, ECS update, log read). It lacks the admin operations (`logs:PutRetentionPolicy`, `sqs:*`, `dynamodb:ListTables`, `events:ListRules`, `s3:ListAllMyBuckets`) needed to verify or change the rest. Documented here so future on-call knows what permission-class is required for which incident.

---

## Appendix B — [NEEDS CONFIRMATION] items

These claims are inferred from prior conversation logs, the local repo state, or pricing tables; they should be verified against the actual AWS console / git state before being relied on. The author's AWS keys were rotated after the build session by design (the keys pasted into the chat had to be deleted), so live AWS queries from this session are not possible. Confirmation paths are listed inline.

1. **Live ECS Fargate ENI IP at report time.** Five distinct IPs appeared in session logs as the task rotated (`54.206.111.36`, `15.135.233.125`, `3.106.121.110`, `13.211.172.87`, `3.25.226.240`). README pins `54.206.111.36`; current value resolves via `make smoke-test` (which calls `aws ecs describe-tasks` + `aws ec2 describe-network-interfaces`).
2. **v2 SageMaker training cost ≈ $0.75.** Successful run is job `moviesentiment-multitask-2026-05-26-22-07-16-892` (confirmed by author — the fifth submission after four short-lived failures whose logs have since been pruned to control CloudWatch cost). Dollar figure derived from on-demand `ml.g4dn.xlarge` list price $0.526/hr × ~1.4 hr. Authoritative: AWS Billing console for May 2026 SageMaker line.
3. **CloudWatch Logs `/ecs/moviesentiment` retention.** Default if untouched is "never expire" — fine for the "keep the logs to understand our working later" intent. Verify with `aws logs describe-log-groups --log-group-name-prefix /ecs/moviesentiment`. Note: SageMaker training job logs from earlier failed v2 attempts were pruned, so they cannot be re-read.
4. **`MS_MULTITASK_S3_PREFIX` set on the live ECS task.** Confirmed: the live `/analyze` works, which means a task revision past rev 5 was registered with both `MS_DVC_BUCKET` and `MS_MULTITASK_S3_PREFIX` set. The committed `deploy/ecs-task-definition.json` is rev 5 and does NOT list them — the live revision is whatever the console shows. Verify with `aws ecs describe-task-definition --task-definition moviesentiment --region ap-southeast-2`.
5. **Hourly insights Lambda + EventBridge schedule rule.** Confirmed deployed (author: anything that produced live results was deployed to AWS, nothing is local-only). Verify the current state with `aws events list-rules` / `aws lambda list-functions`.
6. **Image size estimate (~250 MB).** Derived from `python:3.11-slim` base (~150 MB) + venv (~80 MB) + ONNX models (`distilbert_onnx` FP32 + `distilbert_onnx_int8` ~320 MB) + frontend (~30 KB). True size depends on layer dedup. Verify with `docker images moviesentiment`.
7. **MLflow Production-stage pointer.** State lives in the local SQLite `mlflow.db`; no canonical version checked in. Run `mlflow ui` against the file to inspect.
8. **OSSF Scorecard badge score.** README links the badge but the underlying score is dynamic. Verify at `securityscorecards.dev/viewer/?uri=github.com/Cryptic2-0/Project`.
9. **Total monthly cost ~$6/mo.** Estimated from Fargate ARM64 pricing ($0.04048 × 0.25 vCPU/hr + $0.004445 × 1 GB/hr × 730 hr ≈ $6.39) plus S3 (~$0.05), CloudWatch (~$0.10), and free-tier Lambda/SQS/DynamoDB. Actual: AWS Cost Explorer for month-to-date.
10. **v1 Macro F1 0.939.** Source: `metrics/transformer.json` written by `train_transformer()` and surfaced in `docs/model_card.md`. Verify by reading the file or by running `moviesentiment metrics`.
11. **Test count 15 files + ~86% coverage.** File count verified via `find tests -maxdepth 1 -name 'test_*.py' | wc -l`. Coverage is enforced at 85% by CI; the 86% figure is from the most recent local run reported in commit `ace25b3`. Reproduce with `make test`.
