# MovieSentiment — Future Improvements (deferred)

Date: 2026-05-26

These items from `possible_improvements.md` are **intentionally deferred**. Each is feasible and valuable, but did not make the current pass — kept here so they don't get lost.

---

## 1.13 — Replace plain FastAPI with LitServe
**Status**: deferred · ~half-day effort · $0 · interview value 3/5

LitServe is a FastAPI-based serving framework that auto-batches requests at the worker level. Claim: ~2× throughput on the same hardware, same Pydantic schemas, drop-in.

**Why deferred**: I'd want a load-test artifact comparing FastAPI vs LitServe at p50/p95/p99 on the actual Fargate task before committing to the dependency. The benchmark (1.14) lands first; this lands only if the numbers justify it.

**Concrete plan when picked up**:
1. Add `litserve>=0.2` to `pyproject.toml` (optional extra).
2. Port `src/moviesentiment/serve/api.py` to a `LitAPI` subclass — `setup()` loads ONNX session, `predict()` runs inference, `decode_request`/`encode_response` use the existing Pydantic schemas.
3. Run `scripts/load_test.py` against both servers; commit `docs/litserve_bench.md` with the numbers.
4. Keep both entrypoints behind a `SERVER_BACKEND=fastapi|litserve` env var so the old code path remains.

**Interview line**: *"I benchmarked both — LitServe wins above N RPS because of dynamic batching; below that the FastAPI hot path is faster because there's no queue. We're below N, so I shipped FastAPI."* (need actual N from the benchmark.)

---

## 3.2 — SageMaker Serverless Inference Endpoint (compare against ECS)
**Status**: deferred · ~half-day effort · ~$2/mo · interview value 4/5

Deploy the same ONNX model to a SageMaker Serverless endpoint *next to* Fargate. Generates an honest comparison: cold start, warm latency, $/M requests.

**Why deferred**: Tier-3 item, sits closest to the budget ceiling (+$2/mo realistic at portfolio traffic). Want Phase A/B improvements landing first so the project is already strong before adding the second serving path.

**Concrete plan when picked up**:
1. Package the ONNX INT8 model + tokenizer as a SageMaker model artifact (tarball uploaded to S3).
2. Write `scripts/sagemaker_serverless_deploy.py` — creates `EndpointConfig` with `MemorySizeInMB=1024, MaxConcurrency=4`.
3. Add a CloudWatch metric scraper to `scripts/compare_endpoints.py` that hits both endpoints at 1/5/20 RPS and records cold start + warm latency + cost-per-million.
4. Commit `docs/endpoint_comparison.md` with the table.

**Cost math**: $0.000016 per GB-second + $0.20 per 1M requests. At ≤1k req/day with 1 GB memory and ~100 ms warm: ≈ $1.40 / month before request charges.

**Interview line**: *"Fargate wins on warm latency (6.8 ms vs ~80 ms cold start). SageMaker Serverless wins below ~100 req/day where Fargate's always-on baseline dominates. The README has the cost-per-million chart."*

---

## 3.4 — Multi-language support (es, fr via XLM-R)
**Status**: deferred · ~2-3 day effort · $0 (compute is already accounted for) · interview value 3/5

Switch base model from `distilbert-base-uncased` to XLM-RoBERTa-base for cross-lingual transfer. Train one classifier, evaluate on Spanish and French movie reviews.

**Why deferred**: Doubles training time on SageMaker (mitigated by 1.10 spot training). The IMDb 50K dataset is English-only, so this requires sourcing Spanish (e.g. SemEval-2017 Task 4) and French (e.g. Allociné) review datasets first. Larger model = slower inference (~12 ms p50 instead of 6.8 ms). Only worth it if interviewing for multilingual roles.

**Concrete plan when picked up**:
1. Add data scrapers in `src/moviesentiment/data/scrape_es.py` and `scrape_fr.py`.
2. Swap `params.yaml::transformer.model_name` to `xlm-roberta-base`. Re-quantize.
3. Add per-language F1 to `metrics/transformer.json` and to the model card.
4. Update the live demo to show language detection + per-language confidence.

**Interview line**: *"I cross-lingually transferred sentiment to Spanish and French — F1 0.91 on Allociné, 0.89 on SemEval-2017 ES. Latency moves from 6.8 ms to 12 ms because XLM-R-base is larger; for a sentiment classifier that trade is worth it once you're outside English."*

---

## Pickup order

If the budget allows one of these and you can only ship one before an interview cycle:

1. **3.2 SageMaker Serverless comparison** — highest interview-value per hour. Real cost numbers from two production serving stacks beat almost any other line.
2. **1.13 LitServe** — only if the load-test (1.14) shows queueing under burst load.
3. **3.4 Multi-language** — only for multilingual / international team interviews.

---

## v2 — Review Intelligence (multi-task expansion)

**Status**: code skeleton landed (commit on `main`); training run pending · ~$0.24 one-time + $0/mo · interview value 5/5

**What is in the repo now**
- `src/moviesentiment/models/multitask.py` — shared-encoder DistilBERT with five heads.
- `src/moviesentiment/models/multitask_train.py` — joint trainer with task-weighted loss; missing labels per row use sentinel ignore values so each parquet only needs to supervise the task it cares about.
- `src/moviesentiment/data/multitask_loaders.py` — loaders for GoEmotions → Ekman, Kaggle spoiler CSV, helpfulness proxy, and ABSA distillation from `yangheng/deberta-v3-base-absa-v1.1`.
- `src/moviesentiment/serve/multitask_inference.py` — multi-head ONNX wrapper.
- `src/moviesentiment/serve/insights.py` + `GET /insights/{movie_id}` — aggregates the reservoir, falls back gracefully when the v1 reservoir schema is in use.
- `POST /analyze` — wired into the FastAPI app; 503s until the ONNX artefact lands.
- `params.yaml::multitask.*` — full hyperparameter block.
- Tests: `tests/test_multitask.py` covers schema round-trip, 503 fallback, 422 oversize, and the v1↔v2 insights migration path.

**What still needs to happen**
1. Run the data loaders to produce per-task parquets under `data/interim/multitask/`.
2. `moviesentiment train multitask` on SageMaker spot (estimated ~1.5 hr on `ml.g4dn.xlarge`).
3. Export the trained checkpoint to ONNX with five named outputs (`sentiment`, `aspect`, `emotion`, `spoiler`, `helpfulness`) and DVC-track it as `models/distilbert_multitask_onnx/`.
4. Add a `train_multitask` DVC stage and a CI step to fetch the new artefact.
5. Extend the production reservoir schema to capture per-head outputs (so `/insights` aggregates v2 columns).
6. Optional: hourly Lambda materialising `data/production/insights/*.json` to S3.

Pivot the service from "positive vs negative" to a **multi-head review-intelligence API** that no free / portfolio-tier service currently offers in one call. Shared DistilBERT encoder → five linear heads, one forward pass.

### Heads

| Head | Type | Labels | Data source |
|---|---|---|---|
| `sentiment` | binary | pos / neg | existing IMDb 50K |
| `aspect_sentiment` | 5 × 3-class | {neg, neu, pos} × {acting, plot, visuals, pacing, sound} | distill from `yangheng/deberta-v3-base-absa-v1.1` (offline teacher) |
| `emotion` | 6-class | Ekman | GoEmotions transfer, then DistilBERT distill |
| `spoiler` | binary | spoiler / not | Kaggle IMDB Spoiler Dataset (573 K) |
| `helpfulness` | regression | up / (up + down) | live scraper already captures votes |

### New endpoints

- `POST /analyze` — returns all five heads in one response. Replaces `/predict` over time (kept for backwards-compatibility).
- `GET /insights/{movie_id}` — aggregates the reservoir-sampled reviews for a movie: aspect averages, emotion mix, spoiler share, top-5 topic clusters via BERTopic over CLS embeddings. Offline batch refreshed hourly via Lambda; cached in S3.

### Cost math

| Item | One-time | Recurring |
|---|---|---|
| Teacher distillation on SageMaker `ml.g4dn.xlarge` spot, ~1.5 hr | $0.24 | — |
| DVC storage for 50 MB extra labels + adapter | — | < $0.01 / mo |
| Serving latency (5 × linear heads ≈ 100 KB extra weights) | — | +<0.5 ms p50 |
| Insights aggregation Lambda hourly | — | $0 (free tier) |
| **Total** | **$0.24** | **~$0.01 / mo** |

New monthly total ≈ $7.21 vs the $13 / mo cap. Cleared.

### Why this is rare at this price

- Google Natural Language: sentiment + entities only; $0.50–$2 per 1k calls.
- AWS Comprehend: sentiment + key phrases; no domain emotion, no spoiler, no aspect mix; per-call billing.
- Azure: sentiment + opinion mining (a thin ABSA); no spoiler, no helpfulness, no cluster aggregation.
- No free public service ships spoiler-detection + ABSA + helpfulness + per-title topic clusters together.

### Concrete plan when picked up

1. Schema migration — add `MultiHeadOutput` to `src/moviesentiment/serve/schemas.py`.
2. Model — extend `src/moviesentiment/models/transformer.py` to register five heads on top of the shared encoder; freeze encoder when training a single head; joint fine-tune with task-weighted loss for the final pass.
3. Data — add `src/moviesentiment/data/{spoiler,emotion,absa,helpfulness}.py` loaders. Validation stage runs on each.
4. Training — extend `dvc.yaml` with one stage per task plus a `train_joint` stage. SageMaker spot remains the runner.
5. Serving — new `POST /analyze` route. `/predict` stays as a sentiment-only convenience.
6. Insights — `src/moviesentiment/serve/insights.py` + `deploy/lambda/insights_aggregator.py`. BERTopic runs offline; only the JSON output is in the hot path.
7. Frontend — extend the demo UI to render all five outputs and the `/insights` aggregate.
8. Docs — model card per head; per-head F1 + calibration; updated benchmarks for the multi-head latency.

### Interview line

*"The serving stack is the same INT8 ONNX session, but the service now ships sentiment, aspect-level sentiment, emotion, spoiler-detection, and helpfulness regression in one forward pass — no public free service offers that combination. Topic aggregation per title runs offline so the hot path stays under 10 ms. Total added cost over the 50 MB labels and a one-time $0.24 distillation run was zero."*
