# Interview Prep — MovieSentiment

> Exhaustive Q&A drill book. Every decision in the project, the WHY,
> the trade-off, the interviewer's likely follow-up, and the answer.
>
> Read this end-to-end before any MLE / MLOps interview that references
> this project. The doc is organised by topic; within each topic the
> questions go from "warm-up" → "deep technical" → "curveball".

Last updated: 2026-05-29. Project at v3.0.

Sister docs: `docs/interview_talking_points.md` (60-second answers),
`docs/scaling.md` (tier breakdown), `docs/slos.md`, `docs/runbook.md`,
`final_report.md`.

---

## 0. The opening pitch — your 90-second project overview

> "MovieSentiment is an end-to-end MLOps project for IMDb review
> sentiment. The interesting part isn't the model — it's that this
> isn't a Jupyter notebook. It ships data scrape, DVC versioning,
> SageMaker training, ONNX INT8 export, ARM-Fargate serving, drift
> detection, retraining cron, async path with SQS + Lambda, calibration
> gate, label-drift detection, adversarial robustness tests, an
> active-learning Streamlit labeller, a v2 multi-task model with five
> heads — sentiment plus aspect-based sentiment plus emotion plus
> spoiler plus helpfulness — in one forward pass. Cost ceiling was
> thirteen dollars a month, deliberately, to force the engineering
> decisions. Today it sits at about thirty-seven cents a month with
> the models published on HuggingFace and AWS scaled to storage only.
> The architecture diagram is in the README; full build report in
> `final_report.md`."

Don't memorise word-for-word — own the structure: **scope → constraint →
constraint-derived decisions → current state → where to learn more**.

---

## 1. Architecture decisions

### Q: Walk me through the architecture.

**A.** Open the README's Mermaid diagram. Trace top-to-bottom.

1. **Data path**: IMDb (live GraphQL with HuggingFace fallback) → S3
   via DVC. Two-source pattern because Akamai WAF blocks the live
   endpoint from CI runners.
2. **Pipeline**: DVC stages — scrape → clean → validate_data → split
   → train_baseline / train_transformer / prep_multitask_data →
   train_multitask → export_onnx.
3. **Training**: SageMaker `ml.g4dn.xlarge` for both v1 fine-tune
   ($0.75 one-time) and v2 multi-task ($0.75). On-demand because spot
   quota was zero at run time.
4. **ONNX export + INT8 quantisation**: brings v1 from 25 ms PyTorch
   eager → 6.8 ms INT8 ONNX p50. 4× model size reduction (256 → 64 MB).
5. **Serving**: FastAPI on ARM64 Graviton Fargate (today scaled to
   zero). 0.25 vCPU / 1 GB. Production endpoint
   `/predict`, `/analyze`, `/explain`, `/predict/async`,
   `/insights/{movie_id}`, `/similar`. Auth via `X-API-Key`.
6. **Monitoring**: Prometheus + Grafana, structlog with request-id
   correlation, OpenTelemetry traces to Grafana Cloud free tier.
   Evidently for drift, NannyML for label-free F1 estimation.
7. **Retraining loop**: drift_share > 0.30 → train.yml workflow →
   model promoted if F1 improves ≥ 0.5%.

**Land the closed-loop story at the end.** "Drift triggers retraining"
is the senior-MLE signal.

### Q: Why DistilBERT and not BERT-base or a larger model?

**A.** DistilBERT is **40% smaller and 60% faster than BERT-base while
retaining ~97% of its task performance** on sentiment. The accuracy
gap to BERT-base on IMDb 50K is <1% (0.939 vs ~0.945 macro F1) — not
worth 2.5× the latency.

For a sentiment classifier on short reviews where keywords carry most
of the signal, the smaller model captures enough. RoBERTa / XLM-R
would be ~10× the inference cost for ~1% gain.

### Q: Why ONNX + INT8 quantisation specifically?

**A.** Three reasons stacked:

1. **ONNX Runtime on CPU is ~2× faster than PyTorch eager** for
   inference, because of operator fusion and kernel optimisation.
2. **Dynamic INT8 quantisation (weights only, FP32 activations)** drops
   the model from ~256 MB to ~64 MB (4×) and cuts p50 latency from
   12.3 ms → 6.8 ms — another 1.8×. <1% accuracy drop.
3. **Same ONNX file runs anywhere** — Fargate CPU today, GPU via
   `CUDAExecutionProvider` tomorrow, edge via Cloudflare Workers AI
   later. Zero code change at the serving layer.

The export pipeline is `transformers → optimum/ONNX export →
onnxruntime.quantize_dynamic → ORT InferenceSession`. All four
documented in `src/moviesentiment/models/onnx_export.py`.

### Q: Why is the v2 multi-task model better than 5 separate models?

**A.** Three angles:

1. **Latency**: 5 sequential model calls = ~5 × 7 ms = 35 ms p50.
   Multi-task shared encoder = 7.3 ms p50 (the 0.5 ms is the extra
   linear heads). 5× throughput improvement per request.
2. **Cost**: 5 ONNX sessions × 64 MB = 320 MB memory + 5 forward
   passes. Multi-task = 64 MB + 1 pass. Fits the same 1 GB Fargate
   task.
3. **Joint supervision**: ignore-value sentinels (`-100` for CE,
   `NaN` for MSE) let each parquet supervise only the tasks it has
   labels for. We don't need a single dataset with all 5 head labels.

### Q: Why FastAPI over Flask / Starlette / Django REST?

**A.** Pydantic validation at every API boundary (free request schema
documentation via OpenAPI), built-in async, dependency injection
matches the `Depends(_verify_api_key)` pattern, ecosystem of
production-grade middleware (slowapi rate limiting, prometheus
instrumentator, opentelemetry-instrumentation-fastapi). Flask would
need manual schema validation; Django REST is overkill for a service
with no ORM.

### Q: Why DVC instead of Git LFS / S3-only?

**A.** **Reproducibility and lineage**: `dvc.yaml` declares stage
dependencies (params + code + input data) and outputs. Changing
`params.yaml::baseline.lr_C` re-runs only `train_baseline` —
upstream stages cached. Git LFS doesn't model that DAG; S3-only loses
the parameter → output mapping.

### Q: Why SQLite MLflow backend in v3.0?

**A.** Zero infra cost; one file; sufficient for a solo project.
Documented as a Tier 2 upgrade in `docs/scaling.md` — production
would use Postgres + S3 artefact store. The serving code already
reads from `MS_MLFLOW_TRACKING_URI`, so swapping is one env var.

---

## 2. Why we built it cost-conscious

### Q: Why a $13/mo cost ceiling?

**A.** Three reasons:

1. **It forced principled engineering decisions**. Without a ceiling
   I'd reach for ALB ($18/mo) or always-on Lambda concurrency or
   managed MLflow. With a ceiling, every decision has to justify
   itself against the budget.
2. **It demonstrates senior judgement**. Junior engineers spend; senior
   engineers cut. Interviewers can read this in `final_report.md` §6.
3. **Portfolio realism**. An interview reviewer can fork this and
   actually run it without $200/mo of AWS bills.

The current state ($0.37/mo) is below the ceiling because the
artefacts moved to HF Hub and the compute layer is torn down. A live
demo costs ~$0.02/hour by spinning up the Fargate service for the
demo window — documented in `docs/demo_walkthrough.md`.

### Q: Why ECS Fargate over Lambda?

**A.** Cold-start. Lambda containers cold-start in 2–10 s for an
image with ONNX Runtime + the model. Fargate stays warm at $6/mo per
0.25 vCPU. Below ~1k req/day Lambda is cheaper; above it Fargate
wins on cost-per-request **and** p99 latency.

We documented the comparison in `docs/future_improvements.md` §3.2
(SageMaker Serverless comparison was deferred — same shape of
trade-off).

### Q: Why ARM Graviton over x86 Fargate?

**A.** ARM64 Graviton Fargate is **~30% cheaper** per vCPU-hour.
ONNX Runtime ships ARM builds. The cross-arch CI matters because
Docker BuildKit with QEMU is slower than native x86 builds (~3 min
extra in CI), but the lifetime cost savings on Fargate dominate.

The X86_64→ARM64 migration also surfaced one of the build-journey
bugs (commit message in `final_report.md` Phase 4) — task definition
platform pin saved us from silent failures later.

### Q: Why GitHub Actions cron over Airflow / Dagster?

**A.** **Airflow is overkill for a project with 3 cron jobs**. The
overhead of a scheduler, executor, metadata DB, and web UI dwarfs the
actual pipeline code. The build guide called this out as the "right"
portfolio call — using Airflow here would be an over-engineering red
flag.

For >5 interdependent pipelines or complex fan-out, the upgrade to
Dagster is documented in `docs/scaling.md` Tier 2.

### Q: Why HuggingFace Hub over S3 for model storage?

**A.** Free public hosting; auto-rendered model card from
`docs/model_card.md`; version history with diffs; inference widget so
interviewers can click instead of `curl`. S3 has none of these
properties without extra glue. Cost difference: $0 vs $0.03/mo —
nominal, but the discoverability + interview value is the real reason.

---

## 3. Model + data decisions

### Q: How did you split train/val/test?

**A.** **70/15/15 stratified by label**, fixed seed 42 in
`params.yaml::split.seed`. 70% to train (35k examples), 15% val
(7.5k, for early stopping + hyperparameter tuning), 15% test (7.5k,
held-out for final eval).

Stratification preserves the 50/50 class balance in each split (IMDb
50K is balanced by construction). Fixed seed makes splits
byte-identical across runs.

### Q: Why label threshold rating ≥7 / ≤4?

**A.** Ratings 5–6 are genuinely ambiguous — the model would learn
noise. Dropping the middle third gives cleaner signal. Documented in
`docs/error_analysis.md` with examples of 5-rated reviews that read
positive and 6-rated reviews that read negative.

This makes the test set ~70% of the raw IMDb 50K (after dropping
neutrals + cleaning).

### Q: What's the v2 multi-task loss?

**A.** Task-weighted sum:

```
L = w_sent * CE(sent_logits, y_sent)        // ignore_index = -100
  + w_aspect * CE(aspect_logits, y_aspect)  // 5 aspects × 3 classes
  + w_emotion * CE(emotion_logits, y_emotion)
  + w_spoiler * CE(spoiler_logits, y_spoiler)
  + w_help * MSE(help_score, y_help)         // NaN-masked
```

Weights in `params.yaml::multitask.loss_weights` — sentiment 1.0,
aspect 1.0, emotion 1.0, spoiler 1.0, helpfulness 0.5 (down-weighted
because MSE scale dominates CE otherwise). Each task can be absent
per-row — CE ignores -100, MSE skips NaN.

### Q: How do you handle class imbalance?

**A.** Two layers:

1. **In v1 training**: `LogisticRegression(class_weight='balanced')`
   on the baseline. DistilBERT uses balanced IMDb 50K — no
   imbalance.
2. **In production drift detection**: the reservoir-sampled production
   distribution is compared to the training distribution. Real-world
   reviews skew ~70/30 positive, so the model card flags expected
   false-negative growth. The `label_drift` function in
   `monitor/drift.py` (v2.1) catches this concept drift separately.

In Tier 2+ production we'd add per-segment loss reweighting at
retraining time.

### Q: How calibrated is the model?

**A.** Well-calibrated above 0.85 confidence; below 0.70 it's
essentially chance. Documented in `docs/model_card.md` with the
reliability diagram. CI gate fails the build if Brier score exceeds
0.10 — `scripts/check_calibration.py` was added in v2.1.

Production recommendation: gate downstream actions on confidence
≥0.85 (78% of test set hits this). Treat <0.70 (2%) as "model
declined to answer". The 6% middle band needs human review in real
production.

### Q: Per-genre performance?

**A.** From a live IMDb scrape of the 10 movie_ids in `params.yaml`
(2,525 reviews, 2026-05-29):

```
Adventure  macro F1 0.848
Crime      macro F1 0.840
Animation  macro F1 0.834
Drama      macro F1 0.827
Action     macro F1 0.779  ← weakest
```

Overall macro F1 0.825. **Caveat**: helpfulness-sorted reviews are
~93% positive, so binary F1 on the majority class stays 0.964–0.973
across genres. Action's weakness is on minority-class (negative
review) recall.

A balanced re-scrape (sort by date) would disentangle "genre effect"
from "class-imbalance effect". Documented as a follow-up.

---

## 4. Production engineering

### Q: What happens when a request arrives?

**A.** Trace top-to-bottom:

1. **ALB** (when running) → ECS task ENI on port 8000.
2. **uvicorn worker** dispatches to FastAPI.
3. **Middleware stack** (in order):
   - CORS (configured via `MS_CORS_ALLOW_ORIGINS`)
   - request-id middleware (round-trip `x-request-id` header)
   - slowapi rate limiter (60/min on `/predict`)
   - OpenTelemetry span starts
4. **`Depends(_verify_api_key)`** checks `X-API-Key` against
   `MS_API_KEY`. Empty key = demo mode passthrough.
5. **Pydantic validation** of `PredictRequest` (`max_length=32`,
   `min_length=1`).
6. **Hard caps**: batch size ≤ 32, text length ≤ 5000 chars.
7. **InferenceEngine.predict()**:
   - tokenizer batch encode (NumPy tensors)
   - `session.run()` ONNX forward pass
   - softmax + argmax in NumPy
8. **Reservoir sampler add** (bounded memory, Vitter Algorithm R, k=1000).
9. **Periodic flush** every N inserts to
   `data/production/recent.parquet`.
10. **Prometheus metrics** observe (confidence histogram + class
    counter).
11. **structlog `predict_complete`** with labels + count (no review
    bodies — PII boundary).
12. Response serialised via Pydantic → JSON.

End-to-end **p50 ~12 ms**: 6.8 ms ONNX + ~5 ms HTTP stack overhead.
Documented breakdown in `docs/loadtest.md`.

### Q: How is the API rate-limited?

**A.** Per-route via `slowapi`:

- `/predict`: 60 / minute / IP
- `/analyze`: 30 / minute / IP
- `/explain`: 10 / minute / IP (expensive — K× the latency)
- `/similar`: 30 / minute / IP

Per-IP because that's the available signal pre-auth. Production
would shift to per-user (post-JWT validation) at Tier 2.

The load-test rig hits 429 against itself because all traffic comes
from one IP — documented as a load-test artifact, not a prod bug, in
`docs/loadtest.md`.

### Q: What's the reservoir sampler and why?

**A.** **Vitter's Algorithm R**, k=1000. Reservoir sampling
guarantees uniform random selection over an unbounded stream at
fixed memory.

We use it on `/predict` + `/analyze` to capture a sample of
production inputs for drift detection. The previous approach was
`random.random() < 0.1` (10% sampling) which biases toward early
traffic — the first 10k requests dominate the sample because the
stream isn't stationary.

Reservoir keeps the **last k** with equal probability regardless of
total stream length. Per-flush to `data/production/recent.parquet`
every 100 inserts. Code at `serve/reservoir.py`.

### Q: Async path — how does it work?

**A.** `POST /predict/async` → SQS → Lambda → DynamoDB. Code at
`serve/async_api.py` + `deploy/lambda/async_predict_handler.py`.

1. Client POSTs `{"texts": [...]}` to `/predict/async`.
2. Endpoint generates a `job_id` (UUID), writes pending row to
   DynamoDB (`MS_JOBS_TABLE`), sends SQS message
   (`MS_SQS_QUEUE_URL`) with `{"job_id": ...}`.
3. Lambda worker reads SQS, loads ONNX session (warm or cold), runs
   inference, writes result back to DynamoDB.
4. Client polls `GET /predict/result/{job_id}` for the result.

Returns 503 if env vars unset — local dev works without AWS plumbing.

**Why split sync + async**: a 10-second batch shouldn't tie up
uvicorn threads serving the 50 RPS that need 10 ms each. Sync path
stays cheap. The async path scales-to-zero between requests via
Lambda reserved concurrency = 0 (default).

### Q: How does the v2 multi-task ONNX get loaded?

**A.** Two paths in `serve/multitask_inference.py`:

1. **`from_disk()`** — checks `models/distilbert_multitask_onnx/`. Local
   dev path.
2. **`from_s3(bucket, prefix)`** — downloads the 5 expected files
   (model.onnx, tokenizer.json, tokenizer_config.json,
   special_tokens_map.json, vocab.txt) from
   `s3://{bucket}/{prefix}/`. Used at Fargate boot when
   `MS_DVC_BUCKET` + `MS_MULTITASK_S3_PREFIX` are set.

**Why S3 bootstrap instead of bundling**: the Docker image stays at
v1 size (~250 MB). v2 ONNX rolls independently — no rebuild needed
when retraining the multi-task model.

The lifespan validates the ONNX output schema (5 named outputs:
`sentiment`, `aspect`, `emotion`, `spoiler`, `helpfulness`) on load
so a wrong model file fails loud.

### Q: How do you handle long reviews?

**A.** Hard cap at `max_text_length=5000` chars (Pydantic 422 if
exceeded). Tokenisation truncates at `max_length=512` tokens for v1
(matches DistilBERT's positional embedding limit) and
`max_length=256` for v2 multi-task (matches v2's training setting —
this was a v2.1 bug fix; pre-fix v2 inference truncated at 512 but
the model never saw positions past 256, producing garbage logits on
long inputs).

For >5000 char reviews in production (rare), recommendation is to
chunk client-side + average confidence. Not in code today.

---

## 5. Observability + reliability

### Q: What metrics do you emit?

**A.** Prometheus, three layers:

1. **Auto-instrumented** by `prometheus-fastapi-instrumentator`:
   `http_requests_total`, `http_request_duration_seconds_bucket` by
   path × status. Standard RED metrics.
2. **Custom** (`monitor/prometheus.py`):
   - `prediction_confidence` (histogram)
   - `prediction_class_total{label}` (counter)
   - `model_version_info{model_name,version,git_sha}` (labelled
     gauge — gives you the deployed-model identity in every request
     log)
3. **OpenTelemetry traces** exported via OTLP to Grafana Cloud free
   tier. Each request gets a span tree: middleware → auth → predict
   → engine → reservoir.

### Q: What are your SLOs?

**A.** `docs/slos.md`:

- **Availability**: 99.5% on `/healthz` (3h36 budget per 30-day
  month)
- **p99 latency on `/predict`**: ≤ 75 ms (1% may exceed)
- **5xx rate on `/predict` + `/analyze`**: ≤ 1% (7h12 budget at >1%)
- **Drift → retrain**: ≤ 24 h from detection to workflow start

Numbers chosen for a single 0.25 vCPU Fargate task. Tier 2 tightens
to 99.9% / 30 ms / 0.1% / 1 h.

Error-budget policy in the same doc: within budget = ship freely;
50% consumed = freeze risky changes; budget exhausted = roll back +
incident report.

### Q: How do you detect drift?

**A.** Two signals, both weekly cron:

1. **Input drift** (`drift_share`): Evidently `DataDriftPreset` on
   text_length + word_count features. Wasserstein distance for
   numeric. Drift > 0.30 → retrain trigger.
2. **Label drift** (`label_drift`, added in v2.1): total-variation
   distance between predicted-label shares (reference vs current
   window). Catches concept drift the input-side test misses.

If `drift.yml` workflow flags drift, it triggers `train.yml` →
retrain → promote if F1 improves ≥0.5%.

### Q: What's the retraining loop?

**A.** End-to-end loop in 4 steps:

1. **Drift detection** (weekly cron, `drift.yml`): generate
   Evidently HTML report, compute drift_share + label_drift. If
   threshold exceeded → trigger train.yml.
2. **Training** (`train.yml`): `dvc pull` data, train baseline +
   DistilBERT on SageMaker spot (when quota allows; on-demand
   fallback). Log to MLflow.
3. **Promotion gate**: new F1 ≥ current F1 + 0.5% on held-out val?
   If yes → promote to `Production` stage in MLflow Registry → push
   updated dvc.lock → tag image.
4. **Deploy** (`ci.yml`): build image with new model artefact → push
   to ECR + GHCR → `aws ecs update-service --force-new-deployment` →
   Fargate rolls.

### Q: How do you do explainability?

**A.** `POST /explain` runs **occlusion attribution** — drop each
token in turn, re-run inference, report the delta in
predicted-class confidence. Strong positive delta = "removing this
token weakens the prediction, so the token was supporting the
label."

Cheaper than integrated gradients (no torch at serving time). Works
directly against the ONNX session. Math fixed in v2.1 — pre-fix
label-flip case used `baseline + drop` (values >1), corrected to
`baseline - (1 - drop)` so deltas stay in [-1, 1].

Opt-in route (10/min rate limit) because it costs ~K × the latency
of `/predict` where K is the token count.

### Q: What's `/similar` for?

**A.** **TF-IDF nearest-neighbour lookup** over the reservoir-sampled
production inputs. `GET /similar?text=...&k=5` returns the 5 closest
reviews the model has seen, by cosine similarity. Interpretability
surface: "show me which past reviews are most like this new one."

Lazy in-process index with mtime rebuild. Thread-safe. Empty hits on
cold start (when reservoir is empty). Code at `serve/similar.py`,
added in v2.1.

Tier 3 upgrade: replace sklearn `NearestNeighbors` with **pgvector**
or **Pinecone** at 10M+ embeddings.

---

## 6. Security

### Q: How do you handle auth?

**A.** Shared-secret API key via `MS_API_KEY` + `X-API-Key` header,
applied to `/predict`, `/analyze`, `/similar` via `Depends`. Empty
key (default) = demo mode passthrough so the public curl in the
README works without setup.

Production hardening (Tier 2): OAuth2 / Cognito → JWT validation
middleware that replaces `_verify_api_key`. Same `Depends()` pattern,
zero route changes.

### Q: What's the threat model?

**A.** `SECURITY.md` enumerates:

- Untrusted input → RCE / crash (Pydantic validation, hard length
  caps, ONNX-only inference — no `pickle` or `exec` anywhere at
  request time)
- DoS via traffic flood (slowapi per-IP rate limits)
- Log injection via `x-request-id` header (allowlisted regex, CRLF
  stripped)
- HuggingFace namespace takeover (pinned `MS_HF_REVISION` to exact
  commit; serving loads from local DVC-pulled paths, no Hub fetch at
  request time)
- Supply-chain vulnerable Python deps (`pip-audit` on every CI build
  with documented `--ignore-vuln` entries)
- CORS abuse (`Allow-Origin: *` with `allow_credentials=false` —
  browsers reject the combination, so we can't accidentally expose
  cookies)
- Logs leaking PII (predict_complete logs labels + count, never the
  review text)
- Container privilege escalation (`USER app` uid 1000)
- TLS / cert validation in S3 fetch (pinned pyOpenSSL with documented
  residual CVE — DVC S3 traffic uses AWS-signed URLs so MITM with a
  forged cert against S3 is bounded)

Out of scope: multi-tenant authz, side-channel attacks on the ONNX
model.

### Q: What CVEs are accepted?

**A.** 5 documented in `SECURITY.md` § residual risks:

1. **PYSEC-2025-217** (transformers 4.57.x) — fix is in 5.0.0rc3
   which breaks optimum-onnx. Mitigated by pinning revision on
   `from_pretrained`.
2. **PYSEC-2026-161** (starlette 0.52.1) — fix breaks fastapi
   0.136. Vulnerability path requires multipart we don't use.
3. **MAL-2026-4750** (fastapi 0.136.x) — OSV false-positive.
4. **CVE-2025-69872** (diskcache 5.6.3) — no fixed release. DVC
   transitive, not exposed to user input.
5. **CVE-2026-27448 / 27459** (pyOpenSSL 24.2.1) — bumping breaks
   pydrive2 → DVC S3. CVEs bounded to MITM against AWS S3 (AWS-signed
   URLs).

Each has an "accepted because…" line that an interviewer can read.
Don't bullshit if asked which one — they're listed.

### Q: How are model artefacts trusted?

**A.** Today: SHA-pinned HuggingFace fetches at training time;
serving loads from local DVC-pulled artefacts (no Hub fetch at
request time).

Tier 3 upgrade (`docs/scaling.md`): cosign-signed images,
SLSA provenance attestation, model signatures verified at deploy
time.

---

## 7. Testing + CI

### Q: What does CI do?

**A.** `.github/workflows/ci.yml`:

1. **Lint** (ruff) + **format check** (black)
2. **Type check** (mypy strict mode)
3. **Security scan** (bandit, pip-audit with documented ignores)
4. **Test** (pytest, fast suite, 85% coverage gate)
5. **Calibration regression gate** (allows-missing while no eval
   file exists)
6. **Docker build** (ARM64 via QEMU + Buildx)
7. **Push** to GHCR + ECR (only on main)
8. **`aws ecs update-service --force-new-deployment`** — rolls
   Fargate to new image

Weekly: OSSF Scorecard, dependabot, benchmark workflow.

### Q: How much coverage and where?

**A.** **86.50% gated at 85%**. 15 test files covering:

- Data (clean / split / scrape / validate)
- Models (inference / onnx_export / multitask / explain)
- API (api / api_loaded / async_api / cli)
- Monitoring (drift)
- Reservoir
- Property-based (hypothesis on clean function)
- **Adversarial** (char perturbation + synonym swap + unicode/emoji
  edges)
- **Calibration** (Brier helper + CI script)
- **Similar** (cold-start, hot path, validation)

Coverage omits big files that need GPU + dataset to exercise
(`models/transformer.py`, `models/multitask*.py`, the multitask
inference engine). Test design philosophy: stub engines for API
tests (`_StubEngine`); fake AWS clients for async path; Typer
CliRunner for CLI; hypothesis for property tests.

### Q: What's the weakest part of the test suite?

**A.** Be honest: **the adversarial tests run against a stub
engine, not the real ONNX model**. The structure is the contract;
swap `_StubEngine()` → `InferenceEngine.from_registry(...)` to get a
real accuracy-drop number, but that requires the ONNX on disk and
~30s of runtime. CI doesn't run it because of disk + time cost.

Also: the v2 multi-task model is **not covered end-to-end** — only
schema + 503 fallback. Full pipeline test would need the multitask
ONNX on disk.

### Q: What's NannyML doing in here?

**A.** `monitor/perf_estimate.py` runs **NannyML CBPE** (Confidence-Based
Performance Estimation) to estimate production F1 **without ground
truth labels**. The model registers prediction-confidence as a
calibration signal; CBPE projects expected accuracy onto a confidence
histogram conditioned on a labelled reference set.

Why it matters: in production we don't get labels back in real time
("did this user actually find the review positive?"). NannyML lets us
flag F1 regression days/weeks before drift detection would catch it
via input distribution shift. Cheap insurance.

CLI: `moviesentiment perf-estimate` reads
`data/production/labeled_reference.parquet` + `data/production/recent.parquet`
→ prints estimated metrics.

### Q: What's OSSF Scorecard / what does it measure?

**A.** `.github/workflows/scorecard.yml` runs the OpenSSF Scorecard
weekly. It scores the repo on supply-chain security signals:

- Branch protection
- Code-review enforcement
- CII Best Practices
- Dependency-update tools (we use dependabot — boosts the score)
- Pinned dependencies
- Signed releases
- SAST configuration
- Token permissions in GH Actions

The badge in the README links to the public scorecard page. Score is a
signal to recruiters / interviewers that supply-chain hygiene was
explicitly invested in — not the perfect 10/10 but well above the
ecosystem median.

### Q: What's bandit catching?

**A.** Static analysis for known-risky Python patterns:

- `assert` in production code (B101 — skipped in tests)
- Bind on 0.0.0.0 (B104 — skipped because intentional in container)
- `random` for non-crypto reservoir (B311 — skipped, reservoir is
  statistical sampling not crypto)
- `subprocess` with `shell=True` (B602 — caught + denied)
- `pickle.loads` on untrusted data (B301 — caught)
- HuggingFace `from_pretrained` on remote ID (B615 — annotated with
  `# nosec` where the path is local DVC-pulled)
- `torch.load` on untrusted file (B614 — annotated for the multitask
  ONNX export, which loads from a SageMaker-produced local checkpoint)

Each annotation has a reason in the source. Documented choices, not
suppressions.

### Q: What's dependabot doing?

**A.** `.github/dependabot.yml`:

- Weekly Python package update PRs, grouped by ecosystem
- Weekly GitHub Actions version bumps
- Weekly Docker base image bumps (python:3.11-slim)

Grouping reduces PR noise — instead of 30 individual transformers /
huggingface_hub / tokenizers PRs, you get one "ML stack" PR per week.

Combined with `pip-audit` in CI, this closes the "dep with known CVE"
loop on a roughly weekly cadence.

### Q: What's the calibration regression gate?

**A.** `compute_and_log_brier` in `eval/metrics.py` writes
`metrics/brier.json` after every full eval. `scripts/check_calibration.py`
reads it and fails CI if `brier > 0.10`. CI step is
`--allow-missing` so the gate is no-op until a full eval lands the
file — that gives the metric a soft landing in the pipeline.

In Tier 2+ production: gate every retrain. The 0.10 threshold is
conservative — well-calibrated models on this task land around 0.04.

---

## 8. The build journey + pivots

### Q: Walk me through the pivots that mattered.

**A.** Six. Each one was forced by a real failure mode (see
`final_report.md` §9):

1. **Fly.io → AWS ECS Fargate ARM64**. Fly cold-starts on
   `auto_stop_machines=true` blew p99. ECS Graviton was 30% cheaper
   per vCPU + always-warm.
2. **GDrive → S3 DVC remote**. GDrive OAuth flow is fragile in CI
   runners. S3 with IAM keys just works.
3. **`datasets` + pyarrow → plain Python list-of-dicts** in v2
   multi-task. pyarrow's schema unification coerced `-100` sentinels
   and `NaN` to `null`, which surfaced in collate as `Could not infer
   dtype of NoneType`. Fix: keep values in pure Python until the
   collate function. 3-commit chase.
4. **Colab → SageMaker on-demand `ml.g4dn.xlarge`**. Reproducibility
   + DVC artefact tracking + 5-commit dep-pinning chase (transformers
   version cap, accelerate install, `--no-deps`, `sys.path` injection).
5. **Single-head DistilBERT → 5-head multi-task**. Service surface
   no free public API ships in one call. $0.75 one-time training cost.
6. **X86_64 → ARM64 Graviton**. ~30% cheaper Fargate. Required
   `docker/setup-qemu-action` + `linux/arm64` platform pin in CI.

Each one is a commit + a failure mode you can cite by hash.

### Q: What was the hardest debugging story?

**A.** The pyarrow None coercion (v2 multi-task). Symptom: training
crashed in collate with `Could not infer dtype of NoneType`. First
two attempts (cast in collate, then per-field helpers) didn't catch
every path because pyarrow was returning `null` for some rows and
Python None for others depending on whether the schema had been
inferred yet.

Fix that worked: drop the `datasets` library entirely. Use a plain
Python `_ListDataset(list[dict[str, Any]])`. Pyarrow never touches
the values; Python ints stay ints, `float('nan')` stays a float.

Lesson: when sentinel values need to travel through a typed schema
layer, **don't trust the schema layer**.

### Q: What's a decision you'd reverse with hindsight?

**A.** **Live IMDb scraping**. It was unreliable from the start
(Akamai WAF) and rotted twice within a month (persisted-query SHA
rotated, then the response schema renamed `rating` → `authorRating`).
Time spent revivifying it could have gone toward extending the
sentiment scope (multi-language, longer context).

The fix was to make the live path opt-in via `params.yaml::scrape.source`
with HuggingFace as the reliable default. That's what we should
have done day 1.

---

## 9. Curveballs + traps

### Q: What if your model serves a confidently wrong prediction?

**A.** Three layers of defence:

1. **Confidence gating** at the application layer — anything <0.70
   surfaced as "uncertain", not classified. Documented in the model
   card.
2. **Drift detection** catches population-level wrong-confidence
   shifts in <24 h.
3. **Reservoir sampler + Streamlit labeller** (`apps/annotate/`)
   surface low-confidence reviews for human labelling, feed back into
   retraining data. Active-learning loop.

No way to prevent a single wrong prediction. Question is whether the
system is calibrated and whether it learns from the mistake.

### Q: Your test set is 50/50 balanced but real traffic is 70/30
positive. How do you adapt?

**A.** Two things:

1. **At eval time**: report macro F1 (not accuracy) so the imbalance
   doesn't hide minority-class regression. Per-confidence-band
   accuracy + per-genre slice already documented.
2. **At retraining**: re-weighted loss (`class_weight='balanced'` on
   the baseline; sample weights on transformer fine-tune) once the
   reservoir reflects production distribution. Not in v3.0 — it's the
   next step.

The drift report flags this drift as it grows so retraining can react.

### Q: What if HuggingFace deletes your repo or rate-limits the
download?

**A.** Three fallbacks:

1. **S3 mirror** — already in place at
   `s3://moviesentiment-dvc-soumya/multitask_onnx/` (until you run
   the AWS teardown). Cost: $0.03/mo.
2. **DVC `models/distilbert_onnx_int8.dvc`** points at the S3 hash —
   `dvc pull` recovers.
3. **Retrain from source** — SageMaker pipeline + the recorded
   `params.yaml` + `dvc.lock` reproduces the artefact byte-for-byte.

Vendor lock-in is bounded.

### Q: What stops me from billing your AWS account from your demo?

**A.** Three controls:

1. **`MS_API_KEY` rate-limited per-IP** via slowapi (60/min on
   `/predict`).
2. **Hard input caps**: batch size 32, text length 5000 chars.
3. **AWS Budgets alarm at $13/mo** from `scripts/setup_aws_budget.py` —
   email alert at 80% forecast + 100% actual. Won't stop billing but
   will alert in time to scale down.

For real prod: shift to per-user JWT + per-tenant quota + WAF rule
matching abuse signatures. Documented in `docs/scaling.md` Tier 2.

### Q: How would you sell this to your manager?

**A.** Frame by ROI:

- **Effort**: 5 calendar days of build time → portfolio-grade MLOps
  surface.
- **Cost**: $0.37/mo recurring with full demo path; $6/mo for
  always-on live URL.
- **Recruitment leverage**: clickable demo + model card + per-genre
  F1 + scaling doc end-to-end. Hire signal.
- **Knowledge transfer**: `docs/runbook.md` + `docs/slos.md` are
  on-call-ready; new team members onboard against documented
  failure modes, not tribal knowledge.

### Q: Why didn't you use LLM-as-judge / GPT for evaluation?

**A.** Two reasons:

1. **Cost**: 50k eval calls at $0.001/call = $50 per full eval. Run
   weekly = $200/mo. Above the budget ceiling.
2. **Signal noise**: GPT-as-judge has known biases on sentiment
   (over-weights positive framing). For a quantifiable F1 the held-out
   labelled test set is the trusted reference.

LLM-as-judge would be useful for **out-of-distribution detection** —
"is this even a movie review?" That's the next slice. Documented as a
deferred item.

### Q: What's the most over-engineered part?

**A.** Honest answer: the **OpenTelemetry traces to Grafana Cloud**.
The traffic doesn't justify distributed tracing — single Fargate
task, single uvicorn worker, no service-to-service calls. The OTel
instrumentation cost ~1 ms p50 on every request.

Kept it because (a) the load test showed it was usable to break down
the 5 ms HTTP overhead, and (b) Tier 2 onwards traces become
essential. But on a Tier 0 budget, structlog + Prometheus would be
enough.

### Q: What's the most under-engineered part?

**A.** **No HTTPS terminator**. The ECS task is HTTP-only on port 8000.
ALB ($18/mo) is above the budget ceiling. Real production would
absolutely need TLS. Documented in `docs/runbook.md` §3 and
`docs/scaling.md` Tier 2.

Honest about it. Don't pretend otherwise.

### Q: Suppose the model card claims 0.939 F1 but production sees 0.82. What's your debugging order?

**A.** Four checks in order, fastest first:

1. **Schema mismatch** — Pydantic 422s tell you if validation is
   rejecting real traffic. Look at slowapi 429 vs server 500 vs 422
   ratios in Prometheus.
2. **Tokenizer mismatch** — model card says `max_length=512` but if
   the ONNX was exported with 256, anything past 256 is garbage. Check
   `Settings.max_text_length` vs the engine's truncation point.
3. **Distribution drift** — run `moviesentiment drift` against the
   reservoir. If `drift_share > 0.30`, the train set doesn't represent
   prod. NannyML CBPE will agree. Retrain.
4. **Calibration drift** — Brier score regression catches "model still
   answers but confidence is no longer trustworthy". 30-day comparison.

The cheap checks land first because they're a 5-minute fix; the
expensive ones (retraining) only if cheap ones don't explain it.

### Q: Walk me through what happens during a SageMaker training failure.

**A.** Recovery path documented in `docs/runbook.md` plus the build
journey:

1. **Job submitted via `scripts/sagemaker_launch.py`** — boto3
   `sagemaker.estimator.PyTorch.fit()` returns a job name.
2. **Failure surface**: `aws sagemaker describe-training-job
   --training-job-name <name>` returns `FailureReason` + last
   CloudWatch log lines.
3. **Common failure modes** (each documented in the build journey):
   - Source bundle too big → exclude ONNX dirs from `source_dir`
   - SageMaker's CUDA torch clobbered by user requirements → use
     `pip install --no-deps`
   - `transformers 5.x` breaks `optimum-onnx` → cap `<5.0`
   - `src/` not on `sys.path` → inject before import
   - DVC remote naming mismatch → align across workflows
4. **Spot interruption recovery** — `max_wait` + `checkpoint_s3_uri`
   in the estimator. SageMaker resumes from the last saved checkpoint.

The v2 multi-task run failed 4 times before succeeding. Each failure
fixed a downstream issue. **The model code was the easy part — the
plumbing was the hard part.**

### Q: How do you keep the cost ceiling enforced over time?

**A.** Three controls compounding:

1. **AWS Budgets alarm at $13/mo** (`scripts/setup_aws_budget.py`):
   email at 80% forecast + 100% actual. Idempotent script so you can
   re-tune the limit.
2. **Documented per-service cost in `final_report.md` §11 + §7**:
   every line item has a $/mo annotation. Re-running `aws ce
   get-cost-and-usage` periodically validates the table didn't drift.
3. **`docs/aws_teardown.md`** — one-command "$0 right now" recipe.
   If costs spike unexpectedly, you can tear down in 60 seconds and
   redeploy from HF + CI.

This isn't theoretical: the project today sits at $0.37/mo because the
teardown was applied during the cleanup phase. The $13/mo ceiling
held over five days of active build + monitoring.

### Q: What happens if you 10× the request volume tomorrow?

**A.** Four-stage response:

1. **Immediate (today's infra)**: load test in `docs/loadtest.md`
   shows the single Fargate task saturates at ~120 RPS. 10× = 1.2k
   RPS = 503s. The slowapi rate limiter would cap incoming traffic at
   60/min/IP, but a real DDoS bypasses by IP rotation.
2. **Vertical scale**: bump Fargate task to 0.5 vCPU (+$3.50/mo).
   Doubles headroom to ~240 RPS. Inside the budget ceiling.
3. **Horizontal scale via ECS service** to 10 tasks (+$54/mo). Linear
   scale-out behind ALB. Above the ceiling — requires raising the
   budget.
4. **Burst spikes → SQS async path**. `POST /predict/async` decouples
   client latency from inference throughput; Lambda scales-to-zero
   when idle. For batch / non-realtime requests, this is the right
   answer regardless of scale.

At 100×: documented in `docs/scaling.md` Tier 2 onwards — Triton +
dynamic batching + Karpenter + multi-region.

### Q: The interviewer says "this is way too much work for a sentiment classifier — why didn't you just use a managed service like AWS Comprehend?"

**A.** Three responses, escalating:

1. **Cost**: AWS Comprehend is $0.0001 per 100-char request. At 10k
   req/day = $0.30/day = $9/mo *just for sentiment*. The full stack
   here is $0.37/mo with sentiment + ABSA + emotion + spoiler +
   helpfulness — a ~25× capability surface at a fraction of the cost.
2. **Lock-in**: Comprehend doesn't expose the model, the training
   data, or the calibration. You can't retrain on your domain, can't
   slice F1 by genre, can't run adversarial robustness tests. For an
   MLE / MLOps role, the interview signal is precisely the
   end-to-end ownership Comprehend hides.
3. **Honesty**: for a real *sentiment-only* use case in a startup
   with no MLE team, Comprehend is the right answer for the first
   year. This project doesn't claim otherwise — see the comparison
   table in `final_report.md` §10. The point is that *this is the
   lifecycle you'd build once Comprehend stopped being enough*.

### Q: What's the worst failure mode you haven't fully mitigated?

**A.** **Cold-start on the ECS service**. When the service scales from
0 → 1 task (interview-day pattern), the new task takes ~90 s to:

1. Pull the ECR image (3.79 GB)
2. Start uvicorn
3. Run the lifespan loader (load v1 ONNX, attempt S3 bootstrap for v2)
4. Pass the ALB health check on `/healthz`

If the demo URL is hit during that 90-second window, requests time out.
Documented in `docs/runbook.md` §3 with the spin-up command preceded
by `--desired-count 1` 5 min before the demo.

Real prod (Tier 2+): keep min_capacity=1 in the auto-scaling group.
Costs $6/mo to eliminate cold-starts entirely.

### Q: How do you know your data is correct after the live IMDb scrape?

**A.** Three layers of validation:

1. **`src/moviesentiment/data/validate.py`** — Great Expectations
   patterns checked at the DVC `validate_data` stage. Schema (cols
   present), types, null thresholds, value ranges (rating ∈ {1..10}),
   character-set validity. Output: `metrics/data_quality.json`. Fails
   the pipeline if expectations are violated.
2. **Cleaning function** (`data/clean.py`) — lowercases, strips HTML,
   removes URLs, drops duplicates. Property-based tests
   (`tests/test_clean_hypothesis.py`) verify no HTML tags survive,
   text length ≤ 5000, no null injection.
3. **Split function** (`data/split.py`) — fixed-seed 70/15/15
   stratified. Asserts each split has both labels present.

For the live scrape: per-movie row count printed in the script; visual
sanity check on the rating distribution. If a movie has <50 reviews,
it's flagged in the per-genre aggregation as a small-N caveat.

### Q: How do you reproduce a specific MLflow run?

**A.** Three artefacts pinned per run:

1. **`params.yaml`** — every hyperparameter (logged via
   `mlflow.log_params(params)`)
2. **`dvc.lock`** — data pin (hashes of every input parquet)
3. **Git SHA** — code pin (logged as MLflow tag `git_sha`)

To replay run X: `git checkout <sha>`, `dvc checkout <dvc.lock>`,
`python -m moviesentiment.cli train transformer`. Byte-identical
output (modulo GPU non-determinism — set
`PYTHONHASHSEED=42 CUDA_DETERMINISTIC=1` if you need exact bit
reproducibility).

### Q: One sentence — what's the single most important thing in this project?

**A.** "The closed loop from production drift back to deployed model
without human intervention, documented and tested at every step."

Variation for the interviewer who only wants the headline: "Shipping
the lifecycle, not the model."

---

## 10. Behavioral

### Q: Why this project?

**A.** Three reasons:

1. **Demonstrates the full lifecycle**, not just modelling.
   Interviewers reliably ask "can I curl it" — notebook portfolios
   fail. This one passes.
2. **Forces engineering judgement**. The $13/mo ceiling made every
   decision earn its keep.
3. **Sentiment is a deliberately boring problem**. Interviewers
   focus on how I ship, not on whether I picked a novel modelling
   challenge.

### Q: What did you learn?

**A.** Three lessons that transfer to any future project:

1. **Cloud training is mostly path / version plumbing**, not model
   code. The 6-commit SageMaker chase is the case study.
2. **Don't trust typed schema layers with sentinel values**. Pyarrow
   coercing `-100` → `null` cost three commits to chase.
3. **Pivots compound**. Fly → ECS unlocked Graviton; Graviton
   unlocked $6/mo; $6/mo unlocked "real prod feel under $13/mo
   ceiling". Each forced an architectural reconsideration that
   produced better answers than the original choice.

### Q: What would you do differently next time?

**A.** Two:

1. **Skip the live scraper entirely** until I had a stable model.
   Use HuggingFace dataset from day 1. The Akamai WAF dance was a
   distraction.
2. **Use HuggingFace Hub as the model store from day 1** instead of
   migrating after the fact. S3 + DVC for code/data + Hub for models
   is the right separation.

### Q: How do you handle disagreements with reviewers?

**A.** Bring evidence. Example: when the CI calibration gate
threshold was being set, the choice between 0.05 (tight) and 0.10
(loose) was settled by running the v1 ONNX over 100 bootstrap
resamples and showing the Brier 95% CI was [0.039, 0.058]. 0.10
buys headroom for retraining without churning on noise. 0.05 would
have failed CI on every retrain.

Disagreements that don't resolve with evidence → defer to the call
that's cheapest to reverse. Document the rationale either way.

---

## 11. Anti-patterns I deliberately avoided

When asked "what would you NOT do":

- **Kubernetes for a single service**. Documented in
  `MOVIESENTIMENT_BUILD_GUIDE.md` §10 — "adding K8s because MLOps"
  is the senior-engineer red flag.
- **Airflow for 3 cron jobs**. Same.
- **Storing model binaries in git**. DVC + HF Hub instead.
- **Training inside CI on every push**. Manual + cron only. Compute
  cost is real.
- **Putting credentials in `.env.example`**. GitHub Secrets only;
  `.env` is gitignored.
- **A separate notebook per experiment**. MLflow logs everything.
- **`--no-verify` on git commits**. Pre-commit hooks exist for a
  reason. Failure to pass means fix the issue, not skip it.
- **Premature abstractions**. Three similar lines is better than a
  premature factory.

---

## 12. Pre-interview checklist

Run through this 24 hours before:

- [ ] Pull latest `main` (`git fetch && git checkout v3.0`)
- [ ] `huggingface-cli download` both repos so `make serve` works
      offline
- [ ] `make serve` + curl `/predict` end-to-end — verify it works
- [ ] Open `http://localhost:8000/ui/` Custom IP → localhost — verify
      frontend renders
- [ ] Skim README, `docs/model_card.md`, `final_report.md` §13
      summary
- [ ] Re-read this doc's §1, §3, §4, §9 — the highest-density
      questions
- [ ] Confirm GitHub repo + HF repos load on a guest profile
- [ ] One-line answers ready for: "biggest tradeoff", "would do
      differently", "scale to 100×"
