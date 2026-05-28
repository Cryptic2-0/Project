# Changelog

All notable changes to MovieSentiment. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [v3.1] — 2026-05-29

Interview-prep + scaling documentation pass. No code changes.

### Added

- **`docs/scaling.md`** — 4-tier scaling ladder. Tier 0 ($0–6/mo, today)
  → Tier 1 (high-end single GPU) → Tier 2 (small team prod, ~$250–500/mo)
  → Tier 3 (mid-size company, ~$2k/mo) → Tier 4 (enterprise, $2–10M/yr).
  Cross-tier table of what stays the same vs what scales.
- **`docs/interview_prep.md`** — 12-section exhaustive Q&A drill book.
  Every decision + likely interviewer follow-up + curveballs.
  ~50 Q&A pairs covering architecture, cost choices, model & data,
  production engineering, observability, security, testing, build
  journey, behavioral.

### Changed

- README + `final_report.md` link to the two new docs.

---

## [v3.0] — 2026-05-29

Project marked **COMPLETE**. Models published to HuggingFace Hub; AWS
scaled to storage-only (~$0.37/mo); demo + teardown docs added.

### Added

- **`docs/demo_walkthrough.md`** — full step-by-step to run the model
  + frontend UI + Grafana stack locally. Replaces the now-stale "live
  AWS URL" demo path. Includes troubleshooting + interview-flow
  checklist.
- **`docs/aws_teardown.md`** — step-by-step to drive AWS spend to $0/mo.
  HF Hub is the source of truth for ONNX models, so deleting S3 + ECR
  is safe. Includes a 60-second "$0 right now" recipe.

### Changed

- `final_report.md`: marked COMPLETE (v3.0); top-of-file public-artefacts
  table with HF URLs + demo + teardown doc links.
- README.md: pointer to demo_walkthrough.md + aws_teardown.md from the
  Quickstart block.

### Removed

- Deleted `models/distilbert_multitask_onnx/model_fp32.onnx` (253 MB,
  redundant — INT8 is the production artefact; FP32 was only the
  intermediate from the legacy TorchScript exporter).

---

## [v2.3] — 2026-05-29

HuggingFace Hub publish of v1 + v2 ONNX models. Replaces S3 as the
public-facing artefact store.

### Added

- **`scripts/push_to_hf.py`** — idempotent push of both INT8 ONNX
  bundles to HF Hub. Stages `docs/model_card.md` as the repo README.
  Skips FP32 ONNX to save bandwidth.
- **`make hf-push`** target wrapping the script.
- README: HF badges + quickstart now offers HuggingFace as the primary
  artefact source (Option A); DVC + S3 as the alternative.

### Published

- https://huggingface.co/Cryptic2-0/moviesentiment-distilbert-onnx-int8
- https://huggingface.co/Cryptic2-0/moviesentiment-multitask-onnx-int8

---

## [v2.2] — 2026-05-29

Admin-key AWS audit pass + CloudWatch retention bump. No code changes
to source.

### Changed

- **CloudWatch retention** on `/ecs/moviesentiment` bumped from unset
  (never expire) to **90 days** via
  `aws logs put-retention-policy --retention-in-days 90`. Verified.
- Appendix C of `final_report.md` finalised with full verified AWS state
  (SQS empty, DynamoDB empty, EventBridge empty, Lambda empty in
  ap-southeast-2 + us-east-1, S3 + ECR + task-def family alive).

### Documented

- Permission gap inventory: `moviesentiment-ci` IAM user is CI-scope
  only; lacks `logs:PutRetentionPolicy`, `sqs:ListQueues`,
  `dynamodb:ListTables`, `events:ListRules`, `s3:ListAllMyBuckets`.
  `moviesentiment-admin` user created for the audit and rotated within
  the same session.

---

## [v2.1] — 2026-05-29

Portfolio-hardening pass: correctness fixes, shared-secret auth, ops docs,
new endpoints (`/similar`), regression gates (calibration, label drift,
adversarial), Streamlit active-learning labeller, live IMDb scrape revived
with per-genre F1.

### Added

- **API key auth** via `MS_API_KEY` + `X-API-Key` header on `/predict`,
  `/analyze`, `/similar`. Empty default = demo mode preserved.
- **`GET /similar?text=...&k=N`** — TF-IDF nearest-neighbour over the
  reservoir. Lazy in-process index with mtime rebuild. Thread-safe.
- **Calibration regression gate** — `compute_and_log_brier` in
  `eval/metrics.py`, CI step (`scripts/check_calibration.py`,
  `--allow-missing` while no eval file is present).
- **Label-drift detection** — `monitor/drift.py::label_drift` returns
  total-variation distance between reference and current label shares.
  Complements the existing input-feature drift.
- **Adversarial robustness tests** — `tests/test_adversarial.py`. Char
  perturbation, synonym swap, unicode/emoji/case edges. 8 tests.
- **AWS Budget setup script** — `scripts/setup_aws_budget.py`. Idempotent;
  creates a $13/mo budget with email alerts at 80% forecast + 100% actual.
- **Genre-enrichment script** — `scripts/enrich_with_genre.py`. Joins an
  IMDb genre CSV onto a reviews parquet.
- **Streamlit active-learning labeller** — `apps/annotate/app.py`. Reads
  low-confidence reservoir rows, single-row UI, appends to
  `data/labeled/augmentation.parquet`. `streamlit` added to `[dev]` extras.
- **Documentation**: `docs/slos.md`, `docs/runbook.md` (9 playbooks),
  `docs/shadow_canary.md`, `docs/demo_script.md`, `final_report.md`
  (full build report).
- **Per-genre F1 numbers** in `docs/model_card.md` from a live IMDb
  scrape — 2,525 reviews across 10 movies. Action lowest macro F1
  (0.779), Adventure highest (0.848).

### Changed

- **Live IMDb scraper** revived (`src/moviesentiment/data/scrape.py`):
  rotated persisted-query SHA to `286aee...389505`, switched POST → GET
  per the imdb-web-next-localized client, cookies via `MS_IMDB_COOKIE`
  env var. New schema key `authorRating` accepted alongside legacy
  `rating`.
- **Dockerfile**: drop `.[train]` extras from the runtime image; the
  SageMaker container installs them itself. ~150 MB saved at serve time.
- **Multi-task inference** `max_length` 512 → 256 to match training.
- **README**: live-demo banner reflects torn-down ECS state; new endpoint
  rows for `/analyze`, `/similar`, `/insights`; new docs index entries;
  `X-API-Key` auth note.
- **SECURITY.md**: API-key section + hardened-config block updated.

### Fixed

- **Occlusion attribution** (`serve/explain.py`): label-flip math used
  `baseline + drop` which produced values >1. Replaced with
  `baseline - (1 - drop)` so deltas stay in [-1, 1]. Test updated.
- **Multi-task final-loss division** (`models/multitask_train.py`):
  off-by-N denominator when `step` was a multiple of 50 at the end of
  training. Tracked `window_steps` separately.
- **Defensive `async_api` import** in `serve/api.py`: wrapped in
  try/except so an env-config failure cannot prevent app startup.

### Documented (no code change)

- **AWS live state** verified 2026-05-29 (Appendix C of `final_report.md`).
  ECS service + Lambdas + SQS + DynamoDB + EventBridge all torn down
  post-build; S3 + ECR + log group + task definition family alive.
  `multitask_onnx/` artefacts intact (~67 MB).
- **CloudWatch retention** on `/ecs/moviesentiment` bumped to 90 days
  (was unset / never expire). 842 KB currently stored.
- **Permission gaps** for the `moviesentiment-ci` IAM user catalogued
  alongside the per-operation IAM-class requirements.

---

## [v2.0] — 2026-05-26

ECS deploy, mermaid arch diagram, `/version` endpoint fix. See git tag
`v2.0` for full diff.

## [v1.0] — 2026-05-25

First production-ready release. SageMaker-trained DistilBERT, ONNX INT8,
ECR + ECS push pipeline. See git tag `v1.0`.

## Earlier tags

`v0.3-deployed` — drift detection + retraining pipeline.
`v0.2-serving`  — multi-stage Dockerfile + Grafana provisioning.
`v0.1-baseline` — eval rigor (ROC, PR, calibration, error analysis).
`v0.0-scaffold` — initial project scaffold.
