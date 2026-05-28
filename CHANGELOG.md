# Changelog

All notable changes to MovieSentiment. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
  ECS service + Lambdas torn down post-build; S3 + ECR + log group +
  task definition family alive. `multitask_onnx/` artefacts intact.
- **Permission gaps** for the `moviesentiment-ci` IAM user catalogued
  with the operations each blocks.

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
