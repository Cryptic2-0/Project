# Interview Talking Points

Reference for the "walk me through your project" question. 2–3 sentences per bullet.

---

## Architecture decisions

**Why DistilBERT over BERT-base?**
DistilBERT is 40% smaller and 60% faster than BERT-base while retaining ~97% of its performance. For a sentiment classification task on short reviews, the accuracy gap is <1%. This tradeoff matters in production: lower latency = fewer idle uvicorn workers = lower Fly.io bill.

**Why ONNX + INT8 quantization?**
ONNX Runtime on CPU is ~2x faster than PyTorch eager mode for inference. Dynamic INT8 quantization (weights only) reduces model size from ~260 MB to ~65 MB and cuts p50 latency further with <1% accuracy drop. The full pipeline is: HF Trainer → ONNX export via `optimum` → `quantize_dynamic` → ORT session.

**Why SQLite for MLflow?**
Zero-infra, ships in one file (`mlflow.db`), sufficient for a solo project. In production I'd use Postgres (concurrent writes, proper locking) with S3 artifact store. The code is already parameterized via `MS_MLFLOW_TRACKING_URI` so swapping is one env-var change.

**Why Fly.io over Heroku/Render?**
Fly deploys a real Docker container (not a buildpack), supports persistent volumes, and has a generous free tier with 3 shared-CPU VMs. More importantly, the `fly.toml` + `flyctl deploy --image` flow mirrors what you'd do with ECS or Cloud Run — it's transferable knowledge.

**Why GitHub Actions cron instead of Airflow?**
Airflow is overkill for a project with 3 pipelines running once a week. The overhead of a scheduler, executor, metadata DB, and web UI would dwarf the actual pipeline code. For >5 interdependent pipelines or complex fan-out I'd reach for Dagster.

---

## Data decisions

**Why Parquet?**
Parquet is columnar, compressed, and pandas-native. For a 25k-row dataset it's 5x smaller than CSV. DVC can hash Parquet files efficiently. The schema is enforced at write time.

**Why 70/15/15 split?**
Standard for NLP. Val set is large enough (~3750 samples) to get stable F1 estimates without overfitting training decisions to test. Fixed seed in `params.yaml` makes the split reproducible across runs.

**Why label threshold ≥7 / ≤4?**
Ratings 5–6 are genuinely ambiguous — the model would learn noise. Dropping the middle third gives cleaner signal. This is documented in `docs/error_analysis.md` with examples.

---

## MLOps decisions

**What does DVC buy you here?**
Reproducibility and data lineage. Anyone can `dvc pull` + `dvc repro` and get byte-identical artifacts. The pipeline DAG in `dvc.yaml` documents dependencies explicitly — CI knows exactly which stages to re-run when `params.yaml` changes.

**How does model promotion work?**
New models register in MLflow as `Staging`. The `train.yml` workflow compares new F1 vs `CURRENT_PROD_F1` (a GitHub repo variable). If improvement ≥0.5%, it promotes to `Production` and triggers a Fly.io redeploy. The API reads the `Production`-stage model at startup.

**How do you detect drift?**
Evidently `DataDriftPreset` runs on production request logs (sampled 10%) vs the training distribution. Drift is measured on text length and word count (statistical test: Wasserstein distance). If `drift_share > 0.3` the weekly retrain triggers early via `workflow_dispatch`.

---

## Things that would be different at scale

| Portfolio | Real production |
|-----------|-----------------|
| SQLite MLflow | Postgres MLflow + S3 |
| 10% random sampling | Reservoir sampling + stratified by confidence |
| 1 Fly.io VM | ECS/GKE with HPA |
| GitHub Actions cron | Dagster/Airflow |
| Single-model serving | Shadow deploy → A/B → promote |
| No auth on /predict | API key + rate limiting |
