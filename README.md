# MovieSentiment

> Production-style sentiment analysis for IMDb reviews. Built end-to-end as an MLOps portfolio project.

[![ci](https://github.com/Cryptic2-0/moviesentiment/actions/workflows/ci.yml/badge.svg)](https://github.com/Cryptic2-0/moviesentiment/actions/workflows/ci.yml)

**Status:** Work in progress — Week 1 (data pipeline + baseline model)

## Architecture

```
IMDb (web) → raw Parquet (DVC) → clean → split
                                         ↓
                          TF-IDF + LR   DistilBERT FT
                                    ↘ ↙
                             MLflow Tracking + Registry
                                     ↓
                              FastAPI (ONNX, Docker)
                                     ↓
                              Fly.io / HF Spaces
                                     ↓
                         Prometheus + Evidently drift
```

## Quickstart

```bash
git clone https://github.com/Cryptic2-0/moviesentiment
cd moviesentiment
uv pip install -e ".[dev]"
dvc pull
make train
make serve
```

## Results

| Model | Accuracy | Macro F1 | p95 latency (CPU) | Size |
|---|---|---|---|---|
| TF-IDF + LR | TBD | TBD | TBD | TBD |
| DistilBERT ONNX-INT8 | TBD | TBD | TBD | TBD |

## What I'd do differently in real production

- Replace the SQLite MLflow backend with a managed Postgres + S3 artifact store.
- Move from GitHub Actions cron to Airflow/Dagster once we have >5 pipelines.
- Add a feature store (Feast) once features are shared across models.
- Shadow-deploy new models for 24h before promotion instead of relying on offline metrics alone.

## Roadmap

- [ ] Multi-language support (es, fr)
- [ ] Active learning loop on low-confidence predictions
- [ ] A/B testing harness

## License

MIT
