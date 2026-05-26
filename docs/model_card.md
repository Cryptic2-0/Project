# Model Card — MovieSentiment DistilBERT INT8

Following the [Mitchell et al. 2019 Model Card](https://arxiv.org/abs/1810.03993) template.

## Model details

- **Model name**: `moviesentiment-classifier`
- **Architecture**: DistilBERT-base-uncased (66M parameters), HuggingFace `distilbert-base-uncased`
- **Head**: 2-class linear classifier on `[CLS]` pooled output
- **Quantization**: ONNX dynamic INT8 (8-bit weights, FP32 activations) via `optimum.onnxruntime.ORTQuantizer` with `AutoQuantizationConfig.avx512_vnni(is_static=False)`
- **Size**: 64 MB (down from 256 MB FP32; 4× compression)
- **Framework versions**: `transformers>=4.46`, `optimum[onnxruntime]>=1.17`, `onnxruntime>=1.17`
- **Owner**: Soumya Sarkar — `github.com/Cryptic2-0`
- **License**: MIT
- **Date**: 2026-05-26 (last retrained from main)
- **Git SHA**: returned at `GET /version`

## Intended use

**Primary use**: classify English IMDb-style movie reviews into positive/negative sentiment.

**Primary users**: this project is a portfolio MLOps demonstration. The API is publicly reachable but is not intended for downstream products.

**Out-of-scope**:
- Non-English text — see *Quantitative analyses* below for the cliff-edge accuracy drop.
- Product, food, news, or any non-movie domain — not evaluated; results unreliable.
- Long-form text > 5000 characters — truncated at 512 tokens.
- Multi-class sentiment (neutral, mixed, sarcastic) — binary output only.
- Any safety-, health-, or finance-sensitive decision.

## Factors

| Factor | Buckets evaluated |
|---|---|
| Review length | short (<200 chars), medium (200–600), long (>600) |
| Sentiment intensity | strong (>0.9 confidence), weak (0.55–0.7) |
| Genre signal | comedy / drama / action / horror (from movie metadata, where available) |
| Named entities | reviews mentioning specific actors / directors |
| Sarcasm proxy | reviews flagged by a heuristic sarcasm detector (manual seed list) |

## Metrics

Test set: held-out 5,000 reviews from IMDb 50K.

| Metric | DistilBERT INT8 | TF-IDF + LR (baseline) |
|---|---|---|
| Macro F1 | **0.939** | 0.904 |
| Accuracy | 0.940 | 0.905 |
| ROC AUC | 0.984 | 0.962 |
| Latency p50 | 6.8 ms | <5 ms |
| Latency p99 | 14 ms | <8 ms |
| Size on disk | 64 MB | 18 MB |

CI for F1 (bootstrap, 1000 resamples): **[0.934, 0.944]**.

## Quantitative analyses

### Performance by review length

| Length bucket | F1 | n |
|---|---|---|
| Short (<200 chars) | 0.917 | 1,402 |
| Medium (200–600) | 0.940 | 2,217 |
| Long (>600) | 0.951 | 1,381 |

Model is weakest on very short reviews (terse one-liners with weak signal). This matches intuition — fewer tokens, fewer keywords.

### Performance by confidence bucket

| Confidence | Accuracy | Coverage |
|---|---|---|
| >0.95 | 0.984 | 78% of test |
| 0.85–0.95 | 0.913 | 14% |
| 0.70–0.85 | 0.760 | 6% |
| <0.70 | 0.512 | 2% |

The model is **well-calibrated above 0.85** — high-confidence predictions are reliable. The thin (~2%) low-confidence band is essentially chance-level and should be treated as "model declined to answer."

### Sarcasm slice

Reviews flagged by the heuristic sarcasm seed list (~340 examples): **F1 0.823**, vs 0.939 overall — a 12-point absolute drop. This is the most prominent failure mode and is documented on the live demo card.

### Non-English

Spanish translations of the test set (machine-translated via NLLB): **F1 0.51** (chance is 0.50). Treat any non-English input as untrusted. The frontend does not gate on language; this is intentional (interview material), but a production deployment should.

## Ethical considerations

- **Training data is movie-only**. Generalization to other "subjective text" domains is not implied.
- **No demographic factors are inferred or stored**. The pipeline does not capture reviewer identity.
- **Reservoir-sampled production inputs** are stored under `data/production/recent.parquet` for drift detection. The current sampler is in-memory and flushed every 100 inserts; users who post personally identifying text are subject to this storage for the lifetime of the running container.
- **No human-in-the-loop**. Predictions are returned directly.

## Caveats and recommendations

1. Use confidence to gate downstream actions — anything below 0.70 should be surfaced as "uncertain", not classified.
2. The model was trained on IMDb 50K which is **balanced** (50/50 positive/negative). Real-world review distributions skew positive (~70/30). Expect more false negatives at deployment time.
3. If you need higher recall on negative reviews (e.g., for moderation), retrain with a re-weighted loss or threshold-tune via the bundled `eval/metrics.py` PR-curve helpers.
4. Retraining loop closes when Evidently drift > 0.30 (`monitor/drift.py`) — fires a GH Actions workflow to scrape, retrain (SageMaker spot + LoRA), re-export ONNX, redeploy.

## Environmental impact

Trained once on `ml.g4dn.xlarge` (T4 GPU) for ~25 minutes. Estimated energy: ~0.05 kWh. With CodeCarbon's default Australia grid factor (~0.66 kgCO2/kWh) that's ~33g CO2 per training run. The retraining loop runs at most weekly.

Inference: ARM Graviton Fargate, 0.25 vCPU, 1 GB RAM, single task. Steady-state ~3 W. Per-prediction CO2 cost is negligible at portfolio traffic.

## Where to find more

- Source: `github.com/Cryptic2-0/Project`
- Live API contract: `OpenAPI /docs` on the running Fargate task
- Drift reports: `docs/drift_reports/` (HTML, generated by `python -m moviesentiment.cli drift`)
- Latency benchmarks: `docs/benchmarks.md`
- Load test report: `docs/load_test_report.html`
- Datasheet for the training data: `docs/datasheet.md`
