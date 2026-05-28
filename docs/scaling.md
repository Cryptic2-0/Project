# Scaling — From $6/mo to Enterprise

> Four tiers. Pick the one that matches your org size + budget. Today's
> v3.0 deployment sits at Tier 0 (zero recurring). Each subsequent tier
> documents the move + the cost + what it buys.

Last updated: 2026-05-29.

---

## Tier 0 — Demo / portfolio (where v3.0 lives today)

Goal: prove the architecture works end-to-end. ~$0–6/mo.

| Component | Choice | Cost |
|---|---|---|
| Inference | Single ECS Fargate 0.25 vCPU / 1 GB or local `make serve` | $0–6/mo |
| Storage | HuggingFace Hub (free) + S3 (~$0.03/mo) | $0–0.05/mo |
| MLflow | SQLite | $0 |
| Drift | Evidently weekly cron via GH Actions | $0 |
| Monitoring | Prometheus + Grafana local (`docker-compose`) | $0 |
| CI/CD | GitHub Actions free tier | $0 |
| Cost ceiling | $13/mo hard ceiling via AWS Budgets | $0 (the alarm) |

**Saturates around 120 RPS** on the single Fargate task per
`docs/loadtest.md`. Documented as the deliberate floor — portfolio
project, not real prod.

---

## Tier 1 — High-end single device / GPU workstation

Goal: max single-box throughput. ~$0 incremental over Tier 0 (uses
existing hardware).

| Lever | Gain | How |
|---|---|---|
| GPU inference | 10–50× throughput | `onnxruntime` provider switch: `CPUExecutionProvider` → `CUDAExecutionProvider`. ~30 ms p50 on RTX 4090 batch 32 (vs 6.8 ms CPU single). GPU wins when batch >= 8. |
| TensorRT backend | 2–4× over ORT-CUDA | `optimum.exporters.onnx → trt` exporter. Kernel fusion + INT8 calibration. Works with the v1 / v2 ONNX directly. |
| Dynamic batching | 5–10× throughput | LitServe (deferred item in `docs/future_improvements.md` §1.13). Same Pydantic schemas. Drop-in. |
| FP8 / INT4 weights | 2× smaller, 1.5× faster | bitsandbytes or AWQ. Need calibration set; expect ~1% accuracy drop. |
| ORT execution providers | Free 10–30% | `OpenVINOExecutionProvider` on Intel CPU; `DmlExecutionProvider` on Windows + iGPU. |

Saturates around **5k QPS on a single RTX 4090** with batch 32, mixed
`/predict` + `/analyze` traffic.

**Best use:** internal team demo, on-prem batch scoring, edge gateway
device, single-customer SaaS.

---

## Tier 2 — Small team / startup production (10–100 QPS sustained)

Goal: real SLOs, multi-AZ, no manual ops. ~$200–500/mo.

### Upgrade path from v3.0

```text
Tier 0                          Tier 2
─────────                       ─────────
1× ECS Fargate task         →   3–10× behind ALB, auto-scale on CPU
SQLite MLflow               →   RDS Postgres MLflow + S3 artefact store
Rotating ENI IP             →   ALB + Route53 + ACM cert (stable URL)
uvicorn workers=1           →   Triton Inference Server (dynamic batching)
No cache                    →   CloudFront in front of ALB
Local Prom + Grafana        →   AWS Managed Prometheus + Grafana Cloud Pro
Weekly cron drift           →   EventBridge + drift-share threshold
GH Actions cron             →   SageMaker Pipelines (once >5 jobs coordinate)
.env secrets                →   AWS Secrets Manager + IRSA
X-API-Key                   →   OAuth2 / Cognito + JWT
```

### Cost breakdown (Tier 2)

| Line | Monthly |
|---|---|
| ECS Fargate 3-10 tasks (auto-scale) | $25–80 |
| ALB + ACM cert + Route53 | $20 |
| RDS Postgres `db.t4g.micro` (MLflow + drift state) | $15 |
| Managed Prometheus + Grafana Cloud Pro | ~$60 |
| S3 + CloudFront + data transfer | $30 |
| SageMaker training (~10 hr/mo spot) | $80 |
| CloudWatch Logs + alarms | $20 |
| **Total** | **~$250–300/mo** |

### SLOs achievable at Tier 2

| SLI | Target |
|---|---|
| Availability | 99.9% (3 9s) |
| `/predict` p99 latency | 30 ms |
| Drift → retrain trigger | <1 h |
| Cold-start to live URL | <90 s |

Same `docs/slos.md` template, tighter numbers.

---

## Tier 3 — Mid-size company (1k–10k QPS, multi-region)

Goal: multi-region, multi-tenant ergonomics, model A/B at scale.
~$2–10k/mo.

### Infrastructure

- **EKS with Karpenter**, replacing Fargate. Spot fleet for 70% of
  nodes. Right-sizing per workload. Bin-pack inference + training onto
  the same nodes overnight.
- **Multi-region active-active**: us-east-1 + eu-west-1 +
  ap-southeast-2. CloudFront with origin failover. Route53
  geo-routing.
- **Service mesh** (Istio or AWS App Mesh): per-route timeouts,
  retries, circuit breakers, mTLS between services. Replaces ad-hoc
  `requests.Session()` resilience.
- **Centralised model registry**: MLflow → Vertex AI Model Registry
  or SageMaker Model Registry with model signing (cosign) and SLSA
  provenance attestation.
- **Vector DB**: replace sklearn `NearestNeighbors` in `serve/similar.py`
  with **pgvector** ($0 if on RDS) or **Pinecone** / **Weaviate** at
  scale. 10M+ embeddings indexed. Latency stays sub-10 ms.

### Inference

- **NVIDIA Triton Inference Server** with `tensorrt` + `onnxruntime` +
  `python` backends. Multi-model on the same GPU. Per-model autoscaling
  + dynamic batching + ensemble graphs. Saves ~$8k/mo vs single-model
  per-node deployments.
- **Model warm pools** by latency tier: free tier → 30 ms warm-from-cold,
  premium → 5 ms always-warm.
- **Speculative decoding** if you move to a generative head later (not
  relevant for sentiment — flag it as future-proofing).

### ML platform

- **Feature store** (Feast or Tecton): online + offline. Lets multiple
  models share features (movie_id × review_count × historical_rating).
  Today each model re-derives. At Tier 3 you can't afford the duplication.
- **Pipeline orchestration**: Dagster (asset-aware) or Airflow.
  Replaces GH Actions cron once >5 jobs coordinate. Managed Dagster
  Cloud or self-hosted on K8s.
- **Distributed training**: Ray + DeepSpeed, or torchrun with
  `accelerate`. SageMaker multi-instance for >100M-param models.
  Gradient checkpointing + activation offload for big models on smaller
  GPUs.
- **Experiment tracking**: W&B or Comet for cross-team experiments.
  MLflow is fine for solo work; W&B's collaboration UI matters with
  multiple data scientists.

### Data + retraining

- **Streaming**: Kafka / Kinesis → Flink or Spark Structured Streaming.
  Real-time feature freshness for `/insights/{movie_id}`. The hourly
  Lambda becomes an exception path for cold movies.
- **Continuous training**: TFX or Kubeflow Pipelines triggered on
  drift_share threshold + label-drift threshold + scheduled. Reuses the
  `label_drift` function from `monitor/drift.py`.
- **Active learning loop**: replace `apps/annotate/` Streamlit with
  **Label Studio** integrated into the data pipeline. Auto-route
  low-confidence rows to a labelling queue, by domain expertise tag.

### Reliability

- **Shadow deploys**: 100% of traffic mirrored to challenger for 24–48 h
  before any shift. Replay-based comparison on KPI dashboards.
  Documented as the next step in `docs/shadow_canary.md`.
- **Multi-armed bandit** for A/B: not flat 90/10 splits — Thompson
  sampling on per-route reward signal.
- **Chaos engineering**: Gremlin or AWS Fault Injection Simulator.
  Quarterly game days.
- **Cost-aware autoscaling**: scale-down at off-peak hours by tenant
  traffic pattern.

### Cost breakdown (Tier 3)

| Line | Monthly |
|---|---|
| EKS control plane + 10 m6g.large workers | $800 |
| RDS Postgres multi-AZ | $200 |
| S3 + CloudFront + Route53 + ACM | $50 |
| Managed Prometheus + Grafana Cloud Pro | $200 |
| MLflow Postgres + S3 | $30 |
| Feature store (Feast on existing infra) | $0 |
| SageMaker training (~50 hr/mo spot) | $400 |
| Dagster Cloud | $50 |
| **Total** | **~$1.7k/mo** |

---

## Tier 4 — Enterprise (100k+ QPS, regulated)

Goal: SOC2 / HIPAA / PCI / GDPR, multi-tenant SaaS, sub-10 ms global
p99. $2–10M+/yr.

### Compliance + isolation

| Concern | Approach |
|---|---|
| Compliance certifications | SOC2 Type II, ISO 27001, PCI DSS, HIPAA depending on industry. AWS Artifact for shared responsibility evidence. Third-party pentests quarterly. |
| Encryption | At rest (KMS CMK per tenant), in transit (mTLS everywhere). Envelope encryption for per-tenant data. |
| Per-tenant isolation | Namespace-per-tenant in K8s, network policies, KMS keys per tenant, S3 prefixes with IAM session tags. Bedrock-style logical isolation. |
| Audit logs | CloudTrail + per-request structured logs to a SOC2-compliant SIEM. The `x-request-id` plumbing in `serve/api.py` makes this cheap. |
| PII | Presidio scrubbing on ingestion; differential privacy if you store user-level features. The reservoir sampler in `serve/reservoir.py` is the seam — bound it by tenant. |

### Global latency

- **Edge inference**: ONNX deployed to AWS Lambda@Edge / Cloudflare
  Workers AI / Fastly Compute. Sub-50 ms RTT globally. INT8 ONNX is
  small enough to ship to edge. v1 (64 MB) fits Cloudflare Workers AI
  free tier (10 ms execution limit). v2 needs Workers AI paid tier (50
  ms) or a dedicated Pop runtime.
- **Per-region model registries** replicated from a primary. Models
  signed with cosign; signature verified at edge.
- **Multi-model serving**: KServe or Seldon Core. Inference graphs
  (ensemble + business logic). Per-model autoscaling.

### A/B + ML governance

- **Real-time A/B + bandit**: Statsig / Eppo / LaunchDarkly with custom
  metrics. Multi-armed bandit at the API gateway.
- **Model lineage + governance**: Weights & Biases Models, ModelDB, or
  Vertex AI Model Registry. Track every artefact's lineage from raw
  data → training run → deployed model → predictions. Reproducible from
  any audit point.
- **Data lake**: Iceberg / Delta tables on S3 with Athena / Trino. PII
  tagging, lineage via OpenLineage.
- **Bias + fairness**: pre-deploy fairness eval (Fairlearn / Aequitas)
  per protected attribute. Audit reports per release.
- **Adversarial robustness**: TextFooler / DeepWordBug in CI. Min
  accuracy under attack as a gate. `tests/test_adversarial.py` is the
  scaffold for this.

### Cost governance

- Per-team / per-tenant cost allocation tags. FinOps team.
- Reserved instances + Savings Plans for steady-state.
- Spot for non-critical training. Capacity Blocks for guaranteed GPU.
- Tag-based budgets with PagerDuty integration on breach.

### Org structure

- **Platform team** (infra + ML platform engineers) maintains the
  serving stack, registry, pipelines, feature store.
- **Per-product ML teams** build models against the platform's APIs.
- **Productionising any model** = ticket → platform pipeline. No
  raw-serving stacks per team.

---

## Cross-tier: what the v3.0 project already gets right

These don't need to change as you scale up — they were intentional choices.

| Concern | Why it scales |
|---|---|
| ONNX INT8 serving | Same artefact runs on Pi, Fargate, EKS GPU, Cloudflare Workers, NVIDIA NIM. Format-agnostic. |
| Pydantic schemas at API boundary | Triton + KServe + edge runtimes all consume them. No rewrite. |
| Structured logging with `x-request-id` | Distributed tracing maps directly. Datadog, Honeycomb, Tempo all key on the same header. |
| Reservoir sampling bounded by `k` | Memory ceiling stays the same at 1 RPS or 10k RPS. |
| Drift detection on input features (length, word count) AND label distribution | Both signals work at any scale. Add per-segment drift at Tier 3+. |
| DVC for data versioning | Replace S3 remote with shared cluster storage at Tier 3. Same `dvc.yaml`. |
| Reproducible CI build → image push → ECS update | Same pattern works for ArgoCD + Helm at Tier 3. |
| API key via header | Replace with JWT validation middleware at Tier 2. Same `Depends()` pattern. |
| 5-task v2 multi-head model | Single ONNX session for 5 outputs scales linearly. Adding a 6th head is a `nn.Linear` away. |

---

## Comparison snapshot

| | v3.0 (today) | Tier 2 | Tier 3 | Tier 4 |
|---|---|---|---|---|
| Model artefact storage | HF Hub | HF or private S3 + signing | Vertex Model Registry + cosign + SLSA | Per-tenant CMK encryption |
| Inference | Fargate 1 task | ECS auto-scale + ALB | EKS + Triton + Karpenter | KServe + edge POPs |
| ML platform | DVC + MLflow SQLite | MLflow + RDS + S3 | Vertex / SageMaker Pipelines | Vertex enterprise + feature store + lineage |
| Observability | Prom + Grafana local | AMP + Grafana Cloud Pro | Datadog APM + SLO autoscale | Splunk + Datadog + SIEM |
| Auth | X-API-Key | OAuth2 / JWT via Cognito | Service mesh mTLS + JWT | mTLS + per-tenant IAM + KMS |
| Retraining | GH Actions cron | EventBridge + Pipelines | Dagster + drift triggers | Continuous training + bandit-driven |
| Drift | Evidently weekly | Real-time NannyML | Per-segment + concept-drift | Per-tenant + privacy-preserving |
| Cost | $0–6/mo | $250–500/mo | ~$2k/mo | $2–10M/yr |
| QPS sustained | 120 | 1k | 10k | 100k+ |
| Multi-region | No | Single-region multi-AZ | 3 regions active-active | Global edge |
| Compliance | None | Basic GDPR data minimisation | SOC2 + GDPR | SOC2 + HIPAA / PCI + per-region |

---

## Interview soundbite

> "Today it's a single Fargate task at six dollars a month deliberately,
> to force the engineering decisions interviewers actually want to see.
> The scaling ladder is documented end-to-end. At the next budget tier
> I'd add ALB + auto-scale, then Triton with dynamic batching, then a
> feature store and multi-region with EKS + Karpenter, then edge
> inference and per-tenant isolation. The model serving stack stays the
> same INT8 ONNX session at each tier; what changes is the orchestrator,
> the autoscaler, and the data-flow guarantees."
