# 3-Minute Demo Script — MovieSentiment

> Use this as a Loom voiceover script. Time budget: **3 min total**.
> Tabs / windows ready before recording:
> 1. README on GitHub (full screen).
> 2. Architecture diagram from the README (Mermaid render).
> 3. Terminal with `make smoke-test` ready.
> 4. MLflow UI on `localhost:5000`.
> 5. Grafana dashboard tab.
> 6. `docs/drift_reports/<latest>.html` open in a browser.
> 7. GitHub Actions runs page (most recent CI green run).

---

## [0:00 – 0:20] Opening + the problem (20 s)

> "MovieSentiment is an end-to-end MLOps project for IMDb review sentiment.
> The interesting part isn't the model — it's that this is a full lifecycle:
> data scrape, training, ONNX export, ECS deploy, monitoring, drift detection,
> and automatic retraining. All for about six dollars a month."

Show: README hero block + the live `curl` snippet.

---

## [0:20 – 0:50] Architecture (30 s)

> "The data path runs from IMDb through DVC and S3 into a SageMaker training
> job. The trained model gets quantized to INT8 ONNX. The serving image goes
> through ECR and lands on a single ARM64 Fargate task. The same task serves
> both v1 binary sentiment and v2 multi-task — sentiment plus aspect-based
> sentiment plus emotion plus spoiler detection plus helpfulness regression —
> in one forward pass."

Show: Mermaid diagram. Trace the loop with the cursor while talking.

---

## [0:50 – 1:20] Live demo (30 s)

> "Here's the live API."

Run: `make smoke-test`

> "That resolves the current Fargate task's public IP and hits `/predict`
> with two reviews. You see binary sentiment back. If I hit `/analyze`
> instead..."

Run:
```bash
curl -s -X POST "http://<ip>:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text":"The hero dies and the villain wins but the cinematography is breathtaking."}' | jq
```

> "...I get all five heads in one response. Notice the spoiler probability
> picks up on the plot reveal, the aspects rate visuals high, and the
> overall sentiment is mixed."

---

## [1:20 – 1:50] MLflow + retraining loop (30 s)

> "Every training run is logged to MLflow. Here's the baseline TF-IDF run
> at 0.904 F1, and DistilBERT at 0.939. The v2 multi-task run is here, with
> per-task losses."

Show: MLflow runs page; click the v2 multi-task run.

> "When drift detection in production sees the input distribution shift by
> more than 30%, the weekly cron triggers `train.yml`. If the new model
> improves F1 by at least half a percent, it auto-promotes to production
> and the next ECS deploy rolls it out."

Show: `.github/workflows/drift.yml` briefly.

---

## [1:50 – 2:20] Monitoring (30 s)

> "Prometheus scrapes the FastAPI metrics endpoint. Grafana shows latency,
> request rate, and per-class prediction distribution. Custom metrics
> include a prediction-confidence histogram and a model-version gauge so I
> can tell exactly which model SHA served any given request."

Show: Grafana dashboard.

> "Drift detection writes an Evidently HTML report every Monday — here's
> the latest one."

Show: `docs/drift_reports/<latest>.html`.

---

## [2:20 – 2:45] CI + the bits that make it a real project (25 s)

> "CI runs on every push: lint, type-check with mypy strict, pytest with
> an 85% coverage gate, bandit + pip-audit for security, OSSF Scorecard
> weekly. Build pushes ARM64 image to both GHCR and ECR, then
> `update-service --force-new-deployment` rolls Fargate."

Show: a green CI run page.

> "Production cost is six dollars a month: one Fargate task, free-tier
> Lambda + SQS + DynamoDB for the async path, S3 for DVC artefacts. The
> AWS Budget alert at the thirteen-dollar ceiling is in
> `scripts/setup_aws_budget.py`."

---

## [2:45 – 3:00] Close (15 s)

> "Repo's on GitHub at Cryptic2-0/Project. README has the live curl, the
> reproduce steps, the model card, the security threat model, the SLOs,
> the runbook, and the deferred-features list. Thanks for watching."

Show: README top of page.

---

## Things to NOT skip

- Use the **real** Fargate IP, not a placeholder. `make smoke-test`
  produces it for you on the fly.
- Hit `/analyze` with a phrase that actually contains a spoiler — the
  demo loses impact otherwise.
- Don't run `dvc repro` live (takes ~25 min for the transformer stage).
  Show the `dvc.yaml` DAG instead.
- Don't open VS Code with secrets visible. Keep `.env` shut.

## Things to mention if there's time

- The pivots that mattered: Fly → ECS, GDrive → S3, dataset+pyarrow →
  plain list, single-head → multi-head.
- The reservoir sampler (Vitter Algorithm R) for bounded-memory drift.
- ONNX export wraps the multi-task dataclass with `_OnnxWrap` so the
  legacy TorchScript exporter sees a flat tuple (`dynamo=False`).
- API key auth via `MS_API_KEY` is wired in but disabled in the demo so
  the curl works without a header.
