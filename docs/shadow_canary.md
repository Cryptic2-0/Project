# Shadow / Canary Deploy Plan — MovieSentiment

> Plan, not code. Implementing this needs an Application Load Balancer
> (~$18/mo) which sits above the $13/mo budget ceiling. Documented here so the
> path is concrete the day the project leaves portfolio mode.

Last updated: 2026-05-28.

---

## What we have today

```
   client ── HTTP ──> ENI public IP ── port 8000 ──> single Fargate task
```

One task, one image tag, one model artefact in S3. Deploys roll via
`aws ecs update-service --force-new-deployment`. No traffic split, no
shadow, no canary. The error budget in `docs/slos.md` accommodates a
single-task topology.

---

## Target architecture

```
                              ┌──────────────────────────┐
                              │   ALB (HTTPS terminator) │
                              └─────┬────────────┬───────┘
                          weight 90%│            │weight 10%
                                    ▼            ▼
                  ┌──────────────────────┐ ┌──────────────────────┐
                  │ TG: champion         │ │ TG: challenger       │
                  │ moviesentiment:N     │ │ moviesentiment:N+1   │
                  │ task def rev X       │ │ task def rev X+1     │
                  └──────────┬───────────┘ └──────────┬───────────┘
                             ▼                        ▼
                       Fargate task              Fargate task
                       (current model)           (new model)
```

Two target groups, both registered on the ALB on the same listener path
(`/`). Traffic weights live in the listener rule action. Health-check on
`/healthz`.

---

## Workflow

1. **Promote a new model in MLflow Registry** to `Staging`.
2. **Build + push** a new image tagged `moviesentiment:<git-sha>`.
3. **Register a new ECS task definition revision** pointing at the new image.
4. **Create / update the challenger service** to point at the new task def.
   The challenger inherits the same network config as the champion but is
   attached to its own target group.
5. **Update the ALB listener rule** to forward 10% weight to the challenger
   target group, 90% to the champion. Use AWS-Console (or
   `aws elbv2 modify-listener`) to set:
   ```json
   "ForwardConfig": {
     "TargetGroups": [
       { "TargetGroupArn": "<champion-tg>", "Weight": 90 },
       { "TargetGroupArn": "<challenger-tg>", "Weight": 10 }
     ]
   }
   ```
6. **Watch the metrics for 24 h:**
   - Prometheus 5xx-rate, p99 latency, prediction-confidence histogram
     (split per target group via the `model_version_info` gauge).
   - Drift report on the challenger's reservoir slice — guard against the
     new model fitting a different distribution.
7. **Promote (or abort):**
   - If green: shift to 50/50, then 0/100 over a second 24-h window. The
     challenger becomes the new champion. Old champion service stays
     scaled-to-zero for fast rollback.
   - If red: shift to 100/0, archive the challenger MLflow run, capture the
     incident in `docs/incidents/`.

---

## Concrete commands (when wiring lands)

```bash
# Targeting infrastructure that does not exist yet.
make canary-promote IMAGE_TAG=$(git rev-parse --short HEAD) WEIGHT=10
# under the hood:
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.json
aws ecs update-service --service moviesentiment-challenger \
  --task-definition moviesentiment:<new-rev> --force-new-deployment
aws elbv2 modify-listener --listener-arn <arn> --default-actions \
  Type=forward,ForwardConfig='{TargetGroups=[...]}'
```

`make canary-promote` would live in `Makefile` next to the existing
`smoke-test` target.

---

## Why this is not in code today

| Constraint | Cost | Hits the $13/mo ceiling? |
|---|---|---|
| Application Load Balancer (always on) | ~$18/mo | Yes. |
| Two parallel Fargate tasks (champion + challenger) | +$6/mo | Yes (already at $6 with one). |
| Cross-AZ data transfer between ALB + Fargate | ~$1/mo | Marginal. |
| ACM TLS certificate | $0 | No. |

Two paid lines above the budget. The right next step is to:

1. Move HTTPS termination + path-based routing to an ALB **first** (also
   solves the rotating ENI IP problem documented in §7 of the report).
2. Once ALB exists, the canary path is a free incremental addition.

The README "What I'd do differently in production" already names this as
the upgrade path, and `docs/future_improvements.md` §3.2 (SageMaker
Serverless comparison) is the alternative — also above budget but on the
SLA-first side rather than the latency-first side.

---

## What would I A/B?

Once the rig exists, the experiments queue is:

1. **v1 INT8 vs v1 INT8-AWQ** — activation-aware weight quantization.
   Expected: same accuracy, ~5% faster inference. Risk: AWQ tooling is
   newer than dynamic quant; portability is unknown on ARM64.
2. **v1 vs v2** — replace `/predict` (sentiment-only) with the
   single-head extraction from the multi-task model. Expected: same
   accuracy, slightly higher latency (shared encoder + linear head vs.
   dedicated single-head model). Validate before retiring v1.
3. **DistilBERT vs ELECTRA-small** — both fit the latency budget; ELECTRA
   sometimes wins on short inputs.
4. **English-only vs XLM-R multilingual** — for the "multi-language"
   roadmap item.

Each lives ~3 days behind the canary at 5%, 10%, 50%, 100% with explicit
KPI gates (F1 ≥ champion-0.5%, p99 ≤ champion+10%, calibration Brier ≤
champion+0.02).
