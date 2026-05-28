# Service Level Objectives — MovieSentiment

> Audience: the on-call (currently the author) and any interview panel reading
> the project. The SLOs here are the contract the service holds itself to.

Last reviewed: 2026-05-28. Reviewed monthly or after any incident.

---

## In scope

The live FastAPI service on ECS Fargate (`ap-southeast-2`) exposing `/predict`,
`/analyze`, `/explain`, `/predict/async`, `/insights/{movie_id}`, and the ops
endpoints (`/healthz`, `/readyz`, `/version`, `/metrics`).

Out of scope: training jobs (best-effort, weekly cron); CI; the local
docker-compose stack; the static frontend at `/ui/` (best-effort).

---

## SLIs (Service Level Indicators)

| SLI | Measurement | Source |
|---|---|---|
| **Availability** | Fraction of `/healthz` probes returning 2xx within a 30-second window. | ECS health check + uptime monitor (`UptimeRobot` free tier, 5-min interval). |
| **Sync latency p99** | 99th percentile of `/predict` (batch=1) end-to-end response time. | Prometheus `http_request_duration_seconds_bucket{path="/predict"}`. |
| **Error rate** | Fraction of `/predict` + `/analyze` responses with HTTP status ≥ 500 inside any rolling 5-minute window. | Prometheus `http_requests_total{status=~"5.."}` / `http_requests_total`. |
| **Drift detection latency** | Time from a drift-share crossing of 0.30 to a `train.yml` workflow start. | GitHub Actions `drift-detection` cron + manual `workflow_dispatch`. |

---

## SLOs (targets, monthly window)

| SLO | Target | Error budget per 30-day month |
|---|---|---|
| Availability of `/healthz` | **≥ 99.5%** | 3 h 36 min downtime allowed |
| `/predict` p99 latency | **≤ 75 ms** | 1% of requests may exceed 75 ms |
| `/predict` + `/analyze` 5xx rate | **≤ 1%** | 7 h 12 min of >1% error allowed |
| Drift → retrain trigger | **≤ 24 h** | One missed cron tolerated |

Numbers chosen to be tight enough to be meaningful and loose enough to be
achievable on a single 0.25 vCPU / 1 GB Fargate task. Tightening the latency
target requires either a CPU bump or LitServe-style batching — both deferred
in `docs/future_improvements.md`.

---

## Error budget policy

- **Within budget:** ship freely. Risky changes (model swap, dependency major
  bump, ECS task definition rewrite) preferred mid-month after the burn-rate
  is known.
- **Over 50% of budget consumed:** freeze risky changes; restrict deploys to
  bug fixes and CVE patches.
- **Budget exhausted:** roll back to the last green commit on `main`,
  open an incident in `docs/incidents/YYYY-MM-DD.md`, and update the budget
  policy if the same failure mode appears twice in a quarter.

The single-task topology means a Fargate task crash burns ~5 min of
availability budget. Two consecutive crashes within a 30-day window still
leaves >3 hours of headroom — meaningful for a portfolio project, not for a
real SLA.

---

## What does NOT count against the budget

- Planned maintenance windows announced in the README at least 24 h ahead.
- Upstream outages: AWS ECS, ECR, or `ap-southeast-2`-wide events. Documented
  in the runbook with the AWS Health link.
- Synthetic traffic from the load test (Locust scenarios in
  `scripts/load_test.py`) — known to push p99 past target on a single task.

---

## How we measure

- **Local development:** `make loadtest` runs Locust against a localhost task;
  results in `docs/load_test_report.html`.
- **Production:** Prometheus + Grafana dashboard at
  `docs/load_test_report.html` (mirrored from Grafana Cloud) tracks the
  rolling p99 and 5xx rate. Burn-rate alerts fire at 2× and 5× the monthly
  budget consumption rate.
- **Drift SLO:** GitHub Actions run history for `drift.yml` + `train.yml`.

If any of these instruments fail to report for >24 h, treat the SLO as
unknown for that window — do not assume green.

---

## Review cadence

- **Monthly:** Author reviews the error-budget burn-down and adjusts targets
  if real traffic patterns push consistently above or below. Logged in
  `docs/progress/` with the month tag.
- **Per incident:** Runbook (`docs/runbook.md`) is updated with the new
  failure mode + recovery time so the next on-call sees the playbook.
- **Per major release (`vX.0`):** SLOs are re-evaluated against the new
  architecture; this doc updates in the same PR.
