# Grafana Cloud (free) — OTLP traces + metrics from production Fargate

Set up once. Free tier covers portfolio traffic forever: 50 GB logs + 50 GB traces + 10k Prometheus series + 14-day retention. $0/month.

## Why

Local docker-compose has Prometheus + Tempo and is fine for development. Production traces / metrics need to leave the Fargate task somewhere — but standing up our own Tempo/Loki on AWS would blow the cost budget.

Grafana Cloud is the free escape hatch. From the Fargate task, OTLP-export both traces and metrics to Grafana Cloud's hosted ingest. The same Grafana UI displays both. Done.

## One-time setup

1. **Sign up**: `grafana.com/auth/sign-up/create-user` → "Free forever" plan.
2. Create a new Grafana Cloud stack. Note the stack URL (`<slug>.grafana.net`).
3. In **Connections → Add new connection → OpenTelemetry**: copy the OTLP endpoint and the auto-generated Basic auth header.
4. The endpoint looks like `https://otlp-gateway-prod-<region>.grafana.net/otlp` and the auth header looks like `Authorization=Basic <base64-of-instanceID:token>`.

## Wire into ECS

Add two environment variables to `deploy/ecs-task-definition.json` (use ECS Secrets Manager for the auth value, not hardcoded):

```json
{
  "environment": [
    {"name": "PORT", "value": "8000"},
    {"name": "OTEL_EXPORTER_OTLP_ENDPOINT", "value": "https://otlp-gateway-prod-<region>.grafana.net/otlp"},
    {"name": "OTEL_SERVICE_NAME", "value": "moviesentiment"},
    {"name": "MS_ENV", "value": "prod"}
  ],
  "secrets": [
    {"name": "OTEL_EXPORTER_OTLP_HEADERS", "valueFrom": "arn:aws:secretsmanager:ap-southeast-2:<account>:secret:moviesentiment/grafana-cloud-otlp"}
  ]
}
```

Create the secret once:

```bash
aws secretsmanager create-secret \
  --name moviesentiment/grafana-cloud-otlp \
  --secret-string 'Authorization=Basic <base64-of-instanceID:token>' \
  --region ap-southeast-2
```

Give the task execution role permission to read it:

```bash
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite   # tighten in prod
```

Redeploy.

## Verifying

`src/moviesentiment/serve/tracing.py::setup_tracing` is a no-op when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset, and lights up automatically when the env var appears. Inside the Fargate task, after a `/predict` you should see a trace appear in Grafana Cloud → Explore → Tempo within ~30 s.

## Cost ceiling

Hard rule from `possible_improvements.md`: total bill stays ≤ $13/mo. Grafana Cloud adds **$0**. Free tier capacity:
- 50 GB traces / mo (this project: ~50 MB at portfolio traffic — 1000× under)
- 50 GB logs / mo (this project: ~100 MB — 500× under)
- 10k Prometheus active series (this project: ~30 series — 333× under)

If usage ever crosses any free-tier threshold, the worst case is throttled ingest, not surprise billing. Grafana Cloud does not bill above the free tier without an explicit plan upgrade.

## Local equivalent (no setup, no account)

`deploy/docker-compose.yml` already runs Prometheus and Grafana for development. To also run Tempo locally, append:

```yaml
  tempo:
    image: grafana/tempo:2.5.0
    command: ["-config.file=/etc/tempo.yaml"]
    volumes:
      - ./tempo.yaml:/etc/tempo.yaml
    ports:
      - "3200:3200"
      - "4317:4317"
```

Then set `OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317` in the API container. Grafana picks Tempo up as a datasource automatically.
