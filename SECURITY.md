# Security policy

## Reporting

Open a private security advisory: [github.com/Cryptic2-0/Project/security/advisories/new](https://github.com/Cryptic2-0/Project/security/advisories/new). Public issues for security bugs are not appropriate — please use the advisory flow so a fix lands before disclosure.

## Threat model (in scope)

This is a portfolio MLOps demo with a public-internet API. We defend against:

| Threat | Where | Mitigation |
|---|---|---|
| Untrusted client input → server crash / RCE | `POST /predict`, `/explain`, `/predict/async` | Pydantic validation, hard length caps (`max_batch_size`, `max_text_length`), ONNX-only inference (no `eval`/`pickle`/`exec`). |
| Abuse / DoS via traffic flood | All endpoints | `slowapi` per-IP rate limits (`/predict` 60/min, `/explain` 10/min). |
| Log injection via headers | `x-request-id` | Header is allowlisted to `[A-Za-z0-9._-]{,64}`; CRLF/control chars stripped. |
| HuggingFace namespace takeover serving malicious weights | training pipeline | `MS_HF_REVISION` (or `params.yaml::transformer.revision`) pins an exact commit. CI sets this before training jobs. Serving loads from local DVC-pulled artifacts (no Hub fetch at runtime). |
| Supply-chain — vulnerable Python deps | `pyproject.toml` | `pip-audit` runs on every CI build with documented `--ignore-vuln` entries; Dependabot opens weekly updates. |
| Supply-chain — typosquatted / malicious pip packages | install | All deps version-pinned and resolved from lock files (`scripts/compile_requirements.sh`); pre-commit `bandit` flags risky calls. |
| CORS abuse from arbitrary origins with cookies | browser clients | Default `Allow-Origin: *` with `allow_credentials=false`. Tightened in production via `MS_CORS_ALLOW_ORIGINS=https://cryptic2-0.github.io,...` which auto-enables credentials. |
| Logs leaking PII (review text) | structlog | `predict_complete` logs counts + labels only — review body is not logged. Production reservoir-sampled bodies live in `data/production/` with no public S3 prefix. |
| Container privilege escalation | Fargate task | Image runs as `app` (uid 1000), read-only filesystem-friendly, no root shell tools. |
| TLS / cert validation in S3 fetch (DVC) | `pyOpenSSL` indirect dep | Pinned to `<=24.2.1` due to `pydrive2` constraint; CVE-2026-27448/27459 tracked, mitigated in practice because DVC S3 traffic uses long-lived AWS-signed URLs (no cross-domain cert risk). |

## Out of scope

- Multi-tenant authorization. The API supports a shared-secret API key via `MS_API_KEY` + `X-API-Key` header (disabled by default for the public demo). Multi-tenant scope / per-user quotas / OAuth still require OPA / JWT before any non-demo use.
- Side-channel attacks on the ONNX model (membership inference, model extraction). Realistic for higher-stakes models; not addressed here.
- Cost-control on the AWS account at large. Budget alarms are the operator's responsibility.

## Hardened production config

```bash
MS_CORS_ALLOW_ORIGINS=https://cryptic2-0.github.io
MS_HF_REVISION=<exact-commit-sha-of-distilbert-base-uncased>
MS_API_KEY=<generated-shared-secret>     # /predict + /analyze require X-API-Key
MS_ENV=prod
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-prod-<region>.grafana.net/otlp
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <token>
```

## Audit cadence

- `bandit` + `pip-audit` run on every PR via CI (`.github/workflows/ci.yml`).
- OSSF Scorecard runs weekly (`.github/workflows/scorecard.yml`).
- Dependabot proposes weekly dep + Docker base image bumps (`.github/dependabot.yml`).
- Pre-commit blocks risky patterns (`bandit`, `detect-private-key`).

## Known residual risks (accepted)

| ID | Component | Reason accepted |
|---|---|---|
| PYSEC-2025-217 / CVE-2026-1839 | transformers 4.57.x | Fix is in 5.0.0rc3 which removes `optimum-onnx` compat used by the ONNX export path. Exploitable only via untrusted model loading — we pin `revision=` for all `from_pretrained` calls touching remote sources. |
| PYSEC-2026-161 | starlette 0.52.1 | Fix landed in `1.0.1`; `fastapi==0.136.3` (latest available) pins `starlette<1.0`. Vulnerability path requires multipart upload handler we don't use (`POST` endpoints take JSON only). |
| MAL-2026-4750 | fastapi 0.136.x | OSV malicious-package flag appears to be a false positive against the current 0.136.x line; no actual indicator-of-compromise observable. Tracking upstream. |
| CVE-2025-69872 | diskcache 5.6.3 | No fixed release available. Only used as a DVC transitive dep for local cache; not exposed to user input. |
| CVE-2026-27448 / 27459 | pyOpenSSL 24.2.1 | Bumping to 26.0.0 breaks `pydrive2` which `dvc[s3]` requires. CVEs affect TLS certificate validation paths; only invoked by DVC's S3 transport which uses AWS SigV4 over HTTPS — exposure is bounded to MITM with a forged cert against AWS S3. |
