# Runbook — MovieSentiment on-call

> One page per failure mode. Each entry: **symptom → diagnose → fix → verify**.
> Aim to resolve any item below in <30 min without paging upstream.

Owner: Soumya Sarkar (`Cryptic2-0`). Last reviewed 2026-05-28.

Region: `ap-southeast-2`. Cluster: `moviesentiment`. Service: `moviesentiment`.

---

## 0. Pre-flight: what's running where

```bash
# Locate the live ECS task + its ENI IP in one shot.
make smoke-test
# Same thing manually:
aws ecs list-tasks --cluster moviesentiment --service-name moviesentiment \
  --region ap-southeast-2 --query 'taskArns[0]' --output text
aws ecs describe-tasks --cluster moviesentiment --tasks <task-arn> \
  --region ap-southeast-2 --query 'tasks[0].attachments[0].details'
aws ec2 describe-network-interfaces --network-interface-ids <eni-id> \
  --region ap-southeast-2 --query 'NetworkInterfaces[0].Association.PublicIp'
```

If `smoke-test` returns no task ARN, jump to §3 (no task running).

---

## 1. 5xx spike on `/predict` or `/analyze`

**Symptom:** error-budget burn-rate alert; users see 500/503.

1. **Diagnose**
   - `curl -sf http://<eni-ip>:8000/readyz` — if 503, the inference engine
     failed to load. Look at CloudWatch Logs `/ecs/moviesentiment` for
     `model_load_failed` or `multitask_model_unavailable`.
   - `aws ecs describe-tasks --cluster moviesentiment --tasks <task-arn> \
     --query 'tasks[0].containers[0].lastStatus'` — `STOPPED` means the
     container died; check `exitCode` + `reason`.

2. **Fix**
   - **Engine failed to load (v1):** verify `models/distilbert_onnx_int8/` is
     in the image; if not, the CI's `dvc pull` step failed. Re-run the
     pipeline: `gh workflow run ci.yml --ref main` then
     `aws ecs update-service --cluster moviesentiment --service moviesentiment
     --force-new-deployment --region ap-southeast-2`.
   - **v2 multi-task unavailable:** confirm `MS_DVC_BUCKET` +
     `MS_MULTITASK_S3_PREFIX` are set in the live task definition; confirm
     the S3 keys exist under that prefix. `/analyze` 503 is graceful — v1
     `/predict` still works.
   - **OOM:** see §4.

3. **Verify**
   - `curl -sf http://<eni-ip>:8000/healthz && curl -sf
     http://<eni-ip>:8000/readyz`
   - Grafana 5xx rate panel returns to baseline within 5 min.

---

## 2. Drift spike (`drift_share > 0.3`)

**Symptom:** Monday-morning `drift-detection` workflow logs `should_retrain=true`;
weekly retrain job fires.

1. **Diagnose**
   - Open the latest report under `docs/drift_reports/YYYY-MM-DD.html`. Identify
     which input features moved (text length, word count, vocabulary).
   - Compare against `data/processed/train.parquet` to confirm whether the
     scraper picked up new movies / new vocabulary.

2. **Fix**
   - If drift is genuine (new movies, new vocab): let the retrain job run.
     The `train.yml` flow auto-promotes a model whose F1 improves by ≥0.5%.
   - If drift is spurious (scraping bug, parquet schema change): roll back
     the offending `scrape.py` change, push, re-run `dvc repro`. Do NOT let
     the auto-promote run while the data is suspect — set
     `vars.CURRENT_PROD_F1` artificially high in repo settings to fail the
     improvement gate.

3. **Verify**
   - Next Monday's drift report shows `drift_share < 0.3`.
   - `mlflow ui` shows the new model registered in `Staging` or `Production`.

---

## 3. No ECS task running

**Symptom:** `make smoke-test` returns empty; `aws ecs list-tasks` empty.

1. **Diagnose**
   - `aws ecs describe-services --cluster moviesentiment --services
     moviesentiment` — read `events[]`. Common cause: image pull failure or
     subnet/SG misconfiguration after a task definition edit.
   - Check ECR for the latest image tag matching the deployed task def:
     `aws ecr describe-images --repository-name moviesentiment
     --region ap-southeast-2`.

2. **Fix**
   - **Image pull failed:** verify the task execution role has
     `AmazonECSTaskExecutionRolePolicy` attached. Re-tag and re-push if the
     manifest is missing the right platform: ARM64 manifests need
     `linux/arm64`, X86 needs `linux/amd64`.
   - **Bad task definition:** revert the service to the previous task def
     revision: `aws ecs update-service --task-definition
     moviesentiment:<previous-rev> --force-new-deployment`.

3. **Verify**
   - `aws ecs list-tasks` returns a task ARN.
   - The new ENI IP responds to `/healthz` within 90 s of task start.

---

## 4. Task OOM-killed

**Symptom:** Container `exitCode=137` (OOM); rolling restart loop.

1. **Diagnose**
   - CloudWatch metric
     `ECS/ContainerInsights/MemoryUtilization{ServiceName=moviesentiment}`
     pinned at 100% before the kill.
   - Inspect logs immediately preceding the kill: large batch (`POST /predict`
     with `texts` near the batch limit) or a long-running occlusion
     attribution job.

2. **Fix**
   - **One-off bad request:** the request was already rejected by the
     `max_batch_size` (32) and `max_text_length` (5000) Pydantic gates. If
     it slipped through, validate `serve/schemas.py`.
   - **Genuine load:** bump the task definition memory from 1 GB to 2 GB
     (+$1/mo). Edit `deploy/ecs-task-definition.json` → `"memory": "2048"`,
     register, then `update-service`.

3. **Verify**
   - Steady-state memory utilisation < 70% over a 30-minute window.

---

## 5. Stuck ECS deployment (rollout stalls)

**Symptom:** `aws ecs describe-services` shows `runningCount < desiredCount`
for >5 min after a `--force-new-deployment`.

1. **Diagnose**
   - Health-check probe failing: the new task is starting but `/healthz`
     never returns 2xx. Look for stack traces in CloudWatch immediately
     after container start.
   - The most common cause is a config-driven import error (e.g. a new
     `MS_*` env var the code requires but the task definition does not
     supply).

2. **Fix**
   - Roll back: `aws ecs update-service --task-definition
     moviesentiment:<last-known-good-rev> --force-new-deployment`.
   - Then add the missing env var to the new task definition and re-deploy.

3. **Verify**
   - `runningCount == desiredCount` in `describe-services`.
   - `/version` returns the expected git SHA.

---

## 6. MLflow rollback (need to revert to an older model)

**Symptom:** A new model promoted to `Production` underperforms in the wild.

1. **Diagnose**
   - `mlflow ui` (local against the SQLite DB) — find the run ID of the
     prior `Production` model.

2. **Fix**
   - Demote the bad model: `mlflow models transition-version --name
     moviesentiment-classifier --version <bad-version> --stage Archived`.
   - Promote the prior good version to `Production`.
   - Re-export ONNX from the good run if the artefact isn't already on disk:
     `python -m moviesentiment.models.onnx_export`.
   - DVC-push then redeploy: `dvc push && aws ecs update-service
     --force-new-deployment`.

3. **Verify**
   - `/version` reports the rolled-back model's stage / version.
   - Smoke test confirms expected accuracy on a known-good test batch.

---

## 7. S3 ONNX artefact corrupted / missing (v2 multi-task)

**Symptom:** Fargate boot logs `multitask_model_unavailable` with an S3 error.

1. **Diagnose**
   - `aws s3 ls s3://moviesentiment-dvc-soumya/multitask_onnx/` — confirm
     the five expected keys (`model.onnx`, `tokenizer.json`,
     `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`).

2. **Fix**
   - Re-export from local checkpoint:
     `python -m moviesentiment.models.onnx_export_multitask`. The script
     produces both `model_fp32.onnx` and quantized `model.onnx`.
   - Re-upload: `aws s3 cp models/distilbert_multitask_onnx/
     s3://moviesentiment-dvc-soumya/multitask_onnx/ --recursive --exclude
     "*_fp32.onnx"`.
   - Force ECS redeploy to re-run the lifespan bootstrap.

3. **Verify**
   - `POST /analyze` with a sample text returns all five heads.
   - Logs show `multitask_model_loaded_from_s3`.

---

## 8. Drift workflow won't trigger retrain

**Symptom:** `drift_share` rises above 0.3 but `train.yml` does not fire.

1. **Diagnose**
   - Open `.github/workflows/drift.yml`. Check the `should_retrain` output
     in the most recent run.
   - The retrain job has `if: needs.detect.outputs.should_retrain == 'true'`.
     A truthy string but with whitespace breaks the comparison.

2. **Fix**
   - The drift workflow writes to `$GITHUB_OUTPUT` — check the trailing
     newline / whitespace handling in the inline Python.
   - As a workaround, manually trigger: `gh workflow run train.yml --ref
     main`.

3. **Verify**
   - `gh run list --workflow=train.yml` shows a new run.

---

## 9. Cost alarm fired (`moviesentiment-monthly` budget)

**Symptom:** Email from AWS Budgets: forecasted 80% or actual 100% breach.

1. **Diagnose**
   - AWS Cost Explorer → filter by service. The expected line items are
     Fargate (~$6/mo) and trace amounts for S3 + CloudWatch.
   - Unexpected lines: data transfer (someone pulled lots of S3), SageMaker
     (a stuck training job), NAT (we don't use one — investigate VPC).

2. **Fix**
   - Stop any running SageMaker job: `aws sagemaker stop-training-job
     --training-job-name <name>`.
   - If Fargate spend ballooned: confirm there's exactly one task with one
     revision; kill any duplicates from a failed rolling deploy.

3. **Verify**
   - Next-day Cost Explorer estimate trends back under $13.
   - `scripts/setup_aws_budget.py` is re-runnable; bumping the alarm email
     or limit is one command.

---

## Escalation

This is a portfolio project. Escalation = ask Soumya. There is no pager rota.
For a real production deployment of this stack:

- Wire the AWS Budgets alert to PagerDuty (or equivalent).
- Add a synthetic monitor (UptimeRobot free tier; Pingdom for production).
- Add CloudWatch alarms on the Prometheus 5xx-rate metric (via CW Metric
  Filter on the log group).
- Document the on-call rotation here.
