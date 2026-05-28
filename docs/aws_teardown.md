# AWS Teardown — Stop All Billing

> Step-by-step to drive the project's AWS spend to **$0/month**.

Last updated: 2026-05-29. Region: `ap-southeast-2`. Account: `375259955411`.

---

## What's billable today

As of 2026-05-29 the project's AWS spend is **~$0.37/month** (ECR storage
~$0.33 + S3 ~$0.03 + CloudWatch ~$0.01). Everything compute is already
torn down. This doc walks through removing the storage too if you want
$0 instead of $0.37.

Run all commands with admin AWS credentials. The `moviesentiment-ci` user
does not have delete permissions on most of these resources.

```powershell
# Set creds for the session (PowerShell):
$env:AWS_ACCESS_KEY_ID = "AKIA..."
$env:AWS_SECRET_ACCESS_KEY = "..."
$env:AWS_REGION = "ap-southeast-2"

# Verify:
aws sts get-caller-identity
```

---

## Step 1 — Verify nothing is running (sanity check, ~10 s)

```powershell
aws ecs list-clusters --region ap-southeast-2
# Expected: { "clusterArns": [] }

aws lambda list-functions --region ap-southeast-2
# Expected: { "Functions": [] }

aws sqs list-queues --region ap-southeast-2
# Expected: empty

aws dynamodb list-tables --region ap-southeast-2
# Expected: { "TableNames": [] }

aws events list-rules --region ap-southeast-2
# Expected: []
```

If any of these come back non-empty, delete those first
(`aws ecs delete-service --force`, `aws lambda delete-function`, etc.).

---

## Step 2 — Delete the ECR images (~$0.33/mo)

This is your biggest spend.

```powershell
# List images
aws ecr describe-images --repository-name moviesentiment --region ap-southeast-2

# Delete all images in the repo
aws ecr batch-delete-image --repository-name moviesentiment --region ap-southeast-2 ^
  --image-ids "$(aws ecr list-images --repository-name moviesentiment --region ap-southeast-2 --query 'imageIds[*]' --output json)"

# Delete the repo itself
aws ecr delete-repository --repository-name moviesentiment --force --region ap-southeast-2
```

Caveat: if you want to redeploy later, you'll have to rebuild + push.
CI does this automatically on every `main` push (build-image job in
`.github/workflows/ci.yml`).

**Saves: ~$0.33/mo.**

---

## Step 3 — Delete the S3 bucket (~$0.03/mo)

The bucket holds:
* `dvc/` — DVC artefacts for older datasets / models.
* `multitask_onnx/` — v2 ONNX. **Already mirrored to HuggingFace**:
  https://huggingface.co/Cryptic2-0/moviesentiment-multitask-onnx-int8

```powershell
# Delete all objects (multipart-uploads + versions if versioned)
aws s3 rm s3://moviesentiment-dvc-soumya --recursive

# Confirm empty
aws s3 ls s3://moviesentiment-dvc-soumya/

# Delete bucket
aws s3api delete-bucket --bucket moviesentiment-dvc-soumya --region ap-southeast-2
```

**Saves: ~$0.03/mo.**

If you want to keep DVC working for retraining: leave the bucket and just
clean specific prefixes. The retraining workflow re-uploads what it needs.

---

## Step 4 — Delete CloudWatch logs (~$0.01/mo)

```powershell
aws logs delete-log-group --log-group-name /ecs/moviesentiment --region ap-southeast-2
```

Caveat: the log group is set to 90-day retention, so it auto-deletes at
zero cost. You can skip this step and the bill stays under $0.01/mo.

**Saves: <$0.01/mo.**

---

## Step 5 — Deregister ECS task definitions (free, but tidy)

Task definitions don't bill but they clutter the console.

```powershell
# List all revisions
aws ecs list-task-definitions --family-prefix moviesentiment --region ap-southeast-2

# Deregister every revision (one at a time)
# Example for rev 5:
aws ecs deregister-task-definition --task-definition moviesentiment:5 --region ap-southeast-2

# Then delete the family entirely:
aws ecs delete-task-definitions --task-definitions moviesentiment:1 moviesentiment:2 ... --region ap-southeast-2
```

**Saves: $0.** Cosmetic only.

---

## Step 6 — Delete IAM users + roles (free, optional)

If you don't plan to redeploy, you can delete the IAM principals:

```powershell
# Detach all policies from a user first
aws iam list-attached-user-policies --user-name moviesentiment-ci
aws iam detach-user-policy --user-name moviesentiment-ci --policy-arn arn:aws:iam::aws:policy/...

# Delete inline policies
aws iam list-user-policies --user-name moviesentiment-ci
aws iam delete-user-policy --user-name moviesentiment-ci --policy-name <name>

# Delete access keys
aws iam list-access-keys --user-name moviesentiment-ci
aws iam delete-access-key --user-name moviesentiment-ci --access-key-id <id>

# Delete the user
aws iam delete-user --user-name moviesentiment-ci

# Repeat for moviesentiment-admin
```

For roles (`ecsTaskExecutionRole`, `ECSTaskRole-moviesentiment`,
`SageMakerRole-moviesentiment`):

```powershell
# Detach all policies first, then:
aws iam delete-role --role-name <role-name>
```

**Saves: $0.** IAM is free.

---

## Step 7 — Confirm $0 billing

```powershell
# Wait 24 hours for the bill to recalculate, then:
aws ce get-cost-and-usage ^
  --time-period "Start=$(Get-Date -Format 'yyyy-MM-01'),End=$(Get-Date -Format 'yyyy-MM-dd')" ^
  --granularity MONTHLY ^
  --metrics BlendedCost ^
  --region us-east-1
```

The `Total.BlendedCost.Amount` value should be at or near 0.

Also set a precautionary AWS Budget so you get alerted if anything
sneaks back:

```powershell
# From the repo:
$env:MS_BUDGET_EMAIL = "you@example.com"
python scripts/setup_aws_budget.py --limit 1
# (Budget at $1/mo; alarms at 80% forecast + 100% actual.)
```

---

## If you want to redeploy later

Everything you deleted is reproducible:

1. **ECR image** → CI rebuilds + pushes on every `main` push.
2. **S3 bucket** → `aws s3 mb s3://moviesentiment-dvc-soumya`, then
   `make hf-push` is replaced by pulling the models from HF and
   re-uploading to S3 if you really want the S3 path back.
3. **ECS service** → see `docs/runbook.md` §3 for the
   `create-cluster` + `register-task-definition` + `create-service`
   commands.

The HF Hub repos (v1 + v2) are the source of truth for the ONNX models
now. Anything else can be regenerated from the git repo.

---

## Quick "$0 right now" recipe

If you want to clear the entire bill in two commands and don't care
about preserving anything:

```powershell
# Nuke ECR + S3 contents in 1 minute
aws ecr delete-repository --repository-name moviesentiment --force --region ap-southeast-2
aws s3 rm s3://moviesentiment-dvc-soumya --recursive
aws s3api delete-bucket --bucket moviesentiment-dvc-soumya --region ap-southeast-2

# Optional cosmetic cleanup
aws logs delete-log-group --log-group-name /ecs/moviesentiment --region ap-southeast-2
```

Done. Bill drops to $0 within 24 h.
