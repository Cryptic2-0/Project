# External Setup Checklist

Things that live **outside the codebase** — accounts, secrets, installs, and
manual one-time steps. Check each item off as you complete it.

---

## Immediate (needed before the pipeline can run end-to-end)

- [x] **Push local commits to GitHub**
  The repo currently has 3 unpushed commits (scraper + Day 3 tests).
  ```powershell
  git push origin main
  ```

- [x] **Configure a DVC remote**
  Without a remote, `dvc push` / `dvc pull` don't work and the pipeline
  can't be reproduced by anyone else (or by CI).
  Easiest free option — Google Drive:
  ```powershell
  dvc remote add -d gdrive gdrive://<your-folder-id>
  dvc remote modify gdrive gdrive_acknowledge_abuse true
  dvc push   # first push after running dvc repro
  ```
  Alternative: a local path remote works for solo use but won't help in CI.
  ```powershell
  dvc remote add -d localremote C:/dvc-store
  ```
  > Once set, commit `.dvc/config` so CI can read the remote URL.

---

## Week 1 (before Day 4 baseline training)

- [x] **Run the DVC pipeline locally for the first time**
  This downloads the HuggingFace dataset (~100 MB) and produces all
  intermediate Parquet files.
  ```powershell
  .\.venv\Scripts\dvc.exe repro
  ```
  Then push the data artifacts to your DVC remote:
  ```powershell
  .\.venv\Scripts\dvc.exe push
  ```

- [x] **Install Docker Desktop** (needed Week 2, Day 10 — install now so it
  finishes updating in the background)
  ```powershell
  winget install Docker.DockerDesktop
  ```

---

## Week 2 (Days 6–10)

- [ ] **GPU training — SageMaker Training Job (ml.g4dn.xlarge T4, ~$0.30)**
  Artifacts land directly in S3. No browser babysitting, no manual zip download.

  **One-time IAM setup (manual):**
  1. AWS Console → IAM → Roles → Create role
  2. Trusted entity: **AWS service → SageMaker**
  3. Attach policy: `AmazonSageMakerFullAccess`
  4. Add inline policy (S3 access for DVC remote):
     ```json
     {
       "Effect": "Allow",
       "Action": "s3:*",
       "Resource": [
         "arn:aws:s3:::moviesentiment-dvc-soumya",
         "arn:aws:s3:::moviesentiment-dvc-soumya/*"
       ]
     }
     ```
  5. Copy the role ARN (format: `arn:aws:iam::<account-id>:role/<role-name>`)

  **Submit training job:**
  ```powershell
  pip install sagemaker   # one-time; or: pip install -e ".[train]"
  $env:SAGEMAKER_ROLE_ARN = "arn:aws:iam::<account-id>:role/<role-name>"
  python scripts/sagemaker_launch.py
  ```
  Logs stream live. ~25 min. Final line prints the S3 artifact path.

  **After job completes:**
  ```powershell
  # Copy the S3 path printed by sagemaker_launch.py, then:
  $S3_PATH = "s3://moviesentiment-dvc-soumya/sagemaker-output/<job-name>/output/model.tar.gz"
  aws s3 cp $S3_PATH distilbert_artifacts.tar.gz
  tar -xzf distilbert_artifacts.tar.gz
  Move-Item distilbert models\distilbert -Force
  New-Item -ItemType Directory -Force metrics | Out-Null
  Move-Item transformer.json metrics\transformer.json -Force
  .venv\Scripts\dvc.exe add models\distilbert
  .venv\Scripts\dvc.exe push
  git add models\distilbert.dvc metrics\transformer.json dvc.lock
  git commit -m "feat(day6): distilbert fine-tuning artifacts"
  git push
  ```

- [ ] **Verify Docker Desktop is running** after install, then test:
  ```powershell
  docker run --rm hello-world
  ```

- [ ] **GitHub Container Registry (GHCR) write access**
  GHCR uses `GITHUB_TOKEN` automatically — no extra token needed.
  Just make sure your GitHub Actions workflow has:
  ```yaml
  permissions:
    packages: write
  ```
  This is already in the CI template in `MOVIESENTIMENT_BUILD_GUIDE.md`.

---

## Week 3 (Days 11–15)

- [ ] **Create a Fly.io account** (free, no credit card for hobby tier)
  https://fly.io/app/sign-up
  Then install the CLI:
  ```powershell
  winget install superfly.flyctl
  flyctl auth login
  ```

- [ ] **Add `FLY_API_TOKEN` to GitHub Secrets**
  1. `flyctl auth token` → copy the token.
  2. Go to: GitHub repo → Settings → Secrets and variables → Actions →
     New repository secret.
  3. Name: `FLY_API_TOKEN`, Value: the token from step 1.

- [ ] **Add DVC remote credentials (AWS S3) to GitHub Secrets**
  DVC remote is S3 (`moviesentiment-dvc-soumya`, `ap-southeast-2`).
  CI needs the same IAM keys to `dvc pull`.
  1. GitHub repo → Settings → Secrets and variables → Actions.
  2. Add two secrets:
     - `AWS_ACCESS_KEY_ID` — your IAM key ID
     - `AWS_SECRET_ACCESS_KEY` — your IAM secret key
  3. In the CI workflow add:
     ```yaml
     env:
       AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
       AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
     ```

- [ ] **Grafana Cloud free tier** (optional — screenshots in README is fine)
  https://grafana.com/auth/sign-up
  Free tier supports Prometheus remote-write and 10 K series.
  You'll point `deploy/prometheus.yml` scrape target at the Fly.io URL.

---

## Week 4 (Days 16–21)

- [ ] **Loom account** for the 3-minute demo video
  https://www.loom.com/signup (free tier is enough)
  Record: architecture diagram → MLflow UI → live curl → Grafana → drift
  report → one CI run.

- [ ] **Excalidraw** for the architecture diagram (no install, browser-based)
  https://excalidraw.com
  Export as PNG → save to `docs/architecture.png`.

- [ ] **LinkedIn update**
  After tagging `v1.0`: add the live Fly.io URL + Loom link to your profile
  and/or a post. Link directly to the GitHub repo.

- [ ] **Pin the repo on your GitHub profile**
  GitHub profile → Customize your pins → add `moviesentiment`.

- [ ] **Open `v1.0` self-PR**
  ```powershell
  gh pr create --title "v1.0 release" --body "Release notes: ..."
  ```
  Interviewers check PR hygiene — this signals you follow the same process
  you'd follow on a team.

---

## Optional / nice-to-have

- [ ] **HuggingFace Hub token** — only needed if you want to push the
  fine-tuned model to the Hub as a public artifact.
  https://huggingface.co/settings/tokens → New token (write scope).
  Add as GitHub Secret `HF_TOKEN`.

- [ ] **Custom domain on Fly.io** — not needed; `*.fly.dev` URL is fine for
  an interviewer demo.

- [ ] **Dagshub** — free hosted MLflow + DVC remote in one place. Good
  alternative to Google Drive + local MLflow if you want everything in the
  cloud from day one.
  https://dagshub.com

---

## Status summary

| Item | Status | Week |
|------|--------|------|
| Push local commits | done | Now |
| DVC remote configured | done | Now |
| DVC pipeline run locally | done | Week 1 |
| Docker Desktop installed | done | Week 1 |
| GPU training (SageMaker ml.g4dn.xlarge) | pending | Week 2 |
| Fly.io account + flyctl | pending | Week 3 |
| FLY_API_TOKEN in GitHub Secrets | pending | Week 3 |
| DVC credentials in GitHub Secrets (S3/AWS) | pending | Week 3 |
| Grafana Cloud (optional) | pending | Week 3 |
| Loom account | pending | Week 4 |
| Excalidraw diagram | pending | Week 4 |
| LinkedIn update | pending | Week 4 |
| Pin repo on GitHub profile | pending | Week 4 |
| Open v1.0 self-PR | pending | Week 4 |
