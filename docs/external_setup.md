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

- [ ] **Install Docker Desktop** (needed Week 2, Day 10 — install now so it
  finishes updating in the background)
  ```powershell
  winget install Docker.DockerDesktop
  ```

---

## Week 2 (Days 6–10)

- [ ] **GPU access for DistilBERT fine-tuning**
  There is no local GPU on this machine. Options (pick one):
  - **Google Colab free tier** — upload `src/` + `data/processed/`, run
    `transformer.py`, download `models/` back.
  - **Kaggle Notebooks** — free T4 GPU, 30 h/week.
  - **GitHub Codespaces with GPU** (paid, ~$0.36/h for T4).
  The training script is already wired to detect CUDA automatically.

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

- [ ] **Add DVC remote credentials to GitHub Secrets** (so CI can `dvc pull`)
  Depends on your remote choice:
  - GDrive: export `GDRIVE_CREDENTIALS_DATA` (JSON from OAuth flow).
  - Local path: not usable in CI — switch to GDrive or an S3-compatible
    store before CI setup.
  Secret name: `DVC_GDRIVE_CREDENTIALS_DATA` (or whatever the dvc remote
  modifier expects).

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
| Docker Desktop installed | pending | Week 1 |
| GPU access (Colab / Kaggle) | pending | Week 2 |
| Fly.io account + flyctl | pending | Week 3 |
| FLY_API_TOKEN in GitHub Secrets | pending | Week 3 |
| DVC credentials in GitHub Secrets | pending | Week 3 |
| Grafana Cloud (optional) | pending | Week 3 |
| Loom account | pending | Week 4 |
| Excalidraw diagram | pending | Week 4 |
| LinkedIn update | pending | Week 4 |
| Pin repo on GitHub profile | pending | Week 4 |
| Open v1.0 self-PR | pending | Week 4 |
