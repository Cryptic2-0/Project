# Demo Walkthrough — Run + Show the Project

> Step-by-step guide to run MovieSentiment locally and demo it. Replaces the
> "live AWS URL" demo path now that the ECS service is torn down for cost
> control. Works on Windows / macOS / Linux.

Last updated: 2026-05-29.

---

## Prerequisites (one-time setup, ~5 min)

```powershell
# 1. Clone the repo if you haven't already
git clone https://github.com/Cryptic2-0/Project moviesentiment
cd moviesentiment

# 2. Create the virtualenv + install everything
pip install uv
uv pip install -e ".[dev]"

# 3. Pull the ONNX models from HuggingFace (public, no auth, ~130 MB)
huggingface-cli download Cryptic2-0/moviesentiment-distilbert-onnx-int8 ^
    --local-dir models/distilbert_onnx_int8
huggingface-cli download Cryptic2-0/moviesentiment-multitask-onnx-int8 ^
    --local-dir models/distilbert_multitask_onnx
```

If `huggingface-cli` is missing: `pip install -U huggingface_hub`.

Verify:

```powershell
dir models/distilbert_onnx_int8/model.onnx       # ~66 MB
dir models/distilbert_multitask_onnx/model.onnx  # ~66 MB
```

---

## Quick demo (terminal-only, no UI) — 30 seconds

```powershell
# Start the API
make serve
# OR if make is not on PATH:
uvicorn moviesentiment.serve.api:app --host 0.0.0.0 --port 8000

# In another terminal:
curl -X POST http://localhost:8000/predict ^
     -H "Content-Type: application/json" ^
     -d "{\"texts\":[\"A complete masterpiece.\",\"Worst film I have ever seen.\"]}"
```

Expected output:

```json
{"predictions":[
  {"text":"A complete masterpiece.","label":"positive","confidence":0.99},
  {"text":"Worst film I have ever seen.","label":"negative","confidence":0.98}
]}
```

For the v2 multi-task endpoint:

```powershell
curl -X POST http://localhost:8000/analyze ^
     -H "Content-Type: application/json" ^
     -d "{\"text\":\"The hero dies and the villain wins but the cinematography is breathtaking.\"}"
```

Returns all 5 heads: sentiment + aspects + emotion + spoiler_prob + helpfulness.

---

## Full demo (with frontend UI) — interview-grade, 2 min

The repo ships a static frontend at `frontend/index.html`. It is bundled into
the Fargate image and served at `/ui/` when the app is running.

### Mode A — Frontend served by the same FastAPI process (recommended)

```powershell
# Start the API (frontend mounts automatically at /ui)
make serve
```

Open in browser:

* **API docs (Swagger)**: http://localhost:8000/docs
* **Frontend UI**: http://localhost:8000/ui/

The UI has a 3-mode toggle in the top-right:

1. **Mock** — keyword-based stub (works offline, useful when explaining
   the architecture without spinning up the model).
2. **Live (auto)** — reads `api.json` to find a deployed URL (currently
   points at the torn-down ECS task; falls back to localhost).
3. **Custom IP** — paste any host/port; useful if you redeploy to ECS
   for the day.

For a local demo: pick **Custom IP** → enter `http://localhost:8000` → click
through the sample reviews. The UI animates the architecture diagram,
shows the live prediction confidence as ASCII bars, and tracks request
counts.

### Mode B — Full Docker stack (Prometheus + Grafana + API)

If the interviewer asks "where are your dashboards?":

```powershell
make docker-up
# OR:
docker compose -f deploy/docker-compose.yml up --build
```

Brings up:

* **API**         → http://localhost:8000 (same endpoints as Mode A)
* **Frontend**    → http://localhost:8000/ui/
* **Prometheus**  → http://localhost:9090 (target list shows the API)
* **Grafana**     → http://localhost:3000 (login `admin` / `admin`,
                   dashboard auto-provisioned with prediction-confidence
                   histogram + class distribution + request rate)

Send a handful of predictions in the API tab so the Grafana panels have
data, then switch to Grafana to show the live metrics. This is the
single best 30-second visual for an interview.

---

## What to actually click through for an interview (3 min)

1. **Architecture diagram** (README.md Mermaid block) — point at the
   data → train → serve → monitor → retrain loop.
2. **Open the UI** at http://localhost:8000/ui/ in Custom IP mode pointing
   at localhost. Show two reviews getting scored live.
3. **Hit `/analyze`** with a phrase that has a real spoiler + aspect
   mixture. Walk through the JSON: "see how we get sentiment, aspect
   scores, emotion mix, spoiler probability, helpfulness regression in
   one forward pass."
4. **MLflow runs**: `mlflow ui` in another terminal. Open the runs page,
   show baseline vs DistilBERT vs multi-task. Point at the F1 numbers.
5. **Grafana** (only if Mode B is up): show the prediction-confidence
   histogram, class distribution, and the request-rate panel.
6. **Drift report**: `docs/drift_reports/<latest>.html` in the browser.
   Or run `moviesentiment drift` to generate a fresh one.
7. **CI**: GitHub Actions page (github.com/Cryptic2-0/Project/actions) —
   show the green check on the latest commit + the 85% coverage gate.

The interview-talking-points doc at `docs/interview_talking_points.md`
has the 60-second answer for every decision.

---

## Stopping the demo

```powershell
# Mode A (uvicorn): Ctrl+C in the make serve terminal.

# Mode B (Docker): in a new terminal
docker compose -f deploy/docker-compose.yml down -v

# Confirm nothing is left running:
docker ps
```

Reclaim disk if Docker left a lot of layers behind:

```powershell
docker system prune -af --volumes
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `make serve` fails: "model not found" | You skipped step 3 in Prerequisites. Run the two `huggingface-cli download` commands. |
| `/predict` returns 503 | The ONNX model didn't load. Check the uvicorn log for `model_load_failed`. |
| `/analyze` returns 503 | v2 multi-task model not on disk. Run the second `huggingface-cli download`. |
| Grafana shows no data | Hit `/predict` a few times to generate metrics; Prometheus scrape interval is 15 s. |
| Frontend "Live (auto)" mode shows offline | Expected — the deployed ECS task is currently scaled to zero. Use Custom IP → `http://localhost:8000`. |
| `huggingface-cli` not found | `pip install -U huggingface_hub`. |
| Port 8000 already in use | `make serve PORT=8001` (uvicorn) or change `deploy/docker-compose.yml` port mapping. |
