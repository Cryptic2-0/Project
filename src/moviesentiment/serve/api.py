"""FastAPI inference service."""

from __future__ import annotations

import os
import subprocess
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from moviesentiment.config import settings
from moviesentiment.monitor.prometheus import (
    model_version_info,
    prediction_class_total,
    prediction_confidence,
)
from moviesentiment.serve.inference import InferenceEngine
from moviesentiment.serve.insights import compute_insights
from moviesentiment.serve.logging_setup import bind_request_id, configure_logging, log
from moviesentiment.serve.multitask_inference import MultiTaskInferenceEngine
from moviesentiment.serve.reservoir import ReservoirSampler
from moviesentiment.serve.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    InsightsResponse,
    PredictRequest,
    PredictResponse,
    TokenAttribution,
)
from moviesentiment.serve.tracing import setup_tracing


def _git_sha() -> str:
    """Return git SHA: env var (set at image build), or git command, or 'unknown'."""
    if env_sha := os.environ.get("GIT_SHA"):
        return env_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


engine: InferenceEngine | None = None
multitask_engine: MultiTaskInferenceEngine | None = None
sampler = ReservoirSampler(k=1000)
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    global engine, multitask_engine
    try:
        engine = InferenceEngine.from_registry(settings.model_name, settings.model_stage)
        model_version_info.labels(
            model_name=settings.model_name,
            version="onnx-int8",
            git_sha=_git_sha(),
        ).set(1)
        log.info("model_loaded", model=settings.model_name, stage=settings.model_stage)
    except Exception as exc:
        log.warning("model_load_failed", error=str(exc))

    # v2 multi-task model is optional; /analyze returns 503 when absent so the
    # v1 deployment path keeps working without the new artefact.
    try:
        multitask_engine = MultiTaskInferenceEngine.from_disk()
        log.info("multitask_model_loaded")
    except Exception as exc:
        log.info("multitask_model_unavailable", reason=str(exc))
    yield


app = FastAPI(
    title="MovieSentiment",
    version="1.0.0",
    description="IMDb sentiment analysis API",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

_origins = settings.cors_origins_list()
# Wildcard + credentials is rejected by browsers; only set allow_credentials when the
# allowlist is explicit. Default config is no-credentials wildcard for the public demo.
_allow_credentials = bool(_origins) and "*" not in _origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
    allow_credentials=_allow_credentials,
    max_age=600,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    rid = bind_request_id(request.headers.get("x-request-id"))
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/readyz", response_model=HealthResponse, tags=["ops"])
def readyz() -> HealthResponse:
    if engine is None or not engine.is_ready():
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(status="ready")


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
@limiter.limit("60/minute")
def predict(request: Request, req: PredictRequest) -> PredictResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if len(req.texts) > settings.max_batch_size:
        raise HTTPException(status_code=422, detail=f"batch size > {settings.max_batch_size}")
    for t in req.texts:
        if len(t) > settings.max_text_length:
            raise HTTPException(status_code=422, detail="text exceeds max length")

    resp = PredictResponse(predictions=engine.predict(req.texts))

    for p in resp.predictions:
        prediction_confidence.observe(p.confidence)
        prediction_class_total.labels(label=p.label).inc()

    # Reservoir sampling: keep a uniform sample of N production inputs at fixed memory.
    # Replaces an earlier random.random() < 0.1 sample which biased toward early traffic.
    for text, pred in zip(req.texts, resp.predictions, strict=False):
        record = {
            "text": text,
            "label": pred.label,
            "confidence": pred.confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        sampler.add(record)

    sampler.maybe_flush(settings.data_dir / "production" / "recent.parquet")

    log.info(
        "predict_complete",
        n=len(req.texts),
        labels=[p.label for p in resp.predictions],
    )

    return resp


@app.post("/explain", response_model=ExplainResponse, tags=["inference"])
@limiter.limit("10/minute")
def explain(request: Request, req: ExplainRequest) -> ExplainResponse:
    """Per-token attribution for a single review via occlusion (drop-one-token).

    Opt-in interpretability surface — costs ~K× the latency of /predict where K is the
    token count. Kept separate so the hot /predict path stays cheap.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if len(req.text) > settings.max_text_length:
        raise HTTPException(status_code=422, detail="text exceeds max length")

    from moviesentiment.serve.explain import occlusion_attribution

    base, attrs = occlusion_attribution(engine, req.text, top_k=req.top_k)
    return ExplainResponse(
        text=req.text,
        label=base.label,
        confidence=base.confidence,
        attributions=[TokenAttribution(token=t, attribution=a) for t, a in attrs],
    )


@app.get("/version", tags=["ops"])
def version() -> dict[str, str]:
    return {
        "model_name": settings.model_name,
        "model_stage": settings.model_stage,
        "git_sha": _git_sha(),
    }


@app.get("/sample", tags=["ops"])
def sample_state() -> JSONResponse:
    """Return reservoir-sampler stats (size, seen, flushed). For debugging only."""
    return JSONResponse(sampler.stats())


@app.post("/analyze", response_model=AnalyzeResponse, tags=["inference"])
@limiter.limit("30/minute")
def analyze(request: Request, req: AnalyzeRequest) -> AnalyzeResponse:
    """v2 multi-task analyse: sentiment + ABSA + emotion + spoiler + helpfulness.

    Returns 503 until the multi-task ONNX artefact is trained and bundled. See
    docs/future_improvements.md for the training plan.
    """
    if multitask_engine is None:
        raise HTTPException(status_code=503, detail="multitask model not loaded")
    if len(req.text) > settings.max_text_length:
        raise HTTPException(status_code=422, detail="text exceeds max length")
    return multitask_engine.analyze(req.text)


@app.get("/insights/{movie_id}", response_model=InsightsResponse, tags=["inference"])
def insights(movie_id: str) -> InsightsResponse:
    """Aggregated per-movie review intelligence over the production reservoir."""
    result = compute_insights(movie_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no data for that movie yet")
    return result


Instrumentator().instrument(app).expose(app)

setup_tracing(app)

# Async inference endpoints (SQS+Lambda). Returns 503 if MS_SQS_QUEUE_URL+MS_JOBS_TABLE
# are not set, so local dev works without AWS plumbing.
from moviesentiment.serve.async_api import async_router  # noqa: E402

app.include_router(async_router)

# Optional: serve the static frontend from the same Fargate task. Mounted last so it
# doesn't shadow JSON routes. Skipped silently if frontend/index.html isn't bundled.
_frontend_dir = settings.project_root / "frontend"
if (_frontend_dir / "index.html").exists():
    from fastapi.staticfiles import StaticFiles  # noqa: E402

    app.mount("/ui", StaticFiles(directory=_frontend_dir, html=True), name="ui")
