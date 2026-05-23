"""FastAPI inference service."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from moviesentiment.config import settings
from moviesentiment.serve.inference import InferenceEngine
from moviesentiment.serve.schemas import HealthResponse, PredictRequest, PredictResponse

engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global engine
    try:
        engine = InferenceEngine.from_registry(settings.model_name, settings.model_stage)
    except Exception as exc:
        import logging

        logging.warning(f"Model not loaded at startup: {exc}")
    yield


app = FastAPI(
    title="MovieSentiment",
    version="1.0.0",
    description="IMDb sentiment analysis API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/readyz", response_model=HealthResponse, tags=["ops"])
def readyz() -> HealthResponse:
    if engine is None or not engine.is_ready():
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(status="ready")


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if len(req.texts) > settings.max_batch_size:
        raise HTTPException(status_code=422, detail=f"batch size > {settings.max_batch_size}")
    for t in req.texts:
        if len(t) > settings.max_text_length:
            raise HTTPException(status_code=422, detail="text exceeds max length")
    return PredictResponse(predictions=engine.predict(req.texts))


@app.get("/version", tags=["ops"])
def version() -> dict[str, str]:
    import subprocess

    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    return {"model_name": settings.model_name, "model_stage": settings.model_stage, "git_sha": sha}


Instrumentator().instrument(app).expose(app)
