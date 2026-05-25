"""FastAPI inference service."""

from __future__ import annotations

import random
import subprocess
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from moviesentiment.config import settings
from moviesentiment.monitor.prometheus import (
    model_version_info,
    prediction_class_total,
    prediction_confidence,
)
from moviesentiment.serve.inference import InferenceEngine
from moviesentiment.serve.schemas import HealthResponse, PredictRequest, PredictResponse

engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global engine
    try:
        engine = InferenceEngine.from_registry(settings.model_name, settings.model_stage)
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        model_version_info.labels(
            model_name=settings.model_name,
            version="onnx-int8",
            git_sha=sha,
        ).set(1)
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

    resp = PredictResponse(predictions=engine.predict(req.texts))

    for p in resp.predictions:
        prediction_confidence.observe(p.confidence)
        prediction_class_total.labels(label=p.label).inc()

    if random.random() < 0.1:
        _log_production(req.texts, resp)

    return resp


@app.get("/version", tags=["ops"])
def version() -> dict[str, str]:
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    return {"model_name": settings.model_name, "model_stage": settings.model_stage, "git_sha": sha}


def _log_production(texts: list[str], resp: PredictResponse) -> None:
    import pandas as pd

    prod_dir = settings.data_dir / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "text": t,
            "label": p.label,
            "confidence": p.confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for t, p in zip(texts, resp.predictions, strict=False)
    ]
    df = pd.DataFrame(records)
    path = prod_dir / "recent.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)


Instrumentator().instrument(app).expose(app)
