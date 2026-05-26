"""Pydantic request/response schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=32)

    model_config = {
        "json_schema_extra": {
            "examples": [{"texts": ["A complete masterpiece.", "Two hours I will never get back."]}]
        }
    }


class Prediction(BaseModel):
    text: str
    label: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=10, ge=1, le=50)


class TokenAttribution(BaseModel):
    token: str
    attribution: float


class ExplainResponse(BaseModel):
    text: str
    label: str
    confidence: float
    attributions: list[TokenAttribution]
