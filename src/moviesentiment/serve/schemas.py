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


# --- v2 multi-task "review intelligence" schemas -----------------------------

ASPECTS = ("acting", "plot", "visuals", "pacing", "sound")
EMOTIONS = ("joy", "anger", "fear", "sadness", "surprise", "disgust")


class AspectScores(BaseModel):
    """Per-aspect ternary sentiment: P(negative), P(neutral), P(positive)."""

    acting: list[float] = Field(..., min_length=3, max_length=3)
    plot: list[float] = Field(..., min_length=3, max_length=3)
    visuals: list[float] = Field(..., min_length=3, max_length=3)
    pacing: list[float] = Field(..., min_length=3, max_length=3)
    sound: list[float] = Field(..., min_length=3, max_length=3)


class EmotionScores(BaseModel):
    """Ekman 6-class probabilities."""

    joy: float
    anger: float
    fear: float
    sadness: float
    surprise: float
    disgust: float


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class AnalyzeResponse(BaseModel):
    text: str
    sentiment: Prediction
    aspects: AspectScores
    emotions: EmotionScores
    spoiler_prob: float = Field(..., ge=0.0, le=1.0)
    helpfulness: float = Field(..., ge=0.0, le=1.0)


class InsightsResponse(BaseModel):
    """Aggregated stats for a movie, computed offline over the reservoir sample."""

    movie_id: str
    n_reviews: int
    sentiment_positive_share: float
    aspect_means: dict[str, float]  # aspect -> mean signed score in [-1, 1]
    emotion_mix: dict[str, float]  # emotion -> share, sums to ~1
    spoiler_share: float
    helpfulness_mean: float
    topics: list[str] = Field(default_factory=list)  # populated when BERTopic batch runs


class SimilarHit(BaseModel):
    text: str
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class SimilarResponse(BaseModel):
    query: str
    hits: list[SimilarHit]
