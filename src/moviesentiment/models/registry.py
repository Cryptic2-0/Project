"""MLflow model registry helpers: promote, load, compare."""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from moviesentiment.config import settings


def promote_model(run_id: str, stage: str = "Staging") -> None:
    """Register a run's model artifact and promote it to the given stage."""
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    result = mlflow.register_model(f"runs:/{run_id}/model", settings.model_name)
    client.transition_model_version_stage(
        name=settings.model_name,
        version=result.version,
        stage=stage,
    )


def load_production_model() -> object:
    """Load the Production-stage model from the registry."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.pyfunc.load_model(f"models:/{settings.model_name}/{settings.model_stage}")
