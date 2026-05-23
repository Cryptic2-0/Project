"""TF-IDF + Logistic Regression baseline, tracked with MLflow."""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from moviesentiment.config import settings
from moviesentiment.eval.metrics import log_confusion_matrix


def build_pipeline(params: dict[str, object]) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(params["tfidf_ngram_min"], params["tfidf_ngram_max"]),
                    max_features=params["tfidf_max_features"],
                    sublinear_tf=True,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    C=params["lr_C"],
                    max_iter=params["lr_max_iter"],
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_baseline() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())["baseline"]

    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name="baseline-tfidf-lr", tags={"model": "baseline"}):
        mlflow.log_params(params)

        pipe = build_pipeline(params)
        pipe.fit(train["text"], train["label"])

        preds = pipe.predict(val["text"])
        proba = pipe.predict_proba(val["text"])[:, 1]

        acc = accuracy_score(val["label"], preds)
        f1 = f1_score(val["label"], preds, average="macro")
        auc = roc_auc_score(val["label"], proba)

        mlflow.log_metrics({"accuracy": acc, "macro_f1": f1, "roc_auc": auc})
        log_confusion_matrix(val["label"], preds)

        model_path = settings.model_dir / "baseline.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_path)
        mlflow.sklearn.log_model(
            pipe, artifact_path="model", registered_model_name=settings.model_name
        )

        Path("metrics/baseline.json").parent.mkdir(exist_ok=True)
        import json

        Path("metrics/baseline.json").write_text(
            json.dumps({"accuracy": acc, "macro_f1": f1, "roc_auc": auc})
        )

        print(f"Baseline — acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")
