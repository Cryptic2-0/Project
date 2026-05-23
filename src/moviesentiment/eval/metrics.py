"""Evaluation utilities: confusion matrix, ROC, PR curve, calibration, slice analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def log_confusion_matrix(y_true: pd.Series[Any], y_pred: np.ndarray[Any, Any]) -> None:
    """Plot and log confusion matrix as an MLflow artifact."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    ax.set_title("Confusion Matrix")
    tmp = Path("metrics/confusion_matrix.png")
    tmp.parent.mkdir(exist_ok=True)
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(tmp))


def slice_by_length(df: pd.DataFrame, preds: np.ndarray[Any, Any]) -> pd.DataFrame:
    """Return accuracy broken down by review-length quartile."""
    df = df.copy()
    df["pred"] = preds
    df["correct"] = df["pred"] == df["label"]
    df["length_bin"] = pd.qcut(df["text"].str.len(), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    return df.groupby("length_bin")["correct"].mean().rename("accuracy").reset_index()
