"""Evaluation utilities: confusion matrix, ROC, PR curve, calibration, slice analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)

_METRICS_DIR = Path("metrics")


def _save_and_log(fig: Figure, name: str) -> None:
    _METRICS_DIR.mkdir(exist_ok=True)
    path = _METRICS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path))


def log_confusion_matrix(y_true: pd.Series[Any], y_pred: np.ndarray[Any, Any]) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    ax.set_title("Confusion Matrix")
    _save_and_log(fig, "confusion_matrix.png")


def log_roc_curve(y_true: pd.Series[Any], y_prob: np.ndarray[Any, Any]) -> None:
    """Log ROC curve; y_prob is probability of positive class."""
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC Curve")
    _save_and_log(fig, "roc_curve.png")


def log_pr_curve(y_true: pd.Series[Any], y_prob: np.ndarray[Any, Any]) -> None:
    """Log precision-recall curve; y_prob is probability of positive class."""
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("Precision-Recall Curve")
    _save_and_log(fig, "pr_curve.png")


def log_calibration_plot(y_true: pd.Series[Any], y_prob: np.ndarray[Any, Any]) -> None:
    """Log reliability diagram (calibration curve)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, ax=ax)
    ax.set_title("Calibration Plot")
    _save_and_log(fig, "calibration.png")


def log_classification_report(y_true: pd.Series[Any], y_pred: np.ndarray[Any, Any]) -> None:
    """Log per-class precision/recall/F1 as MLflow metrics and a text artifact."""
    report_str = classification_report(y_true, y_pred, target_names=["negative", "positive"])
    _METRICS_DIR.mkdir(exist_ok=True)
    report_path = _METRICS_DIR / "classification_report.txt"
    report_path.write_text(report_str)
    mlflow.log_artifact(str(report_path))

    report_dict = classification_report(
        y_true, y_pred, target_names=["negative", "positive"], output_dict=True
    )
    for label in ("negative", "positive"):
        for metric in ("precision", "recall", "f1-score"):
            mlflow.log_metric(f"{label}_{metric.replace('-', '_')}", report_dict[label][metric])


def log_all_eval(
    y_true: pd.Series[Any],
    y_pred: np.ndarray[Any, Any],
    y_prob: np.ndarray[Any, Any] | None = None,
) -> None:
    """Convenience wrapper: log all eval artifacts in one call."""
    log_confusion_matrix(y_true, y_pred)
    log_classification_report(y_true, y_pred)
    if y_prob is not None:
        log_roc_curve(y_true, y_prob)
        log_pr_curve(y_true, y_prob)
        log_calibration_plot(y_true, y_prob)


def slice_by_length(df: pd.DataFrame, preds: np.ndarray[Any, Any]) -> pd.DataFrame:
    """Return accuracy broken down by review-length quartile."""
    df = df.copy()
    df["pred"] = preds
    df["correct"] = df["pred"] == df["label"]
    df["length_bin"] = pd.qcut(df["text"].str.len(), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    return df.groupby("length_bin")["correct"].mean().rename("accuracy").reset_index()
