"""NannyML performance estimation — predict production F1/accuracy without ground truth.

Input drift (Evidently) tells you the data has shifted. It does NOT tell you whether the
model still works on the shifted data. NannyML's CBPE (Confidence-Based Performance
Estimation) uses the confidence distribution of production predictions plus per-class
calibration from a reference set to estimate the expected F1/accuracy. When ground truth
arrives (delayed labels) you can back-test the estimate.

This module emits the estimated metrics as Prometheus gauges so the same Grafana dashboard
that shows drift can show expected-perf-under-drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from prometheus_client import Gauge

if TYPE_CHECKING:
    import pandas as pd


estimated_f1 = Gauge(
    "model_estimated_f1",
    "CBPE-estimated F1 on production traffic (no labels required)",
    labelnames=["metric"],
)


def _prep(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Reshape predict-log rows into NannyML's required columns."""
    import pandas as pd

    out = pd.DataFrame(
        {
            "y_pred_proba": df["confidence"].astype(float),
            "y_pred": (df["label"] == "positive").astype(int),
        }
    )
    if "true_label" in df.columns:
        out["y_true"] = (df["true_label"] == "positive").astype(int)
    if "timestamp" in df.columns:
        out["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    out["chunk"] = kind
    return out


def estimate(reference: Path, current: Path) -> dict[str, float]:
    """Run CBPE; reference must have ground-truth labels; current may be unlabeled.

    Returns the estimated F1 + ROC AUC for the current chunk. Also pushes the F1 to the
    Prometheus gauge so /metrics scrapes pick it up.
    """
    import nannyml as nml
    import pandas as pd

    ref_raw = pd.read_parquet(reference)
    cur_raw = pd.read_parquet(current)
    if "true_label" not in ref_raw.columns:
        raise ValueError("reference data must include a 'true_label' column")

    ref = _prep(ref_raw, "reference")
    cur = _prep(cur_raw, "analysis")

    cbpe = nml.CBPE(
        y_pred_proba="y_pred_proba",
        y_pred="y_pred",
        y_true="y_true",
        problem_type="classification_binary",
        metrics=["f1", "roc_auc"],
        chunk_size=max(50, len(cur) // 10 or 50),
    )
    cbpe.fit(ref)
    result = cbpe.estimate(cur)
    df = result.to_df()

    out: dict[str, float] = {}
    for metric in ("f1", "roc_auc"):
        col = (metric, "value")
        if col in df.columns:
            val = float(df[col].mean())
            out[metric] = val
            estimated_f1.labels(metric=metric).set(val)
    return out
