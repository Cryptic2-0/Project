"""Evidently drift detection — compares production inputs to reference distribution."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add text length and word count features for drift analysis."""
    import pandas as _pd

    out: _pd.DataFrame = df.copy()
    out["text_length"] = out["text"].str.len()
    out["word_count"] = out["text"].str.split().str.len()
    return out


def run_drift_report(reference: Path, current: Path, out_dir: Path) -> Path:
    """Generate an Evidently DataDrift HTML report. Returns the report path."""
    import pandas as pd

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        # Evidently 0.7+ relocated the legacy API under evidently.legacy.*.
        from evidently.legacy.metric_preset import DataDriftPreset
        from evidently.legacy.report import Report

    ref_df = _add_text_features(pd.read_parquet(reference)[["text"]])
    cur_df = _add_text_features(pd.read_parquet(current)[["text"]])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today()}.html"
    report.save_html(str(out_path))
    return out_path


def drift_share(reference: Path, current: Path) -> float:
    """Return the share of drifted columns (0.0–1.0) without writing a report."""
    import pandas as pd

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        # Evidently 0.7+ relocated the legacy API under evidently.legacy.*.
        from evidently.legacy.metric_preset import DataDriftPreset
        from evidently.legacy.report import Report

    ref_df = _add_text_features(pd.read_parquet(reference)[["text"]])
    cur_df = _add_text_features(pd.read_parquet(current)[["text"]])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    result = report.as_dict()
    try:
        share: float = result["metrics"][0]["result"]["share_of_drifted_columns"]
        return share
    except (KeyError, IndexError):
        return 0.0
