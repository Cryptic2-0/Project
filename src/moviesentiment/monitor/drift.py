"""Evidently drift detection — compares production inputs to reference distribution."""

from __future__ import annotations

from datetime import date
from pathlib import Path


def run_drift_report(reference: Path, current: Path, out_dir: Path) -> Path:
    """Generate an Evidently DataDrift HTML report. Returns the report path."""
    import pandas as pd
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    ref_df = pd.read_parquet(reference)[["text"]]
    cur_df = pd.read_parquet(current)[["text"]]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today()}.html"
    report.save_html(str(out_path))
    return out_path
