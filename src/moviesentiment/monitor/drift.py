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


def label_drift(reference: Path, current: Path, label_col: str = "label") -> float:
    """Concept-drift signal: total-variation distance between predicted-label shares.

    Complements `drift_share`, which only inspects input features. A model can
    sit on a stable input distribution and still produce a drifting output
    distribution (concept drift) when the underlying mapping moves — e.g. a
    new movie genre that the encoder happens to project to a different region.

    Returns a value in [0, 1]. 0 = identical class proportions, 1 = disjoint.
    A practical retrain trigger is `> 0.15` on a binary head; tune per task.
    """
    import pandas as pd

    ref_df = pd.read_parquet(reference)
    cur_df = pd.read_parquet(current)
    if label_col not in ref_df.columns or label_col not in cur_df.columns:
        return 0.0
    ref_share = ref_df[label_col].value_counts(normalize=True)
    cur_share = cur_df[label_col].value_counts(normalize=True)
    labels = set(ref_share.index) | set(cur_share.index)
    tv = 0.5 * sum(abs(ref_share.get(label, 0.0) - cur_share.get(label, 0.0)) for label in labels)
    return float(tv)
