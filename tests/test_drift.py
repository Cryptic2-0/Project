"""Drift detector — synthetic data with known divergence asserts share > 0.0."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from moviesentiment.monitor.drift import _add_text_features, drift_share


def test_add_text_features() -> None:
    df = pd.DataFrame({"text": ["hello world", "foo bar baz", ""]})
    out = _add_text_features(df)
    assert "text_length" in out.columns
    assert "word_count" in out.columns
    assert out.loc[0, "text_length"] == 11
    assert out.loc[1, "word_count"] == 3


@pytest.mark.slow
def test_drift_share_synthetic_drift(tmp_path: Path) -> None:
    """Reference has short reviews; current has long reviews. Drift should be > 0."""
    ref = pd.DataFrame({"text": ["a b c"] * 100 + ["short review"] * 100})
    cur = pd.DataFrame(
        {"text": ["a b c d e f g h i j k l m n o p" * 5] * 100 + ["different text"] * 100}
    )
    ref_p = tmp_path / "ref.parquet"
    cur_p = tmp_path / "cur.parquet"
    ref.to_parquet(ref_p)
    cur.to_parquet(cur_p)

    share = drift_share(ref_p, cur_p)
    # text-length and word-count distributions are radically different — at least one
    # of the two engineered features must register as drifted
    assert share > 0.0, f"expected drift > 0, got {share}"


def test_drift_share_no_drift_when_identical(tmp_path: Path) -> None:
    """Same distribution on both sides should NOT register drift."""
    pytest.importorskip("evidently.metric_preset")
    df = pd.DataFrame({"text": [f"review {i} text " * 5 for i in range(100)]})
    ref_p = tmp_path / "ref.parquet"
    cur_p = tmp_path / "cur.parquet"
    df.to_parquet(ref_p)
    df.to_parquet(cur_p)

    share = drift_share(ref_p, cur_p)
    assert share == 0.0, f"expected zero drift on identical data, got {share}"
