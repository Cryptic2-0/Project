"""Drift detector — synthetic data with known divergence asserts share > 0.0."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from moviesentiment.monitor.drift import _add_text_features, drift_share, label_drift


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


def _have_evidently() -> bool:
    try:
        import evidently.legacy.metric_preset  # noqa: F401

        return True
    except ImportError:
        try:
            import evidently.metric_preset  # noqa: F401

            return True
        except ImportError:
            return False


@pytest.mark.skipif(not _have_evidently(), reason="evidently not installed")
def test_drift_share_no_drift_when_identical(tmp_path: Path) -> None:
    """Same distribution on both sides should NOT register drift."""
    df = pd.DataFrame({"text": [f"review {i} text " * 5 for i in range(100)]})
    ref_p = tmp_path / "ref.parquet"
    cur_p = tmp_path / "cur.parquet"
    df.to_parquet(ref_p)
    df.to_parquet(cur_p)

    share = drift_share(ref_p, cur_p)
    assert share == 0.0, f"expected zero drift on identical data, got {share}"


def test_label_drift_identical_zero(tmp_path: Path) -> None:
    """Concept-drift signal is zero when reference and current label shares match."""
    df = pd.DataFrame({"label": ["positive"] * 50 + ["negative"] * 50})
    ref_p = tmp_path / "ref.parquet"
    cur_p = tmp_path / "cur.parquet"
    df.to_parquet(ref_p)
    df.to_parquet(cur_p)
    assert label_drift(ref_p, cur_p) == 0.0


def test_label_drift_disjoint_one(tmp_path: Path) -> None:
    """Total-variation distance hits 1.0 when ref is all-positive and current is all-negative."""
    ref = pd.DataFrame({"label": ["positive"] * 100})
    cur = pd.DataFrame({"label": ["negative"] * 100})
    ref_p = tmp_path / "ref.parquet"
    cur_p = tmp_path / "cur.parquet"
    ref.to_parquet(ref_p)
    cur.to_parquet(cur_p)
    assert abs(label_drift(ref_p, cur_p) - 1.0) < 1e-9


def test_label_drift_skew_partial(tmp_path: Path) -> None:
    ref = pd.DataFrame({"label": ["positive"] * 50 + ["negative"] * 50})
    cur = pd.DataFrame({"label": ["positive"] * 80 + ["negative"] * 20})
    ref_p = tmp_path / "ref.parquet"
    cur_p = tmp_path / "cur.parquet"
    ref.to_parquet(ref_p)
    cur.to_parquet(cur_p)
    # 50/50 -> 80/20 => |0.5-0.8| + |0.5-0.2| = 0.6 ; TV = 0.5 * 0.6 = 0.3
    assert abs(label_drift(ref_p, cur_p) - 0.3) < 1e-6


def test_label_drift_missing_column_returns_zero(tmp_path: Path) -> None:
    df = pd.DataFrame({"text": ["x"] * 10})
    ref_p = tmp_path / "ref.parquet"
    cur_p = tmp_path / "cur.parquet"
    df.to_parquet(ref_p)
    df.to_parquet(cur_p)
    assert label_drift(ref_p, cur_p) == 0.0


@pytest.mark.skipif(not _have_evidently(), reason="evidently not installed")
def test_run_drift_report_writes_html(tmp_path: Path) -> None:
    """Smoke-test run_drift_report end-to-end on tiny synthetic data."""
    from moviesentiment.monitor.drift import run_drift_report

    df = pd.DataFrame({"text": [f"review {i} text " * 5 for i in range(50)]})
    ref_p = tmp_path / "ref.parquet"
    cur_p = tmp_path / "cur.parquet"
    df.to_parquet(ref_p)
    df.to_parquet(cur_p)

    out_path = run_drift_report(ref_p, cur_p, tmp_path / "out")
    assert out_path.exists()
    assert out_path.suffix == ".html"
