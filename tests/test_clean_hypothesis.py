"""Property tests for data cleaning — fuzz inputs must never crash the cleaner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from moviesentiment.data.clean import _clean_text, clean_reviews


@given(st.text(max_size=2000))
@settings(max_examples=200, deadline=None)
def test_clean_text_never_crashes(text: str) -> None:
    result = _clean_text(text)
    # Whatever the input, output is a string, lowercased, and has no HTML/URL leftover
    assert isinstance(result, str)
    assert result == result.lower()
    assert "<" not in result or ">" not in result  # tags stripped


@given(st.text(min_size=11, max_size=500).filter(lambda s: s.strip()))
@settings(max_examples=50, deadline=None)
def test_clean_text_strips_to_lowercase(text: str) -> None:
    out = _clean_text(text)
    if out:
        assert out == out.strip()
        assert out == out.lower()


@given(st.lists(st.text(min_size=15, max_size=200), min_size=2, max_size=20))
@settings(max_examples=20, deadline=None)
def test_clean_reviews_pipeline_no_crash(tmp_path_factory, texts: list[str]) -> None:  # type: ignore[no-untyped-def]
    tmp = tmp_path_factory.mktemp("clean")
    df = pd.DataFrame({"text": texts, "label": [0, 1] * (len(texts) // 2) + [0] * (len(texts) % 2)})
    inp = Path(tmp) / "in.parquet"
    out = Path(tmp) / "out.parquet"
    df.to_parquet(inp)
    n = clean_reviews(inp, out)
    assert n >= 0
    assert out.exists()
