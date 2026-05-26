"""Tests for data cleaning logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from moviesentiment.data.clean import _clean_text, clean_reviews

# ---------------------------------------------------------------------------
# _clean_text unit tests
# ---------------------------------------------------------------------------


def test_strips_html() -> None:
    assert "<b>" not in _clean_text("A <b>great</b> film")


def test_removes_url() -> None:
    assert "http" not in _clean_text("check https://example.com for details")


def test_lowercases() -> None:
    result = _clean_text("GREAT Film")
    assert result == result.lower()


def test_collapses_whitespace() -> None:
    result = _clean_text("too   many    spaces")
    assert "  " not in result


def test_empty_after_cleaning_is_short() -> None:
    result = _clean_text("<b></b>")
    assert len(result) <= 10


# ---------------------------------------------------------------------------
# clean_reviews() end-to-end tests
# ---------------------------------------------------------------------------


def _make_parquet(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    p = tmp_path / "raw.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


@pytest.fixture()
def basic_inp(tmp_path: Path) -> Path:
    return _make_parquet(
        tmp_path,
        [
            {
                "review_id": "r1",
                "movie_id": "tt1",
                "text": "A <b>truly</b> remarkable film. https://example.com",
                "rating": 9,
                "scraped_at": "2024-01-01",
                "label": 1,
            },
            {
                "review_id": "r2",
                "movie_id": "tt1",
                "text": "Absolute garbage, boring and terribly slow.",
                "rating": 2,
                "scraped_at": "2024-01-01",
                "label": 0,
            },
        ],
    )


def test_clean_reviews_strips_html_and_urls(tmp_path: Path, basic_inp: Path) -> None:
    out = tmp_path / "clean.parquet"
    clean_reviews(basic_inp, out)
    df = pd.read_parquet(out)
    assert all("<" not in t for t in df["text"])
    assert all("http" not in t for t in df["text"])


def test_clean_reviews_deduplicates(tmp_path: Path) -> None:
    inp = _make_parquet(
        tmp_path,
        [
            {
                "review_id": "r1",
                "movie_id": "tt1",
                "text": "identical review text here",
                "rating": 8,
                "scraped_at": "2024-01-01",
                "label": 1,
            },
            {
                "review_id": "r2",
                "movie_id": "tt1",
                "text": "identical review text here",
                "rating": 8,
                "scraped_at": "2024-01-01",
                "label": 1,
            },
            {
                "review_id": "r3",
                "movie_id": "tt1",
                "text": "a completely different opinion on this film",
                "rating": 2,
                "scraped_at": "2024-01-01",
                "label": 0,
            },
        ],
    )
    out = tmp_path / "clean.parquet"
    n = clean_reviews(inp, out)
    assert n == 2


def test_clean_reviews_drops_short_text(tmp_path: Path) -> None:
    inp = _make_parquet(
        tmp_path,
        [
            {
                "review_id": "r1",
                "movie_id": "tt1",
                "text": "ok",
                "rating": 5,
                "scraped_at": "2024-01-01",
                "label": 0,
            },
            {
                "review_id": "r2",
                "movie_id": "tt1",
                "text": "This is a wonderful movie with great acting and story.",
                "rating": 9,
                "scraped_at": "2024-01-01",
                "label": 1,
            },
        ],
    )
    out = tmp_path / "clean.parquet"
    n = clean_reviews(inp, out)
    assert n == 1


def test_clean_reviews_output_schema(tmp_path: Path, basic_inp: Path) -> None:
    out = tmp_path / "clean.parquet"
    clean_reviews(basic_inp, out)
    df = pd.read_parquet(out)
    assert {"review_id", "movie_id", "text", "rating", "scraped_at", "label"}.issubset(df.columns)
