"""Tests for the IMDb scraper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from moviesentiment.data.scrape import scrape_reviews


def _edge(review_id: str, rating: int | None, text: str) -> dict[str, object]:
    return {
        "node": {
            "id": review_id,
            "rating": rating,
            "text": {"originalText": {"plaidHtml": text}},
        }
    }


def _mock_post(edges: list[dict[str, object]], has_next: bool = False) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "data": {
            "title": {
                "reviews": {
                    "edges": edges,
                    "pageInfo": {
                        "hasNextPage": has_next,
                        "endCursor": "cursor123" if has_next else None,
                    },
                }
            }
        }
    }
    return mock


@pytest.fixture()  # type: ignore[misc]
def tmp_parquet(tmp_path: Path) -> Path:
    return tmp_path / "reviews.parquet"


def test_empty_movie_list_writes_empty_parquet(tmp_parquet: Path) -> None:
    n = scrape_reviews(movie_ids=[], out_path=tmp_parquet)
    assert n == 0
    df = pd.read_parquet(tmp_parquet)
    assert len(df) == 0


def test_neutral_ratings_are_dropped(tmp_parquet: Path) -> None:
    edges = [
        _edge("rw001", 9, "Loved it"),
        _edge("rw002", 3, "Terrible"),
        _edge("rw003", 5, "Meh"),
        _edge("rw004", 6, "OK"),
        _edge("rw005", None, "No rating"),
    ]
    with (
        patch("requests.Session.get"),
        patch("requests.Session.post", return_value=_mock_post(edges)),
    ):
        n = scrape_reviews(movie_ids=["tt0111161"], out_path=tmp_parquet)

    assert n == 2
    df = pd.read_parquet(tmp_parquet)
    assert set(df["review_id"].tolist()) == {"rw001", "rw002"}


def test_label_assignment(tmp_parquet: Path) -> None:
    edges = [
        _edge("pos", 8, "Great"),
        _edge("neg", 2, "Awful"),
    ]
    with (
        patch("requests.Session.get"),
        patch("requests.Session.post", return_value=_mock_post(edges)),
    ):
        scrape_reviews(movie_ids=["tt0111161"], out_path=tmp_parquet)

    df = pd.read_parquet(tmp_parquet)
    assert df.loc[df["review_id"] == "pos", "label"].item() == 1
    assert df.loc[df["review_id"] == "neg", "label"].item() == 0


def test_output_schema(tmp_parquet: Path) -> None:
    edges = [_edge("rw999", 8, "Great film")]
    with (
        patch("requests.Session.get"),
        patch("requests.Session.post", return_value=_mock_post(edges)),
    ):
        scrape_reviews(movie_ids=["tt9999999"], out_path=tmp_parquet)

    df = pd.read_parquet(tmp_parquet)
    assert {"review_id", "movie_id", "text", "rating", "scraped_at", "label"}.issubset(df.columns)
    assert df["movie_id"].iloc[0] == "tt9999999"
    assert df["text"].iloc[0] == "Great film"
