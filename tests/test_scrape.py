"""Tests for the IMDb scraper (both 'hf' and 'live' sources)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from moviesentiment.data.scrape import scrape_reviews

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


@pytest.fixture()
def tmp_parquet(tmp_path: Path) -> Path:
    return tmp_path / "reviews.parquet"


# ---------------------------------------------------------------------------
# live source tests
# ---------------------------------------------------------------------------


def test_empty_movie_list_writes_empty_parquet(tmp_parquet: Path) -> None:
    n = scrape_reviews(movie_ids=[], out_path=tmp_parquet, source="live")
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
        n = scrape_reviews(movie_ids=["tt0111161"], out_path=tmp_parquet, source="live")

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
        scrape_reviews(movie_ids=["tt0111161"], out_path=tmp_parquet, source="live")

    df = pd.read_parquet(tmp_parquet)
    assert df.loc[df["review_id"] == "pos", "label"].item() == 1
    assert df.loc[df["review_id"] == "neg", "label"].item() == 0


def test_live_output_schema(tmp_parquet: Path) -> None:
    edges = [_edge("rw999", 8, "Great film")]
    with (
        patch("requests.Session.get"),
        patch("requests.Session.post", return_value=_mock_post(edges)),
    ):
        scrape_reviews(movie_ids=["tt9999999"], out_path=tmp_parquet, source="live")

    df = pd.read_parquet(tmp_parquet)
    assert {"review_id", "movie_id", "text", "rating", "scraped_at", "label"}.issubset(df.columns)
    assert df["movie_id"].iloc[0] == "tt9999999"
    assert df["text"].iloc[0] == "Great film"


def test_graphql_error_raises(tmp_parquet: Path) -> None:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"errors": [{"message": "PersistedQueryNotFound"}]}
    with (
        patch("requests.Session.get"),
        patch("requests.Session.post", return_value=mock),
    ):
        with pytest.raises(RuntimeError, match="PersistedQueryNotFound"):
            scrape_reviews(movie_ids=["tt0111161"], out_path=tmp_parquet, source="live")


# ---------------------------------------------------------------------------
# hf source tests
# ---------------------------------------------------------------------------


def test_hf_source_schema(tmp_parquet: Path) -> None:
    fake_dataset = {
        "train": [{"text": "Amazing film", "label": 1}, {"text": "Boring movie", "label": 0}],
        "test": [{"text": "Pretty good", "label": 1}],
    }
    with patch("datasets.load_dataset", return_value=fake_dataset):
        n = scrape_reviews(out_path=tmp_parquet, source="hf")

    assert n == 3
    df = pd.read_parquet(tmp_parquet)
    assert {"review_id", "movie_id", "text", "rating", "scraped_at", "label"}.issubset(df.columns)
    assert df["movie_id"].unique().tolist() == ["hf_imdb"]
    assert set(df["label"].tolist()) == {0, 1}
    assert df["rating"].isna().all()


def test_hf_review_ids_are_unique(tmp_parquet: Path) -> None:
    fake_dataset = {
        "train": [{"text": f"Review {i}", "label": i % 2} for i in range(5)],
        "test": [{"text": f"Test review {i}", "label": i % 2} for i in range(3)],
    }
    with patch("datasets.load_dataset", return_value=fake_dataset):
        scrape_reviews(out_path=tmp_parquet, source="hf")

    df = pd.read_parquet(tmp_parquet)
    assert df["review_id"].nunique() == len(df)


def test_invalid_source_raises(tmp_parquet: Path) -> None:
    with pytest.raises(ValueError, match="Unknown source"):
        scrape_reviews(out_path=tmp_parquet, source="foobar")
