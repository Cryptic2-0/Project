"""IMDb review scraper — adapted from Imdb_review_scrapper-2026-."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

_LOG = logging.getLogger(__name__)

_GRAPHQL_URL = "https://caching.graphql.imdb.com/"
_PERSISTED_HASH = "d389bc70c27f09c00b663705f0112254e8a7c75cde1cfd30e63a2d98c1080c87"
_BASE_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
_PAGE_SIZE = 25
_SLEEP_S = 0.8


def _prime_session(movie_id: str) -> requests.Session:
    session = requests.Session()
    session.get(
        f"https://www.imdb.com/title/{movie_id}/reviews/",
        headers={"User-Agent": _BASE_UA, "Accept-Language": "en-US,en;q=0.9"},
        timeout=30,
    )
    return session


def _graphql_headers(movie_id: str) -> dict[str, str]:
    return {
        "User-Agent": _BASE_UA,
        "Accept": "application/graphql+json, application/json",
        "Content-Type": "application/json",
        "Origin": "https://www.imdb.com",
        "Referer": f"https://www.imdb.com/title/{movie_id}/reviews/",
        "x-imdb-client-name": "imdb-web-next",
        "x-imdb-client-version": "1.0.0",
    }


def _build_payload(movie_id: str, cursor: str | None) -> dict[str, Any]:
    return {
        "operationName": "TitleReviewsRefine",
        "variables": {
            "after": cursor,
            "const": movie_id,
            "filter": {},
            "first": _PAGE_SIZE,
            "locale": "en-US",
            "sort": {"by": "HELPFULNESS_SCORE", "order": "DESC"},
        },
        "extensions": {"persistedQuery": {"sha256Hash": _PERSISTED_HASH, "version": 1}},
    }


def _fetch_page(
    session: requests.Session,
    movie_id: str,
    cursor: str | None,
) -> tuple[list[Any], str | None]:
    resp = session.post(
        _GRAPHQL_URL,
        headers=_graphql_headers(movie_id),
        json=_build_payload(movie_id, cursor),
        timeout=30,
    )
    resp.raise_for_status()
    block: dict[str, Any] = resp.json()["data"]["title"]["reviews"]
    next_cursor: str | None = (
        block["pageInfo"]["endCursor"] if block["pageInfo"]["hasNextPage"] else None
    )
    return block["edges"], next_cursor


def _parse_edge(edge: dict[str, Any], movie_id: str, scraped_at: str) -> dict[str, Any] | None:
    node: dict[str, Any] = edge["node"]
    rating: int | None = node.get("rating")
    if rating is None or 5 <= rating <= 6:
        return None
    plaid: Any = (node.get("text") or {}).get("originalText") or {}
    text = str(plaid.get("plaidHtml", "")) if isinstance(plaid, dict) else ""
    return {
        "review_id": node.get("id", ""),
        "movie_id": movie_id,
        "text": text,
        "rating": rating,
        "scraped_at": scraped_at,
        "label": 1 if rating >= 7 else 0,
    }


def scrape_reviews(
    movie_ids: list[str] | None = None,
    out_path: Path = Path("data/raw/reviews.parquet"),
    max_pages: int | None = None,
) -> int:
    """Scrape IMDb reviews for the given movie IDs and write to Parquet.

    Output schema: review_id, movie_id, text, rating, scraped_at, label
    Label: rating >= 7 → 1 (positive), rating <= 4 → 0 (negative). Neutrals dropped.
    Returns number of reviews written.
    """
    if movie_ids is None:
        movie_ids = []

    scraped_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []

    for movie_id in movie_ids:
        _LOG.info("Scraping %s", movie_id)
        session = _prime_session(movie_id)
        cursor: str | None = None
        page = 1

        while True:
            edges, cursor = _fetch_page(session, movie_id, cursor)
            _LOG.info("  page %d: %d edges", page, len(edges))
            for edge in edges:
                row = _parse_edge(edge, movie_id, scraped_at)
                if row is not None:
                    records.append(row)
            if cursor is None or (max_pages is not None and page >= max_pages):
                break
            page += 1
            time.sleep(_SLEEP_S)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    _LOG.info("Wrote %d reviews to %s", len(records), out_path)
    return len(records)
