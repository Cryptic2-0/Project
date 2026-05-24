"""IMDb review scraper.

Two data sources:
  hf   — HuggingFace stanfordnlp/imdb (50 K reviews, default, reproducible).
  live — Live GraphQL scrape from IMDb using movie_ids from params.yaml.
         Note: IMDb uses AWS WAF protection that blocks automated requests;
         a browser-automation layer (e.g. Playwright) is required for live use.
"""

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
_BASE_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_PAGE_SIZE = 25
_SLEEP_S = 0.8


# ---------------------------------------------------------------------------
# HuggingFace source
# ---------------------------------------------------------------------------


def _load_hf(out_path: Path) -> int:
    """Download stanfordnlp/imdb and write Parquet with canonical schema."""
    from datasets import load_dataset

    _LOG.info("Downloading stanfordnlp/imdb from HuggingFace …")
    dataset: Any = load_dataset("stanfordnlp/imdb")
    scraped_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []

    for split_name in ("train", "test"):
        split: Any = dataset[split_name]
        for i, item in enumerate(split):
            records.append(
                {
                    "review_id": f"hf_{split_name}_{i}",
                    "movie_id": "hf_imdb",
                    "text": str(item["text"]),
                    "rating": None,
                    "scraped_at": scraped_at,
                    "label": int(item["label"]),
                }
            )
        _LOG.info("  %s: %d reviews", split_name, len(split))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out_path, index=False)
    _LOG.info("Wrote %d reviews to %s", len(records), out_path)
    return len(records)


# ---------------------------------------------------------------------------
# Live IMDb GraphQL source
# ---------------------------------------------------------------------------


def _prime_session(movie_id: str) -> requests.Session:
    session = requests.Session()
    session.get(
        f"https://www.imdb.com/title/{movie_id}/reviews/",
        headers={
            "User-Agent": _BASE_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
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
    data: dict[str, Any] = resp.json()
    if "errors" in data:
        msgs = [str(e.get("message", "")) for e in data.get("errors", [])]
        raise RuntimeError(f"IMDb GraphQL errors for {movie_id}: {msgs}")
    block: dict[str, Any] = data["data"]["title"]["reviews"]
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


def _scrape_live(
    movie_ids: list[str],
    out_path: Path,
    max_pages: int | None,
) -> int:
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
    pd.DataFrame(records).to_parquet(out_path, index=False)
    _LOG.info("Wrote %d reviews to %s", len(records), out_path)
    return len(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scrape_reviews(
    movie_ids: list[str] | None = None,
    out_path: Path = Path("data/raw/reviews.parquet"),
    max_pages: int | None = None,
    source: str = "hf",
) -> int:
    """Fetch IMDb reviews and write to Parquet.

    source='hf'   — HuggingFace stanfordnlp/imdb (50 K reviews, default, reproducible).
    source='live' — Live GraphQL scrape using movie_ids. Requires WAF bypass.

    Output schema: review_id, movie_id, text, rating, scraped_at, label
    Returns number of reviews written.
    """
    if source == "hf":
        return _load_hf(out_path)
    if source == "live":
        return _scrape_live(movie_ids or [], out_path, max_pages)
    raise ValueError(f"Unknown source {source!r}. Use 'hf' or 'live'.")
