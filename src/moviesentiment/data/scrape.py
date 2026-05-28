"""IMDb review scraper.

Two data sources:
  hf   — HuggingFace stanfordnlp/imdb (50 K reviews, default, reproducible).
  live — Live GraphQL scrape from IMDb using movie_ids from params.yaml.
         Note: IMDb uses AWS WAF protection that blocks automated requests;
         a browser-automation layer (e.g. Playwright) is required for live use.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

_LOG = logging.getLogger(__name__)

_GRAPHQL_URL = "https://caching.graphql.imdb.com/"
# Persisted-query SHA for the TitleReviewsRefine operation. IMDb rotates this
# on every imdb-web-next release; capture a fresh value from the browser
# Network tab + paste here (or set MS_IMDB_PERSISTED_HASH at runtime).
_PERSISTED_HASH = "286aee4ac14648e42c02c576e0cd29c33e9113f022290145cb1872968b389505"
_BASE_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)
_PAGE_SIZE = 25
_SLEEP_S = 0.8


# ---------------------------------------------------------------------------
# HuggingFace source
# ---------------------------------------------------------------------------


def _load_hf(out_path: Path) -> int:
    """Download stanfordnlp/imdb and write Parquet with canonical schema."""
    from datasets import load_dataset

    # Pin dataset revision via MS_HF_REVISION to defend against namespace takeover.
    # Empty default = main; CI / training jobs should set this to a concrete commit.
    from moviesentiment.config import settings as _settings

    rev = _settings.hf_revision or None
    _LOG.info("Downloading stanfordnlp/imdb from HuggingFace (revision=%s) …", rev or "main")
    dataset: Any = load_dataset("stanfordnlp/imdb", revision=rev)
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
    cookie = os.environ.get("MS_IMDB_COOKIE", "")
    if cookie:
        session.headers["Cookie"] = cookie
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
    headers = {
        "User-Agent": _BASE_UA,
        "Accept": "application/graphql+json, application/json",
        "Content-Type": "application/json",
        "Origin": "https://www.imdb.com",
        "Referer": "https://www.imdb.com/",
        "x-imdb-client-name": "imdb-web-next-localized",
        "x-imdb-user-country": "US",
        "x-imdb-user-language": "en-US",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
    }
    cookie = os.environ.get("MS_IMDB_COOKIE", "")
    if cookie:
        headers["Cookie"] = cookie
    return headers


def _build_query_params(movie_id: str, cursor: str | None) -> dict[str, str]:
    """Build URL query-string params for the GraphQL GET call.

    IMDb's imdb-web-next-localized client sends operationName + JSON-encoded
    `variables` + JSON-encoded `extensions.persistedQuery` as URL parameters.
    """
    variables = {
        "after": cursor or "",
        "const": movie_id,
        "filter": {},
        "first": _PAGE_SIZE,
        "locale": "en-US",
        "sort": {"by": "HELPFULNESS_SCORE", "order": "DESC"},
    }
    if not cursor:
        variables.pop("after")
    sha = os.environ.get("MS_IMDB_PERSISTED_HASH", _PERSISTED_HASH)
    extensions = {"persistedQuery": {"sha256Hash": sha, "version": 1}}
    return {
        "operationName": "TitleReviewsRefine",
        "variables": json.dumps(variables, separators=(",", ":")),
        "extensions": json.dumps(extensions, separators=(",", ":")),
    }


def _fetch_page(
    session: requests.Session,
    movie_id: str,
    cursor: str | None,
) -> tuple[list[Any], str | None]:
    url = f"{_GRAPHQL_URL}?{urlencode(_build_query_params(movie_id, cursor))}"
    resp = session.get(
        url,
        headers=_graphql_headers(movie_id),
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
    # IMDb renamed `rating` -> `authorRating` in the imdb-web-next-localized
    # client. Accept either for forward / backward compatibility.
    rating_raw = node.get("authorRating", node.get("rating"))
    if rating_raw is None:
        return None
    rating: int = int(rating_raw)
    if 5 <= rating <= 6:
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
