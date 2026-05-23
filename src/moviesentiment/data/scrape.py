"""IMDb review scraper — adapted from Imdb_review_scrapper-2026-."""

from __future__ import annotations

from pathlib import Path


def scrape_reviews(
    movie_ids: list[str] | None = None,
    out_path: Path = Path("data/raw/reviews.parquet"),
) -> int:
    """Scrape IMDb reviews for the given movie IDs and write to Parquet.

    Schema: review_id, movie_id, text, rating, scraped_at
    Label rule: rating >= 7 → positive, rating <= 4 → negative, drop neutrals.

    Returns number of reviews written.
    """
    raise NotImplementedError("Port scraper logic from Imdb_review_scrapper-2026-")
