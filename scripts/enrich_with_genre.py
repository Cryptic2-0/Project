"""Join IMDb genre metadata onto the cleaned reviews parquet.

Outputs a new parquet with one extra column, `genre`, where each value is the
first listed genre from the source CSV. Movies missing in the source CSV are
labelled "unknown" rather than dropped — the slice analysis in the model card
should call out the unknown bucket explicitly so the gap is visible.

Why this exists: the model card's "Bias / fairness" section reports per-genre
F1, but the test parquet only carries `movie_id`. Run this once to enrich,
then re-eval.

Source CSV format (e.g. IMDb non-commercial dump `title.basics.tsv.gz` -> csv):

    movie_id,title,genres
    tt0111161,The Shawshank Redemption,Drama
    tt0068646,The Godfather,Crime|Drama

The script picks the first genre after splitting on `|`.

Usage:

    python scripts/enrich_with_genre.py \
        --reviews data/processed/test.parquet \
        --genres  data/external/genres.csv \
        --out     data/processed/test_with_genre.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def enrich(reviews: Path, genres_csv: Path, out: Path) -> int:
    import pandas as pd

    reviews_df = pd.read_parquet(reviews)
    if "movie_id" not in reviews_df.columns:
        print(
            f"ERROR: {reviews} missing 'movie_id' column; cannot join",
            file=sys.stderr,
        )
        return 1

    genres_df = pd.read_csv(genres_csv)
    expected = {"movie_id", "genres"}
    missing = expected - set(genres_df.columns)
    if missing:
        print(
            f"ERROR: {genres_csv} missing columns {sorted(missing)}",
            file=sys.stderr,
        )
        return 1

    genres_df["genre"] = (
        genres_df["genres"].fillna("").astype(str).str.split("|").str[0].replace("", "unknown")
    )
    merged = reviews_df.merge(genres_df[["movie_id", "genre"]], on="movie_id", how="left").fillna(
        {"genre": "unknown"}
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)

    bucket_counts = merged["genre"].value_counts().to_dict()
    print(f"Enriched {len(merged)} reviews -> {out}")
    print("Per-genre counts:")
    for genre, n in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {genre:20s} {n}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--genres", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return enrich(args.reviews, args.genres, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
