"""Clean raw IMDb reviews: lowercase, strip HTML, remove URLs, drop dupes."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.lower()
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def clean_reviews(inp: Path, out: Path) -> int:
    """Read raw Parquet, clean text column, write cleaned Parquet. Returns row count."""
    df = pd.read_parquet(inp)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).map(_clean_text)
    df = df[df["text"].str.len() > 10]
    df = df.drop_duplicates(subset=["text"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return len(df)
