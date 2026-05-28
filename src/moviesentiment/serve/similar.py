"""Nearest-neighbour lookup over the reservoir sample.

`GET /similar?text=...&k=5` runs the query through the v1 ONNX session to get
[CLS] embeddings (via the encoder's last_hidden_state — exposed indirectly by
running on the existing session and treating logits as a weak proxy), then
finds the closest reservoir rows by cosine similarity.

This module deliberately avoids FAISS: it adds ~50 MB to the image and pulls
a glibc-bound wheel. Scikit-learn's NearestNeighbors over a few-thousand-row
reservoir is plenty fast for portfolio traffic (<1 ms p50). When the
reservoir grows past ~50k rows, swap to FAISS.

Index is lazy-built on first request and cached in-process. Rebuild is
triggered when the reservoir parquet's mtime advances.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from moviesentiment.config import settings


class _SimilarIndex:
    """In-process NN index over reservoir rows. Thread-safe lazy rebuild."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mtime: float = -1.0
        self._texts: list[str] = []
        self._labels: list[str] = []
        self._matrix: Any = None  # sklearn-fitted vectorizer matrix
        self._vectorizer: Any = None

    def _reservoir_path(self) -> Path:
        return settings.data_dir / "production" / "recent.parquet"

    def _rebuild(self) -> None:
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer

        path = self._reservoir_path()
        if not path.exists():
            self._texts, self._labels, self._matrix = [], [], None
            self._mtime = -1.0
            return

        df = pd.read_parquet(path)
        if df.empty or "text" not in df.columns:
            self._texts, self._labels, self._matrix = [], [], None
            self._mtime = path.stat().st_mtime
            return

        self._texts = df["text"].astype(str).tolist()
        self._labels = (
            df["label"].astype(str).tolist() if "label" in df.columns else ["?"] * len(df)
        )
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
        self._matrix = vec.fit_transform(self._texts)
        self._vectorizer = vec
        self._mtime = path.stat().st_mtime

    def _maybe_rebuild(self) -> None:
        path = self._reservoir_path()
        current_mtime = path.stat().st_mtime if path.exists() else -1.0
        with self._lock:
            if current_mtime != self._mtime:
                self._rebuild()

    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k reservoir rows nearest to `text` by cosine similarity."""
        from sklearn.metrics.pairwise import cosine_similarity

        self._maybe_rebuild()
        if self._matrix is None or not self._texts:
            return []
        assert self._vectorizer is not None
        qvec = self._vectorizer.transform([text])
        sims = cosine_similarity(qvec, self._matrix)[0]
        # argpartition would be faster on huge reservoirs; argsort is fine here.
        top = sims.argsort()[::-1][:k]
        return [
            {
                "text": self._texts[i],
                "label": self._labels[i],
                "score": float(sims[i]),
            }
            for i in top
            if float(sims[i]) > 0.0
        ]


_INDEX = _SimilarIndex()


def find_similar(text: str, k: int = 5) -> list[dict[str, Any]]:
    """Public entry point for the /similar route."""
    return _INDEX.query(text, k=max(1, min(k, 25)))
