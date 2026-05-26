"""Hourly Lambda that materialises per-movie insights to S3.

Trigger: EventBridge schedule `rate(1 hour)`.

Pipeline:
    1. Download the production reservoir parquet from
       s3://<MS_DVC_BUCKET>/production/recent.parquet to /tmp.
    2. Group by `movie_id`, compute aggregates (same code path as the live
       `/insights/{movie_id}` route).
    3. Upload each result JSON to
       s3://<INSIGHTS_BUCKET>/insights/<movie_id>.json.
    4. (Optional) Run BERTopic if the parquet has >= MIN_DOCS rows per movie
       and write topics back under
       s3://<INSIGHTS_BUCKET>/topics/<movie_id>.json.

Cost (portfolio traffic): well within Lambda free tier — one invocation per
hour, < 5 s wall clock, < 256 MB memory.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import boto3

DVC_BUCKET = os.environ["MS_DVC_BUCKET"]
DVC_KEY = os.environ.get("MS_RESERVOIR_KEY", "production/recent.parquet")
INSIGHTS_BUCKET = os.environ.get("INSIGHTS_BUCKET", DVC_BUCKET)
INSIGHTS_PREFIX = os.environ.get("INSIGHTS_PREFIX", "insights")
TOPICS_PREFIX = os.environ.get("TOPICS_PREFIX", "topics")
MIN_DOCS_FOR_TOPICS = int(os.environ.get("MIN_DOCS_FOR_TOPICS", "20"))


def _download(s3: Any, local: Path) -> bool:
    try:
        s3.download_file(DVC_BUCKET, DVC_KEY, str(local))
        return True
    except s3.exceptions.ClientError:
        return False


def _aggregate(df: Any, movie_id: str) -> dict[str, Any]:
    sub = df[df["movie_id"] == movie_id] if "movie_id" in df.columns else df
    n = int(len(sub))
    pos_share = float((sub["label"] == "positive").mean()) if "label" in sub.columns else 0.0
    aspect_means = {
        col.removeprefix("aspect_"): float(sub[col].mean())
        for col in sub.columns
        if col.startswith("aspect_")
    }
    emotion_mix: dict[str, float] = {}
    if "emotion_top" in sub.columns:
        counts = sub["emotion_top"].value_counts(normalize=True).to_dict()
        emotion_mix = {str(k): float(v) for k, v in counts.items()}
    return {
        "movie_id": movie_id,
        "n_reviews": n,
        "sentiment_positive_share": pos_share,
        "aspect_means": aspect_means,
        "emotion_mix": emotion_mix,
        "spoiler_share": (
            float(sub.get("spoiler_prob", []).mean()) if "spoiler_prob" in sub.columns else 0.0
        ),
        "helpfulness_mean": (
            float(sub.get("helpfulness", []).mean()) if "helpfulness" in sub.columns else 0.0
        ),
    }


def _topics(sub: Any, top_k: int = 5) -> list[str]:
    """Lightweight topic extraction. BERTopic is heavy for a Lambda layer, so
    use scikit-learn TF-IDF + NMF as a portfolio-budget substitute."""
    import numpy as np
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = sub["text"].astype(str).tolist()
    if len(docs) < MIN_DOCS_FOR_TOPICS:
        return []
    vec = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(docs)
    n_topics = min(top_k, matrix.shape[0])
    nmf = NMF(n_components=n_topics, init="nndsvda", max_iter=200, random_state=42)
    nmf.fit(matrix)
    feature_names = np.array(vec.get_feature_names_out())
    topics: list[str] = []
    for topic_vec in nmf.components_:
        top_idx = topic_vec.argsort()[::-1][:3]
        topics.append(" / ".join(feature_names[top_idx]))
    return topics


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    import pandas as pd

    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "recent.parquet"
        if not _download(s3, local):
            return {"status": "no-reservoir", "processed": 0}
        df = pd.read_parquet(local)
        if df.empty:
            return {"status": "empty", "processed": 0}
        if "movie_id" not in df.columns:
            df["movie_id"] = "all"

        processed = 0
        for movie_id, sub in df.groupby("movie_id"):
            agg = _aggregate(df, str(movie_id))
            s3.put_object(
                Bucket=INSIGHTS_BUCKET,
                Key=f"{INSIGHTS_PREFIX}/{movie_id}.json",
                Body=json.dumps(agg).encode("utf-8"),
                ContentType="application/json",
            )
            topics = _topics(sub)
            if topics:
                s3.put_object(
                    Bucket=INSIGHTS_BUCKET,
                    Key=f"{TOPICS_PREFIX}/{movie_id}.json",
                    Body=json.dumps({"topics": topics}).encode("utf-8"),
                    ContentType="application/json",
                )
            processed += 1
    return {"status": "ok", "processed": processed}
