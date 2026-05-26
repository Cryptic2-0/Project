"""Per-movie aggregation over the reservoir-sampled production logs.

This module is called by both the `/insights/{movie_id}` route and an offline
batch (e.g. an hourly Lambda) that materialises the result to S3. The route
implementation reads either the materialised JSON or computes on the fly when
sample size is small.

The reservoir parquet (written by `ReservoirSampler`) carries one row per
sampled prediction. The v1 schema captures sentiment only; v2 will add
`aspect_*`, `emotion_top`, `spoiler_prob`, `helpfulness`, and `movie_id`. This
function tolerates either schema so the endpoint works during the v1 -> v2
migration window.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from moviesentiment.config import settings
from moviesentiment.models.multitask import ASPECTS, EMOTIONS
from moviesentiment.serve.schemas import InsightsResponse

if TYPE_CHECKING:
    import pandas as pd


_RESERVOIR_PATH = settings.data_dir / "production" / "recent.parquet"


def _load_reservoir() -> pd.DataFrame | None:
    if not _RESERVOIR_PATH.exists():
        return None
    import pandas as pd

    return pd.read_parquet(_RESERVOIR_PATH)


def compute_insights(movie_id: str) -> InsightsResponse | None:
    """Return aggregated insights for `movie_id`, or None if no data yet."""
    df = _load_reservoir()
    if df is None or df.empty:
        return None

    if "movie_id" in df.columns:
        sub = df[df["movie_id"] == movie_id]
    else:
        # v1 reservoir doesn't carry movie_id; treat the whole sample as the
        # movie under analysis. Useful during the migration window.
        sub = df

    n = int(len(sub))
    if n == 0:
        return None

    if "label" in sub.columns:
        positive_share = float((sub["label"] == "positive").mean())
    else:
        positive_share = 0.0

    aspect_means: dict[str, float] = {}
    for aspect in ASPECTS:
        col = f"aspect_{aspect}"
        if col in sub.columns:
            aspect_means[aspect] = float(sub[col].mean())
        else:
            aspect_means[aspect] = 0.0

    emotion_mix: dict[str, float] = {}
    if "emotion_top" in sub.columns:
        counts = sub["emotion_top"].value_counts(normalize=True)
        emotion_mix = {e: float(counts.get(e, 0.0)) for e in EMOTIONS}
    else:
        emotion_mix = {e: 0.0 for e in EMOTIONS}

    spoiler_share = float(sub["spoiler_prob"].mean()) if "spoiler_prob" in sub.columns else 0.0
    helpfulness_mean = float(sub["helpfulness"].mean()) if "helpfulness" in sub.columns else 0.0

    topics_path = settings.data_dir / "production" / "topics" / f"{movie_id}.json"
    topics: list[str] = []
    if topics_path.exists():
        import json

        topics = list(json.loads(topics_path.read_text()).get("topics", []))[:5]

    return InsightsResponse(
        movie_id=movie_id,
        n_reviews=n,
        sentiment_positive_share=positive_share,
        aspect_means=aspect_means,
        emotion_mix=emotion_mix,
        spoiler_share=spoiler_share,
        helpfulness_mean=helpfulness_mean,
        topics=topics,
    )


def materialise_all(out_dir: Path | None = None) -> int:
    """Offline batch: dump insights for every movie_id in the reservoir.

    Returns the number of movies materialised. Wired up by the hourly Lambda
    (deploy/lambda/insights_aggregator.py) or via the `moviesentiment insights`
    CLI. Returns 0 if the reservoir is missing.
    """
    df = _load_reservoir()
    if df is None or df.empty or "movie_id" not in df.columns:
        return 0
    out = out_dir or settings.data_dir / "production" / "insights"
    out.mkdir(parents=True, exist_ok=True)

    import json

    n = 0
    for movie_id, _ in df.groupby("movie_id"):
        ins = compute_insights(str(movie_id))
        if ins is None:
            continue
        (out / f"{movie_id}.json").write_text(json.dumps(ins.model_dump(), indent=2))
        n += 1
    return n
