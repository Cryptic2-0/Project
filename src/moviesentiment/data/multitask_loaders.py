"""Data loaders for the v2 multi-task heads.

Each loader downloads or shapes a public dataset into the canonical
two-column parquet schema (`text`, `label`) expected by
`moviesentiment.models.multitask_train`. Run them once locally (or inside the
SageMaker job) before invoking `moviesentiment train multitask`.

Sources:
    spoiler      : Kaggle "IMDB Spoiler Dataset" (rmisra / 573k) -- requires the
                   user to download the CSV and pass its path. Reuses the
                   Kaggle CLI when present.
    emotion      : HuggingFace `go_emotions` -- mapped to Ekman six by taking
                   the union of the GoEmotions->Ekman mapping shipped here.
    absa         : Synthetic distillation from a teacher (`yangheng/deberta-v3-
                   base-absa-v1.1`) over the existing cleaned IMDb corpus.
                   Runs offline; teacher download is gated by
                   MS_HF_REVISION just like the main model.
    helpfulness  : Live IMDb scraper already captures `rating` and could be
                   extended to vote counts. For now uses the rating as a proxy:
                   `label = clip((rating - 1) / 9, 0, 1)`.

None of these loaders ship pre-baked data with the repo -- they are
deterministic scripts that produce parquet under `data/interim/multitask/`.
"""

from __future__ import annotations

from pathlib import Path

_OUT_DIR = Path("data/interim/multitask")

# GoEmotions -> Ekman 6 (joy, anger, fear, sadness, surprise, disgust).
_GO_TO_EKMAN: dict[str, str] = {
    "admiration": "joy",
    "amusement": "joy",
    "approval": "joy",
    "caring": "joy",
    "desire": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "joy": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "fear": "fear",
    "nervousness": "fear",
    "sadness": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    "disgust": "disgust",
    "embarrassment": "disgust",
}
_EKMAN = ("joy", "anger", "fear", "sadness", "surprise", "disgust")
_EKMAN_IDX = {name: i for i, name in enumerate(_EKMAN)}


def load_emotion(out: Path | None = None) -> Path:
    """Materialise the GoEmotions split mapped to Ekman six."""
    import pandas as pd
    from datasets import load_dataset

    from moviesentiment.config import settings

    out_path = out or _OUT_DIR / "emotion.parquet"
    rev = settings.hf_revision or None
    ds = load_dataset("go_emotions", "simplified", split="train", revision=rev)
    label_names = ds.features["labels"].feature.names

    rows: list[dict[str, object]] = []
    for ex in ds:
        labels = ex["labels"]
        if not labels:
            continue
        primary = label_names[labels[0]]
        ekman = _GO_TO_EKMAN.get(primary)
        if ekman is None:
            continue
        rows.append({"text": ex["text"], "label": _EKMAN_IDX[ekman]})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path


def load_spoiler(csv_path: Path, out: Path | None = None) -> Path:
    """Reshape Kaggle IMDB Spoiler Dataset CSV into (text, label)."""
    import pandas as pd

    out_path = out or _OUT_DIR / "spoiler.parquet"
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("review_text") or cols.get("text")
    spoiler_col = cols.get("is_spoiler") or cols.get("spoiler")
    if text_col is None or spoiler_col is None:
        raise ValueError(f"Spoiler CSV missing text/spoiler columns; got {list(df.columns)}")
    out_df = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": df[spoiler_col].astype(int),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return out_path


def load_helpfulness(reviews_parquet: Path, out: Path | None = None) -> Path:
    """Map the existing scraped IMDb reviews into a helpfulness proxy parquet.

    Uses rating as a proxy: `label = clip((rating - 1) / 9, 0, 1)`. Real
    helpfulness (upvote ratio) lands later when the live scraper persists vote
    counts. Rows with no rating are dropped.
    """
    import pandas as pd

    out_path = out or _OUT_DIR / "helpfulness.parquet"
    df = pd.read_parquet(reviews_parquet)
    df = df.dropna(subset=["text", "rating"])
    df["label"] = ((df["rating"].astype(float) - 1.0) / 9.0).clip(0.0, 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label"]].to_parquet(out_path, index=False)
    return out_path


def distill_absa(
    text_parquet: Path,
    out: Path | None = None,
    teacher_model: str = "yangheng/deberta-v3-base-absa-v1.1",
    aspects: tuple[str, ...] = ("acting", "plot", "visuals", "pacing", "sound"),
    batch_size: int = 16,
) -> Path:
    """Distill an ABSA teacher into (text, label=[5 ints]) parquet.

    Each label is a length-5 list in {-100, 0, 1, 2}; -100 means the aspect was
    not mentioned (loss ignored) and {0, 1, 2} maps to {neg, neu, pos}.
    Runs on whatever device torch finds; expect ~30 min for 50K rows on T4.
    """
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from moviesentiment.config import settings

    out_path = out or _OUT_DIR / "absa.parquet"
    df = pd.read_parquet(text_parquet)[["text"]].dropna()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rev = settings.hf_revision or None
    tok = AutoTokenizer.from_pretrained(teacher_model, revision=rev)
    model = (
        AutoModelForSequenceClassification.from_pretrained(teacher_model, revision=rev)
        .to(device)
        .eval()
    )

    @torch.no_grad()
    def _score(texts: list[str], aspect: str) -> list[int]:
        pairs = [(t, aspect) for t in texts]
        enc = tok(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        return logits.argmax(dim=-1).cpu().tolist()

    rows: list[dict[str, object]] = []
    texts = df["text"].astype(str).tolist()
    aspect_preds: dict[str, list[int]] = {a: [] for a in aspects}
    for aspect in aspects:
        for i in range(0, len(texts), batch_size):
            aspect_preds[aspect].extend(_score(texts[i : i + batch_size], aspect))

    for i, t in enumerate(texts):
        labels = [aspect_preds[a][i] for a in aspects]
        rows.append({"text": t, "label": labels})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path
