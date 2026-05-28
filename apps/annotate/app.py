"""Streamlit active-learning labeller for low-confidence production predictions.

Workflow:

  1. Load the reservoir parquet (`data/production/recent.parquet`).
  2. Filter to rows where the model's confidence is below a threshold
     (default 0.70 — the model card flags this band as essentially
     coin-flip).
  3. Show one row at a time; user clicks "positive" / "negative" / "skip".
  4. Labels append to `data/labeled/augmentation.parquet` for the next
     training run to pick up.

Run locally with:

    streamlit run apps/annotate/app.py

The app is local-only by design — no network deps beyond reading + writing
the parquets — so it does not need to ship with the Fargate image.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RESERVOIR = _REPO_ROOT / "data" / "production" / "recent.parquet"
_DEFAULT_OUT = _REPO_ROOT / "data" / "labeled" / "augmentation.parquet"


def _load_low_confidence(path: Path, threshold: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty or "confidence" not in df.columns:
        return pd.DataFrame()
    return df[df["confidence"] < threshold].reset_index(drop=True)


def _load_labeled(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["text", "label", "labeled_at", "labeller"])
    return pd.read_parquet(path)


def _append_label(out: Path, row: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_labeled(out)
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_parquet(out, index=False)


def main() -> None:
    st.set_page_config(page_title="MovieSentiment Labeller", layout="centered")
    st.title("Active-Learning Labeller")
    st.caption(
        "Low-confidence rows from the production reservoir. Each label you "
        "append goes to `data/labeled/augmentation.parquet` and is consumed by "
        "the next training run."
    )

    reservoir_path = Path(st.sidebar.text_input("Reservoir path", str(_DEFAULT_RESERVOIR)))
    out_path = Path(st.sidebar.text_input("Output parquet", str(_DEFAULT_OUT)))
    threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.70, 0.05)
    labeller = st.sidebar.text_input("Your name", "anonymous")

    candidates = _load_low_confidence(reservoir_path, threshold)
    labeled = _load_labeled(out_path)
    already_labeled = set(labeled["text"]) if "text" in labeled.columns else set()
    pending = (
        candidates[~candidates["text"].isin(already_labeled)]
        if not candidates.empty
        else (candidates)
    )

    st.sidebar.metric("Pending", len(pending))
    st.sidebar.metric("Already labeled", len(labeled))

    if pending.empty:
        st.info("No pending low-confidence rows. Reduce the threshold or wait for traffic.")
        return

    if "cursor" not in st.session_state:
        st.session_state.cursor = 0

    idx = st.session_state.cursor % len(pending)
    row = pending.iloc[idx]
    st.subheader(f"Row {idx + 1} of {len(pending)}")
    st.write(row["text"])
    st.caption(
        f"Model said **{row.get('label', '?')}** with confidence "
        f"**{float(row.get('confidence', 0.0)):.2f}**"
    )

    col_pos, col_neg, col_skip = st.columns(3)

    def _record(label: str) -> None:
        _append_label(
            out_path,
            {
                "text": row["text"],
                "label": label,
                "labeled_at": datetime.now(timezone.utc).isoformat(),
                "labeller": labeller,
            },
        )
        st.session_state.cursor += 1
        st.rerun()

    if col_pos.button("positive"):
        _record("positive")
    if col_neg.button("negative"):
        _record("negative")
    if col_skip.button("skip"):
        st.session_state.cursor += 1
        st.rerun()


if __name__ == "__main__":
    main()
