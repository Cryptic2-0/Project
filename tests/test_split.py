"""Tests for stratified train/val/test split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from moviesentiment.data.split import split_dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clean_parquet(tmp_path: Path, n: int = 200) -> Path:
    """Create a balanced Parquet with n rows (n/2 pos, n/2 neg)."""
    half = n // 2
    df = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(n)],
            "movie_id": ["tt1"] * n,
            "text": [f"review text number {i}" for i in range(n)],
            "rating": [8] * half + [2] * half,
            "scraped_at": ["2024-01-01"] * n,
            "label": [1] * half + [0] * half,
        }
    )
    p = tmp_path / "clean.parquet"
    df.to_parquet(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_split_counts_sum_to_total(tmp_path: Path) -> None:
    inp = _make_clean_parquet(tmp_path)
    counts = split_dataset(inp, tmp_path / "processed")
    assert counts["train"] + counts["val"] + counts["test"] == 200


def test_split_output_files_exist(tmp_path: Path) -> None:
    inp = _make_clean_parquet(tmp_path)
    out_dir = tmp_path / "processed"
    split_dataset(inp, out_dir)
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "val.parquet").exists()
    assert (out_dir / "test.parquet").exists()


def test_split_no_row_leakage(tmp_path: Path) -> None:
    """Every review_id appears in exactly one split."""
    inp = _make_clean_parquet(tmp_path)
    out_dir = tmp_path / "processed"
    split_dataset(inp, out_dir)

    train_ids = set(pd.read_parquet(out_dir / "train.parquet")["review_id"])
    val_ids = set(pd.read_parquet(out_dir / "val.parquet")["review_id"])
    test_ids = set(pd.read_parquet(out_dir / "test.parquet")["review_id"])

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0
    assert len(train_ids | val_ids | test_ids) == 200


def test_split_approximate_sizes(tmp_path: Path) -> None:
    """Test split is ~15% and val split is ~15% of total (params.yaml defaults)."""
    inp = _make_clean_parquet(tmp_path, n=200)
    counts = split_dataset(inp, tmp_path / "processed")
    total = 200
    assert abs(counts["test"] / total - 0.15) < 0.03
    assert abs(counts["val"] / total - 0.15) < 0.03


def test_split_label_balance_preserved(tmp_path: Path) -> None:
    """Stratification keeps each split balanced (50/50 labels)."""
    inp = _make_clean_parquet(tmp_path, n=200)
    out_dir = tmp_path / "processed"
    split_dataset(inp, out_dir)

    for fname in ("train.parquet", "val.parquet", "test.parquet"):
        df = pd.read_parquet(out_dir / fname)
        ratio = df["label"].mean()
        assert 0.4 <= ratio <= 0.6, f"{fname} label ratio {ratio:.2f} out of range"
