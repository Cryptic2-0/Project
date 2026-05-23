"""Stratified train/val/test split with fixed seed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def split_dataset(inp: Path, out_dir: Path) -> dict[str, int]:
    """Split inp Parquet → train/val/test Parquet files under out_dir."""
    params = yaml.safe_load(Path("params.yaml").read_text())["split"]
    test_size: float = params["test_size"]
    val_size: float = params["val_size"]
    seed: int = params["seed"]

    df = pd.read_parquet(inp)
    trainval, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    # val_size is relative to original; adjust for the trainval subset
    val_adjusted = val_size / (1.0 - test_size)
    train, val = train_test_split(
        trainval, test_size=val_adjusted, stratify=trainval["label"], random_state=seed
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    return {"train": len(train), "val": len(val), "test": len(test)}
