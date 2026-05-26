"""Data-quality gate — verifies validate() catches schema, null, length, and balance failures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from moviesentiment.data.validate import validate


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)


def test_passes_clean_data(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "text": [f"this is review number {i} with sufficient length" for i in range(200)],
            "label": [i % 2 for i in range(200)],
        }
    )
    inp = tmp_path / "in.parquet"
    out = tmp_path / "report.json"
    _write(df, inp)
    assert validate(inp, out) == 0
    report = json.loads(out.read_text())
    assert report["failed"] == []


def test_fails_on_missing_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"text": ["a longer review text"] * 10})  # no label
    inp = tmp_path / "in.parquet"
    out = tmp_path / "report.json"
    _write(df, inp)
    assert validate(inp, out) == 1
    report = json.loads(out.read_text())
    assert "schema" in report["failed"]


def test_fails_on_label_imbalance(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "text": [f"this is review {i} with sufficient text" for i in range(100)],
            "label": [1] * 90 + [0] * 10,
        }
    )
    inp = tmp_path / "in.parquet"
    out = tmp_path / "report.json"
    _write(df, inp)
    assert validate(inp, out) == 1
    report = json.loads(out.read_text())
    assert "label_balance" in report["failed"]


def test_fails_on_short_text(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "text": ["short"] * 100,
            "label": [0, 1] * 50,
        }
    )
    inp = tmp_path / "in.parquet"
    out = tmp_path / "report.json"
    _write(df, inp)
    assert validate(inp, out) == 1
