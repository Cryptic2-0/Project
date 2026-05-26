"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def sample_reviews_path() -> Generator[Path, None, None]:
    yield Path(__file__).parent / "fixtures" / "sample_reviews.csv"


@pytest.fixture()
def sample_reviews_df(sample_reviews_path: Path) -> Generator[pd.DataFrame, None, None]:
    yield pd.read_csv(sample_reviews_path)
