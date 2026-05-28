"""Nearest-neighbour endpoint covers cold-start, hot path, rebuild, and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from moviesentiment.serve import similar as similar_mod
from moviesentiment.serve.api import app


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    # Point the index at a temp reservoir; rebuild fires on first query.
    monkeypatch.setattr(similar_mod, "_INDEX", similar_mod._SimilarIndex())
    from moviesentiment.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(similar_mod.settings, "data_dir", tmp_path)
    return TestClient(app)


def test_similar_returns_empty_on_cold_start(client: TestClient) -> None:
    r = client.get("/similar", params={"text": "a film", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "a film"
    assert body["hits"] == []


def test_similar_finds_lexical_overlap(client: TestClient, tmp_path: Path) -> None:
    prod_dir = tmp_path / "production"
    prod_dir.mkdir()
    df = pd.DataFrame(
        {
            "text": [
                "a masterpiece film about cinema",
                "boring trash, awful film",
                "this is a masterpiece of the genre",
                "neutral review with no signal",
            ],
            "label": ["positive", "negative", "positive", "negative"],
        }
    )
    df.to_parquet(prod_dir / "recent.parquet")

    r = client.get("/similar", params={"text": "masterpiece cinema", "k": 2})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) <= 2
    # Top hit must overlap on "masterpiece".
    assert any("masterpiece" in h["text"].lower() for h in hits)


def test_similar_rejects_oversize_text(client: TestClient) -> None:
    r = client.get("/similar", params={"text": "x" * 6000})
    assert r.status_code == 422


def test_similar_rejects_empty_text(client: TestClient) -> None:
    r = client.get("/similar", params={"text": ""})
    assert r.status_code == 422


def test_similar_clamps_k(client: TestClient, tmp_path: Path) -> None:
    prod_dir = tmp_path / "production"
    prod_dir.mkdir()
    df = pd.DataFrame(
        {
            "text": [f"review number {i} about cinema" for i in range(50)],
            "label": ["positive"] * 50,
        }
    )
    df.to_parquet(prod_dir / "recent.parquet")
    # k=999 is silently clamped to 25.
    r = client.get("/similar", params={"text": "cinema", "k": 999})
    assert r.status_code == 200
    assert len(r.json()["hits"]) <= 25
