"""Reservoir sampler — verifies bounded memory, uniform sampling, and flush behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from moviesentiment.serve.reservoir import ReservoirSampler


def test_size_caps_at_k() -> None:
    s = ReservoirSampler(k=10, flush_every=1_000_000, seed=0)
    for i in range(1000):
        s.add({"i": i})
    stats = s.stats()
    assert stats["n_in_reservoir"] == 10
    assert stats["n_seen"] == 1000


def test_uniform_distribution() -> None:
    """k/n probability: across many trials, late items appear in the reservoir
    roughly as often as early items (within a wide tolerance for randomness)."""
    K = 50
    N = 500
    counts = [0] * N
    trials = 200
    for trial in range(trials):
        s = ReservoirSampler(k=K, flush_every=1_000_000, seed=trial)
        for i in range(N):
            s.add(i)
        for item in s._reservoir:
            counts[item] += 1

    expected = trials * K / N
    # early-half vs late-half average appearance should be close (chi-square-style)
    early = sum(counts[: N // 2]) / (N // 2)
    late = sum(counts[N // 2 :]) / (N // 2)
    assert abs(early - late) / expected < 0.20, f"bias detected: early={early}, late={late}"


def test_flush_writes_parquet(tmp_path: Path) -> None:
    s = ReservoirSampler(k=100, flush_every=5, seed=0)
    out = tmp_path / "sample.parquet"
    for i in range(20):
        s.add({"i": i, "text": f"row-{i}"})
        s.maybe_flush(out)
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) <= 100
    assert "text" in df.columns


def test_stats_keys() -> None:
    s = ReservoirSampler(k=3, flush_every=1_000_000, seed=0)
    s.add(1)
    s.add(2)
    stats = s.stats()
    assert set(stats) == {"k", "n_seen", "n_in_reservoir", "flushes"}
    assert stats["n_seen"] == 2
