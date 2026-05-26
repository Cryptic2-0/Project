"""Reservoir sampling (Vitter's Algorithm R) for fixed-memory production input capture."""

from __future__ import annotations

import random
import threading
from pathlib import Path
from typing import Any


class ReservoirSampler:
    """Algorithm R reservoir sampler.

    Maintains a uniform random sample of size <= k over an unbounded stream. After n
    items have been seen, every item has identical probability k/n of being in the
    reservoir — independent of arrival order. Memory is O(k).

    Periodically flushes to parquet so the sample survives process restart. Flush
    cadence: every `flush_every` inserts AND when the reservoir is full (whichever
    fires first); kept here so callers don't have to wire up a timer.
    """

    def __init__(self, k: int = 1000, flush_every: int = 100, seed: int | None = None) -> None:
        self._k = k
        self._flush_every = flush_every
        self._reservoir: list[Any] = []
        self._n = 0
        self._writes_since_flush = 0
        self._flushed_total = 0
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def add(self, item: Any) -> None:
        with self._lock:
            self._n += 1
            if len(self._reservoir) < self._k:
                self._reservoir.append(item)
            else:
                j = self._rng.randrange(self._n)
                if j < self._k:
                    self._reservoir[j] = item
            self._writes_since_flush += 1

    def maybe_flush(self, path: Path) -> bool:
        """Flush to parquet if pending writes exceed `flush_every`. Returns True if flushed."""
        with self._lock:
            if self._writes_since_flush < self._flush_every:
                return False
            self._flush_locked(path)
            return True

    def flush(self, path: Path) -> None:
        with self._lock:
            self._flush_locked(path)

    def _flush_locked(self, path: Path) -> None:
        if not self._reservoir:
            self._writes_since_flush = 0
            return
        import pandas as pd

        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(list(self._reservoir))
        df.to_parquet(path, index=False)
        self._flushed_total += 1
        self._writes_since_flush = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "k": self._k,
                "n_seen": self._n,
                "n_in_reservoir": len(self._reservoir),
                "flushes": self._flushed_total,
            }
