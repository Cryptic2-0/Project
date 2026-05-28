"""CI gate: fail the build if the last evaluation's Brier score regressed.

Reads `metrics/brier.json` (written by `compute_and_log_brier`) and exits
1 if `brier > threshold`. Run in CI after `pytest`/training:

    python scripts/check_calibration.py

Threshold defaults to `BRIER_FAIL_THRESHOLD` from `moviesentiment.eval.metrics`
(0.10). Override via `--threshold` for special cases (e.g. a known-bad release
that ships with a calibration warning).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from moviesentiment.eval.metrics import BRIER_FAIL_THRESHOLD

DEFAULT_PATH = Path("metrics/brier.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--threshold", type=float, default=BRIER_FAIL_THRESHOLD)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit 0 when the file is absent (useful for incremental rollouts).",
    )
    args = parser.parse_args()

    if not args.path.exists():
        if args.allow_missing:
            print(f"[calibration] {args.path} missing; --allow-missing -> skip")
            return 0
        print(f"[calibration] FAIL: {args.path} missing", file=sys.stderr)
        return 1

    data = json.loads(args.path.read_text())
    brier = float(data.get("brier", 1.0))
    print(f"[calibration] brier={brier:.4f} threshold={args.threshold:.4f}")
    if brier > args.threshold:
        print(
            f"[calibration] FAIL: brier {brier:.4f} > threshold {args.threshold:.4f}",
            file=sys.stderr,
        )
        return 1
    print("[calibration] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
