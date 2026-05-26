"""Data quality gate (Great Expectations) — halts the DVC pipeline on schema/dist drift.

Runs on the cleaned dataset before split/train. Catches problems at ingest, not at
inference time. Failing the gate sets a non-zero exit code so `dvc repro` aborts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _expectations() -> list[dict[str, object]]:
    return [
        {"name": "schema", "rule": "columns must contain {text,label}"},
        {"name": "no_null_text", "rule": "text non-null"},
        {"name": "no_null_label", "rule": "label non-null"},
        {"name": "text_length", "rule": "len(text) between 10 and 5000"},
        {"name": "label_values", "rule": "label in {0,1}"},
        {"name": "label_balance", "rule": "0.40 <= P(label=1) <= 0.60"},
    ]


def validate(inp: Path, out: Path) -> int:
    import pandas as pd

    df = pd.read_parquet(inp)

    results: list[dict[str, object]] = []
    failed: list[str] = []

    def check(name: str, ok: object, detail: str = "") -> None:
        ok_py = bool(ok)
        results.append({"name": name, "ok": ok_py, "detail": detail})
        if not ok_py:
            failed.append(name)

    check("schema", {"text", "label"} <= set(df.columns), detail=str(list(df.columns)))
    check("no_null_text", bool(df["text"].notna().all()) if "text" in df else False)
    check("no_null_label", bool(df["label"].notna().all()) if "label" in df else False)

    if "text" in df:
        lens = df["text"].astype(str).str.len()
        check(
            "text_length",
            bool(((lens >= 10) & (lens <= 5000)).all()),
            detail=f"min={lens.min()} max={lens.max()}",
        )

    if "label" in df:
        unique = set(df["label"].dropna().unique().tolist())
        check("label_values", unique.issubset({0, 1}), detail=str(sorted(unique)))

        pos_share = float((df["label"] == 1).mean())
        check(
            "label_balance",
            0.40 <= pos_share <= 0.60,
            detail=f"P(label=1)={pos_share:.3f}",
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"rows": int(len(df)), "checks": results, "failed": failed},
            indent=2,
        )
    )

    return 1 if failed else 0


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: validate.py <input.parquet> <report.json>", file=sys.stderr)
        sys.exit(2)
    code = validate(Path(sys.argv[1]), Path(sys.argv[2]))
    if code != 0:
        print("data validation FAILED — see report for failing checks", file=sys.stderr)
    sys.exit(code)


if __name__ == "__main__":
    main()
