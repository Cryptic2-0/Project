"""Push the v1 + v2 ONNX models to HuggingFace Hub.

Replaces S3 as the public artefact store. Idempotent: re-running re-uploads
changed files only (HF dedupes by content hash) and skips repos that already
exist at the requested target.

Why HF over S3:
    * Free public hosting.
    * Auto-rendered model card (docs/model_card.md) + version history.
    * Interview-clickable URL with a built-in inference widget for
      the FP32/text-classification head.

Usage:

    export HF_TOKEN=hf_xxxxxxxxxxxxx
    python scripts/push_to_hf.py            # push both v1 + v2
    python scripts/push_to_hf.py --only v1  # push only v1
    python scripts/push_to_hf.py --only v2  # push only v2
    python scripts/push_to_hf.py --user Cryptic2-0  # override HF user

The repos default to:
    huggingface.co/<user>/moviesentiment-distilbert-onnx-int8
    huggingface.co/<user>/moviesentiment-multitask-onnx-int8

Both default to public. Pass --private to flip.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _push(
    folder: Path,
    repo_id: str,
    token: str,
    private: bool,
    commit_message: str,
    model_card: Path | None,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    # Stage the README from docs/model_card.md so HF renders it as the repo
    # landing page. HF requires README.md at repo root.
    if model_card and model_card.exists():
        staged_readme = folder / "README.md"
        staged_readme.write_text(model_card.read_text(encoding="utf-8"), encoding="utf-8")

    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=["*.git*", "*_fp32.onnx"],  # skip FP32 (saves bandwidth)
    )
    print(f"Pushed {folder} -> https://huggingface.co/{repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default=os.environ.get("HF_USER", "Cryptic2-0"))
    parser.add_argument("--only", choices=["v1", "v2"], default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF write token. Falls back to HF_TOKEN env.",
    )
    args = parser.parse_args()

    if not args.token:
        print("ERROR: pass --token or set HF_TOKEN.", file=sys.stderr)
        return 2

    model_card = REPO_ROOT / "docs" / "model_card.md"

    plans = [
        (
            "v1",
            REPO_ROOT / "models" / "distilbert_onnx_int8",
            f"{args.user}/moviesentiment-distilbert-onnx-int8",
            "v1 binary sentiment INT8 ONNX (DistilBERT, SageMaker fine-tune, Macro F1 0.939)",
        ),
        (
            "v2",
            REPO_ROOT / "models" / "distilbert_multitask_onnx",
            f"{args.user}/moviesentiment-multitask-onnx-int8",
            "v2 multi-task INT8 ONNX (sentiment + ABSA + emotion + spoiler + helpfulness)",
        ),
    ]

    for tag, folder, repo_id, message in plans:
        if args.only and args.only != tag:
            continue
        if not folder.exists():
            print(f"SKIP {tag}: {folder} not found", file=sys.stderr)
            continue
        _push(folder, repo_id, args.token, args.private, message, model_card)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
