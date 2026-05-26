"""SageMaker training entrypoint for the v2 multi-task DistilBERT.

Pulls processed data via DVC (sentiment), and emotion + spoiler parquets from
the staging S3 prefix (uploaded by scripts/sagemaker_launch_multitask.py).
Runs `train_multitask`. Copies the resulting checkpoint to /opt/ml/model.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path("/opt/ml/code")
MODEL_OUTPUT_DIR = Path("/opt/ml/model")
STAGING_BUCKET = os.environ.get("MS_DVC_BUCKET", "moviesentiment-dvc-soumya")
STAGING_PREFIX = os.environ.get("MS_STAGING_PREFIX", "staging/multitask")


def _run(cmd: list[str], **kwargs: object) -> None:
    subprocess.run(cmd, check=True, **kwargs)  # type: ignore[call-overload]


def main() -> None:
    os.chdir(CODE_DIR)

    _run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"])
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "transformers>=4.46,<5.0",
            "accelerate>=0.26.0",
            "datasets>=2.18",
            "mlflow>=2.11",
            "dvc[s3]>=3.48",
            "pydantic>=2.6",
            "pydantic-settings>=2.2",
            "pyyaml",
            "--upgrade",
        ]
    )

    src_dir = str(CODE_DIR / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    if not (CODE_DIR / ".git").exists():
        _run(["git", "init", "-q"])
        _run(["git", "config", "user.email", "sagemaker@local"])
        _run(["git", "config", "user.name", "sagemaker"])

    dvc = Path(sys.executable).parent / "dvc"
    _run(
        [
            str(dvc),
            "pull",
            "data/processed/train.parquet",
            "data/processed/val.parquet",
            "data/processed/test.parquet",
        ]
    )

    # Pull staged task parquets (emotion + spoiler) from S3.
    import boto3

    s3 = boto3.client("s3")
    multitask_dir = CODE_DIR / "data" / "interim" / "multitask"
    multitask_dir.mkdir(parents=True, exist_ok=True)
    for name in ("emotion.parquet", "spoiler.parquet"):
        key = f"{STAGING_PREFIX}/{name}"
        local = multitask_dir / name
        s3.download_file(STAGING_BUCKET, key, str(local))
        print(f"pulled s3://{STAGING_BUCKET}/{key} -> {local}")

    from moviesentiment.models.multitask_train import train_multitask

    train_multitask()

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_src = CODE_DIR / "models" / "distilbert_multitask"
    if model_src.exists():
        shutil.copytree(model_src, MODEL_OUTPUT_DIR / "distilbert_multitask")

    metric = CODE_DIR / "metrics" / "multitask.json"
    if metric.exists():
        shutil.copy(metric, MODEL_OUTPUT_DIR / "multitask.json")


if __name__ == "__main__":
    main()
