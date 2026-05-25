"""SageMaker training entrypoint for DistilBERT fine-tuning."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path("/opt/ml/code")
MODEL_OUTPUT_DIR = Path("/opt/ml/model")


def _run(cmd: list[str], **kwargs: object) -> None:
    subprocess.run(cmd, check=True, **kwargs)  # type: ignore[call-overload]


def main() -> None:
    os.chdir(CODE_DIR)

    # IMPORTANT: --no-deps so pip does NOT reinstall torch. The SageMaker PyTorch
    # 2.2 image ships a CUDA-enabled torch; pip would otherwise pull the CPU
    # wheel from PyPI and break transformers' PyTorch backend detection.
    _run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"])
    # Install only runtime deps the base image lacks. transformers gets
    # explicitly bumped because mlflow.transformers needs >=4.40 and the image
    # ships ~4.38.
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "transformers>=4.46,<5.0",
            "datasets>=2.18",
            "mlflow>=2.11",
            "dvc[s3]>=3.48",
            "pydantic>=2.6",
            "pydantic-settings>=2.2",
            "pyyaml",
            "--upgrade",
        ]
    )

    # Belt-and-suspenders: editable install may not update sys.path in the
    # already-running process, so insert src/ explicitly before the import.
    src_dir = str(CODE_DIR / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # DVC requires an SCM root (git repo) unless core.no_scm is set. Source
    # bundle ships only .dvc/config, no .git -- init an empty repo to satisfy it.
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

    from moviesentiment.models.transformer import train_transformer

    train_transformer()

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_src = CODE_DIR / "models" / "distilbert"
    if model_src.exists():
        shutil.copytree(model_src, MODEL_OUTPUT_DIR / "distilbert")

    for src, dst_name in [
        (CODE_DIR / "metrics" / "transformer.json", "transformer.json"),
        (CODE_DIR / "mlflow.db", "mlflow.db"),
    ]:
        if src.exists():
            shutil.copy(src, MODEL_OUTPUT_DIR / dst_name)


if __name__ == "__main__":
    main()
