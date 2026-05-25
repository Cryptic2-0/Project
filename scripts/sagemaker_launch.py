"""Submit DistilBERT fine-tuning as a SageMaker training job.

Prerequisites:
  pip install sagemaker
  export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/<role>

Cost: ~$0.30 on ml.g4dn.xlarge (~25 min on T4 GPU).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")
BUCKET = "moviesentiment-dvc-soumya"
REGION = "ap-southeast-2"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Whole repo is ~880MB (models/, data/, mlruns/, .venv, .dvc/cache).
# SageMaker tars source_dir verbatim and has no ignore-file support, so we
# stage a minimal directory containing only what the entrypoint actually needs.
SOURCE_FILES = ["pyproject.toml", "params.yaml", "dvc.yaml", "dvc.lock"]
SOURCE_DIRS = ["scripts", "src"]
DVC_CONFIG_FILES = ["config", ".gitignore"]


def _stage_source() -> Path:
    stage = Path(tempfile.mkdtemp(prefix="sm-source-"))
    for name in SOURCE_FILES:
        shutil.copy2(REPO_ROOT / name, stage / name)
    for name in SOURCE_DIRS:
        shutil.copytree(REPO_ROOT / name, stage / name)
    dvc_dst = stage / ".dvc"
    dvc_dst.mkdir()
    for name in DVC_CONFIG_FILES:
        src = REPO_ROOT / ".dvc" / name
        if src.exists():
            shutil.copy2(src, dvc_dst / name)
    return stage


def main() -> None:
    if not ROLE_ARN:
        raise SystemExit(
            "Set SAGEMAKER_ROLE_ARN env var.\n"
            "  IAM -> Roles -> Create role -> SageMaker -> AmazonSageMakerFullAccess\n"
            "  + inline S3 policy for s3://moviesentiment-dvc-soumya/*\n"
            "  Then: $env:SAGEMAKER_ROLE_ARN = 'arn:aws:iam::<account>:role/<name>'"
        )

    boto_session = boto3.Session(region_name=REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)

    stage = _stage_source()
    print(
        f"Staged source at {stage} ({sum(f.stat().st_size for f in stage.rglob('*') if f.is_file()) // 1024} KB)"
    )

    estimator = PyTorch(
        entry_point="scripts/train_entrypoint.py",
        source_dir=str(stage),
        role=ROLE_ARN,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        framework_version="2.2.0",
        py_version="py310",
        output_path=f"s3://{BUCKET}/sagemaker-output/",
        base_job_name="moviesentiment-distilbert",
        sagemaker_session=sm_session,
    )

    print("Submitting training job (~25 min on ml.g4dn.xlarge T4)...")
    try:
        estimator.fit(wait=True)
        print(f"\nArtifacts at: {estimator.model_data}")
        print("Next: run the download + dvc add steps in docs/external_setup.md")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
