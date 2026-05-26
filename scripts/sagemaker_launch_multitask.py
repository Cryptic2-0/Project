"""Submit v2 multi-task DistilBERT fine-tuning as a SageMaker training job.

Prerequisites:
  - SAGEMAKER_ROLE_ARN env var with sagemaker.amazonaws.com trust + S3 access.
  - Task parquets staged at s3://<bucket>/staging/multitask/ (see prep
    helpers under src/moviesentiment/data/multitask_loaders.py).

Cost: ml.g4dn.xlarge spot ~$0.16/hr; ~130K examples × 2 epochs at batch 16
runs in ~1.5 hr → ~$0.25.
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
BUCKET = os.environ.get("MS_DVC_BUCKET", "moviesentiment-dvc-soumya")
REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILES = ["pyproject.toml", "params.yaml", "dvc.yaml", "dvc.lock"]
SOURCE_DIRS = ["scripts", "src"]
DVC_CONFIG_FILES = ["config", ".gitignore"]


def _stage_source() -> Path:
    stage = Path(tempfile.mkdtemp(prefix="sm-mt-"))
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
        raise SystemExit("Set SAGEMAKER_ROLE_ARN env var.")

    boto_session = boto3.Session(region_name=REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)

    stage = _stage_source()
    print(
        f"Staged source at {stage} "
        f"({sum(f.stat().st_size for f in stage.rglob('*') if f.is_file()) // 1024} KB)"
    )

    use_spot = os.environ.get("SAGEMAKER_USE_SPOT", "1") != "0"
    max_run = 60 * 60 * 3  # 3-hour ceiling

    estimator = PyTorch(
        entry_point="scripts/train_entrypoint_multitask.py",
        source_dir=str(stage),
        role=ROLE_ARN,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        framework_version="2.2.0",
        py_version="py310",
        output_path=f"s3://{BUCKET}/sagemaker-output-multitask/",
        base_job_name="moviesentiment-multitask",
        sagemaker_session=sm_session,
        use_spot_instances=use_spot,
        max_run=max_run,
        max_wait=max_run * 2 if use_spot else None,
        checkpoint_s3_uri=(f"s3://{BUCKET}/sagemaker-checkpoints-multitask/" if use_spot else None),
        environment={
            "MS_DVC_BUCKET": BUCKET,
            "MS_STAGING_PREFIX": "staging/multitask",
            "AWS_DEFAULT_REGION": REGION,
        },
    )

    wait = os.environ.get("SAGEMAKER_WAIT", "1") != "0"
    print(f"Submitting multi-task training (wait={wait}, ~1.5 hr on ml.g4dn.xlarge T4)...")
    try:
        estimator.fit(wait=wait)
        print(f"\nJob name: {estimator.latest_training_job.name}")
        if wait:
            print(f"Artefacts: {estimator.model_data}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
