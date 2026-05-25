"""Submit DistilBERT fine-tuning as a SageMaker training job.

Prerequisites:
  pip install sagemaker
  export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/<role>

Cost: ~$0.30 on ml.g4dn.xlarge (~25 min on T4 GPU).
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")
BUCKET = "moviesentiment-dvc-soumya"
REGION = "ap-southeast-2"
REPO_ROOT = str(Path(__file__).resolve().parents[1])


def main() -> None:
    if not ROLE_ARN:
        raise SystemExit(
            "Set SAGEMAKER_ROLE_ARN env var.\n"
            "  IAM → Roles → Create role → SageMaker → AmazonSageMakerFullAccess\n"
            "  + inline S3 policy for s3://moviesentiment-dvc-soumya/*\n"
            "  Then: $env:SAGEMAKER_ROLE_ARN = 'arn:aws:iam::<account>:role/<name>'"
        )

    boto_session = boto3.Session(region_name=REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)

    estimator = PyTorch(
        entry_point="scripts/train_entrypoint.py",
        source_dir=REPO_ROOT,
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
    estimator.fit(wait=True)

    print(f"\nArtifacts at: {estimator.model_data}")
    print("Next: run the download + dvc add steps in docs/external_setup.md")


if __name__ == "__main__":
    main()
