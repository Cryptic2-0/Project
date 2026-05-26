"""Async batch inference endpoints — queues to SQS, Lambda processes, DynamoDB stores.

Mounted on the main FastAPI app via `app.include_router(async_router)` once AWS infra is
provisioned. Keeps the sync /predict path independent.

Env vars required:
    MS_SQS_QUEUE_URL    SQS queue URL
    MS_JOBS_TABLE       DynamoDB table name
    AWS_REGION          AWS region

If any are unset, the router still mounts but every call returns 503 — useful for local
dev where the AWS plumbing isn't there.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from moviesentiment.serve.schemas import Prediction

async_router = APIRouter(prefix="/predict", tags=["inference-async"])


class AsyncPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=10_000)


class AsyncPredictResponse(BaseModel):
    job_id: str
    status: str


class AsyncResultResponse(BaseModel):
    job_id: str
    status: str
    predictions: list[Prediction] | None = None
    error: str | None = None


def _aws_clients() -> tuple[Any, Any]:
    queue_url = os.environ.get("MS_SQS_QUEUE_URL")
    table_name = os.environ.get("MS_JOBS_TABLE")
    if not queue_url or not table_name:
        raise HTTPException(status_code=503, detail="async path not configured")
    import boto3

    return (
        boto3.client("sqs"),
        boto3.resource("dynamodb").Table(table_name),
    )


@async_router.post("/async", response_model=AsyncPredictResponse)
def submit(req: AsyncPredictRequest) -> AsyncPredictResponse:
    """`/predict` is sync for the UI; this queues to Lambda so a 10s batch doesn't tie up Fargate threads."""
    sqs, table = _aws_clients()
    job_id = uuid.uuid4().hex

    table.put_item(
        Item={
            "job_id": job_id,
            "status": "queued",
            "texts": req.texts,
        }
    )
    sqs.send_message(
        QueueUrl=os.environ["MS_SQS_QUEUE_URL"],
        MessageBody=json.dumps({"job_id": job_id}),
    )
    return AsyncPredictResponse(job_id=job_id, status="queued")


@async_router.get("/result/{job_id}", response_model=AsyncResultResponse)
def result(job_id: str) -> AsyncResultResponse:
    _, table = _aws_clients()
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if item is None:
        raise HTTPException(status_code=404, detail="job not found")

    preds_raw = item.get("predictions") or []
    preds = [Prediction(**p) for p in preds_raw] if item["status"] == "complete" else None
    return AsyncResultResponse(
        job_id=job_id,
        status=item["status"],
        predictions=preds,
        error=item.get("error"),
    )
