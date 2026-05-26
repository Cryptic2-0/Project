"""SQS-triggered Lambda handler for async batch inference.

Architecture (2.3 from possible_improvements.md):

    POST /predict/async (FastAPI)
        -> writes job to DynamoDB (status=queued)
        -> publishes job_id to SQS

    SQS event source mapping
        -> this Lambda fires
        -> downloads ONNX model from S3 (cached on /tmp across warm invocations)
        -> runs inference
        -> writes result back to DynamoDB (status=complete)

    GET /predict/result/{job_id} (FastAPI)
        -> reads DynamoDB

Cost: Lambda free tier covers 1M req + 400k GB-sec/mo. SQS free tier 1M/mo. DynamoDB
on-demand free tier covers portfolio traffic. Realistic monthly: $0.

To deploy: bundle this file + the ONNX model into a Lambda layer (or use container
images via ECR; the INT8 model is 64 MB so the 250 MB unzipped Lambda limit allows the
zip-based deployment too).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DDB_TABLE = os.environ["JOBS_TABLE"]
MODEL_S3_BUCKET = os.environ["MODEL_S3_BUCKET"]
MODEL_S3_KEY = os.environ.get("MODEL_S3_KEY", "moviesentiment/distilbert_onnx_int8/")

_LABELS = {0: "negative", 1: "positive"}
_CACHE_DIR = Path(tempfile.gettempdir()) / "moviesentiment"
_session = None
_tokenizer = None


def _ensure_model() -> None:
    """Lazily download model files to /tmp on cold start; reuse on warm invocations."""
    global _session, _tokenizer
    if _session is not None:
        return

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=MODEL_S3_BUCKET, Prefix=MODEL_S3_KEY)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        local = _CACHE_DIR / Path(key).name
        if not local.exists():
            logger.info("downloading %s -> %s", key, local)
            s3.download_file(MODEL_S3_BUCKET, key, str(local))

    import onnxruntime as ort
    from transformers import AutoTokenizer

    _session = ort.InferenceSession(
        str(_CACHE_DIR / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    _tokenizer = AutoTokenizer.from_pretrained(str(_CACHE_DIR))


def _predict(texts: list[str]) -> list[dict[str, Any]]:
    _ensure_model()
    import numpy as np

    assert _tokenizer is not None and _session is not None  # nosec B101 - post _ensure_model
    enc = _tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=512)
    logits = _session.run(None, dict(enc))[0]
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)

    out: list[dict[str, Any]] = []
    for i, t in enumerate(texts):
        row = probs[i]
        label_id = int(row.argmax())
        out.append({"text": t, "label": _LABELS[label_id], "confidence": float(row[label_id])})
    return out


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """SQS-triggered. Each record is a job_id; jobs are stored in DynamoDB by the API."""
    ddb = boto3.resource("dynamodb").Table(DDB_TABLE)

    processed = 0
    for record in event.get("Records", []):
        body = json.loads(record["body"])
        job_id = body["job_id"]

        item = ddb.get_item(Key={"job_id": job_id}).get("Item")
        if item is None:
            logger.warning("job %s missing from DynamoDB", job_id)
            continue

        texts = item["texts"]
        try:
            predictions = _predict(list(texts))
            ddb.update_item(
                Key={"job_id": job_id},
                UpdateExpression="SET #s = :s, predictions = :p, completed_at = :t",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":s": "complete",
                    ":p": predictions,
                    ":t": context.aws_request_id,
                },
            )
            processed += 1
        except Exception as exc:
            logger.exception("inference failed for job %s", job_id)
            ddb.update_item(
                Key={"job_id": job_id},
                UpdateExpression="SET #s = :s, error = :e",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":s": "failed", ":e": str(exc)},
            )

    return {"processed": processed}
