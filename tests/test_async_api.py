"""Async path (/predict/async, /predict/result/{id}) — happy + edge cases."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from moviesentiment.serve.api import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class _FakeTable:
    def __init__(self, items: dict[str, dict[str, Any]] | None = None) -> None:
        self.items: dict[str, dict[str, Any]] = items or {}
        self.puts: list[dict[str, Any]] = []

    def put_item(self, Item: dict[str, Any]) -> None:  # noqa: N803 (boto3 kwarg)
        self.items[Item["job_id"]] = Item
        self.puts.append(Item)

    def get_item(self, Key: dict[str, str]) -> dict[str, Any]:  # noqa: N803
        item = self.items.get(Key["job_id"])
        return {"Item": item} if item is not None else {}


class _FakeSQS:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send_message(self, QueueUrl: str, MessageBody: str) -> None:  # noqa: N803
        self.sent.append({"QueueUrl": QueueUrl, "MessageBody": MessageBody})


def _wire_fakes(monkeypatch: pytest.MonkeyPatch, table: _FakeTable, sqs: _FakeSQS) -> None:
    """Patch _aws_clients to return our fakes regardless of env state."""
    from moviesentiment.serve import async_api as aa

    monkeypatch.setenv("MS_SQS_QUEUE_URL", "https://sqs.test/queue")
    monkeypatch.setenv("MS_JOBS_TABLE", "moviesentiment-jobs")
    monkeypatch.setattr(aa, "_aws_clients", lambda: (sqs, table))


def test_submit_writes_dynamodb_and_publishes_sqs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    table, sqs = _FakeTable(), _FakeSQS()
    _wire_fakes(monkeypatch, table, sqs)

    r = client.post("/predict/async", json={"texts": ["hello world"]})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]
    # DynamoDB put + SQS send both happened with matching job_id.
    assert table.puts[0]["job_id"] == job_id
    assert table.puts[0]["texts"] == ["hello world"]
    import json as _json

    msg = _json.loads(sqs.sent[0]["MessageBody"])
    assert msg["job_id"] == job_id


def test_result_unknown_id_returns_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    table, sqs = _FakeTable(), _FakeSQS()
    _wire_fakes(monkeypatch, table, sqs)
    r = client.get("/predict/result/unknown-id")
    assert r.status_code == 404


def test_result_queued_job_returns_status_without_predictions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _FakeTable({"abc": {"job_id": "abc", "status": "queued", "texts": ["x"]}})
    _wire_fakes(monkeypatch, table, _FakeSQS())
    r = client.get("/predict/result/abc")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert body["predictions"] is None


def test_result_complete_job_returns_predictions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _FakeTable(
        {
            "done": {
                "job_id": "done",
                "status": "complete",
                "texts": ["good film"],
                "predictions": [
                    {"text": "good film", "label": "positive", "confidence": 0.9},
                ],
            }
        }
    )
    _wire_fakes(monkeypatch, table, _FakeSQS())
    r = client.get("/predict/result/done")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete"
    assert body["predictions"][0]["label"] == "positive"


def test_result_failed_job_includes_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _FakeTable({"f": {"job_id": "f", "status": "failed", "texts": ["x"], "error": "boom"}})
    _wire_fakes(monkeypatch, table, _FakeSQS())
    r = client.get("/predict/result/f")
    body = r.json()
    assert body["status"] == "failed"
    assert body["error"] == "boom"


def test_async_503_when_env_unset(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MS_SQS_QUEUE_URL", raising=False)
    monkeypatch.delenv("MS_JOBS_TABLE", raising=False)
    r = client.post("/predict/async", json={"texts": ["x"]})
    assert r.status_code == 503
