"""Tests for the FastAPI inference service."""

import pytest
from fastapi.testclient import TestClient
from moviesentiment.serve.api import app


@pytest.fixture()  # type: ignore[misc]
def client() -> TestClient:
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_without_model(client: TestClient) -> None:
    r = client.get("/readyz")
    assert r.status_code == 503


def test_predict_without_model(client: TestClient) -> None:
    r = client.post("/predict", json={"texts": ["great movie"]})
    assert r.status_code == 503


def test_predict_batch_too_large(client: TestClient) -> None:
    r = client.post("/predict", json={"texts": ["x"] * 33})
    assert r.status_code == 422
