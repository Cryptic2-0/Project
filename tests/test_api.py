"""Tests for the FastAPI inference service."""

import pytest
from fastapi.testclient import TestClient

from moviesentiment.serve.api import app


@pytest.fixture()
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


def test_request_id_header_round_trip(client: TestClient) -> None:
    r = client.get("/healthz", headers={"x-request-id": "abc12345"})
    assert r.headers.get("x-request-id") == "abc12345"


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    r = client.get("/healthz")
    rid = r.headers.get("x-request-id")
    assert rid is not None and len(rid) >= 8


def test_version_endpoint(client: TestClient) -> None:
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert {"model_name", "model_stage", "git_sha"} <= set(body)


def test_sample_endpoint(client: TestClient) -> None:
    r = client.get("/sample")
    assert r.status_code == 200
    assert {"k", "n_seen", "n_in_reservoir", "flushes"} <= set(r.json())


def test_explain_without_model(client: TestClient) -> None:
    r = client.post("/explain", json={"text": "great", "top_k": 3})
    assert r.status_code == 503


def test_async_endpoint_returns_503_without_aws(client: TestClient) -> None:
    r = client.post("/predict/async", json={"texts": ["x"]})
    assert r.status_code == 503


def test_async_result_endpoint_returns_503_without_aws(client: TestClient) -> None:
    r = client.get("/predict/result/abc123")
    assert r.status_code == 503


def test_frontend_ui_mount_when_bundled(client: TestClient) -> None:
    """If frontend/index.html exists, /ui/ serves it."""
    from pathlib import Path

    if not (Path(__file__).resolve().parent.parent / "frontend" / "index.html").exists():
        return  # frontend not bundled in this checkout
    r = client.get("/ui/")
    assert r.status_code == 200
    assert "MovieSentiment" in r.text


def test_metrics_endpoint_exposes_prometheus(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"# HELP" in r.content


def test_predict_invalid_payload(client: TestClient) -> None:
    r = client.post("/predict", json={"wrong_key": ["x"]})
    assert r.status_code == 422


def test_predict_empty_batch(client: TestClient) -> None:
    r = client.post("/predict", json={"texts": []})
    assert r.status_code == 422


def test_request_id_sanitizer_strips_unsafe_chars() -> None:
    """Sanitizer-level test: CRLF, quotes, spaces, colons stripped; safe chars kept."""
    from moviesentiment.serve.logging_setup import _sanitize_rid

    malicious = 'abc\r\nLog-Forged: yes "evil"'
    out = _sanitize_rid(malicious)
    for bad in ("\r", "\n", '"', " ", ":"):
        assert bad not in out
    assert len(out) <= 64
    assert _sanitize_rid("req_abc-123.v2") == "req_abc-123.v2"


def test_request_id_sanitizer_caps_length() -> None:
    from moviesentiment.serve.logging_setup import _sanitize_rid

    assert len(_sanitize_rid("a" * 1000)) == 64


def test_request_id_sanitizer_generates_uuid_for_unsafe_only() -> None:
    from moviesentiment.serve.logging_setup import _sanitize_rid

    out = _sanitize_rid("!!! @@@")
    assert len(out) >= 8
    assert all(c.isalnum() or c in "._-" for c in out)


def test_request_id_safe_value_round_trips(client: TestClient) -> None:
    r = client.get("/healthz", headers={"x-request-id": "safe_abc-123"})
    assert r.headers.get("x-request-id") == "safe_abc-123"


def test_cors_preflight_does_not_echo_arbitrary_origin(client: TestClient) -> None:
    """With default `*` allowlist, CORS does NOT include allow_credentials."""
    r = client.options(
        "/healthz",
        headers={
            "origin": "https://evil.example",
            "access-control-request-method": "GET",
        },
    )
    assert r.headers.get("access-control-allow-credentials") != "true"
