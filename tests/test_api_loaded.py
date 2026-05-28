"""Cover the /predict + /explain success paths by patching the engine."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from moviesentiment.serve.api import app
from moviesentiment.serve.schemas import Prediction


class _StubEngine:
    KEYWORD = "masterpiece"

    def is_ready(self) -> bool:
        return True

    def predict(self, texts: list[str]) -> list[Prediction]:
        return [
            Prediction(
                text=t,
                label="positive" if self.KEYWORD in t else "negative",
                confidence=0.92 if self.KEYWORD in t else 0.61,
            )
            for t in texts
        ]


@pytest.fixture()
def client_with_engine(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from moviesentiment.serve import api as api_mod

    monkeypatch.setattr(api_mod, "engine", _StubEngine())
    # Avoid reservoir disk writes — patch flush.
    monkeypatch.setattr(api_mod.sampler, "maybe_flush", lambda _p: False)
    return TestClient(app)


def test_readyz_returns_200_when_engine_ready(client_with_engine: TestClient) -> None:
    r = client_with_engine.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_predict_happy_path_returns_labels(client_with_engine: TestClient) -> None:
    r = client_with_engine.post("/predict", json={"texts": ["a masterpiece film", "boring trash"]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert preds[0]["label"] == "positive"
    assert preds[1]["label"] == "negative"


def test_predict_rejects_batch_too_large(client_with_engine: TestClient) -> None:
    r = client_with_engine.post("/predict", json={"texts": ["x"] * 33})
    assert r.status_code == 422


def test_predict_rejects_text_too_long(client_with_engine: TestClient) -> None:
    r = client_with_engine.post("/predict", json={"texts": ["x" * 6000]})
    assert r.status_code == 422


def test_explain_happy_path_returns_attributions(
    client_with_engine: TestClient,
) -> None:
    r = client_with_engine.post(
        "/explain", json={"text": "this is a masterpiece of cinema", "top_k": 5}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "positive"
    assert len(body["attributions"]) <= 5
    assert all("token" in a and "attribution" in a for a in body["attributions"])


def test_explain_rejects_oversize_text(client_with_engine: TestClient) -> None:
    r = client_with_engine.post("/explain", json={"text": "x" * 6000, "top_k": 5})
    assert r.status_code == 422
