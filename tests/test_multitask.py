"""Tests for the v2 multi-task surface: /analyze, /insights, schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from moviesentiment.serve.api import app
from moviesentiment.serve.schemas import (
    ASPECTS,
    EMOTIONS,
    AnalyzeResponse,
    AspectScores,
    EmotionScores,
    InsightsResponse,
    Prediction,
)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_multitask_engine_from_s3_downloads_expected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """from_s3 should pull every required tokenizer file and the model.onnx."""
    from moviesentiment.serve import multitask_inference as mi

    monkeypatch.setattr(mi.settings, "model_dir", tmp_path)

    pulled: list[tuple[str, str, str]] = []

    class _FakeS3:
        def download_file(self, bucket: str, key: str, local: str) -> None:
            pulled.append((bucket, key, local))
            Path(local).write_bytes(b"\x00")

    class _FakeBoto:
        @staticmethod
        def client(name: str) -> _FakeS3:
            assert name == "s3"
            return _FakeS3()

    monkeypatch.setattr(mi, "boto3", _FakeBoto, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "boto3", _FakeBoto)

    # Make construction succeed without onnxruntime: stub cls.__init__.
    def _fake_init(self: object, onnx_path: Path, tokenizer_path: Path) -> None:
        self._session = None  # type: ignore[attr-defined]
        self._tokenizer = None  # type: ignore[attr-defined]

    monkeypatch.setattr(mi.MultiTaskInferenceEngine, "__init__", _fake_init)

    mi.MultiTaskInferenceEngine.from_s3("bkt", "multitask_onnx")
    keys = [k for _, k, _ in pulled]
    assert "multitask_onnx/model.onnx" in keys
    assert "multitask_onnx/tokenizer.json" in keys
    assert "multitask_onnx/vocab.txt" in keys


def test_analyze_returns_503_without_multitask_model(client: TestClient) -> None:
    r = client.post("/analyze", json={"text": "A wonderful film."})
    assert r.status_code == 503


def test_analyze_writes_reservoir_row_when_engine_present(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch the multi-task engine; assert /analyze returns 200 and writes a
    v2 reservoir row carrying movie_id + per-head columns."""
    from moviesentiment.serve import api as api_mod

    canned = AnalyzeResponse(
        text="great",
        sentiment=Prediction(text="great", label="positive", confidence=0.93),
        aspects=AspectScores(
            acting=[0.05, 0.05, 0.9],
            plot=[0.1, 0.1, 0.8],
            visuals=[0.05, 0.15, 0.8],
            pacing=[0.2, 0.2, 0.6],
            sound=[0.1, 0.2, 0.7],
        ),
        emotions=EmotionScores(
            joy=0.62, anger=0.05, fear=0.02, sadness=0.05, surprise=0.21, disgust=0.05
        ),
        spoiler_prob=0.08,
        helpfulness=0.72,
    )

    class _FakeEngine:
        def analyze(self, text: str) -> AnalyzeResponse:  # noqa: D401
            return canned

    monkeypatch.setattr(api_mod, "multitask_engine", _FakeEngine())
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(api_mod.sampler, "add", lambda rec: captured.append(rec))
    monkeypatch.setattr(api_mod.sampler, "maybe_flush", lambda _p: False)

    r = client.post("/analyze?movie_id=tt0111161", json={"text": "great"})
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment"]["label"] == "positive"
    assert body["spoiler_prob"] == pytest.approx(0.08)
    assert captured, "reservoir.add was not called"
    row = captured[0]
    assert row["movie_id"] == "tt0111161"
    assert row["emotion_top"] == "joy"
    assert row["spoiler_prob"] == pytest.approx(0.08)
    assert row["aspect_acting"] == pytest.approx(0.9 - 0.05)
    assert "timestamp" in row


def test_analyze_rejects_oversized_text(client: TestClient) -> None:
    r = client.post("/analyze", json={"text": "x" * 6000})
    # 422 from Pydantic max_length=5000 before the route sees it.
    assert r.status_code == 422


def test_analyze_response_schema_round_trips() -> None:
    """Construct the multi-head response in isolation; protects the schema."""
    resp = AnalyzeResponse(
        text="ok",
        sentiment=Prediction(text="ok", label="positive", confidence=0.91),
        aspects=AspectScores(
            acting=[0.05, 0.05, 0.9],
            plot=[0.1, 0.2, 0.7],
            visuals=[0.05, 0.15, 0.8],
            pacing=[0.2, 0.3, 0.5],
            sound=[0.05, 0.1, 0.85],
        ),
        emotions=EmotionScores(
            joy=0.7, anger=0.05, fear=0.02, sadness=0.05, surprise=0.15, disgust=0.03
        ),
        spoiler_prob=0.02,
        helpfulness=0.78,
    )
    blob = resp.model_dump()
    assert set(blob["aspects"]) == set(ASPECTS)
    assert set(blob["emotions"]) == set(EMOTIONS)
    assert 0.0 <= blob["spoiler_prob"] <= 1.0
    assert 0.0 <= blob["helpfulness"] <= 1.0


def test_insights_returns_404_when_reservoir_missing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the reservoir path to a non-existent file.
    monkeypatch.setattr(
        "moviesentiment.serve.insights._RESERVOIR_PATH", tmp_path / "missing.parquet"
    )
    r = client.get("/insights/tt0111161")
    assert r.status_code == 404


def test_insights_computes_aggregates_from_v1_reservoir(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1 reservoir lacks per-movie columns; the helper still produces a payload."""
    sample = pd.DataFrame(
        [
            {"text": "great film", "label": "positive", "confidence": 0.95},
            {"text": "boring", "label": "negative", "confidence": 0.81},
            {"text": "loved it", "label": "positive", "confidence": 0.88},
        ]
    )
    path = tmp_path / "recent.parquet"
    sample.to_parquet(path, index=False)
    monkeypatch.setattr("moviesentiment.serve.insights._RESERVOIR_PATH", path)

    r = client.get("/insights/tt0111161")
    assert r.status_code == 200
    body = r.json()
    assert body["movie_id"] == "tt0111161"
    assert body["n_reviews"] == 3
    assert body["sentiment_positive_share"] == pytest.approx(2 / 3, rel=1e-3)
    assert set(body["aspect_means"]) == set(ASPECTS)
    assert set(body["emotion_mix"]) == set(EMOTIONS)
    InsightsResponse(**body)  # round-trips through the Pydantic schema.


def test_insights_groups_by_movie_when_column_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from moviesentiment.serve.insights import compute_insights

    sample = pd.DataFrame(
        [
            {"text": "a", "label": "positive", "movie_id": "tt1"},
            {"text": "b", "label": "negative", "movie_id": "tt1"},
            {"text": "c", "label": "positive", "movie_id": "tt2"},
        ]
    )
    path = tmp_path / "recent.parquet"
    sample.to_parquet(path, index=False)
    monkeypatch.setattr("moviesentiment.serve.insights._RESERVOIR_PATH", path)

    ins_tt1 = compute_insights("tt1")
    ins_tt2 = compute_insights("tt2")
    ins_none = compute_insights("ttX")

    assert ins_tt1 is not None
    assert ins_tt1.n_reviews == 2
    assert ins_tt2 is not None
    assert ins_tt2.n_reviews == 1
    assert ins_none is None


def test_materialise_all_writes_per_movie_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from moviesentiment.serve import insights as ins_mod

    sample = pd.DataFrame(
        [
            {"text": "a", "label": "positive", "movie_id": "tt1"},
            {"text": "b", "label": "positive", "movie_id": "tt1"},
            {"text": "c", "label": "negative", "movie_id": "tt2"},
        ]
    )
    path = tmp_path / "recent.parquet"
    sample.to_parquet(path, index=False)
    monkeypatch.setattr(ins_mod, "_RESERVOIR_PATH", path)

    out = tmp_path / "insights"
    n = ins_mod.materialise_all(out)
    assert n == 2
    assert (out / "tt1.json").exists()
    assert (out / "tt2.json").exists()
    payload = json.loads((out / "tt1.json").read_text())
    assert payload["n_reviews"] == 2


def test_materialise_all_noop_when_no_movie_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from moviesentiment.serve import insights as ins_mod

    sample = pd.DataFrame([{"text": "a", "label": "positive"}])
    path = tmp_path / "recent.parquet"
    sample.to_parquet(path, index=False)
    monkeypatch.setattr(ins_mod, "_RESERVOIR_PATH", path)

    assert ins_mod.materialise_all(tmp_path / "out") == 0


def test_insights_topics_loaded_when_materialised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the offline BERTopic batch dropped topics JSON, the endpoint surfaces it."""
    from moviesentiment.serve.insights import compute_insights

    sample = pd.DataFrame([{"text": "x", "label": "positive", "movie_id": "tt9"}])
    path = tmp_path / "recent.parquet"
    sample.to_parquet(path, index=False)
    monkeypatch.setattr("moviesentiment.serve.insights._RESERVOIR_PATH", path)

    # Patch settings.data_dir so the topics path resolves under tmp_path.
    from moviesentiment.serve import insights as ins_mod

    monkeypatch.setattr(ins_mod.settings, "data_dir", tmp_path)

    topics_dir = tmp_path / "production" / "topics"
    topics_dir.mkdir(parents=True)
    (topics_dir / "tt9.json").write_text(json.dumps({"topics": ["pacing", "score"]}))

    result = compute_insights("tt9")
    assert result is not None
    assert result.topics == ["pacing", "score"]
