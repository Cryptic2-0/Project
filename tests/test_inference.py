"""Tests for the ONNX inference engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from moviesentiment.serve.inference import InferenceEngine
from moviesentiment.serve.schemas import Prediction


def test_from_registry_raises_when_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """from_registry raises FileNotFoundError if ONNX files don't exist."""
    monkeypatch.setattr("moviesentiment.serve.inference.settings.model_dir", tmp_path)
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        InferenceEngine.from_registry("moviesentiment-classifier", "Production")


def test_predict_returns_predictions() -> None:
    """predict() returns correct labels given raw logits from mock session."""
    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([[-1.0, 1.0], [1.0, -1.0]])]

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": np.array([[1, 2, 3], [1, 2, 3]])}

    engine = object.__new__(InferenceEngine)
    engine._session = mock_session
    engine._tokenizer = mock_tokenizer
    engine._ready = True

    preds = engine.predict(["great movie", "awful film"])

    assert len(preds) == 2
    assert preds[0].label == "positive"
    assert preds[0].confidence > 0.7
    assert preds[1].label == "negative"
    assert preds[1].confidence > 0.7


def test_predict_returns_prediction_objects() -> None:
    """predict() items are Prediction instances."""
    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([[0.3, 0.7]])]
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": np.array([[1, 2]])}

    engine = object.__new__(InferenceEngine)
    engine._session = mock_session
    engine._tokenizer = mock_tokenizer
    engine._ready = True

    results = engine.predict(["test text"])
    assert isinstance(results[0], Prediction)
    assert results[0].text == "test text"
    assert results[0].label in ("positive", "negative")
    assert 0.0 <= results[0].confidence <= 1.0


def test_is_ready_true_after_init() -> None:
    engine = object.__new__(InferenceEngine)
    engine._ready = True
    assert engine.is_ready()
