"""ONNX inference engine — loads model once at startup, handles batched requests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from moviesentiment.config import settings
from moviesentiment.serve.schemas import Prediction

_LABELS: dict[int, str] = {0: "negative", 1: "positive"}


class InferenceEngine:
    """Wraps an ONNX Runtime session for batched sentiment inference."""

    def __init__(self, onnx_path: Path, tokenizer_path: Path) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self._ready = True

    @classmethod
    def from_registry(cls, model_name: str, stage: str) -> InferenceEngine:
        """Load ONNX model from disk: INT8 preferred, then FP32."""
        int8_path = settings.model_dir / "distilbert_onnx_int8" / "model.onnx"
        fp32_path = settings.model_dir / "distilbert_onnx" / "model.onnx"

        if int8_path.exists():
            return cls(int8_path, int8_path.parent)
        if fp32_path.exists():
            return cls(fp32_path, fp32_path.parent)

        raise FileNotFoundError(
            "ONNX model not found. Run: python -m moviesentiment.models.onnx_export"
        )

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, texts: list[str]) -> list[Prediction]:
        import numpy as np

        enc: dict[str, Any] = self._tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )
        logits: np.ndarray[Any, np.dtype[np.float32]] = self._session.run(None, dict(enc))[0]

        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs: np.ndarray[Any, np.dtype[np.float32]] = exp / exp.sum(axis=-1, keepdims=True)

        results: list[Prediction] = []
        for i, text in enumerate(texts):
            row = probs[i]
            label_id = int(row.argmax())
            confidence = float(row[label_id])
            results.append(Prediction(text=text, label=_LABELS[label_id], confidence=confidence))
        return results
