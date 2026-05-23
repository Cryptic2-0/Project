"""ONNX inference engine — loads model once at startup, handles batched requests."""

from __future__ import annotations

from pathlib import Path


class InferenceEngine:
    """Wraps an ONNX Runtime session for batched sentiment inference."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._ready = True

    @classmethod
    def from_registry(cls, model_name: str, stage: str) -> InferenceEngine:
        raise NotImplementedError("Implement in Week 2 after ONNX export")

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, texts: list[str]) -> list[dict[str, object]]:
        raise NotImplementedError("Implement in Week 2")
