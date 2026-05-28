"""Multi-task ONNX inference engine — five heads in one forward pass.

ONNX export of the multi-task model should produce a single graph with five
outputs (sentiment, aspect, emotion, spoiler, helpfulness). This wrapper
loads that graph and decodes each output into the v2 schema types.

If the multi-task ONNX file is not present on disk, `from_disk` raises
FileNotFoundError; the serving layer catches that and exposes /analyze as 503.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from moviesentiment.config import settings as settings  # explicit re-export
from moviesentiment.models.multitask import ASPECTS, EMOTIONS
from moviesentiment.serve.schemas import (
    AnalyzeResponse,
    AspectScores,
    EmotionScores,
    Prediction,
)

_SENTIMENT_LABELS = {0: "negative", 1: "positive"}
_MULTITASK_DIR = "distilbert_multitask_onnx"


def _softmax(x: Any) -> Any:
    import numpy as np

    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


class MultiTaskInferenceEngine:
    """Wraps the multi-task ONNX session. One session per process."""

    def __init__(self, onnx_path: Path, tokenizer_path: Path) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        # nosec B615 - local DVC-pulled path, not a Hub identifier.
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))  # nosec B615
        # Validate the graph shape so a wrong model file fails loud at startup.
        out_names = {o.name for o in self._session.get_outputs()}
        expected = {"sentiment", "aspect", "emotion", "spoiler", "helpfulness"}
        missing = expected - out_names
        if missing:
            raise RuntimeError(
                f"multi-task ONNX missing outputs {sorted(missing)}; "
                f"export the model with onnx_export_multitask first"
            )

    @classmethod
    def from_s3(cls, bucket: str, prefix: str) -> MultiTaskInferenceEngine:
        """Download the multi-task artefact from S3 to settings.model_dir,
        then construct as usual. Used at Fargate boot when the image does
        not bundle the v2 ONNX.
        """
        import boto3

        d = settings.model_dir / _MULTITASK_DIR
        d.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client("s3")
        wanted = (
            "model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        )
        for name in wanted:
            local = d / name
            if local.exists():
                continue
            s3.download_file(bucket, f"{prefix.rstrip('/')}/{name}", str(local))
        return cls(d / "model.onnx", d)

    @classmethod
    def from_disk(cls) -> MultiTaskInferenceEngine:
        d = settings.model_dir / _MULTITASK_DIR
        onnx_path = d / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"multi-task ONNX not at {onnx_path}")
        return cls(onnx_path, d)

    # Multi-task was trained with max_length=256 (see params.yaml::multitask).
    # Inference matches so the model never sees positions it didn't train on.
    _MAX_LENGTH = 256

    def analyze(self, text: str) -> AnalyzeResponse:
        import numpy as np

        enc = self._tokenizer(
            [text],
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self._MAX_LENGTH,
        )
        outputs = self._session.run(None, dict(enc))
        out_names = [o.name for o in self._session.get_outputs()]
        result: dict[str, Any] = dict(zip(out_names, outputs, strict=True))

        sent_probs = _softmax(result["sentiment"])[0]
        sent_label_id = int(sent_probs.argmax())
        sentiment = Prediction(
            text=text,
            label=_SENTIMENT_LABELS[sent_label_id],
            confidence=float(sent_probs[sent_label_id]),
        )

        aspect_logits = result["aspect"][0]  # (5, 3)
        aspect_probs = _softmax(aspect_logits)
        aspect_kwargs = {
            name: [float(p) for p in aspect_probs[i]] for i, name in enumerate(ASPECTS)
        }
        aspects = AspectScores(**aspect_kwargs)

        emotion_probs = _softmax(result["emotion"])[0]
        emotions = EmotionScores(
            **{name: float(emotion_probs[i]) for i, name in enumerate(EMOTIONS)}
        )

        spoiler_probs = _softmax(result["spoiler"])[0]
        spoiler_prob = float(spoiler_probs[1])

        helpfulness = float(np.clip(result["helpfulness"][0], 0.0, 1.0))

        return AnalyzeResponse(
            text=text,
            sentiment=sentiment,
            aspects=aspects,
            emotions=emotions,
            spoiler_prob=spoiler_prob,
            helpfulness=helpfulness,
        )
