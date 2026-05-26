"""Occlusion-based per-token attribution for the inference engine.

Cheaper than integrated gradients and works directly against the ONNX session — no
torch required at serving time. For each token in the input, drop it, re-run inference,
and report the delta in the predicted-class probability. Strong positive attribution
means "dropping this token weakens the prediction → it was supporting the label".
"""

from __future__ import annotations

import re

from moviesentiment.serve.inference import InferenceEngine
from moviesentiment.serve.schemas import Prediction

_TOKEN_RE = re.compile(r"\S+")


def _tokenize_words(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def _drop_index(tokens: list[str], idx: int) -> str:
    return " ".join(tokens[:idx] + tokens[idx + 1 :])


def occlusion_attribution(
    engine: InferenceEngine, text: str, top_k: int = 10
) -> tuple[Prediction, list[tuple[str, float]]]:
    """Run baseline + N occlusions in a single batch. Returns (baseline, top-k attributions)."""
    tokens = _tokenize_words(text)
    if not tokens:
        baseline = engine.predict([text])[0]
        return baseline, []

    batch = [text] + [_drop_index(tokens, i) for i in range(len(tokens))]
    preds = engine.predict(batch)
    baseline = preds[0]
    target_label = baseline.label

    deltas: list[tuple[str, float]] = []
    for i, p in enumerate(preds[1:]):
        # Positive delta = baseline_conf - drop_conf (for the baseline label).
        # If the drop changed the label, count the full confidence loss against the baseline.
        if p.label == target_label:
            delta = baseline.confidence - p.confidence
        else:
            delta = baseline.confidence + p.confidence  # label flipped
        deltas.append((tokens[i], float(delta)))

    deltas.sort(key=lambda x: x[1], reverse=True)
    return baseline, deltas[:top_k]
