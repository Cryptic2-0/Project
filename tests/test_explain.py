"""Occlusion attribution — verifies shape, ordering, and edge cases."""

from __future__ import annotations

from moviesentiment.serve.explain import _drop_index, _tokenize_words


def test_tokenize_words_basic() -> None:
    assert _tokenize_words("hello world") == ["hello", "world"]
    assert _tokenize_words("  spaced   out  ") == ["spaced", "out"]
    assert _tokenize_words("") == []


def test_drop_index_middle() -> None:
    tokens = ["a", "b", "c", "d"]
    assert _drop_index(tokens, 1) == "a c d"


def test_drop_index_first() -> None:
    assert _drop_index(["a", "b", "c"], 0) == "b c"


def test_drop_index_last() -> None:
    assert _drop_index(["a", "b", "c"], 2) == "a b"


# --- occlusion_attribution with a stubbed engine -----------------------------

from moviesentiment.serve.explain import occlusion_attribution  # noqa: E402
from moviesentiment.serve.schemas import Prediction  # noqa: E402


class _StubEngine:
    """InferenceEngine-compatible stub. Returns deterministic confidence that
    depends on whether the 'masterpiece' keyword remains in the text."""

    KEYWORD = "masterpiece"

    def predict(self, texts: list[str]) -> list[Prediction]:
        out: list[Prediction] = []
        for t in texts:
            if self.KEYWORD in t:
                out.append(Prediction(text=t, label="positive", confidence=0.95))
            else:
                out.append(Prediction(text=t, label="negative", confidence=0.55))
        return out


def test_occlusion_empty_text_returns_no_attributions() -> None:
    base, attrs = occlusion_attribution(_StubEngine(), "")  # type: ignore[arg-type]
    assert base.label == "negative"
    assert attrs == []


def test_occlusion_ranks_keyword_first_when_drop_flips_label() -> None:
    base, attrs = occlusion_attribution(  # type: ignore[arg-type]
        _StubEngine(), "this film is a masterpiece", top_k=3
    )
    assert base.label == "positive"
    assert attrs[0][0] == "masterpiece"
    # Stub: baseline.confidence=0.95 (positive), drop.confidence=0.55 (negative).
    # P_drop(positive) = 1 - 0.55 = 0.45 → delta = 0.95 - 0.45 = 0.50.
    assert abs(attrs[0][1] - 0.50) < 1e-6


def test_occlusion_same_label_uses_confidence_delta() -> None:
    class _SameLabel:
        def predict(self, texts: list[str]) -> list[Prediction]:
            return [
                Prediction(text=t, label="positive", confidence=0.95 if i == 0 else 0.90)
                for i, t in enumerate(texts)
            ]

    _, attrs = occlusion_attribution(_SameLabel(), "a b c")  # type: ignore[arg-type]
    assert len(attrs) == 3
    for _tok, delta in attrs:
        assert abs(delta - 0.05) < 1e-6


def test_occlusion_top_k_limits_output() -> None:
    _, attrs = occlusion_attribution(  # type: ignore[arg-type]
        _StubEngine(), "one two three four five", top_k=2
    )
    assert len(attrs) == 2
