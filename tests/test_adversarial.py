"""Adversarial robustness tests for the sentiment engine.

These tests do NOT need a real model. They run against `_StubEngine` (same
pattern as `tests/test_api_loaded.py`) and verify that the *pipeline* is
robust to a small set of attack families: character perturbation and
synonym swap.

The stub is keyword-driven, so the test asserts:

  1. Perturbations that preserve the keyword preserve the label.
  2. Perturbations that destroy the keyword flip the label as expected.
  3. The pipeline does not crash on unicode look-alikes, emoji, or
     leading / trailing whitespace.

When we wire a real ONNX engine here later, the same test bodies will
report *real* accuracy drop under attack; the structure stays put.

Why this exists: every production NLP service should know how much
accuracy it loses under a 1% input perturbation. Documented in
`docs/model_card.md` under "Quantitative analyses".
"""

from __future__ import annotations

import random
import string

import pytest

from moviesentiment.serve.schemas import Prediction

_KEYWORD = "masterpiece"


class _StubEngine:
    """Keyword-driven stub: positive iff KEYWORD is in the text."""

    def is_ready(self) -> bool:
        return True

    def predict(self, texts: list[str]) -> list[Prediction]:
        return [
            Prediction(
                text=t,
                label="positive" if _KEYWORD in t.lower() else "negative",
                confidence=0.92 if _KEYWORD in t.lower() else 0.61,
            )
            for t in texts
        ]


def _char_perturb(text: str, rate: float, seed: int) -> str:
    """Random character swap / insert / delete at the given per-char rate."""
    rng = random.Random(seed)
    out: list[str] = []
    for c in text:
        roll = rng.random()
        if roll < rate / 3:
            out.append(rng.choice(string.ascii_lowercase))
        elif roll < 2 * rate / 3:
            out.append(c + rng.choice(string.ascii_lowercase))
        elif roll < rate:
            continue
        else:
            out.append(c)
    return "".join(out)


_SYNONYMS = {
    "good": ["great", "fine", "decent"],
    "bad": ["awful", "poor", "weak"],
    "film": ["movie", "picture", "flick"],
    "amazing": ["incredible", "stunning", "remarkable"],
}


def _synonym_swap(text: str, seed: int) -> str:
    """Swap one word per known synonym group, deterministically."""
    rng = random.Random(seed)
    words = text.split()
    for i, w in enumerate(words):
        lower = w.lower().strip(string.punctuation)
        if lower in _SYNONYMS:
            words[i] = rng.choice(_SYNONYMS[lower])
    return " ".join(words)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_low_rate_perturbation_preserves_label_when_keyword_survives(seed: int) -> None:
    text = "this is a masterpiece of cinema with great direction"
    perturbed = _char_perturb(text, rate=0.02, seed=seed)
    if _KEYWORD not in perturbed.lower():
        pytest.skip(f"seed={seed} destroyed the keyword; not a robustness case")

    engine = _StubEngine()
    base = engine.predict([text])[0]
    adv = engine.predict([perturbed])[0]
    assert base.label == adv.label, (
        f"label flipped under 2% char perturbation despite surviving keyword: "
        f"base={base.label!r} adv={adv.label!r} text={perturbed!r}"
    )


def test_keyword_destroying_perturbation_flips_label() -> None:
    text = "this is a masterpiece of cinema"
    destroyed = text.replace("masterpiece", "garbage")
    engine = _StubEngine()
    assert engine.predict([text])[0].label == "positive"
    assert engine.predict([destroyed])[0].label == "negative"


def test_synonym_swap_does_not_crash() -> None:
    text = "this is an amazing film with good acting and a bad villain"
    swapped = _synonym_swap(text, seed=7)
    engine = _StubEngine()
    preds = engine.predict([text, swapped])
    assert len(preds) == 2
    for p in preds:
        assert p.label in {"positive", "negative"}
        assert 0.0 <= p.confidence <= 1.0


@pytest.mark.parametrize(
    "edge",
    [
        "   masterpiece   ",
        "\tmasterpiece\n",
        "MASTERPIECE",
        "MaStErPiEcE",
        "masterpiece! amazing!! incredible!!!",
        "a" * 1024 + " masterpiece " + "z" * 1024,
        "🎬 masterpiece 🎥",
        "café masterpiece naïve",
    ],
)
def test_engine_robust_to_whitespace_case_unicode(edge: str) -> None:
    engine = _StubEngine()
    pred = engine.predict([edge])[0]
    assert pred.label in {"positive", "negative"}


def test_adversarial_accuracy_drop_reportable() -> None:
    """Aggregate metric a real ONNX-backed test could surface.

    With the stub, the accuracy is 100% by construction unless the
    perturbation destroys the keyword. The shape of the test is the
    contract: replace `_StubEngine()` with a real `InferenceEngine` to
    get a real number.
    """
    engine = _StubEngine()
    inputs = [
        ("this masterpiece moved me", "positive"),
        ("a complete masterpiece of cinema", "positive"),
        ("absolutely awful, worst film ever", "negative"),
        ("boring, pointless, a waste of time", "negative"),
    ]
    rng = random.Random(42)
    clean_correct = 0
    adv_correct = 0
    for text, expected in inputs:
        if engine.predict([text])[0].label == expected:
            clean_correct += 1
        adv = _char_perturb(text, rate=0.03, seed=rng.randint(0, 1_000_000))
        if engine.predict([adv])[0].label == expected:
            adv_correct += 1
    clean_acc = clean_correct / len(inputs)
    adv_acc = adv_correct / len(inputs)
    drop = clean_acc - adv_acc
    assert clean_acc == 1.0, f"stub baseline broken: {clean_acc}"
    assert drop >= 0.0, f"adversarial accuracy somehow higher than clean: drop={drop}"
