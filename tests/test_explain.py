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
