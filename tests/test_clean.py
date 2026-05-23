"""Tests for data cleaning logic."""

from moviesentiment.data.clean import _clean_text


def test_strips_html() -> None:
    assert "<b>" not in _clean_text("A <b>great</b> film")


def test_removes_url() -> None:
    assert "http" not in _clean_text("check https://example.com for details")


def test_lowercases() -> None:
    result = _clean_text("GREAT Film")
    assert result == result.lower()


def test_collapses_whitespace() -> None:
    result = _clean_text("too   many    spaces")
    assert "  " not in result


def test_empty_after_cleaning_is_short() -> None:
    result = _clean_text("<b></b>")
    assert len(result) <= 10
