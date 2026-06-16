"""Unit tests for the text preprocessing pipeline."""

from __future__ import annotations

import numpy as np

from app.ml.preprocessing import TextCleaner, clean_corpus, clean_text


def test_clean_text_lowercases() -> None:
    assert clean_text("Free OFFER") == clean_text("free offer")


def test_clean_text_removes_urls_and_emails() -> None:
    cleaned = clean_text("Visit https://example.com or mail me at foo@bar.com now")
    assert "http" not in cleaned
    assert "example" not in cleaned
    assert "@" not in cleaned
    assert "foo" not in cleaned


def test_clean_text_strips_non_alpha() -> None:
    cleaned = clean_text("Hello!!! 123 world??? :)")
    tokens = cleaned.split()
    for t in tokens:
        assert t.isalpha()
    assert "world" in cleaned


def test_clean_text_removes_stopwords() -> None:
    cleaned = clean_text("you have got a great prize")
    tokens = cleaned.split()
    for sw in ("you", "have", "a"):
        assert sw not in tokens
    assert "great" in tokens
    assert "prize" in tokens


def test_clean_text_lemmatizes() -> None:
    cleaned = clean_text("messages received here")
    assert "message" in cleaned.split()


def test_clean_text_handles_empty_and_none() -> None:
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_clean_text_handles_emoji_and_unicode() -> None:
    cleaned = clean_text("hello world ðŸ˜€ â¤ï¸")
    assert "hello" in cleaned and "world" in cleaned


def test_clean_corpus_returns_list_of_strings() -> None:
    out = clean_corpus(["Hello!", "spam SPAM spam"])
    assert isinstance(out, list)
    assert all(isinstance(s, str) for s in out)
    assert len(out) == 2


def test_text_cleaner_transformer_returns_ndarray() -> None:
    cleaner = TextCleaner()
    out = cleaner.fit_transform(["Hi there!", "BUY now"])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)


def test_text_cleaner_handles_pandas_series() -> None:
    pd = __import__("pandas")
    s = pd.Series(["Hello world", "free offer"])
    out = TextCleaner().fit_transform(s)
    assert out.shape == (2,)
