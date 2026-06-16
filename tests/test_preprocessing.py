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


def test_clean_text_strips_non_alpha_but_keeps_sentinels() -> None:
    """Digits and stray symbols disappear; punctuation runs become sentinels."""

    cleaned = clean_text("Hello!!! 123 world??? :)")
    tokens = cleaned.split()
    # Every token is letters-only OR one of the punctuation sentinels.
    for t in tokens:
        assert t.isalpha() or t.startswith("_"), t
    assert "world" in tokens
    assert "_excl_" in tokens
    assert "_qst_" in tokens
    # Digits should never survive.
    assert not any(c.isdigit() for c in cleaned)


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


def test_clean_text_keeps_negations() -> None:
    """Negations flip sentiment, so they must not be silently dropped."""

    with_neg = clean_text("i do not hate you").split()
    without_neg = clean_text("i hate you").split()

    assert "not" in with_neg, with_neg
    assert "hate" in with_neg
    assert "not" not in without_neg
    # The two strings must produce different bags-of-words.
    assert with_neg != without_neg

    # A handful of contraction roots that come from `won't`, `don't`, etc.
    assert "don" in clean_text("i don't like spam").split()
    assert "won" in clean_text("you won't believe this").split()


def test_clean_text_emits_punctuation_sentinels() -> None:
    bang = clean_text("FREE!!!").split()
    assert "free" in bang
    assert "_excl_" in bang
    assert bang.count("_excl_") == 1, "a run of !!! collapses to one sentinel"

    qst = clean_text("hi?").split()
    assert "_qst_" in qst

    mixed = clean_text("really?!").split()
    # `?!` is a single run that contains a `!`, so it maps to _excl_.
    assert "_excl_" in mixed


def test_clean_text_does_not_lemmatize_sentinels_or_negations() -> None:
    """Sentinels and contraction roots must keep their identity."""

    out = clean_text("you won't get this !!!").split()
    assert "_excl_" in out
    assert "won" in out  # not lemmatized into "win"
