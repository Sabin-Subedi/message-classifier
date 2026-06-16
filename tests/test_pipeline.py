"""Tests for the FeatureUnion-based classification pipeline.

Verifies pipeline structure (word + char_wb branches) and the user-visible
behaviour we care about: char n-grams should let the classifier generalise
across simple obfuscations like ``idi0t`` -> ``idiot``.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from app.ml.pipeline import build_pipeline


def test_pipeline_has_expected_structure() -> None:
    pipe = build_pipeline(min_df=1)

    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps.keys()) == ["clean", "features", "clf"]

    features = pipe.named_steps["features"]
    assert isinstance(features, FeatureUnion)

    branches = dict(features.transformer_list)
    assert set(branches) == {"word", "char"}
    assert isinstance(branches["word"], TfidfVectorizer)
    assert isinstance(branches["char"], TfidfVectorizer)
    assert branches["word"].analyzer == "word"
    assert branches["char"].analyzer == "char_wb"

    assert isinstance(pipe.named_steps["clf"], MultinomialNB)


def test_pipeline_exposes_ngram_range_params() -> None:
    pipe = build_pipeline(
        word_ngram_range=(1, 1),
        char_ngram_range=(3, 4),
        min_df=1,
    )
    branches = dict(pipe.named_steps["features"].transformer_list)
    assert branches["word"].ngram_range == (1, 1)
    assert branches["char"].ngram_range == (3, 4)


def test_pipeline_feature_union_produces_combined_matrix() -> None:
    """The fitted FeatureUnion should emit a sparse matrix wider than either branch alone."""

    pipe = build_pipeline(min_df=1)
    pipe.fit(
        ["hello world", "free prize now", "you are an idiot"],
        ["normal", "spam", "abusive"],
    )

    features = pipe.named_steps["features"]
    word_dim = len(dict(features.transformer_list)["word"].vocabulary_)
    char_dim = len(dict(features.transformer_list)["char"].vocabulary_)

    cleaned = pipe.named_steps["clean"].transform(["hello world"])
    matrix = features.transform(cleaned)

    assert matrix.shape == (1, word_dim + char_dim)
    assert matrix.shape[1] > word_dim, "char branch should add features"


def test_pipeline_handles_obfuscation_via_char_ngrams() -> None:
    """Char n-grams should bridge between ``idiot`` and ``id1ot`` / ``idi0t``.

    We seed enough samples per class so each char trigram inside ``idiot``
    survives ``min_df=1`` and the classifier can lean on character overlap.
    """

    train: list[tuple[str, str]] = [
        ("how are you doing today friend", "normal"),
        ("can we grab coffee later", "normal"),
        ("see you at the meeting tomorrow", "normal"),
        ("happy birthday have a great day", "normal"),
        ("you are such an idiot stop talking", "abusive"),
        ("shut up you idiot nobody likes you", "abusive"),
        ("get lost you complete idiot", "abusive"),
        ("stop being an idiot already", "abusive"),
    ]
    texts, labels = zip(*train, strict=True)

    pipe = build_pipeline(min_df=1)
    pipe.fit(list(texts), list(labels))

    obfuscated = ["you are such an id1ot", "what an idi0t you are"]
    preds = pipe.predict(obfuscated)
    assert list(preds) == ["abusive", "abusive"]
