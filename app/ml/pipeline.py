"""Sklearn pipeline factory: TextCleaner -> FeatureUnion(word + char_wb) -> MultinomialNB."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from app.ml.preprocessing import TextCleaner


def build_pipeline(
    *,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    alpha: float = 0.3,
) -> Pipeline:
    """Construct the end-to-end text classification pipeline.

    The single :class:`Pipeline` bundles preprocessing, vectorization and
    the classifier so that train-time and serve-time logic stay in lock
    step and a single ``joblib`` artifact captures everything.

    Vectorization is a :class:`FeatureUnion` of two TF-IDF views over the
    same cleaned text:

    * ``word``    -- token n-grams in ``word_ngram_range`` (semantic units).
    * ``char``    -- ``char_wb`` n-grams in ``char_ngram_range`` to catch
                     obfuscations like ``st*pid`` / ``id1ot`` that pure
                     word n-grams miss.

    Both views produce non-negative TF-IDF vectors, so the union remains
    compatible with :class:`MultinomialNB`.
    """

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        lowercase=False,
    )

    features = FeatureUnion(
        transformer_list=[
            ("word", word_vec),
            ("char", char_vec),
        ]
    )

    return Pipeline(
        steps=[
            ("clean", TextCleaner()),
            ("features", features),
            ("clf", MultinomialNB(alpha=alpha)),
        ]
    )
