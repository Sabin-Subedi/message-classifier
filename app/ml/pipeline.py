"""Sklearn pipeline factory: TextCleaner -> TF-IDF -> MultinomialNB."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from app.ml.preprocessing import TextCleaner


def build_pipeline(
    *,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    alpha: float = 0.3,
) -> Pipeline:
    """Construct the end-to-end text classification pipeline.

    The single :class:`Pipeline` bundles preprocessing, vectorization and
    the classifier so that train-time and serve-time logic stay in lock
    step and a single ``joblib`` artifact captures everything.
    """

    return Pipeline(
        steps=[
            ("clean", TextCleaner()),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df,
                    sublinear_tf=sublinear_tf,
                    lowercase=False,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            ("clf", MultinomialNB(alpha=alpha)),
        ]
    )
