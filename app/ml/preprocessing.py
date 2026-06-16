"""Text preprocessing for the classifier.

Implements the steps described in the project report:
    1. Lowercasing
    2. Tokenization
    3. Stopword removal
    4. Lemmatization

The transformation is exposed both as a plain function (`clean_text` /
`clean_corpus`) so it can be used directly, and as a sklearn-compatible
transformer (`TextCleaner`) so it can be embedded inside a `Pipeline`
and persisted alongside the model.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)
# Sentinels emitted in place of `!` / `?` runs so TF-IDF can pick up
# punctuation as a feature without us touching the vectorizer config.
_EXCLAMATION_TOKEN = "_excl_"
_QUESTION_TOKEN = "_qst_"
_PUNCT_RUN_RE = re.compile(r"[!?]+")
# After the sentinels are inserted, strip everything that isn't a letter,
# whitespace, or underscore (so the sentinels survive).
_NON_ALPHA_RE = re.compile(r"[^a-z\s_]+")
_WHITESPACE_RE = re.compile(r"\s+")

# Tokens we explicitly KEEP even though NLTK lists them in the English
# stopword set, because they flip sentiment / signal toxicity.
_NEGATION_KEEP: frozenset[str] = frozenset(
    {
        "no", "not", "nor", "never", "none", "nothing", "nobody",
        "n't",
        "don", "doesn", "didn", "won", "wouldn", "couldn",
        "shouldn", "isn", "aren", "wasn", "weren",
        "hasn", "haven", "hadn",
        "ain", "mightn", "mustn", "needn", "shan",
    }
)

# Tokens whose identity we want to preserve through lemmatization.
_DO_NOT_LEMMATIZE: frozenset[str] = _NEGATION_KEEP | frozenset(
    {_EXCLAMATION_TOKEN, _QUESTION_TOKEN}
)

_REQUIRED_RESOURCES: tuple[tuple[str, str], ...] = (
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
)
_OPTIONAL_RESOURCES: frozenset[str] = frozenset({"punkt_tab", "omw-1.4"})

_NLTK_LOCK = threading.Lock()
_NLTK_READY = False
_LEMMATIZER = None
_STOPWORDS: frozenset[str] = frozenset()


def _resolve_nltk_data_dir() -> Path:
    """Return a writable directory for NLTK corpora.

    Honours ``NLTK_DATA`` if set; otherwise defaults to ``<repo>/.nltk_data``
    so that local dev, CI and the sandbox all have a writable location
    without requiring access to the user's home directory.
    """

    env_dir = os.environ.get("NLTK_DATA")
    if env_dir:
        path = Path(env_dir).expanduser()
    else:
        path = Path(__file__).resolve().parents[2] / ".nltk_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_nltk() -> None:
    """Lazily download required NLTK corpora and instantiate helpers.

    Performed once per process. Safe to call from multiple threads.
    """

    global _NLTK_READY, _LEMMATIZER, _STOPWORDS

    if _NLTK_READY:
        return

    with _NLTK_LOCK:
        if _NLTK_READY:
            return

        import nltk
        from nltk.corpus import stopwords as nltk_stopwords
        from nltk.stem import WordNetLemmatizer

        data_dir = _resolve_nltk_data_dir()
        data_dir_str = str(data_dir)
        if data_dir_str not in nltk.data.path:
            nltk.data.path.insert(0, data_dir_str)

        for resource, path in _REQUIRED_RESOURCES:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(resource, download_dir=data_dir_str, quiet=True)
                except Exception:
                    if resource not in _OPTIONAL_RESOURCES:
                        raise

        _LEMMATIZER = WordNetLemmatizer()
        _STOPWORDS = frozenset(nltk_stopwords.words("english")) - _NEGATION_KEEP
        _NLTK_READY = True


def _tokenize(text: str) -> list[str]:
    """Tokenize using NLTK when available, falling back to whitespace split."""

    try:
        from nltk.tokenize import word_tokenize

        return word_tokenize(text)
    except Exception:
        return text.split()


def _replace_punct_run(match: re.Match[str]) -> str:
    """Map a run of `!`/`?` to a single sentinel token.

    A run that contains any `!` becomes ``_excl_``; otherwise (`?`-only)
    it becomes ``_qst_``. The replacement is wrapped in spaces so it
    survives subsequent tokenization.
    """

    run = match.group(0)
    token = _EXCLAMATION_TOKEN if "!" in run else _QUESTION_TOKEN
    return f" {token} "


def clean_text(text: Any) -> str:
    """Normalize a single message into a space-separated cleaned string.

    Steps: lowercase -> strip URLs/emails -> replace `!`/`?` runs with
    sentinel tokens -> remove other non-letters -> tokenize -> drop
    stopwords/short tokens (preserving negations) -> lemmatize
    (preserving sentinels and negations).
    """

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PUNCT_RUN_RE.sub(_replace_punct_run, text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    _ensure_nltk()

    tokens = _tokenize(text)
    cleaned: list[str] = []
    for tok in tokens:
        if len(tok) < 2:
            continue
        if tok in _STOPWORDS:
            continue
        if tok in _DO_NOT_LEMMATIZE or tok.startswith("_"):
            cleaned.append(tok)
            continue
        lemma = _LEMMATIZER.lemmatize(tok) if _LEMMATIZER is not None else tok
        cleaned.append(lemma)

    return " ".join(cleaned)


def clean_corpus(texts: Iterable[Any]) -> list[str]:
    """Apply :func:`clean_text` to every item in `texts`."""

    return [clean_text(t) for t in texts]


class TextCleaner(BaseEstimator, TransformerMixin):
    """Sklearn transformer wrapping :func:`clean_text`.

    Stateless and pickle-friendly. Inputs may be a list/array/Series of
    strings; output is always a 1-D ``numpy.ndarray`` of strings, suitable
    as input to ``TfidfVectorizer``.
    """

    def fit(self, X: Sequence[Any], y: Any | None = None) -> TextCleaner:  # noqa: D401
        return self

    def transform(self, X: Sequence[Any]) -> np.ndarray:
        if hasattr(X, "tolist"):
            X = X.tolist()
        cleaned = clean_corpus(X)
        return np.asarray(cleaned, dtype=object)

    def get_feature_names_out(self, input_features: Any | None = None) -> np.ndarray:
        return np.asarray(["cleaned_text"], dtype=object)
