"""Tests for the Predictor wrapper using a tiny in-memory pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from app.ml.labels import LABELS
from app.ml.predictor import Prediction, Predictor


def test_predictor_predict_returns_known_label(trained_pipeline) -> None:
    predictor = Predictor(trained_pipeline)
    result = predictor.predict("you have won a free prize click here")
    assert isinstance(result, Prediction)
    assert result.label in LABELS
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities.keys()) == set(predictor.classes)
    total = sum(result.probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_predictor_predict_batch_preserves_order(trained_pipeline) -> None:
    predictor = Predictor(trained_pipeline)
    inputs = [
        "see you at the meeting tomorrow",
        "free entry win cash prize now",
        "you are an idiot",
    ]
    results = predictor.predict_batch(inputs)
    assert len(results) == len(inputs)
    for r in results:
        assert r.label in LABELS


def test_predictor_predict_batch_empty(trained_pipeline) -> None:
    predictor = Predictor(trained_pipeline)
    assert predictor.predict_batch([]) == []


def test_predictor_load_roundtrip(tmp_path: Path, trained_pipeline) -> None:
    artifact = tmp_path / "tiny.joblib"
    joblib.dump(trained_pipeline, artifact)

    predictor = Predictor.load(artifact)
    info = predictor.info
    assert info.classes == predictor.classes
    assert str(artifact) in info.model_path
    result = predictor.predict("hello friend how are you")
    assert result.label in LABELS


def test_predictor_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Predictor.load(tmp_path / "does_not_exist.joblib")
