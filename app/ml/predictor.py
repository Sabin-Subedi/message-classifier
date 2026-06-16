"""Loadable predictor wrapping a persisted sklearn pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.ml.labels import LABELS

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Result of classifying a single text message."""

    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass
class ModelInfo:
    """Metadata about the loaded model."""

    model_path: str
    classes: list[str]
    trained_at: str | None = None
    model_version: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class Predictor:
    """Thread-safe wrapper around a trained sklearn ``Pipeline``."""

    def __init__(
        self,
        pipeline: Any,
        *,
        model_path: Path | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._model_path = model_path
        self._metrics = dict(metrics or {})

        classes = getattr(pipeline, "classes_", None)
        if classes is None:
            clf = getattr(pipeline, "named_steps", {}).get("clf")
            classes = getattr(clf, "classes_", None)
        if classes is None:
            raise ValueError("Loaded pipeline has no `classes_` attribute.")

        self._classes: list[str] = [str(c) for c in classes]

        unknown = [c for c in self._classes if c not in LABELS]
        if unknown:
            logger.warning(
                "Pipeline reports classes outside the canonical label set: %s",
                unknown,
            )

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        metrics_path: str | Path | None = None,
    ) -> Predictor:
        """Load a serialized pipeline (and optional metrics JSON) from disk."""

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        logger.info("Loading model from %s", model_path)
        pipeline = joblib.load(model_path)

        metrics: dict[str, Any] = {}
        if metrics_path is not None:
            metrics_path = Path(metrics_path)
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to read metrics file %s: %s", metrics_path, exc)

        return cls(pipeline, model_path=model_path, metrics=metrics)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            model_path=str(self._model_path) if self._model_path else "<in-memory>",
            classes=self.classes,
            trained_at=self._metrics.get("trained_at"),
            model_version=self._metrics.get("model_version"),
            metrics={k: v for k, v in self._metrics.items() if k not in {"trained_at", "model_version"}},
        )

    def predict(self, text: str) -> Prediction:
        """Classify a single text message."""

        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify a batch of text messages.

        A single batched call to the underlying pipeline is more efficient
        than looping. Returns one :class:`Prediction` per input text in
        the same order.
        """

        if len(texts) == 0:
            return []

        proba_matrix = self._pipeline.predict_proba(list(texts))
        proba_matrix = np.asarray(proba_matrix, dtype=float)

        results: list[Prediction] = []
        for row in proba_matrix:
            best_idx = int(np.argmax(row))
            label = self._classes[best_idx]
            probabilities = {cls: float(p) for cls, p in zip(self._classes, row, strict=False)}
            results.append(
                Prediction(
                    label=label,
                    confidence=float(row[best_idx]),
                    probabilities=probabilities,
                )
            )
        return results
