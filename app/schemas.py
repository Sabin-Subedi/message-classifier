"""Pydantic request and response models for the public API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    app_version: str
    model_path: str
    model_version: str | None = None
    trained_at: str | None = None
    classes: list[str]
    metrics: dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Message text to classify.",
    )

    @field_validator("text")
    @classmethod
    def _strip_and_validate(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace only.")
        return stripped


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of messages to classify.",
    )

    @field_validator("texts")
    @classmethod
    def _strip_and_validate(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for idx, value in enumerate(values):
            if not isinstance(value, str):
                raise ValueError(f"texts[{idx}] must be a string.")
            stripped = value.strip()
            if not stripped:
                raise ValueError(f"texts[{idx}] must not be empty or whitespace only.")
            cleaned.append(stripped)
        return cleaned


class PredictionResult(BaseModel):
    label: str = Field(..., description="Predicted class label.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of the predicted class.",
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Per-class probability distribution.",
    )


class PredictResponse(PredictionResult):
    pass


class BatchPredictResponse(BaseModel):
    results: list[PredictionResult]


class ErrorResponse(BaseModel):
    detail: str
