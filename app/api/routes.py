"""HTTP routes for the classifier service."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app import __version__ as APP_VERSION
from app.core.config import Settings, get_settings
from app.core.security import require_api_key
from app.ml.predictor import Predictor
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResult,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_predictor(request: Request) -> Predictor:
    """Dependency: return the predictor stashed on app.state by the lifespan."""

    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet. Train and place the artifact at MODEL_PATH.",
        )
    return predictor


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Liveness probe (does not require model).",
)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name, version=APP_VERSION)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["meta"],
    summary="Readiness probe (200 only when the model is loaded).",
)
async def ready(request: Request) -> ReadyResponse:
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        load_error: str | None = getattr(request.app.state, "predictor_error", None)
        return _readiness_unavailable(detail=load_error or "Model not loaded.")
    return ReadyResponse(ready=True, model_loaded=True)


def _readiness_unavailable(detail: str) -> ReadyResponse:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=detail,
    )


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    tags=["meta"],
    dependencies=[Depends(require_api_key)],
    summary="Model metadata (classes, training timestamp, metrics).",
)
async def info(predictor: Predictor = Depends(_get_predictor)) -> ModelInfoResponse:
    meta = predictor.info
    return ModelInfoResponse(
        app_version=APP_VERSION,
        model_path=meta.model_path,
        model_version=meta.model_version,
        trained_at=meta.trained_at,
        classes=meta.classes,
        metrics=meta.metrics,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["predict"],
    dependencies=[Depends(require_api_key)],
    summary="Classify a single message.",
)
async def predict(
    payload: PredictRequest,
    settings: Settings = Depends(get_settings),
    predictor: Predictor = Depends(_get_predictor),
) -> PredictResponse:
    if len(payload.text) > settings.max_text_length:
        raise HTTPException(
            status_code=413,
            detail=f"text exceeds max_text_length={settings.max_text_length}",
        )

    result = predictor.predict(payload.text)
    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["predict"],
    dependencies=[Depends(require_api_key)],
    summary="Classify a batch of messages.",
)
async def predict_batch(
    payload: BatchPredictRequest,
    settings: Settings = Depends(get_settings),
    predictor: Predictor = Depends(_get_predictor),
) -> BatchPredictResponse:
    if len(payload.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"batch size {len(payload.texts)} exceeds max_batch_size={settings.max_batch_size}",
        )

    too_long = [
        i for i, t in enumerate(payload.texts) if len(t) > settings.max_text_length
    ]
    if too_long:
        raise HTTPException(
            status_code=413,
            detail=(
                f"texts at indices {too_long[:5]} exceed max_text_length="
                f"{settings.max_text_length}"
            ),
        )

    results = predictor.predict_batch(payload.texts)
    return BatchPredictResponse(
        results=[
            PredictionResult(
                label=r.label,
                confidence=r.confidence,
                probabilities=r.probabilities,
            )
            for r in results
        ]
    )
