"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__ as APP_VERSION
from app.api.routes import router as api_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.ml.predictor import Predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the ML model once on startup so request handling is hot."""

    settings: Settings = get_settings()
    configure_logging(settings.log_level)

    app.state.predictor = None
    app.state.predictor_error = None

    try:
        predictor = Predictor.load(settings.model_path, metrics_path=settings.metrics_path)
        app.state.predictor = predictor
        logger.info(
            "Model loaded from %s (classes=%s)",
            settings.model_path,
            predictor.classes,
        )
    except FileNotFoundError as exc:
        msg = (
            f"Model artifact not found at {settings.model_path}. "
            "The /predict endpoints will return 503 until the file is present."
        )
        logger.warning("%s (%s)", msg, exc)
        app.state.predictor_error = msg
    except Exception as exc:
        msg = f"Failed to load model: {exc}"
        logger.exception(msg)
        app.state.predictor_error = msg

    try:
        yield
    finally:
        app.state.predictor = None


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Message Classifier",
        description=(
            "Intelligent Messaging Safety & Priority Classifier. Categorises text "
            "messages into normal / spam / abusive / hateful using TF-IDF + Naive Bayes."
        ),
        version=APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def _root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": APP_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()
