"""Application configuration loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings.

    Values are read from environment variables (or a local .env during
    development). Defaults are tuned for local docker-compose usage.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "message-classifier"
    app_version: str = "0.1.0"

    model_path: Path = Field(default=Path("models/classifier.joblib"))
    metrics_path: Path = Field(default=Path("models/metrics.json"))

    api_key: str = Field(default="", description="When non-empty, X-API-Key is required.")

    log_level: str = Field(default="INFO")

    max_batch_size: int = Field(default=100, ge=1, le=1000)
    max_text_length: int = Field(default=4000, ge=1, le=100_000)

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
