"""Centralised logging configuration."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with a single stdout handler.

    Idempotent: subsequent calls update the level but do not stack handlers.
    """

    root = logging.getLogger()
    root.setLevel(level.upper())

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setLevel(level.upper())

    logging.getLogger("uvicorn.access").setLevel(level.upper())
