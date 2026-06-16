"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("API_KEY", "")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def tiny_dataset() -> list[tuple[str, str]]:
    """A tiny labelled dataset used to exercise the pipeline end-to-end."""

    samples: list[tuple[str, str]] = [
        ("how are you doing today friend", "normal"),
        ("see you at the meeting tomorrow", "normal"),
        ("can we grab coffee later", "normal"),
        ("happy birthday have a great day", "normal"),
        ("did you finish the homework yet", "normal"),
        ("congrats you won a free prize click here now", "spam"),
        ("limited offer claim your reward now", "spam"),
        ("urgent your account has been compromised verify now", "spam"),
        ("free entry win cash prize call this number", "spam"),
        ("buy cheap meds online no prescription needed", "spam"),
        ("you are such an idiot stop talking", "abusive"),
        ("shut up you moron nobody likes you", "abusive"),
        ("you are stupid and worthless", "abusive"),
        ("get lost loser nobody cares about you", "abusive"),
        ("you are pathetic and dumb", "abusive"),
        ("i hate people like you they should disappear", "hateful"),
        ("those people are inferior and disgusting", "hateful"),
        ("i despise that entire group of humans", "hateful"),
        ("they do not deserve to live among us", "hateful"),
        ("we should not allow them in our country", "hateful"),
    ]
    return samples


@pytest.fixture(scope="session")
def trained_pipeline(tiny_dataset: list[tuple[str, str]]):
    """Train the real pipeline on the tiny dataset (session-scoped)."""

    from app.ml.pipeline import build_pipeline

    texts, labels = zip(*tiny_dataset, strict=True)
    pipe = build_pipeline(min_df=1)
    pipe.fit(list(texts), list(labels))
    return pipe


@pytest.fixture()
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset cached settings between tests when env vars are mutated."""

    from app.core import config

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
