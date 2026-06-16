"""Smoke tests for the FastAPI app, using a tiny pipeline injected at runtime."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app.ml.labels import LABELS


@pytest.fixture()
def trained_artifact_path(tmp_path_factory: pytest.TempPathFactory, trained_pipeline) -> Path:
    """Persist the tiny trained pipeline so the lifespan can load it."""

    out = tmp_path_factory.mktemp("models") / "classifier.joblib"
    joblib.dump(trained_pipeline, out)
    return out


@pytest.fixture()
def client_no_auth(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact_path: Path,
) -> Iterator[TestClient]:
    """TestClient where API auth is disabled and a real model is loaded."""

    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("MODEL_PATH", str(trained_artifact_path))
    monkeypatch.setenv("METRICS_PATH", str(trained_artifact_path.with_name("metrics.json")))
    monkeypatch.setenv("MAX_BATCH_SIZE", "5")
    monkeypatch.setenv("MAX_TEXT_LENGTH", "200")

    import app.core.config as config

    config.get_settings.cache_clear()

    import app.main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        yield client
    config.get_settings.cache_clear()


@pytest.fixture()
def client_with_auth(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact_path: Path,
) -> Iterator[TestClient]:
    """TestClient where an API key is required."""

    monkeypatch.setenv("API_KEY", "secret-key")
    monkeypatch.setenv("MODEL_PATH", str(trained_artifact_path))
    monkeypatch.setenv("MAX_BATCH_SIZE", "100")
    monkeypatch.setenv("MAX_TEXT_LENGTH", "4000")

    import app.core.config as config

    config.get_settings.cache_clear()

    import app.main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        yield client
    config.get_settings.cache_clear()


@pytest.fixture()
def client_no_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """TestClient where the model file does not exist."""

    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))

    import app.core.config as config

    config.get_settings.cache_clear()

    import app.main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        yield client
    config.get_settings.cache_clear()


def test_root_returns_metadata(client_no_auth: TestClient) -> None:
    r = client_no_auth.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["docs"] == "/docs"
    assert body["health"] == "/api/v1/health"


def test_health_ok(client_no_auth: TestClient) -> None:
    r = client_no_auth.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ready_when_model_loaded(client_no_auth: TestClient) -> None:
    r = client_no_auth.get("/api/v1/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["model_loaded"] is True


def test_ready_returns_503_when_model_missing(client_no_model: TestClient) -> None:
    r = client_no_model.get("/api/v1/ready")
    assert r.status_code == 503


def test_predict_happy_path(client_no_auth: TestClient) -> None:
    r = client_no_auth.post("/api/v1/predict", json={"text": "hello friend how are you"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in LABELS
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys())  # non-empty


def test_predict_rejects_empty_text(client_no_auth: TestClient) -> None:
    r = client_no_auth.post("/api/v1/predict", json={"text": "   "})
    assert r.status_code == 422


def test_predict_rejects_missing_field(client_no_auth: TestClient) -> None:
    r = client_no_auth.post("/api/v1/predict", json={})
    assert r.status_code == 422


def test_predict_rejects_text_over_limit(client_no_auth: TestClient) -> None:
    big_text = "spam " * 200
    r = client_no_auth.post("/api/v1/predict", json={"text": big_text})
    assert r.status_code == 413


def test_predict_batch_happy_path(client_no_auth: TestClient) -> None:
    r = client_no_auth.post(
        "/api/v1/predict/batch",
        json={"texts": ["hello there", "you won a free prize"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    for item in body["results"]:
        assert item["label"] in LABELS


def test_predict_batch_rejects_over_max(client_no_auth: TestClient) -> None:
    r = client_no_auth.post(
        "/api/v1/predict/batch",
        json={"texts": ["x"] * 6},
    )
    assert r.status_code in (400, 422)


def test_predict_requires_api_key_when_configured(client_with_auth: TestClient) -> None:
    r = client_with_auth.post(
        "/api/v1/predict",
        json={"text": "hello there"},
    )
    assert r.status_code == 401


def test_predict_with_correct_api_key(client_with_auth: TestClient) -> None:
    r = client_with_auth.post(
        "/api/v1/predict",
        json={"text": "hello there"},
        headers={"X-API-Key": "secret-key"},
    )
    assert r.status_code == 200


def test_info_returns_classes(client_no_auth: TestClient) -> None:
    r = client_no_auth.get("/api/v1/info")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["classes"], list)
    assert len(body["classes"]) >= 2
    assert body["app_version"]
