"""Smoke test for the ``--grid-search`` path in ``scripts.train``.

We intentionally collapse the grid to a single combination (and the CV to
2 folds) via monkeypatching so the test stays well under a second on a
200-row in-memory fixture, while still exercising the GridSearchCV branch
end-to-end (refit, persistence, metrics serialisation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import train as train_mod


def _synthesize_dataset(n_per_class: int = 50) -> pd.DataFrame:
    """Build a small but learnable 4-class dataset by repeating templates."""

    templates: dict[str, list[str]] = {
        "normal": [
            "how are you doing today friend",
            "see you at the meeting tomorrow",
            "can we grab coffee later in the afternoon",
            "happy birthday have a great day",
            "did you finish the homework already",
        ],
        "spam": [
            "congrats you won a free prize click here now",
            "limited offer claim your reward now",
            "urgent your account has been compromised verify now",
            "free entry win cash prize call this number",
            "buy cheap meds online no prescription needed",
        ],
        "abusive": [
            "you are such an idiot stop talking",
            "shut up you moron nobody likes you",
            "you are stupid and worthless",
            "get lost loser nobody cares about you",
            "you are pathetic and dumb",
        ],
        "hateful": [
            "i hate people like you they should disappear",
            "those people are inferior and disgusting",
            "i despise that entire group of humans",
            "they do not deserve to live among us",
            "we should not allow them in our country",
        ],
    }

    rows: list[dict[str, str]] = []
    for label, samples in templates.items():
        for i in range(n_per_class):
            rows.append({"text": samples[i % len(samples)], "label": label})
    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_dataset_csv(tmp_path: Path) -> Path:
    df = _synthesize_dataset(n_per_class=50)
    csv_path = tmp_path / "messages.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def test_train_grid_search_smoke(
    tiny_dataset_csv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the grid-search path on a tiny synthetic dataset and assert artifacts."""

    monkeypatch.setattr(
        train_mod,
        "PARAM_GRID",
        {
            "features__word__ngram_range": [(1, 1)],
            "features__char__ngram_range": [(3, 4)],
            "clf__alpha": [0.3],
        },
    )
    monkeypatch.setattr(train_mod, "GRID_SEARCH_CV_FOLDS", 2)
    # Keep the test single-process; `n_jobs=-1` requires SC_SEM_NSEMS_MAX,
    # which is not available in some sandboxed CI environments.
    monkeypatch.setattr(train_mod, "GRID_SEARCH_N_JOBS", 1)

    model_path = tmp_path / "classifier.joblib"
    metrics_path = tmp_path / "metrics.json"

    payload = train_mod.train(
        dataset_path=tiny_dataset_csv,
        model_path=model_path,
        metrics_path=metrics_path,
        random_state=0,
        grid_search=True,
    )

    assert model_path.exists(), "fitted pipeline should be persisted"
    assert metrics_path.exists(), "metrics file should be written"

    on_disk = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "grid_search" in on_disk
    assert on_disk["grid_search"]["n_combinations"] == 1
    assert on_disk["grid_search"]["cv_folds"] == 2
    assert on_disk["grid_search"]["scoring"] == "f1_macro"

    best_params = on_disk["grid_search"]["best_params"]
    assert "clf__alpha" in best_params
    assert "features__word__ngram_range" in best_params
    assert "features__char__ngram_range" in best_params
    # Tuples become lists when round-tripped through JSON.
    assert best_params["features__word__ngram_range"] == [1, 1]
    assert best_params["features__char__ngram_range"] == [3, 4]

    # Sanity: the in-memory payload matches what was serialised.
    assert payload["grid_search"]["best_cv_score"] == on_disk["grid_search"]["best_cv_score"]


def test_train_without_grid_search_omits_section(
    tiny_dataset_csv: Path,
    tmp_path: Path,
) -> None:
    """The default path must not write a ``grid_search`` block."""

    model_path = tmp_path / "classifier.joblib"
    metrics_path = tmp_path / "metrics.json"

    train_mod.train(
        dataset_path=tiny_dataset_csv,
        model_path=model_path,
        metrics_path=metrics_path,
        random_state=0,
        grid_search=False,
    )

    on_disk = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "grid_search" not in on_disk
    assert "test" in on_disk and "validation" in on_disk
