"""Unit tests for the new HateXplain and DailyDialog loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.ml.labels import LABEL_ABUSIVE, LABEL_HATEFUL, LABEL_NORMAL, LABEL_SPAM
from scripts.build_dataset import (
    ALL_SOURCES,
    ENRON_MAX_CHARS,
    _load_dailydialog,
    _load_enron_spam,
    _load_hatexplain,
    build_dataset,
)


def _annot(label: str) -> dict[str, str]:
    return {"label": label, "annotator_id": "stub", "target": "stub"}


def test_load_hatexplain_majority_vote(tmp_path: Path) -> None:
    payload = {
        "post_h1": {
            "annotators": [_annot("hatespeech"), _annot("hatespeech"), _annot("offensive")],
            "post_tokens": ["i", "really", "hate", "those", "people"],
        },
        "post_o1": {
            "annotators": [_annot("offensive"), _annot("offensive"), _annot("normal")],
            "post_tokens": ["you", "are", "a", "moron"],
        },
        "post_n1": {
            "annotators": [_annot("normal"), _annot("normal"), _annot("normal")],
            "post_tokens": ["see", "you", "tomorrow", "morning"],
        },
        "post_skip_short": {
            "annotators": [_annot("normal"), _annot("normal"), _annot("normal")],
            "post_tokens": ["hi"],
        },
        "post_skip_no_annotators": {
            "annotators": [],
            "post_tokens": ["something", "here"],
        },
    }
    f = tmp_path / "hatexplain.json"
    f.write_text(json.dumps(payload), encoding="utf-8")

    df = _load_hatexplain(f)

    assert set(df.columns) == {"text", "label"}
    assert len(df) == 3, "short posts and posts with no annotators must be dropped"
    by_label = dict(zip(df["label"], df["text"], strict=False))
    assert by_label[LABEL_HATEFUL].startswith("really hate") or "hate" in by_label[LABEL_HATEFUL]
    assert by_label[LABEL_ABUSIVE]
    assert by_label[LABEL_NORMAL]


def test_load_hatexplain_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_hatexplain(tmp_path / "nope.json")


def test_load_dailydialog_explodes_to_utterances(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "dialog": [
                ["hello there friend", "how are you doing"],
                ["see you tomorrow", "ok bye"],
                [],
            ],
            "act": [[1, 1], [1, 1], []],
            "emotion": [[0, 0], [0, 0], []],
        }
    )
    f = tmp_path / "daily_dialog.parquet"
    df.to_parquet(f)

    out = _load_dailydialog(f)

    assert set(out.columns) == {"text", "label"}
    assert (out["label"] == LABEL_NORMAL).all()
    assert len(out) == 4
    assert "hello there friend" in out["text"].tolist()


def test_load_dailydialog_unexpected_schema(tmp_path: Path) -> None:
    f = tmp_path / "broken.parquet"
    pd.DataFrame({"foo": [1, 2]}).to_parquet(f)
    with pytest.raises(ValueError, match="unexpected schema"):
        _load_dailydialog(f)


def test_load_dailydialog_preexploded_utterance_schema(tmp_path: Path) -> None:
    """The current HF mirror serves one utterance per row."""

    df = pd.DataFrame(
        {
            "dialog_id": [0, 0, 1],
            "utterance": [
                "see you at the meeting tomorrow",
                "ok sounds good",
                "x",
            ],
            "turn_type": [1, 1, 1],
            "emotion": [0, 0, 0],
        }
    )
    f = tmp_path / "daily_dialog.parquet"
    df.to_parquet(f)

    out = _load_dailydialog(f)

    assert (out["label"] == LABEL_NORMAL).all()
    assert len(out) == 2  # the 1-token "x" row should be dropped
    assert "see you at the meeting tomorrow" in out["text"].tolist()


def test_build_dataset_exclude_rejects_unknown(tmp_path: Path) -> None:
    """`--exclude` should refuse unknown source names early."""

    with pytest.raises(ValueError, match="Unknown source"):
        build_dataset(
            raw_dir=tmp_path,
            processed_dir=tmp_path,
            exclude=("not-a-real-source",),
        )


def test_all_sources_constant_matches_loader_keys() -> None:
    """Guard: ALL_SOURCES is the canonical list used by the CLI."""

    assert set(ALL_SOURCES) == {"sms", "davidson", "hatexplain", "dailydialog", "enron"}


def test_load_enron_spam_happy_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "enron_spam.csv"
    rows = [
        {
            "Message ID": 0,
            "Subject": "lunch tomorrow",
            "Message": "are we still on for noon",
            "Spam/Ham": "ham",
            "Date": "2001-01-01",
        },
        {
            "Message ID": 1,
            "Subject": "",
            "Message": "see attached q4 numbers please review",
            "Spam/Ham": "Ham",  # case-insensitive
            "Date": "2001-01-02",
        },
        {
            "Message ID": 2,
            "Subject": "free viagra",
            "Message": "click here to claim your prize",
            "Spam/Ham": "spam",
            "Date": "2001-01-03",
        },
        {
            "Message ID": 3,
            "Subject": "limited offer",
            "Message": "lowest mortgage rates apply now",
            "Spam/Ham": "SPAM",
            "Date": "2001-01-04",
        },
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    out = _load_enron_spam(csv_path)

    assert len(out) == 4
    assert sorted(out["label"].unique().tolist()) == sorted([LABEL_NORMAL, LABEL_SPAM])
    assert (out["label"] == LABEL_NORMAL).sum() == 2
    assert (out["label"] == LABEL_SPAM).sum() == 2
    text_with_subject = out["text"].iloc[0]
    assert "lunch" in text_with_subject and "noon" in text_with_subject


def test_load_enron_spam_drops_oversized(tmp_path: Path) -> None:
    csv_path = tmp_path / "enron_spam.csv"
    long_body = "x " * (ENRON_MAX_CHARS + 100)
    pd.DataFrame(
        [
            {
                "Message ID": 0,
                "Subject": "ok subject",
                "Message": "short normal length email body",
                "Spam/Ham": "ham",
                "Date": "2001-01-01",
            },
            {
                "Message ID": 1,
                "Subject": "ignore me",
                "Message": long_body,
                "Spam/Ham": "spam",
                "Date": "2001-01-02",
            },
        ]
    ).to_csv(csv_path, index=False)

    out = _load_enron_spam(csv_path)

    assert len(out) == 1
    assert out["label"].iloc[0] == LABEL_NORMAL


def test_load_enron_spam_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "broken.csv"
    pd.DataFrame([{"Subject": "x", "Message": "y"}]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        _load_enron_spam(csv_path)


def test_load_enron_spam_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_enron_spam(tmp_path / "nope.csv")
