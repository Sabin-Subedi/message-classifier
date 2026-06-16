"""Train the message classifier and persist artifacts.

Outputs:
    models/classifier.joblib  -- the fitted sklearn Pipeline
    models/metrics.json       -- accuracy + per-class P/R/F1 + confusion matrix
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app import __version__ as APP_VERSION
from app.ml.labels import LABELS
from app.ml.pipeline import build_pipeline

logger = logging.getLogger("train")

DEFAULT_DATASET = Path("data/processed/messages.csv")
DEFAULT_MODEL_PATH = Path("models/classifier.joblib")
DEFAULT_METRICS_PATH = Path("models/metrics.json")
DEFAULT_RANDOM_STATE = 42

# 70/20/10 train/val/test split per report Â§3.9.
VAL_FRACTION = 0.20
TEST_FRACTION = 0.10


def _split(
    df: pd.DataFrame, *, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/20/10 train/val/test split."""

    test_size = TEST_FRACTION
    val_size_relative = VAL_FRACTION / (1 - TEST_FRACTION)

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size_relative,
        stratify=train_val["label"],
        random_state=random_state,
    )
    return train, val, test


def _evaluate(
    pipeline,
    X,
    y_true,
    *,
    split_name: str,
) -> dict:
    """Compute the standard report metrics on a single split."""

    y_pred = pipeline.predict(list(X))
    labels_list = list(LABELS)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class_report = classification_report(
        y_true,
        y_pred,
        labels=labels_list,
        target_names=labels_list,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        lbl: {
            "precision": float(per_class_report[lbl]["precision"]),
            "recall": float(per_class_report[lbl]["recall"]),
            "f1": float(per_class_report[lbl]["f1-score"]),
            "support": int(per_class_report[lbl]["support"]),
        }
        for lbl in labels_list
        if lbl in per_class_report
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels_list).tolist()

    logger.info(
        "[%s] accuracy=%.4f f1_macro=%.4f f1_weighted=%.4f",
        split_name,
        accuracy,
        f1_macro,
        f1_weighted,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels_list,
        "n_samples": int(len(y_true)),
    }


def train(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Train the pipeline end-to-end and persist artifacts."""

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_path}. Run scripts/build_dataset.py first."
        )

    df = pd.read_csv(dataset_path, encoding="utf-8")
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Dataset at {dataset_path} contains no usable rows.")

    logger.info("Loaded %d rows. Class counts: %s", len(df), df["label"].value_counts().to_dict())

    train_df, val_df, test_df = _split(df, random_state=random_state)
    logger.info(
        "Split sizes -- train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    pipeline = build_pipeline()
    logger.info("Fitting pipeline on training split...")
    pipeline.fit(list(train_df["text"]), list(train_df["label"]))

    val_metrics = _evaluate(pipeline, val_df["text"], val_df["label"], split_name="val")
    test_metrics = _evaluate(pipeline, test_df["text"], test_df["label"], split_name="test")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Saved model to %s", model_path)

    metrics_payload = {
        "model_version": APP_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "random_state": random_state,
        "validation": val_metrics,
        "test": test_metrics,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s", metrics_path)

    return metrics_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    try:
        train(
            dataset_path=args.dataset,
            model_path=args.model_path,
            metrics_path=args.metrics_path,
            random_state=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
