"""Re-evaluate a trained classifier and (optionally) plot a confusion matrix.

Loads ``models/classifier.joblib``, re-creates the same stratified 70/20/10
split with the recorded random seed, and reports metrics on the held-out
test set. If matplotlib is available, also writes a PNG of the confusion
matrix to ``models/confusion_matrix.png``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from app.ml.labels import LABELS
from scripts.train import (
    DEFAULT_DATASET,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RANDOM_STATE,
    _evaluate,
    _split,
)

logger = logging.getLogger("evaluate")

DEFAULT_PLOT_PATH = Path("models/confusion_matrix.png")


def _plot_confusion_matrix(cm: list[list[int]], labels: list[str], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed; skipping confusion matrix plot.")
        return

    cm_arr = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix (test split)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    threshold = cm_arr.max() / 2.0 if cm_arr.size else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                format(cm_arr[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > threshold else "black",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", out_path)


def evaluate(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    dataset_path: Path = DEFAULT_DATASET,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    plot_path: Path | None = DEFAULT_PLOT_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}.")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")

    pipeline = joblib.load(model_path)
    df = pd.read_csv(dataset_path, encoding="utf-8")
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)

    _, _, test_df = _split(df, random_state=random_state)
    test_metrics = _evaluate(pipeline, test_df["text"], test_df["label"], split_name="test")

    metrics_payload: dict = {}
    if metrics_path.exists():
        try:
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics_payload = {}
    metrics_payload["test"] = test_metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.info("Updated metrics at %s", metrics_path)

    if plot_path is not None:
        _plot_confusion_matrix(test_metrics["confusion_matrix"], test_metrics["labels"], plot_path)

    return test_metrics


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Where to save the confusion matrix PNG. Use 'none' to skip.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    plot_path: Path | None = args.plot_path
    if plot_path is not None and str(plot_path).lower() == "none":
        plot_path = None

    try:
        evaluate(
            model_path=args.model_path,
            dataset_path=args.dataset,
            metrics_path=args.metrics_path,
            plot_path=plot_path,
            random_state=args.seed,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
