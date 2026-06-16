"""Combine raw datasets into a single balanced 4-class messages.csv.

Final label mapping
-------------------
* `normal`   -- SMS `ham` + Davidson `2` (neither hate nor offensive)
* `spam`     -- SMS `spam`
* `abusive`  -- Davidson `1` (offensive language)
* `hateful`  -- Davidson `0` (hate speech)

Per-class sample target defaults to 3000 (matching report Â§3.8). When a
source class is too small the script downsamples to whatever the smallest
class actually contains and logs a warning. Use `--target` to override
or `--strict` to abort if any class falls short.

Output: ``data/processed/messages.csv`` with columns ``text,label``.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from app.ml.labels import LABEL_ABUSIVE, LABEL_HATEFUL, LABEL_NORMAL, LABEL_SPAM, LABELS

logger = logging.getLogger("build_dataset")

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_TARGET_PER_CLASS = 3000
DEFAULT_RANDOM_STATE = 42
MIN_TOKENS = 2

_WHITESPACE = re.compile(r"\s+")


def _normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    s = str(text).strip()
    s = _WHITESPACE.sub(" ", s)
    return s


def _load_sms_spam(path: Path) -> pd.DataFrame:
    """Load SMS Spam Collection (label\\ttext, no header)."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing SMS Spam file: {path}. Run scripts/download_data.py first."
        )
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
    df["text"] = df["text"].map(_normalize_text)
    df = df[df["text"].str.split().str.len() >= MIN_TOKENS]
    df["label"] = df["label"].str.lower().map({"ham": LABEL_NORMAL, "spam": LABEL_SPAM})
    df = df.dropna(subset=["label"])
    return df[["text", "label"]]


def _load_davidson(path: Path) -> pd.DataFrame:
    """Load Davidson hate-speech/offensive dataset.

    Original columns: count, hate_speech, offensive_language, neither, class, tweet.
    `class`: 0=hate, 1=offensive, 2=neither.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Davidson file: {path}. Run scripts/download_data.py first."
        )
    df = pd.read_csv(path, encoding="utf-8")
    if "class" not in df.columns or "tweet" not in df.columns:
        raise ValueError(
            f"Davidson file at {path} does not have expected columns; got {list(df.columns)}"
        )
    mapping = {0: LABEL_HATEFUL, 1: LABEL_ABUSIVE, 2: LABEL_NORMAL}
    df = df[["tweet", "class"]].rename(columns={"tweet": "text"})
    df["label"] = df["class"].map(mapping)
    df = df.drop(columns=["class"])
    df["text"] = df["text"].map(_normalize_text)
    df = df[df["text"].str.split().str.len() >= MIN_TOKENS]
    df = df.dropna(subset=["label"])
    return df[["text", "label"]]


def _balance(
    df: pd.DataFrame,
    *,
    target_per_class: int,
    random_state: int,
    strict: bool,
) -> pd.DataFrame:
    """Sample up to `target_per_class` rows per label.

    Falls back to the size of the smallest class so that the resulting
    dataset stays balanced. When `strict=True`, raises if any class is
    below `target_per_class`.
    """

    counts = df["label"].value_counts().to_dict()
    missing = [lbl for lbl in LABELS if lbl not in counts]
    if missing:
        raise ValueError(f"Missing rows for labels: {missing}")

    smallest = min(counts[lbl] for lbl in LABELS)
    if smallest < target_per_class:
        msg = (
            f"Target {target_per_class}/class but smallest class has {smallest} rows; "
            f"counts={counts}."
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg + " Falling back to smallest-class sample size.")
        sample_size = smallest
    else:
        sample_size = target_per_class

    parts: list[pd.DataFrame] = []
    for lbl in LABELS:
        subset = df[df["label"] == lbl]
        sampled = subset.sample(n=sample_size, random_state=random_state)
        parts.append(sampled)

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def build_dataset(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    target_per_class: int = DEFAULT_TARGET_PER_CLASS,
    random_state: int = DEFAULT_RANDOM_STATE,
    strict: bool = False,
) -> Path:
    """Assemble the combined CSV and return its path."""

    sms = _load_sms_spam(raw_dir / "sms_spam.tsv")
    davidson = _load_davidson(raw_dir / "davidson_labeled_data.csv")

    combined = pd.concat([sms, davidson], ignore_index=True)
    combined = combined.dropna(subset=["text", "label"])
    combined["text"] = combined["text"].map(_normalize_text)
    combined = combined[combined["text"].str.len() > 0]
    combined = combined.drop_duplicates(subset=["text"])

    logger.info("Raw class counts: %s", combined["label"].value_counts().to_dict())

    balanced = _balance(
        combined,
        target_per_class=target_per_class,
        random_state=random_state,
        strict=strict,
    )

    logger.info("Final class counts: %s", balanced["label"].value_counts().to_dict())

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "messages.csv"
    balanced.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote %d rows to %s", len(balanced), out_path)
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET_PER_CLASS,
        help=f"Target samples per class (default: {DEFAULT_TARGET_PER_CLASS}).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any class has fewer rows than --target.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    try:
        build_dataset(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            target_per_class=args.target,
            random_state=args.seed,
            strict=args.strict,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Failed to build dataset: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
