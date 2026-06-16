"""Combine raw datasets into a single balanced 4-class messages.csv.

Final label mapping
-------------------
* `normal`   -- SMS `ham`
              + Davidson `2` (neither hate nor offensive)
              + HateXplain `normal`
              + DailyDialog utterances (clean conversational chat)
              + Enron-Spam `ham`
* `spam`     -- SMS `spam`
              + Enron-Spam `spam`
* `abusive`  -- Davidson `1` (offensive language)
              + HateXplain `offensive`
* `hateful`  -- Davidson `0` (hate speech)
              + HateXplain `hatespeech`

Per-class sample target defaults to 3000 (matching report 3.8). When a
source class is too small the script downsamples to whatever the smallest
class actually contains and logs a warning. Use `--target` to override
or `--strict` to abort if any class falls short.

Output: ``data/processed/messages.csv`` with columns ``text,label``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from app.ml.labels import LABEL_ABUSIVE, LABEL_HATEFUL, LABEL_NORMAL, LABEL_SPAM, LABELS

ALL_SOURCES: tuple[str, ...] = ("sms", "davidson", "hatexplain", "dailydialog", "enron")

ENRON_MIN_CHARS = 10
ENRON_MAX_CHARS = 800

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


def _load_hatexplain(path: Path) -> pd.DataFrame:
    """Load HateXplain (Mathew et al.).

    The raw file is a JSON dict keyed by post id. Each record contains a list
    of three annotators with `label` in {hatespeech, offensive, normal} and a
    pre-tokenized `post_tokens` list. We pick the gold label by majority vote
    and reconstruct the post text by joining the tokens with spaces.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing HateXplain file: {path}. Run scripts/download_data.py first."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    label_map = {
        "hatespeech": LABEL_HATEFUL,
        "offensive": LABEL_ABUSIVE,
        "normal": LABEL_NORMAL,
    }

    rows: list[dict[str, str]] = []
    for rec in raw.values():
        labels = [a.get("label") for a in rec.get("annotators", []) if a.get("label")]
        if not labels:
            continue
        majority = Counter(labels).most_common(1)[0][0]
        mapped = label_map.get(majority)
        if mapped is None:
            continue
        tokens = rec.get("post_tokens") or []
        if not tokens:
            continue
        rows.append({"text": " ".join(tokens), "label": mapped})

    if not rows:
        return pd.DataFrame(columns=["text", "label"])

    df = pd.DataFrame(rows)
    df["text"] = df["text"].map(_normalize_text)
    df = df[df["text"].str.split().str.len() >= MIN_TOKENS]
    return df[["text", "label"]]


def _load_dailydialog(path: Path) -> pd.DataFrame:
    """Load DailyDialog as a flat list of `normal` utterances.

    Two parquet schemas are supported so the loader keeps working as the
    upstream HuggingFace mirror evolves:

    * Pre-exploded (e.g. ``pixelsandpointers/better_daily_dialog``):
      one row per utterance with column ``utterance`` (or ``text``).
    * Original (``dialog: list[str]`` plus ``act`` / ``emotion``):
      we explode the dialog list ourselves.

    Reading parquet requires `pyarrow`, provided by the `data` dependency
    group (see ``pyproject.toml``).
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing DailyDialog file: {path}. Run scripts/download_data.py first."
        )

    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}

    if "utterance" in cols:
        utterances = df[cols["utterance"]].dropna().astype(str).tolist()
    elif "text" in cols:
        utterances = df[cols["text"]].dropna().astype(str).tolist()
    elif "dialog" in cols:
        utterances = df[cols["dialog"]].explode().dropna().astype(str).tolist()
    else:
        raise ValueError(
            f"DailyDialog file at {path} has unexpected schema; got {list(df.columns)}"
        )

    if not utterances:
        return pd.DataFrame(columns=["text", "label"])

    out = pd.DataFrame({"text": utterances, "label": LABEL_NORMAL})
    out["text"] = out["text"].map(_normalize_text)
    out = out[out["text"].str.split().str.len() >= MIN_TOKENS]
    return out[["text", "label"]]


def _load_enron_spam(path: Path) -> pd.DataFrame:
    """Load Enron-Spam (Metsis et al.) from the SetFit CSV mirror.

    Schema: ``Message ID, Subject, Message, Spam/Ham, Date``. Subject and
    Message are concatenated; rows whose final character length falls
    outside ``[ENRON_MIN_CHARS, ENRON_MAX_CHARS]`` are dropped to stay
    closer to the chat-messaging distribution and avoid letting long
    multi-paragraph emails dominate the spam class.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Enron-Spam file: {path}. Run scripts/download_data.py first."
        )

    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    needed = {"Subject", "Message", "Spam/Ham"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"Enron-Spam at {path} missing columns: {sorted(missing)}; got {list(df.columns)}"
        )

    subject = df["Subject"].fillna("").astype(str).str.strip()
    body = df["Message"].fillna("").astype(str).str.strip()
    text = (subject + " " + body).str.strip()
    text = text.map(_normalize_text)

    label = df["Spam/Ham"].astype(str).str.lower().map(
        {"ham": LABEL_NORMAL, "spam": LABEL_SPAM}
    )

    out = pd.DataFrame({"text": text, "label": label}).dropna(subset=["label"])
    out = out[out["text"].str.len().between(ENRON_MIN_CHARS, ENRON_MAX_CHARS)]
    out = out[out["text"].str.split().str.len() >= MIN_TOKENS]
    return out[["text", "label"]]


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
    exclude: tuple[str, ...] = (),
) -> Path:
    """Assemble the combined CSV and return its path.

    `exclude` accepts any of the names in :data:`ALL_SOURCES` to skip
    individual data sources -- handy for ablation runs without re-downloading.
    """

    excluded = {s.lower() for s in exclude}
    unknown = excluded - set(ALL_SOURCES)
    if unknown:
        raise ValueError(
            f"Unknown source(s) in --exclude: {sorted(unknown)}. Allowed: {ALL_SOURCES}."
        )

    frames: list[pd.DataFrame] = []
    if "sms" not in excluded:
        frames.append(_load_sms_spam(raw_dir / "sms_spam.tsv"))
    if "davidson" not in excluded:
        frames.append(_load_davidson(raw_dir / "davidson_labeled_data.csv"))
    if "hatexplain" not in excluded:
        frames.append(_load_hatexplain(raw_dir / "hatexplain.json"))
    if "dailydialog" not in excluded:
        frames.append(_load_dailydialog(raw_dir / "daily_dialog.parquet"))
    if "enron" not in excluded:
        frames.append(_load_enron_spam(raw_dir / "enron_spam.csv"))

    if not frames:
        raise ValueError("All sources excluded; nothing to build.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["text", "label"])
    combined["text"] = combined["text"].map(_normalize_text)
    combined = combined[combined["text"].str.len() > 0]
    combined = combined.drop_duplicates(subset=["text"])

    logger.info(
        "Raw class counts (sources=%s): %s",
        sorted(set(ALL_SOURCES) - excluded),
        combined["label"].value_counts().to_dict(),
    )

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
    parser.add_argument(
        "--exclude",
        nargs="*",
        choices=list(ALL_SOURCES),
        default=[],
        help="Skip one or more sources (handy for ablation runs).",
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
            exclude=tuple(args.exclude),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Failed to build dataset: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
