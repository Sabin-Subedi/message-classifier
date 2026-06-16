"""Download raw datasets used to train the message classifier.

Sources:

* SMS Spam Collection (UCI ML Repository) -- labelled `ham` / `spam`.
  Mirror: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

* Davidson et al., "Hate Speech and Offensive Language" -- three classes
  (0=hate, 1=offensive, 2=neither).
  https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv

* HateXplain (Mathew et al.) -- three classes (hatespeech / offensive / normal),
  three annotators per post. Used to enlarge the `hateful` and `abusive` pools.
  https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json

* DailyDialog (Li et al., 2017) -- ~13k clean dyadic conversations. Every
  utterance becomes a `normal` sample to reduce false positives on benign chat.
  https://huggingface.co/datasets/daily_dialog/resolve/main/data/train-00000-of-00001.parquet

* Enron-Spam (Metsis et al.) -- ~33k labelled emails (~50/50 ham vs spam).
  Used to break the spam-class bottleneck. Subject + Message are concatenated
  by the loader and very long emails are filtered out to stay close to the
  chat-messaging distribution.
  https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/enron_spam_data.csv

The script is idempotent: existing files are reused unless `--force` is
passed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger("download_data")

DEFAULT_RAW_DIR = Path("data/raw")


@dataclass(frozen=True)
class DataSource:
    name: str
    url: str
    filename: str


SOURCES: tuple[DataSource, ...] = (
    DataSource(
        name="sms_spam",
        url="https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
        filename="sms_spam.tsv",
    ),
    DataSource(
        name="davidson_hate_offensive",
        url="https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
        filename="davidson_labeled_data.csv",
    ),
    DataSource(
        name="hatexplain",
        url="https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json",
        filename="hatexplain.json",
    ),
    DataSource(
        # `pixelsandpointers/better_daily_dialog` exposes the standard
        # DailyDialog corpus already exploded into ~87k single utterances
        # (column `utterance`). We use HuggingFace's auto-generated parquet
        # branch directly to avoid going through the redirect API.
        name="daily_dialog",
        url=(
            "https://huggingface.co/datasets/pixelsandpointers/better_daily_dialog/"
            "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
        ),
        filename="daily_dialog.parquet",
    ),
    DataSource(
        # SetFit's mirror of the Metsis et al. Enron-Spam corpus.
        # CSV with columns: Message ID, Subject, Message, Spam/Ham, Date.
        # ~50 MB. Streamed with retries by `_download`.
        name="enron_spam",
        url="https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/enron_spam_data.csv",
        filename="enron_spam.csv",
    ),
)


def _download(url: str, dest: Path, *, timeout: int = 120, retries: int = 3) -> None:
    """Stream `url` to `dest` with a small retry loop for flaky CDNs."""

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        logger.info("Downloading %s -> %s (attempt %d/%d)", url, dest, attempt, retries)
        try:
            with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
                r.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as fp:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fp.write(chunk)
                tmp.replace(dest)
                return
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            logger.warning("Download attempt %d failed: %s", attempt, exc)
    assert last_exc is not None
    raise last_exc


def download_all(raw_dir: Path = DEFAULT_RAW_DIR, *, force: bool = False) -> dict[str, Path]:
    """Download all source datasets, skipping any that already exist."""

    results: dict[str, Path] = {}
    for source in SOURCES:
        target = raw_dir / source.filename
        if target.exists() and not force:
            logger.info("[skip] %s already exists at %s", source.name, target)
        else:
            _download(source.url, target)
        results[source.name] = target
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory to store downloaded files (default: data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    try:
        download_all(args.raw_dir, force=args.force)
    except requests.RequestException as exc:
        logger.error("Download failed: %s", exc)
        return 1
    logger.info("All downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
