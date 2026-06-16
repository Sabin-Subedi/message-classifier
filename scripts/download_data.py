"""Download raw datasets used to train the message classifier.

Sources:

* SMS Spam Collection (UCI ML Repository) — labelled `ham` / `spam`.
  Mirror: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

* Davidson et al., "Hate Speech and Offensive Language" — three classes
  (0=hate, 1=offensive, 2=neither).
  https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv

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
)


def _download(url: str, dest: Path, *, timeout: int = 60) -> None:
    logger.info("Downloading %s -> %s", url, dest)
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fp:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                fp.write(chunk)


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
