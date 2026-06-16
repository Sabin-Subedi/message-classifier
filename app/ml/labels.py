"""Label definitions for the 4-class message classifier."""

from __future__ import annotations

from typing import Final

LABEL_NORMAL: Final = "normal"
LABEL_SPAM: Final = "spam"
LABEL_ABUSIVE: Final = "abusive"
LABEL_HATEFUL: Final = "hateful"

LABELS: Final[tuple[str, ...]] = (
    LABEL_NORMAL,
    LABEL_SPAM,
    LABEL_ABUSIVE,
    LABEL_HATEFUL,
)

LABEL_DESCRIPTIONS: Final[dict[str, str]] = {
    LABEL_NORMAL: "Regular conversational messages without harmful or unwanted content.",
    LABEL_SPAM: "Promotional, fraudulent, or otherwise unwanted messages.",
    LABEL_ABUSIVE: "Aggressive, insulting or harassing language directed at others.",
    LABEL_HATEFUL: "Messages expressing hate or discrimination toward a person or group.",
}


def is_valid_label(label: str) -> bool:
    """Return True if `label` is one of the supported classes."""

    return label in LABELS
