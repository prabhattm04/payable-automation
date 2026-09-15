"""src/utils/normalization.py — text normalisation helpers.

Thin wrappers used across the inventory pipeline.  Kept here so
they can be reused by later phases without duplication.
"""
from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including non-breaking spaces) to a single space."""
    text = text.replace("\xa0", " ")  # NBSP → space
    return re.sub(r"\s+", " ", text).strip()


def remove_control_chars(text: str) -> str:
    """Strip ASCII/Unicode control characters that are not printable."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf") or ch in ("\n", "\t")
    )


def count_words(text: str) -> int:
    """Return approximate word count for extracted text."""
    return len(text.split()) if text.strip() else 0


def count_chars(text: str) -> int:
    """Return non-whitespace character count."""
    return len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
