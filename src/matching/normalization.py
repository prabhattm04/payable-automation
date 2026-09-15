"""src/matching/normalization.py — Normalization helpers for master-data matching.

Phase 8A: Master-Data Loading and Indexing Foundation.

Design Principles:
1. Preserve Meaningful Identity:
   - Canonical name/text normalization uses Unicode NFKC + casefold + whitespace normalization.
   - Diacritics and accents are explicitly preserved in canonical representations.
   - Reference codes are never aggressively normalized (case and punctuation preserved).
   - Identifiers (VAT IDs, tax numbers) preserve country prefixes and alphanumeric streams.
2. Safe Separation of Secondary Normalization:
   - Accent-folding (ASCII decomposition) is isolated as an optional secondary helper,
     never mixed into canonical indexes.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional


def normalize_text(text: Optional[str]) -> str:
    """Canonical text normalization:
    - Unicode NFKC normalization
    - casefold for locale-independent lowercase matching
    - surrounding and repeated whitespace collapsing
    - PRESERVES diacritics / accents.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.casefold()
    return " ".join(s.split())


def normalize_name(name: Optional[str]) -> str:
    """Canonical entity/supplier/location name normalization.
    Preserves all internal characters, diacritics, and symbols (e.g. '&', '-', '.'),
    while trimming surrounding punctuation/whitespace.
    """
    if name is None:
        return ""
    normalized = normalize_text(name)
    # Strip peripheral punctuation wrappers like quotes or brackets without touching internal chars
    return normalized.strip(" \t\n\r\"'.,;:()[]{}")


def normalize_code(code: Optional[str]) -> str:
    """Conservative code normalization for ERP reference codes (supplier_id, tax code, po_id).
    Strictly strips outer whitespace only. Does NOT alter casing, punctuation, or internal format.
    """
    if code is None:
        return ""
    return str(code).strip()


def normalize_identifier(identifier: Optional[str]) -> str:
    """Identifier normalization for VAT IDs, registration numbers, tax identifiers.
    - Strips surrounding whitespace
    - Uppercases
    - Removes formatting whitespace, hyphens, and dots between alphanumeric segments
      (e.g., 'DE 209 177 122' -> 'DE209177122', 'GHA-VAT-887766' -> 'GHAVAT887766')
    - Preserves the full country prefix and alphanumeric sequence.
    """
    if identifier is None:
        return ""
    s = str(identifier).strip().upper()
    if not s:
        return ""
    # Remove separators while preserving alphanumerics
    return re.sub(r"[\s\-\.]+", "", s)


def normalize_currency(curr: Optional[str]) -> str:
    """Conservative currency code normalization (e.g., ' EUR ' -> 'EUR')."""
    if curr is None:
        return ""
    return str(curr).strip().upper()


def normalize_iban(iban: Optional[str]) -> str:
    """Bank IBAN normalization: strips spaces/hyphens and uppercases."""
    if iban is None:
        return ""
    s = str(iban).strip().upper()
    if not s:
        return ""
    return re.sub(r"[\s\-]+", "", s)


def normalize_rate(rate: Any) -> float:
    """Normalize tax rate percentage to float representation rounded to 4 decimal places.
    Handles numeric types, strings with optional '%', etc.
    """
    if rate is None:
        return 0.0
    if isinstance(rate, (int, float)):
        return round(float(rate), 4)
    s = str(rate).replace("%", "").strip()
    if not s:
        return 0.0
    try:
        return round(float(s), 4)
    except ValueError:
        return 0.0


def normalize_ascii_folding(text: Optional[str]) -> str:
    """Secondary, optional ASCII folding normalization (strips diacritics/accents).
    NOTE: This is NOT used for canonical master indexing; provided only for secondary fallback.
    """
    if text is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return normalize_text(no_accents)
