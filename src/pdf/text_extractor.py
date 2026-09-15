"""src/pdf/text_extractor.py — native PDF text extraction via PyMuPDF.

Responsibilities
----------------
* Open a PDF and extract text page-by-page using PyMuPDF (fitz).
* Return structured PageText records with character/word counts and an
  image-only flag.
* Intentionally does NOT perform OCR — that is a later phase.
* Clean abstraction so OCR can be plugged in later without touching
  downstream code: callers receive the same PageText regardless of source.

Design notes
------------
* "native text available" means PyMuPDF extracted ≥ MIN_CHARS characters.
* We treat a page as *image-only* when text character count < IMAGE_ONLY_THRESHOLD.
* We record font/language hints where available (PyMuPDF 1.24+).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.logging import get_logger
from src.utils.normalization import normalize_whitespace, count_words, count_chars

log = get_logger(__name__)

# A page with fewer than this many non-whitespace characters is treated as
# image-only (i.e., no usable native text).
IMAGE_ONLY_THRESHOLD: int = 20


@dataclass
class PageText:
    """Text extraction result for one page of a PDF."""

    page_number: int            # 1-indexed
    raw_text: str               # raw extracted text (may be empty)
    character_count: int        # non-whitespace character count
    word_count: int             # approximate word count
    native_text_available: bool # True if character_count >= IMAGE_ONLY_THRESHOLD
    extraction_method: str      # "native" | "ocr" | "none"
    fonts_detected: list[str]   # distinct font names found on page (may be empty)
    has_images: bool            # True if page contains at least one image object
    width_pt: float             # page width in points
    height_pt: float            # page height in points


@dataclass
class DocumentText:
    """Aggregated text extraction result for a whole PDF."""

    path: str                   # absolute path as string
    filename: str
    page_count: int
    pages: list[PageText]       # one entry per page (may be empty on load failure)
    load_error: str = ""        # non-empty if fitz could not open the file

    # ── derived properties ──────────────────────────────────────────────────
    @property
    def pages_with_native_text(self) -> int:
        return sum(1 for p in self.pages if p.native_text_available)

    @property
    def pages_without_native_text(self) -> int:
        return sum(1 for p in self.pages if not p.native_text_available)

    @property
    def total_characters(self) -> int:
        return sum(p.character_count for p in self.pages)

    @property
    def total_words(self) -> int:
        return sum(p.word_count for p in self.pages)

    @property
    def text_coverage(self) -> float:
        """Fraction of pages that have native text (0.0–1.0)."""
        if not self.pages:
            return 0.0
        return self.pages_with_native_text / len(self.pages)

    @property
    def full_text(self) -> str:
        """All pages' raw text joined by form-feed separators."""
        return "\f".join(p.raw_text for p in self.pages)


def extract_document_text(pdf_path: str | Path) -> DocumentText:
    """Open *pdf_path* with PyMuPDF and extract text from every page.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    DocumentText
        Always returns a DocumentText — even on failure (load_error is set).
        A failure on one page is recorded but does not abort processing of
        remaining pages.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (fitz) is required for PDF text extraction. "
            "Install it with: pip install pymupdf"
        ) from exc

    path = Path(pdf_path).resolve()
    filename = path.name

    log.debug("Opening: %s", filename)

    # ── open document ───────────────────────────────────────────────────────
    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        log.error("Cannot open %s: %s", filename, exc)
        return DocumentText(
            path=str(path),
            filename=filename,
            page_count=0,
            pages=[],
            load_error=str(exc),
        )

    page_count = len(doc)
    pages: list[PageText] = []

    for i in range(page_count):
        page_num = i + 1
        try:
            page = doc[i]
            # Extract text; "text" mode returns plain text with spaces
            raw = page.get_text("text")  # type: ignore[attr-defined]
            normalized = normalize_whitespace(raw)

            char_count = count_chars(raw)
            word_count = count_words(raw)
            native_available = char_count >= IMAGE_ONLY_THRESHOLD

            # Font names
            try:
                font_list = [f[3] for f in page.get_fonts(full=True)]
                fonts = sorted(set(font_list))
            except Exception:
                fonts = []

            # Image presence
            try:
                image_list = page.get_images(full=False)
                has_images = len(image_list) > 0
            except Exception:
                has_images = False

            rect = page.rect
            pages.append(
                PageText(
                    page_number=page_num,
                    raw_text=raw,
                    character_count=char_count,
                    word_count=word_count,
                    native_text_available=native_available,
                    extraction_method="native" if native_available else "none",
                    fonts_detected=fonts,
                    has_images=has_images,
                    width_pt=rect.width,
                    height_pt=rect.height,
                )
            )
            log.debug(
                "  Page %d/%d: chars=%d words=%d native=%s images=%s",
                page_num, page_count, char_count, word_count,
                native_available, has_images,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Error reading page %d of %s: %s", page_num, filename, exc)
            pages.append(
                PageText(
                    page_number=page_num,
                    raw_text="",
                    character_count=0,
                    word_count=0,
                    native_text_available=False,
                    extraction_method="none",
                    fonts_detected=[],
                    has_images=False,
                    width_pt=0.0,
                    height_pt=0.0,
                )
            )

    try:
        doc.close()
    except Exception:
        pass

    log.info(
        "%s: %d pages, %d with native text, %d chars",
        filename, page_count, sum(1 for p in pages if p.native_text_available),
        sum(p.character_count for p in pages),
    )

    return DocumentText(
        path=str(path),
        filename=filename,
        page_count=page_count,
        pages=pages,
    )
