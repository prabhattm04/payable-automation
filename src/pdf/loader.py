"""src/pdf/loader.py — PDF discovery and file-level metadata.

Responsibilities
----------------
* Recursively (or shallowly) enumerate PDF files under a root directory.
* Return lightweight PdfFile dataclass records (path, size, etc.).
* No content reading here — keeps loading concerns separate from text extraction.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PdfFile:
    """Metadata for one discovered PDF file."""

    path: Path                  # absolute Path object
    relative_path: str          # relative to the documents root
    filename: str               # stem + suffix  (e.g. "INV-01.pdf")
    stem: str                   # stem only      (e.g. "INV-01")
    file_size_bytes: int        # file size in bytes

    # Page count is filled later by the text extractor
    page_count: int = 0

    # Error flag — set if the file cannot be opened at all
    load_error: str = ""

    def __repr__(self) -> str:
        return f"PdfFile({self.filename!r}, {self.file_size_bytes} bytes)"


def discover_pdfs(documents_dir: str | Path, recursive: bool = True) -> list[PdfFile]:
    """Return a sorted list of all PDF files found under *documents_dir*.

    Parameters
    ----------
    documents_dir:
        Root directory to search.
    recursive:
        If True (default), walk subdirectories as well.

    Returns
    -------
    list[PdfFile]
        Sorted deterministically by relative path (case-insensitive on Windows).
        Never includes placeholder filenames — only files that physically exist.
    """
    root = Path(documents_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    pdf_files: list[PdfFile] = []

    iterator: Iterator[Path]
    if recursive:
        iterator = root.rglob("*.pdf")
    else:
        iterator = root.glob("*.pdf")

    for path in iterator:
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            log.warning("Cannot stat %s: %s", path, exc)
            size = 0

        rel = str(path.relative_to(root))
        pdf_files.append(
            PdfFile(
                path=path,
                relative_path=rel,
                filename=path.name,
                stem=path.stem,
                file_size_bytes=size,
            )
        )

    # Deterministic sort — case-insensitive relative path
    pdf_files.sort(key=lambda f: f.relative_path.lower())

    log.info("Discovered %d PDF file(s) under %s", len(pdf_files), root)
    return pdf_files
