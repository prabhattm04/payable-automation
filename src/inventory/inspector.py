"""src/inventory/inspector.py — main Phase 5 inventory orchestrator.

Usage
-----
    python -m src.inventory.inspector
    python -m src.inventory.inspector --documents documents --output analysis
    python -m src.inventory.inspector --documents path/to/pdfs --output path/to/out --log-level DEBUG

What it does
------------
1. Discovers all PDF files under --documents (filesystem enumeration only).
2. For each PDF, extracts text page-by-page (native PDF text; no OCR).
3. Runs heuristic classification over the extracted text.
4. Assembles a per-document InventoryRecord.
5. Writes:
     <output>/document_inventory.json
     <output>/document_inventory.csv
     <output>/document_inventory.md

One corrupt / unreadable PDF does NOT crash the run.  Its error is logged
and recorded; all remaining documents are processed.

Design constraints (from assignment brief)
------------------------------------------
* No filename-specific rules.
* No hardcoded supplier/PO/tax IDs.
* No OCR in Phase 5.
* Deterministic and repeatable.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from src.utils.logging import configure_logging, get_logger
from src.pdf.loader import discover_pdfs, PdfFile
from src.pdf.text_extractor import extract_document_text, DocumentText
from src.inventory.classifier import classify_document, DocumentClassification
from src.inventory.report import write_json_inventory, write_csv_inventory, write_markdown_report

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Record builder
# ══════════════════════════════════════════════════════════════════════════

def _record_to_dict(
    pdf_file: PdfFile,
    doc_text: DocumentText,
    classification: Optional[DocumentClassification],
) -> dict:
    """Assemble the per-document inventory record dict.

    This is the canonical record shape written to the JSON/CSV outputs.

    Parameters
    ----------
    pdf_file:
        Filesystem metadata from the loader.
    doc_text:
        Text extraction result from text_extractor.
    classification:
        Heuristic classification (None only if extraction failed so badly
        we had nothing to classify).

    Returns
    -------
    dict
        Serialisable dict matching the documented inventory schema.
    """
    pages_with_native = doc_text.pages_with_native_text
    pages_without_native = doc_text.pages_without_native_text
    total_chars = doc_text.total_characters
    total_words = doc_text.total_words
    coverage = round(doc_text.text_coverage, 4)

    pages_detail = [
        {
            "page_number": p.page_number,
            "native_text_available": p.native_text_available,
            "character_count": p.character_count,
            "word_count": p.word_count,
            "extraction_method": p.extraction_method,
            "has_images": p.has_images,
            "width_pt": round(p.width_pt, 1),
            "height_pt": round(p.height_pt, 1),
        }
        for p in doc_text.pages
    ]

    text_block = {
        "pages_with_native_text": pages_with_native,
        "pages_without_native_text": pages_without_native,
        "total_characters": total_chars,
        "total_words": total_words,
        "text_coverage": coverage,
    }

    if classification is not None:
        cls_block = {
            "document_class": {
                "value": classification.document_class.value,
                "method": classification.document_class.method,
                "confidence": classification.document_class.confidence,
                "evidence": classification.document_class.evidence,
            },
            "payable_status": {
                "value": classification.payable_status.value,
                "method": classification.payable_status.method,
                "confidence": classification.payable_status.confidence,
                "evidence": classification.payable_status.evidence,
            },
            "scripts_detected": classification.scripts_detected,
            "language_hints": classification.language_hints,
            "currency_candidates": classification.currency_candidates,
            "invoice_number_candidates": classification.invoice_number_candidates,
            "date_candidates": classification.date_candidates,
            "po_number_candidates": classification.po_number_candidates,
            "financial_structure": classification.financial_structure,
            "master_data_signals": classification.master_data_signals,
            "page_relationship": classification.page_relationship,
            "keyword_hits": classification.keyword_hits,
        }
    else:
        cls_block = {}

    return {
        "file": pdf_file.relative_path.replace("\\", "/"),
        "filename": pdf_file.filename,
        "stem": pdf_file.stem,
        "file_size_bytes": pdf_file.file_size_bytes,
        "page_count": doc_text.page_count,
        "load_error": doc_text.load_error,
        "text": text_block,
        "pages": pages_detail,
        "classification": cls_block,
    }


# ══════════════════════════════════════════════════════════════════════════
# Core inspection function
# ══════════════════════════════════════════════════════════════════════════

def inspect_pdf(pdf_file: PdfFile) -> dict:
    """Fully inspect one PDF file and return its inventory record dict.

    This function is intentionally self-contained: it catches ALL exceptions
    so that one bad document never aborts the run.

    Parameters
    ----------
    pdf_file:
        A PdfFile descriptor from the loader.

    Returns
    -------
    dict
        The inventory record (with load_error set if anything failed).
    """
    log.info("Inspecting: %s (%d bytes)", pdf_file.filename, pdf_file.file_size_bytes)

    # ── Step 1: extract text ───────────────────────────────────────────────
    try:
        doc_text = extract_document_text(pdf_file.path)
    except Exception as exc:  # noqa: BLE001
        log.error("Fatal extraction error for %s: %s", pdf_file.filename, exc)
        log.debug(traceback.format_exc())
        # Return a minimal record with the error flagged
        from src.pdf.text_extractor import DocumentText
        doc_text = DocumentText(
            path=str(pdf_file.path),
            filename=pdf_file.filename,
            page_count=0,
            pages=[],
            load_error=str(exc),
        )

    # ── Step 2: classify ───────────────────────────────────────────────────
    classification: Optional[DocumentClassification] = None
    if not doc_text.load_error and doc_text.page_count > 0:
        try:
            full_text = doc_text.full_text
            page_texts = [p.raw_text for p in doc_text.pages]
            classification = classify_document(
                full_text=full_text,
                page_texts=page_texts,
                filename=pdf_file.filename,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Classification error for %s: %s", pdf_file.filename, exc)
            log.debug(traceback.format_exc())
            # classification stays None — record will have empty cls_block

    # ── Step 3: assemble record ────────────────────────────────────────────
    record = _record_to_dict(pdf_file, doc_text, classification)
    return record


# ══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════

def run_inventory(documents_dir: str, output_dir: str) -> list[dict]:
    """Discover, inspect, classify, and report all PDFs.

    Parameters
    ----------
    documents_dir:
        Root directory containing PDF files.
    output_dir:
        Directory where inventory outputs will be written.

    Returns
    -------
    list[dict]
        All inventory record dicts (also written to disk).
    """
    docs_path = Path(documents_dir).resolve()
    out_path = Path(output_dir).resolve()

    # ── 1. Discover PDFs ───────────────────────────────────────────────────
    log.info("=== Phase 5 Document Inventory ===")
    log.info("Documents directory: %s", docs_path)
    log.info("Output directory:    %s", out_path)

    try:
        pdf_files = discover_pdfs(docs_path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        log.critical("Cannot access documents directory: %s", exc)
        sys.exit(1)

    if not pdf_files:
        log.warning("No PDF files found in %s — nothing to inventory.", docs_path)
        return []

    log.info("Found %d PDF file(s) to inspect.", len(pdf_files))

    # ── 2. Inspect each PDF ────────────────────────────────────────────────
    records: list[dict] = []
    failed = 0
    for i, pdf_file in enumerate(pdf_files, 1):
        log.info("[%d/%d] %s", i, len(pdf_files), pdf_file.filename)
        try:
            record = inspect_pdf(pdf_file)
            if record.get("load_error"):
                failed += 1
            records.append(record)
        except Exception as exc:  # noqa: BLE001
            # Ultimate safety net — this should never be reached because
            # inspect_pdf itself catches all exceptions, but just in case.
            log.error("Unexpected error processing %s: %s", pdf_file.filename, exc)
            log.debug(traceback.format_exc())
            failed += 1
            # Append a minimal error record so the file is still represented
            records.append({
                "file": pdf_file.relative_path.replace("\\", "/"),
                "filename": pdf_file.filename,
                "stem": pdf_file.stem,
                "file_size_bytes": pdf_file.file_size_bytes,
                "page_count": 0,
                "load_error": str(exc),
                "text": {},
                "pages": [],
                "classification": {},
            })

    log.info("Inspection complete. %d OK, %d failed.", len(records) - failed, failed)

    # ── 3. Write reports ───────────────────────────────────────────────────
    out_path.mkdir(parents=True, exist_ok=True)

    json_out = out_path / "document_inventory.json"
    csv_out = out_path / "document_inventory.csv"
    md_out = out_path / "document_inventory.md"

    write_json_inventory(records, json_out)
    write_csv_inventory(records, csv_out)
    write_markdown_report(records, md_out)

    log.info("=== Inventory complete ===")
    log.info("  JSON: %s", json_out)
    log.info("  CSV:  %s", csv_out)
    log.info("  MD:   %s", md_out)

    return records


# ══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.inventory.inspector",
        description=(
            "Phase 5 — Document Inventory Pipeline\n"
            "Discovers all PDFs under --documents, inspects them for text content,\n"
            "runs heuristic classification, and writes inventory reports to --output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--documents",
        default="documents",
        metavar="DIR",
        help="Root directory containing PDF files (default: documents/)",
    )
    parser.add_argument(
        "--output",
        default="analysis",
        metavar="DIR",
        help="Output directory for inventory reports (default: analysis/)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="FILE",
        help="Optional: also write logs to this file.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(level=level, log_file=args.log_file)

    run_inventory(documents_dir=args.documents, output_dir=args.output)


if __name__ == "__main__":
    main()
