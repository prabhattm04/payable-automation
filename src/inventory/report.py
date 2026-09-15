"""src/inventory/report.py — generate human-readable and machine-readable inventory reports.

Inputs
------
A list of InventoryRecord objects (built by inspector.py).

Outputs
-------
* analysis/document_inventory.json  — stable, documented JSON schema
* analysis/document_inventory.csv   — one row per PDF
* analysis/document_inventory.md    — human-readable summary report
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

log = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# JSON Schema version — bump when the schema changes
# ══════════════════════════════════════════════════════════════════════════
SCHEMA_VERSION = "1.0.0"


# ══════════════════════════════════════════════════════════════════════════
# JSON report
# ══════════════════════════════════════════════════════════════════════════

def write_json_inventory(records: list[dict], output_path: str | Path) -> None:
    """Write full inventory to JSON.

    Parameters
    ----------
    records:
        List of inventory record dicts (as produced by inspector._record_to_dict).
    output_path:
        Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "_schema": {
            "version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "description": (
                "Phase 5 document inventory. "
                "All classification fields are heuristic signals — "
                "not final extraction results."
            ),
        },
        "summary": _build_summary(records),
        "documents": records,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    log.info("JSON inventory written: %s", output_path)


def _build_summary(records: list[dict]) -> dict:
    """Compute aggregate statistics across all records."""
    total = len(records)
    failed = sum(1 for r in records if r.get("load_error"))
    total_pages = sum(r.get("page_count", 0) for r in records)
    single_page = sum(1 for r in records if r.get("page_count", 0) == 1)
    multi_page = sum(1 for r in records if r.get("page_count", 0) > 1)

    text_info = [r.get("text", {}) for r in records]
    fully_text = sum(1 for t in text_info if t.get("text_coverage", 0.0) >= 0.9)
    mixed = sum(1 for t in text_info if 0.0 < t.get("text_coverage", 0.0) < 0.9)
    image_only = sum(1 for t in text_info if t.get("text_coverage", 0.0) == 0.0 and not records[0].get("load_error"))
    # re-count cleanly
    image_only_docs = sum(
        1 for r in records
        if not r.get("load_error") and r.get("text", {}).get("text_coverage", 0.0) == 0.0
    )

    # Document classes
    classes = Counter(
        r.get("classification", {}).get("document_class", {}).get("value", "unknown")
        for r in records if not r.get("load_error")
    )

    # Payable status
    payable_status = Counter(
        r.get("classification", {}).get("payable_status", {}).get("value", "uncertain")
        for r in records if not r.get("load_error")
    )

    # Languages
    lang_counter: Counter = Counter()
    for r in records:
        cls = r.get("classification", {})
        for lang in cls.get("language_hints", []):
            lang_counter[lang] += 1

    # Currencies
    currency_counter: Counter = Counter()
    for r in records:
        cls = r.get("classification", {})
        for cur in cls.get("currency_candidates", []):
            currency_counter[cur] += 1

    needs_ocr = sum(
        1 for r in records
        if not r.get("load_error") and r.get("text", {}).get("text_coverage", 0.0) < 1.0
        and r.get("page_count", 0) > 0
    )

    needs_segmentation = sum(
        1 for r in records
        if r.get("classification", {}).get("page_relationship", {}).get("requires_segmentation", False)
    )

    multi_currency_docs = sum(
        1 for r in records
        if len(r.get("classification", {}).get("currency_candidates", [])) > 1
    )

    complex_tax = sum(
        1 for r in records
        if r.get("classification", {}).get("financial_structure", {}).get("has_withholding", False)
        or r.get("classification", {}).get("financial_structure", {}).get("has_discount", False)
        or r.get("classification", {}).get("financial_structure", {}).get("has_charges", False)
    )

    return {
        "total_documents": total,
        "load_failures": failed,
        "total_pages": total_pages,
        "single_page_documents": single_page,
        "multi_page_documents": multi_page,
        "fully_text_readable": fully_text,
        "mixed_text_and_image": mixed,
        "image_only_documents": image_only_docs,
        "documents_needing_ocr": needs_ocr,
        "documents_needing_segmentation": needs_segmentation,
        "multi_currency_documents": multi_currency_docs,
        "complex_tax_structure_documents": complex_tax,
        "document_classes": dict(classes.most_common()),
        "payable_status_distribution": dict(payable_status.most_common()),
        "languages_observed": dict(lang_counter.most_common()),
        "currencies_observed": dict(currency_counter.most_common()),
    }


# ══════════════════════════════════════════════════════════════════════════
# CSV report
# ══════════════════════════════════════════════════════════════════════════

_CSV_FIELDS = [
    "filename",
    "file_size_bytes",
    "page_count",
    "load_error",
    "pages_with_native_text",
    "pages_without_native_text",
    "total_characters",
    "total_words",
    "text_coverage",
    "document_class",
    "doc_class_confidence",
    "payable_status",
    "payable_confidence",
    "scripts_detected",
    "language_hints",
    "currency_candidates",
    "invoice_number_candidates",
    "date_candidates",
    "po_number_candidates",
    "has_line_items",
    "has_quantity",
    "has_unit_price",
    "has_subtotal",
    "has_discount",
    "has_charges",
    "has_tax",
    "has_withholding",
    "has_total",
    "is_multi_page",
    "requires_segmentation",
    "multi_currency_pages",
]


def write_csv_inventory(records: list[dict], output_path: str | Path) -> None:
    """Write one-row-per-PDF CSV summary.

    Parameters
    ----------
    records:
        List of inventory record dicts.
    output_path:
        Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            text = r.get("text", {})
            cls = r.get("classification", {})
            doc_class = cls.get("document_class", {})
            pay_stat = cls.get("payable_status", {})
            fs = cls.get("financial_structure", {})
            pr = cls.get("page_relationship", {})

            row = {
                "filename": r.get("filename", ""),
                "file_size_bytes": r.get("file_size_bytes", ""),
                "page_count": r.get("page_count", ""),
                "load_error": r.get("load_error", ""),
                "pages_with_native_text": text.get("pages_with_native_text", ""),
                "pages_without_native_text": text.get("pages_without_native_text", ""),
                "total_characters": text.get("total_characters", ""),
                "total_words": text.get("total_words", ""),
                "text_coverage": text.get("text_coverage", ""),
                "document_class": doc_class.get("value", ""),
                "doc_class_confidence": doc_class.get("confidence", ""),
                "payable_status": pay_stat.get("value", ""),
                "payable_confidence": pay_stat.get("confidence", ""),
                "scripts_detected": "|".join(cls.get("scripts_detected", [])),
                "language_hints": "|".join(cls.get("language_hints", [])),
                "currency_candidates": "|".join(cls.get("currency_candidates", [])),
                "invoice_number_candidates": "|".join(cls.get("invoice_number_candidates", [])),
                "date_candidates": "|".join(cls.get("date_candidates", []))[:200],
                "po_number_candidates": "|".join(cls.get("po_number_candidates", [])),
                "has_line_items": fs.get("has_line_items", ""),
                "has_quantity": fs.get("has_quantity", ""),
                "has_unit_price": fs.get("has_unit_price", ""),
                "has_subtotal": fs.get("has_subtotal", ""),
                "has_discount": fs.get("has_discount", ""),
                "has_charges": fs.get("has_charges", ""),
                "has_tax": fs.get("has_tax", ""),
                "has_withholding": fs.get("has_withholding", ""),
                "has_total": fs.get("has_total", ""),
                "is_multi_page": pr.get("is_multi_page", ""),
                "requires_segmentation": pr.get("requires_segmentation", ""),
                "multi_currency_pages": pr.get("multi_currency_pages", ""),
            }
            writer.writerow(row)

    log.info("CSV inventory written: %s", output_path)


# ══════════════════════════════════════════════════════════════════════════
# Markdown report
# ══════════════════════════════════════════════════════════════════════════

def write_markdown_report(records: list[dict], output_path: str | Path) -> None:
    """Write human-readable Markdown inventory report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _build_summary(records)
    lines: list[str] = []

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines += [
        "# Phase 5 — Document Inventory Report",
        "",
        f"> **Generated**: {generated_at}  ",
        f"> **Schema version**: {SCHEMA_VERSION}  ",
        "> All classification values are **heuristic signals** — not final extraction results.",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total PDF documents | **{summary['total_documents']}** |",
        f"| Load failures | {summary['load_failures']} |",
        f"| Total pages | {summary['total_pages']} |",
        f"| Single-page documents | {summary['single_page_documents']} |",
        f"| Multi-page documents | {summary['multi_page_documents']} |",
        "",
        "---",
        "",
        "## 2. Text Readability",
        "",
        f"| Category | Count |",
        f"|---|---|",
        f"| Fully text-readable (≥90% pages native) | {summary['fully_text_readable']} |",
        f"| Mixed (some pages image-only) | {summary['mixed_text_and_image']} |",
        f"| Image-only (0% native text) | {summary['image_only_documents']} |",
        f"| Require OCR (any image pages) | {summary['documents_needing_ocr']} |",
        "",
        "---",
        "",
        "## 3. Languages and Scripts Observed",
        "",
    ]

    if summary["languages_observed"]:
        lines.append("| Language Hint | Documents |")
        lines.append("|---|---|")
        for lang, count in summary["languages_observed"].items():
            lines.append(f"| {lang} | {count} |")
    else:
        lines.append("_No language signals detected (may be image-only dataset)._")

    lines += [
        "",
        "---",
        "",
        "## 4. Currencies Observed",
        "",
    ]

    if summary["currencies_observed"]:
        lines.append("| Currency | Documents |")
        lines.append("|---|---|")
        for cur, count in summary["currencies_observed"].items():
            lines.append(f"| {cur} | {count} |")
    else:
        lines.append("_No currency signals detected._")

    lines += [
        "",
        f"**Multi-currency documents**: {summary['multi_currency_documents']}",
        "",
        "---",
        "",
        "## 5. Document Families (Heuristic)",
        "",
        "| Class | Documents |",
        "|---|---|",
    ]
    for cls_val, cnt in summary["document_classes"].items():
        lines.append(f"| {cls_val} | {cnt} |")

    lines += [
        "",
        "---",
        "",
        "## 6. Payable Status Distribution (Heuristic)",
        "",
        "| Status | Documents |",
        "|---|---|",
    ]
    for status, cnt in summary["payable_status_distribution"].items():
        lines.append(f"| {status} | {cnt} |")

    lines += [
        "",
        "---",
        "",
        "## 7. Structural Flags",
        "",
        f"| Flag | Count |",
        f"|---|---|",
        f"| Require page segmentation | {summary['documents_needing_segmentation']} |",
        f"| Multi-currency pages | {summary['multi_currency_documents']} |",
        f"| Complex tax/discount/charge structures | {summary['complex_tax_structure_documents']} |",
        "",
        "---",
        "",
        "## 8. Per-Document Detail",
        "",
    ]

    # Per-document table
    lines += [
        "| # | File | Pages | Text% | Class | Payable? | Currencies | Lang | Issues |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(records, 1):
        text = r.get("text", {})
        cls = r.get("classification", {})
        doc_class = cls.get("document_class", {})
        pay_stat = cls.get("payable_status", {})
        pr = cls.get("page_relationship", {})

        issues = []
        if r.get("load_error"):
            issues.append("LOAD_ERROR")
        if text.get("text_coverage", 1.0) == 0.0 and not r.get("load_error"):
            issues.append("image-only")
        if pr.get("requires_segmentation"):
            issues.append("segmentation-needed")
        if pr.get("multi_currency_pages"):
            issues.append("multi-currency-pages")

        cov_pct = f"{text.get('text_coverage', 0.0)*100:.0f}%"
        currencies = "|".join(cls.get("currency_candidates", []))[:20] or "—"
        langs = "|".join(cls.get("language_hints", []))[:20] or "—"
        issue_str = ", ".join(issues) if issues else "—"

        lines.append(
            f"| {i} | `{r.get('filename','')}` | {r.get('page_count',0)} | {cov_pct} "
            f"| {doc_class.get('value','?')} ({doc_class.get('confidence',0):.0%}) "
            f"| {pay_stat.get('value','?')} "
            f"| {currencies} | {langs} | {issue_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 9. Potentially Difficult Documents",
        "",
    ]

    difficult = [
        r for r in records
        if r.get("load_error")
        or r.get("text", {}).get("text_coverage", 1.0) == 0.0
        or r.get("classification", {}).get("page_relationship", {}).get("requires_segmentation")
        or r.get("classification", {}).get("document_class", {}).get("value") == "unknown"
        or r.get("classification", {}).get("payable_status", {}).get("value") == "uncertain"
    ]

    if difficult:
        for r in difficult:
            text = r.get("text", {})
            cls = r.get("classification", {})
            doc_class = cls.get("document_class", {})
            pay_stat = cls.get("payable_status", {})
            reasons = []
            if r.get("load_error"):
                reasons.append(f"load error: {r['load_error'][:80]}")
            if text.get("text_coverage", 1.0) == 0.0 and not r.get("load_error"):
                reasons.append("image-only (needs OCR)")
            pr = cls.get("page_relationship", {})
            if pr.get("requires_segmentation"):
                reasons.append("may need page segmentation")
            if doc_class.get("value") == "unknown":
                reasons.append("document type unknown")
            if pay_stat.get("value") == "uncertain":
                reasons.append("payable status uncertain")

            lines.append(f"### `{r.get('filename', '')}`")
            lines.append(f"- **Class**: {doc_class.get('value','?')} (conf: {doc_class.get('confidence',0):.0%})")
            lines.append(f"- **Payable**: {pay_stat.get('value','?')}")
            lines.append(f"- **Reasons**: {'; '.join(reasons)}")
            lines.append("")
    else:
        lines.append("_No documents flagged as particularly difficult._")

    lines += [
        "",
        "---",
        "",
        "## 10. Patterns Discovered",
        "",
        _build_patterns_section(records, summary),
        "",
        "---",
        "",
        "## 11. Recommended Phase 6 Architecture",
        "",
        _build_phase6_recommendations(summary),
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    log.info("Markdown report written: %s", output_path)


def _build_patterns_section(records: list[dict], summary: dict) -> str:
    """Generate the patterns-discovered section."""
    parts = []

    # Document prefix families
    prefixes = Counter(r.get("filename", "")[:3] for r in records)
    parts.append("### Document Name Families\n")
    for prefix, cnt in prefixes.most_common():
        parts.append(f"- **`{prefix}*`**: {cnt} document(s)")
    parts.append("")

    # Image-only docs
    image_only_names = [
        r["filename"] for r in records
        if r.get("text", {}).get("text_coverage", 1.0) == 0.0
        and not r.get("load_error") and r.get("page_count", 0) > 0
    ]
    if image_only_names:
        parts.append(f"\n### Image-Only Documents (Require OCR / Vision)\n")
        for n in image_only_names:
            parts.append(f"- `{n}`")

    # Multi-page docs
    multi_page_names = [
        r["filename"] for r in records if r.get("page_count", 0) > 3
    ]
    if multi_page_names:
        parts.append(f"\n### Large Multi-Page Documents (>3 pages)\n")
        for n in multi_page_names:
            pages = next(r.get("page_count", 0) for r in records if r.get("filename") == n)
            parts.append(f"- `{n}` ({pages} pages)")

    # Withholding tax documents
    wht_docs = [
        r["filename"] for r in records
        if r.get("classification", {}).get("financial_structure", {}).get("has_withholding")
    ]
    if wht_docs:
        parts.append(f"\n### Documents with Withholding Tax Signals\n")
        for n in wht_docs:
            parts.append(f"- `{n}`")

    return "\n".join(parts)


def _build_phase6_recommendations(summary: dict) -> str:
    """Generate Phase 6 architecture recommendations based on observed data."""
    recs = []

    recs.append("Based on the Phase 5 inventory findings:\n")

    if summary["image_only_documents"] > 0:
        recs.append(
            f"1. **OCR is required.** {summary['image_only_documents']} document(s) have "
            f"zero native text and cannot be processed without OCR or a Vision LLM. "
            f"Recommended: integrate PyMuPDF page rendering → Tesseract OCR or a "
            f"multimodal LLM (GPT-4o / Claude Vision) for image-heavy pages."
        )

    if summary["documents_needing_segmentation"] > 0:
        recs.append(
            f"2. **Page segmentation logic needed.** {summary['documents_needing_segmentation']} "
            f"document(s) appear to contain multiple payables or supporting pages. "
            f"Phase 6 must implement a page-segmentation step before extraction."
        )

    if summary["multi_currency_documents"] > 0:
        recs.append(
            f"3. **Multi-currency handling.** {summary['multi_currency_documents']} document(s) "
            f"reference more than one currency. The extraction pipeline must NOT assume "
            f"a single currency per file."
        )

    langs = summary.get("languages_observed", {})
    non_english = [l for l in langs if "English" not in l]
    if non_english:
        recs.append(
            f"4. **Multilingual extraction.** Languages beyond English detected: "
            f"{', '.join(non_english)}. The extraction approach must handle "
            f"non-English invoice terminology."
        )

    if summary["complex_tax_structure_documents"] > 0:
        recs.append(
            f"5. **Complex tax/discount/charge structures.** "
            f"{summary['complex_tax_structure_documents']} document(s) contain withholding, "
            f"discounts, or extra charges. ERP field mapping must be faithful — "
            f"line-level vs header-level taxes must NOT be collapsed."
        )

    recs.append(
        "6. **Generalisation architecture.** Per the assignment brief, the pipeline must NOT "
        "grow filename-specific branches. Build a DocumentEvidence intermediate representation "
        "that captures raw observations (page, text span, value, confidence) and feed it into "
        "a generalised financial extraction step."
    )

    return "\n\n".join(recs)
