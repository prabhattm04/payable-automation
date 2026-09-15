"""src/inventory/classifier.py — heuristic document classification.

This module inspects extracted text and produces INVENTORY SIGNALS only.
Nothing here is a final extraction result.  Every signal is labelled with:
  - value      : the detected value / category
  - method     : "heuristic"
  - confidence : rough estimate (0.0–1.0)

Do NOT:
  - hardcode rules per filename
  - produce final autodraft fields
  - match master data (that is a later phase)
  - perform OCR

Everything in this module is based solely on the text extracted by
src.pdf.text_extractor (native PDF text).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from src.utils.normalization import normalize_whitespace

# ══════════════════════════════════════════════════════════════════════════
# Compiled patterns — defined at module level for reuse
# ══════════════════════════════════════════════════════════════════════════

# Currency symbols and codes
_CURRENCY_SYMBOLS: dict[str, str] = {
    "€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY",
    "₹": "INR", "฿": "THB", "₩": "KRW",
}
_CURRENCY_CODE_RE = re.compile(
    r"\b(EUR|USD|GBP|JPY|INR|THB|KRW|CHF|CAD|AUD|SGD|MYR|GHS|KES|ZAR|TRY|"
    r"DKK|SEK|NOK|PLN|RON|VND|HKD|CNY|NZD)\b"
)
_CURRENCY_TEXT_RE = re.compile(
    r"\b(euro|dollar|pound|yen|baht|ringgit|cedi|shilling|rand|lira|krone|zloty|"
    r"dong|franc|rupee)\b",
    re.IGNORECASE,
)

# Invoice / document indicators
_INVOICE_WORDS_RE = re.compile(
    r"\b(invoice|faktura|lasku|rechnung|factura|facture|fattura|"
    r"счёт|請求書|ใบแจ้งหนี้|فاتورة)\b",
    re.IGNORECASE,
)
_CREDIT_MEMO_RE = re.compile(
    r"\b(credit\s*(?:memo|note)|gutschrift|note\s*de\s*cr[eé]dit|"
    r"nota\s*de\s*cr[eé]dito)\b",
    re.IGNORECASE,
)
_DEBIT_MEMO_RE = re.compile(
    r"\b(debit\s*(?:memo|note)|lastschrift)\b",
    re.IGNORECASE,
)
_RECEIPT_RE = re.compile(
    r"\b(receipt|kuitansi|reçu|quittance|beleg)\b",
    re.IGNORECASE,
)
_PO_RE = re.compile(
    r"\b(purchase\s*order|PO\s*(?:number|no\.?|#)|bestellung|order\s*number)\b",
    re.IGNORECASE,
)
_STATEMENT_RE = re.compile(
    r"\b(statement\s*of\s*account|account\s*statement|kontoauszug)\b",
    re.IGNORECASE,
)
_CUSTOMS_RE = re.compile(
    r"\b(customs|declaration|bill\s*of\s*lading|airway\s*bill|"
    r"packing\s*list|certificate\s*of\s*origin)\b",
    re.IGNORECASE,
)
_REMITTANCE_RE = re.compile(
    r"\b(remittance|payment\s*advice|payment\s*confirmation)\b",
    re.IGNORECASE,
)

# Financial structural signals
_LINE_ITEM_RE = re.compile(
    r"\b(qty|quantity|unit\s*price|item|description|amount|line)\b",
    re.IGNORECASE,
)
_SUBTOTAL_RE = re.compile(r"\b(subtotal|sub[-\s]total|net\s*total)\b", re.IGNORECASE)
_TAX_RE = re.compile(
    r"\b(vat|tax|mwst|gst|sst|hst|nhil|getfund|levy|excise|"
    r"taxe|impuesto|steuer|iva|moms|tva)\b",
    re.IGNORECASE,
)
_WITHHOLDING_RE = re.compile(r"\b(withhold|wht|retention)\b", re.IGNORECASE)
_DISCOUNT_RE = re.compile(r"\b(discount|rabatt|remise|desconto|descuento)\b", re.IGNORECASE)
_FREIGHT_RE = re.compile(r"\b(freight|shipping|transport|delivery\s*charge)\b", re.IGNORECASE)
_TOTAL_RE = re.compile(
    r"\b(total|amount\s*due|due\s*amount|grand\s*total|payable|zu\s*zahlen|"
    r"gesamt|montant\s*total)\b",
    re.IGNORECASE,
)
_PAYMENT_TERMS_RE = re.compile(
    r"\b(payment\s*terms?|due\s*date|due\s*by|zahlungsziel|net\s*\d+|"
    r"\d+\s*days?)\b",
    re.IGNORECASE,
)

# Document reference patterns
_INV_NUMBER_RE = re.compile(
    r"\b(?:invoice\s*(?:no\.?|number|#)|inv(?:oice)?[-#]?\s*)"
    r"([A-Z0-9][A-Z0-9\-/]{2,})\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}|\d{4}[./\-]\d{2}[./\-]\d{2})\b"
)
_PO_NUMBER_RE = re.compile(
    r"\b(?:PO|P\.O\.|purchase\s*order)\s*(?:no\.?|number|#)?\s*"
    r"([A-Z0-9][A-Z0-9\-/]{2,})\b",
    re.IGNORECASE,
)

# Script/language heuristics — Unicode block sampling
def _detect_scripts(text: str) -> set[str]:
    """Return a set of Unicode script families detected in the text."""
    scripts: set[str] = set()
    thai = latin = arabic = cyrillic = cjk = 0
    for ch in text[:5000]:  # sample first 5 000 chars
        cp = ord(ch)
        if 0x0E00 <= cp <= 0x0E7F:
            thai += 1
        elif 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
            latin += 1
        elif 0x0600 <= cp <= 0x06FF:
            arabic += 1
        elif 0x0400 <= cp <= 0x04FF:
            cyrillic += 1
        elif (0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF):
            cjk += 1
    if thai > 5:
        scripts.add("thai")
    if arabic > 5:
        scripts.add("arabic")
    if cyrillic > 5:
        scripts.add("cyrillic")
    if cjk > 5:
        scripts.add("cjk")
    if latin > 10:
        scripts.add("latin")
    return scripts


# Estonian/Finish/German vocabulary snippets for language hints
_ESTONIAN_WORDS = re.compile(r"\b(arve|käibemaks|tasuda|tähtaeg|summa|ostja|müüja|kogus)\b", re.IGNORECASE)
_GERMAN_WORDS = re.compile(r"\b(rechnung|mehrwertsteuer|betrag|lieferant|datum|netto|brutto|mwst)\b", re.IGNORECASE)
_THAI_CHARS = re.compile(r"[\u0E00-\u0E7F]")
_PORTUGUESE_WORDS = re.compile(r"\b(fatura|total|valor|nif|contribuinte|iva|desconto|quantidade)\b", re.IGNORECASE)
_FRENCH_WORDS = re.compile(r"\b(facture|montant|tva|remise|quantité|livraison|échéance)\b", re.IGNORECASE)
_POLISH_WORDS = re.compile(r"\b(faktura|podatek|razem|płatność|nabywca|sprzedawca|ilość)\b", re.IGNORECASE)

# ══════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """A single heuristic signal with provenance."""
    value: object               # detected value (str, bool, list, …)
    method: str = "heuristic"
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)  # snippet(s) that triggered


@dataclass
class DocumentClassification:
    """Full heuristic classification for one document."""

    # ── document family ────────────────────────────────────────────────────
    document_class: Signal = field(default_factory=lambda: Signal("unknown", confidence=0.0))

    # ── payable assessment ─────────────────────────────────────────────────
    payable_status: Signal = field(
        default_factory=lambda: Signal("uncertain", confidence=0.0)
    )

    # ── language/script ────────────────────────────────────────────────────
    scripts_detected: list[str] = field(default_factory=list)
    language_hints: list[str] = field(default_factory=list)

    # ── currencies ─────────────────────────────────────────────────────────
    currency_candidates: list[str] = field(default_factory=list)

    # ── document reference candidates ──────────────────────────────────────
    invoice_number_candidates: list[str] = field(default_factory=list)
    date_candidates: list[str] = field(default_factory=list)
    po_number_candidates: list[str] = field(default_factory=list)

    # ── financial structure signals ────────────────────────────────────────
    financial_structure: dict[str, bool] = field(default_factory=dict)

    # ── master data reference signals ──────────────────────────────────────
    master_data_signals: dict[str, list[str]] = field(default_factory=dict)

    # ── multi-page relationship ────────────────────────────────────────────
    page_relationship: dict = field(default_factory=dict)

    # ── raw keyword hits (for debugging) ──────────────────────────────────
    keyword_hits: dict[str, int] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════
# Main classification function
# ══════════════════════════════════════════════════════════════════════════

def classify_document(
    full_text: str,
    page_texts: list[str],
    filename: str = "",
) -> DocumentClassification:
    """Run all heuristic classifiers on *full_text*.

    Parameters
    ----------
    full_text:
        All pages' text concatenated (may include \\f page separators).
    page_texts:
        Per-page text list (used for multi-page analysis).
    filename:
        Used only for logging, never for classification logic.

    Returns
    -------
    DocumentClassification
        Contains SIGNALS ONLY — not final extraction results.
    """
    cls = DocumentClassification()

    # ── 1. Script and language detection ───────────────────────────────────
    scripts = _detect_scripts(full_text)
    cls.scripts_detected = sorted(scripts)
    cls.language_hints = _detect_language_hints(full_text, scripts)

    # ── 2. Currency detection ───────────────────────────────────────────────
    cls.currency_candidates = _detect_currencies(full_text)

    # ── 3. Document family classification ──────────────────────────────────
    cls.document_class = _classify_document_type(full_text)

    # ── 4. Payable status ──────────────────────────────────────────────────
    cls.payable_status = _assess_payable_status(full_text, cls.document_class)

    # ── 5. Document reference candidates ──────────────────────────────────
    cls.invoice_number_candidates = _find_invoice_numbers(full_text)
    cls.date_candidates = _find_dates(full_text)
    cls.po_number_candidates = _find_po_numbers(full_text)

    # ── 6. Financial structure signals ─────────────────────────────────────
    cls.financial_structure = _detect_financial_structure(full_text)

    # ── 7. Keyword hit counts (for debugging/reporting) ───────────────────
    cls.keyword_hits = _count_keyword_hits(full_text)

    # ── 8. Multi-page relationship ─────────────────────────────────────────
    cls.page_relationship = _analyse_page_relationship(page_texts)

    # ── 9. Master data signals ─────────────────────────────────────────────
    cls.master_data_signals = _detect_master_data_signals(full_text)

    return cls


# ══════════════════════════════════════════════════════════════════════════
# Sub-classifiers
# ══════════════════════════════════════════════════════════════════════════

def _detect_language_hints(text: str, scripts: set[str]) -> list[str]:
    hints: list[str] = []
    if "thai" in scripts:
        hints.append("Thai")
    if _ESTONIAN_WORDS.search(text):
        hints.append("Estonian")
    if _GERMAN_WORDS.search(text):
        hints.append("German")
    if _PORTUGUESE_WORDS.search(text):
        hints.append("Portuguese")
    if _FRENCH_WORDS.search(text):
        hints.append("French")
    if _POLISH_WORDS.search(text):
        hints.append("Polish")
    if not hints and "latin" in scripts:
        hints.append("English (assumed)")
    return hints


def _detect_currencies(text: str) -> list[str]:
    currencies: set[str] = set()

    # Symbol detection
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in text:
            currencies.add(code)

    # ISO code detection
    for m in _CURRENCY_CODE_RE.finditer(text):
        currencies.add(m.group(1).upper())

    # GHS-specific: "Ghana cedi" / "GH₵"
    if "GH₵" in text or re.search(r"\bGH[Cc]\b", text):
        currencies.add("GHS")

    return sorted(currencies)


def _classify_document_type(text: str) -> Signal:
    """Classify document type via heuristic keyword matching."""
    evidence: list[str] = []

    # Score each category
    scores: dict[str, float] = {
        "invoice": 0.0,
        "credit_memo": 0.0,
        "debit_memo": 0.0,
        "purchase_order": 0.0,
        "receipt": 0.0,
        "statement": 0.0,
        "customs_document": 0.0,
        "remittance_document": 0.0,
        "unknown": 0.0,
    }

    if _CREDIT_MEMO_RE.search(text):
        scores["credit_memo"] += 2.0
        evidence.append("credit memo keyword")
    if _DEBIT_MEMO_RE.search(text):
        scores["debit_memo"] += 2.0
        evidence.append("debit memo keyword")
    if _INVOICE_WORDS_RE.search(text):
        scores["invoice"] += 1.5
        evidence.append("invoice keyword")
    if _PO_RE.search(text) and not _INVOICE_WORDS_RE.search(text):
        scores["purchase_order"] += 1.5
        evidence.append("PO keyword (no invoice)")
    if _RECEIPT_RE.search(text):
        scores["receipt"] += 1.0
        evidence.append("receipt keyword")
    if _STATEMENT_RE.search(text):
        scores["statement"] += 1.5
        evidence.append("statement keyword")
    if _CUSTOMS_RE.search(text):
        scores["customs_document"] += 1.5
        evidence.append("customs keyword")
    if _REMITTANCE_RE.search(text):
        scores["remittance_document"] += 1.5
        evidence.append("remittance keyword")

    # Boost invoice score if financial structure is present
    if _TOTAL_RE.search(text) and _TAX_RE.search(text):
        scores["invoice"] += 0.5

    best_class = max(scores, key=lambda k: scores[k])
    best_score = scores[best_class]

    if best_score < 0.5:
        return Signal("unknown", confidence=0.1, evidence=["no strong signals"])

    confidence = min(0.95, best_score / 4.0)
    return Signal(best_class, confidence=round(confidence, 2), evidence=evidence)


def _assess_payable_status(text: str, doc_class: Signal) -> Signal:
    """Produce a payable-status signal.

    Returns one of: "likely_payable", "likely_non_payable", "uncertain"
    """
    doc_type = str(doc_class.value)
    evidence: list[str] = []

    # Non-payable families
    if doc_type in ("purchase_order", "customs_document", "remittance_document"):
        evidence.append(f"document class is {doc_type}")
        return Signal("likely_non_payable", confidence=0.7, evidence=evidence)

    if doc_type == "statement":
        evidence.append("statement — typically non-payable")
        return Signal("likely_non_payable", confidence=0.6, evidence=evidence)

    # Payable indicators
    score = 0.0
    if doc_type in ("invoice", "credit_memo", "debit_memo"):
        score += 1.5
        evidence.append(f"document class is {doc_type}")

    if _TOTAL_RE.search(text):
        score += 0.5
        evidence.append("total/amount-due keyword")
    if _PAYMENT_TERMS_RE.search(text):
        score += 0.5
        evidence.append("payment terms keyword")
    if re.search(r"\b(supplier|vendor|from|bill\s*to|invoice\s*to)\b", text, re.IGNORECASE):
        score += 0.3
        evidence.append("supplier/vendor keyword")
    if re.search(r"\b(invoice\s*(?:no|number|#))\b", text, re.IGNORECASE):
        score += 0.5
        evidence.append("invoice number keyword")

    if score >= 2.0:
        return Signal("likely_payable", confidence=min(0.9, score / 4.0), evidence=evidence)
    if score >= 0.8:
        return Signal("uncertain", confidence=0.5, evidence=evidence)
    return Signal("likely_non_payable", confidence=0.5, evidence=evidence)


def _find_invoice_numbers(text: str) -> list[str]:
    """Extract candidate invoice number strings (de-duplicated, sorted)."""
    candidates: set[str] = set()
    for m in _INV_NUMBER_RE.finditer(text):
        val = m.group(1).strip()
        if len(val) >= 3:
            candidates.add(val)
    return sorted(candidates)[:10]  # cap at 10


def _find_dates(text: str) -> list[str]:
    """Extract candidate date strings."""
    candidates: set[str] = set()
    for m in _DATE_RE.finditer(text):
        candidates.add(m.group(1))
    return sorted(candidates)[:15]


def _find_po_numbers(text: str) -> list[str]:
    """Extract candidate PO number strings."""
    candidates: set[str] = set()
    for m in _PO_NUMBER_RE.finditer(text):
        val = m.group(1).strip()
        if len(val) >= 3:
            candidates.add(val)
    return sorted(candidates)[:10]


def _detect_financial_structure(text: str) -> dict[str, bool]:
    """Flag which financial structural elements appear in the document text."""
    return {
        "has_line_items": bool(_LINE_ITEM_RE.search(text)),
        "has_quantity": bool(re.search(r"\b(qty|quantity|menge|hoeveelheid|antal)\b", text, re.IGNORECASE)),
        "has_unit_price": bool(re.search(r"\b(unit\s*price|unitaire|einzelpreis|price\s*per)\b", text, re.IGNORECASE)),
        "has_line_amount": bool(re.search(r"\b(line\s*(?:total|amount)|amount|ext(?:ension)?\.?\s*price)\b", text, re.IGNORECASE)),
        "has_subtotal": bool(_SUBTOTAL_RE.search(text)),
        "has_discount": bool(_DISCOUNT_RE.search(text)),
        "has_charges": bool(_FREIGHT_RE.search(text) or re.search(r"\b(charges?|extra|surcharge)\b", text, re.IGNORECASE)),
        "has_tax": bool(_TAX_RE.search(text)),
        "has_withholding": bool(_WITHHOLDING_RE.search(text)),
        "has_total": bool(_TOTAL_RE.search(text)),
    }


def _count_keyword_hits(text: str) -> dict[str, int]:
    """Count occurrences of key pattern groups for debugging."""
    return {
        "invoice_words": len(_INVOICE_WORDS_RE.findall(text)),
        "tax_words": len(_TAX_RE.findall(text)),
        "total_words": len(_TOTAL_RE.findall(text)),
        "date_patterns": len(_DATE_RE.findall(text)),
        "currency_codes": len(_CURRENCY_CODE_RE.findall(text)),
    }


def _analyse_page_relationship(page_texts: list[str]) -> dict:
    """Heuristically determine multi-page document relationships."""
    n = len(page_texts)
    if n <= 1:
        return {
            "is_multi_page": False,
            "page_count": n,
            "possible_supporting_pages": [],
            "requires_segmentation": False,
            "multi_currency_pages": False,
            "note": "single page or no text",
        }

    # Check if each page has a separate invoice-like header
    invoice_headers_per_page = []
    currencies_per_page = []
    for pt in page_texts:
        invoice_headers_per_page.append(bool(_INVOICE_WORDS_RE.search(pt)))
        currencies_per_page.append(_detect_currencies(pt))

    # Pages that look like they have their own invoice header
    header_pages = [i + 1 for i, h in enumerate(invoice_headers_per_page) if h]

    # Pages that appear to be supporting (customs, packing list, etc.)
    supporting_pages = [i + 1 for i, pt in enumerate(page_texts) if _CUSTOMS_RE.search(pt)]

    # Multiple distinct currencies across pages?
    all_page_currencies = [set(c) for c in currencies_per_page]
    unique_currencies = set().union(*all_page_currencies) if all_page_currencies else set()
    multi_currency = len(unique_currencies) > 1

    requires_segmentation = len(header_pages) > 1 or len(supporting_pages) > 0

    return {
        "is_multi_page": True,
        "page_count": n,
        "invoice_header_pages": header_pages,
        "possible_supporting_pages": supporting_pages,
        "requires_segmentation": requires_segmentation,
        "multi_currency_pages": multi_currency,
        "currencies_by_page": [list(c) for c in all_page_currencies],
        "note": "heuristic — verify manually",
    }


def _detect_master_data_signals(text: str) -> dict[str, list[str]]:
    """Surface obvious master-data reference candidates.

    IMPORTANT: These are SIGNALS only.  Full matching happens later.
    We never guess a code; we only surface string candidates.
    """
    signals: dict[str, list[str]] = {
        "supplier_name_hints": [],
        "vat_id_candidates": [],
        "iban_candidates": [],
        "po_reference_candidates": [],
        "payment_term_hints": [],
        "buyer_org_hints": [],
    }

    # VAT IDs
    for m in re.finditer(
        r"\b([A-Z]{2}\d{8,12}|[A-Z]{2}[0-9A-Z]{8,15})\b", text
    ):
        val = m.group(1)
        signals["vat_id_candidates"].append(val)

    # IBAN patterns
    for m in re.finditer(r"\b([A-Z]{2}\d{2}[A-Z0-9]{4,30})\b", text):
        val = m.group(1)
        if len(val) >= 15:
            signals["iban_candidates"].append(val)

    # Payment term hints
    for m in re.finditer(r"\b(net\s*\d+|\d+\s*days?|due\s*on\s*receipt|immediately)\b", text, re.IGNORECASE):
        signals["payment_term_hints"].append(m.group(0).lower())

    # Buyer org hints (Bolt entities)
    for m in re.finditer(r"\b(bolt\s+\w+(?:\s+\w+)?|vana[-\s]louna|tallinn\s*hq)\b", text, re.IGNORECASE):
        signals["buyer_org_hints"].append(m.group(0))

    # De-duplicate and cap
    for key in signals:
        signals[key] = sorted(set(signals[key]))[:10]

    return signals
