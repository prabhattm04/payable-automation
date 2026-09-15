"""src/understanding/page_classifier.py — Page-Level Document Understanding Layer.

Phase 7B: Determines the page role and payable relevance of individual document
pages using Phase 7A Evidence + Provenance without cross-page grouping,
accounting calculations, or Qwen LLM routing.

Key Architecture & Principles
-----------------------------
1. Controlled Vocabularies:
   - PageRole: Semantic type/state of this individual page.
   - PayableRelevance: Orthogonal financial payable relevance.
2. Complete Orthogonality:
   PageRole and PayableRelevance are evaluated independently.
   An invoice page can be payable_candidate, ambiguous, or supporting.
3. No Fabricated Confidence:
   Deterministic rule-based classification produces uncalibrated results.
   `confidence` strictly remains None (null in JSON).
4. Data-Driven Signal Vocabularies:
   Terminology is organized into categorized semantic dictionaries covering
   multilingual patterns (EN, DE, FR, PT, ES, ET, TH).
5. PO Document vs PO Reference Disambiguation:
   Customer PO references (e.g. "Customer PO: 12345", "PO Number: 88") on an
   invoice are recognized as metadata references, NOT purchase order documents.
6. Full Traceability & Explainability:
   PageUnderstanding links:
   classification -> decision_reasons -> signals -> evidence_ids -> original Evidence.
7. Zero Document-Specific Rules:
   The classifier is invariant to document IDs, filenames, or page-number shortcuts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.understanding.evidence import Evidence, PageEvidence
from src.utils.logging import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Controlled Vocabularies
# ══════════════════════════════════════════════════════════════════════════

class PageRole(str, Enum):
    """Controlled vocabulary for page-level document role/state."""
    INVOICE = "invoice"
    CREDIT_MEMO = "credit_memo"
    DEBIT_MEMO = "debit_memo"
    CONTINUATION = "continuation"
    SUPPORTING_DOCUMENT = "supporting_document"
    PURCHASE_ORDER = "purchase_order"
    RECEIPT = "receipt"
    REMITTANCE = "remittance"
    UNKNOWN = "unknown"


class PayableRelevance(str, Enum):
    """Orthogonal assessment of actionable payable status for this page."""
    PAYABLE_CANDIDATE = "payable_candidate"
    SUPPORTING = "supporting"
    NON_PAYABLE = "non_payable"
    AMBIGUOUS = "ambiguous"


# ══════════════════════════════════════════════════════════════════════════
# Data-Driven Multilingual Semantic Pattern Dictionaries
# ══════════════════════════════════════════════════════════════════════════

# 1. Invoice Document Titles / Explicit Header Terms
_INVOICE_TITLE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:tax\s+)?invoice\b", re.IGNORECASE),
    re.compile(r"\bcommercial\s+invoice\b", re.IGNORECASE),
    re.compile(r"\bcustoms\s+(?:consolidated|detailed)?\s*invoice\b", re.IGNORECASE),
    re.compile(r"\bbill\s+of\s+sale\b", re.IGNORECASE),
    re.compile(r"\brechnung\b", re.IGNORECASE),
    re.compile(r"\bsteuerrechnung\b", re.IGNORECASE),
    re.compile(r"\bfaktura\b", re.IGNORECASE),
    re.compile(r"\bfacture(?:\s+commerciale)?\b", re.IGNORECASE),
    re.compile(r"\b(?:total\s+da\s+)?fa[ck]tura\b", re.IGNORECASE),
    re.compile(r"\barve\b", re.IGNORECASE),
    re.compile(r"ใบแจ้งหนี้", re.IGNORECASE),
    re.compile(r"ใบวางบิล", re.IGNORECASE),
    re.compile(r"ใบกำกับภาษี", re.IGNORECASE),
]

# 2. Invoice Structural Metadata (Bill-to, Invoice No, Due Date)
_INVOICE_METADATA_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:invoice|rechnung|factura|facture|arve)\s*(?:no\.?|nr\.?|number|#)", re.IGNORECASE),
    re.compile(r"\b(?:invoice|rechnungs|facture)\s*date\b", re.IGNORECASE),
    re.compile(r"\brechnungsdatum\b", re.IGNORECASE),
    re.compile(r"\b(?:bill\s*to|billed\s*to|invoice\s*to)\b", re.IGNORECASE),
    re.compile(r"\b(?:due\s*date|payment\s*terms?|fälligkeitsdatum|zahlungsziel)\b", re.IGNORECASE),
    re.compile(r"\bdata\s+de\s+vencimento\b", re.IGNORECASE),
    re.compile(r"\bmaksetähtaeg\b", re.IGNORECASE),
    re.compile(r"\bkunden-?nr\.?\b", re.IGNORECASE),
]

# 3. Credit Memo Patterns
_CREDIT_MEMO_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bcredit\s*(?:memo|note|advice)\b", re.IGNORECASE),
    re.compile(r"\bgutschrift\b", re.IGNORECASE),
    re.compile(r"\bnote\s*de\s*cr[eé]dit\b", re.IGNORECASE),
    re.compile(r"\bnota\s*de\s*cr[eé]dito\b", re.IGNORECASE),
    re.compile(r"\bkreeditarve\b", re.IGNORECASE),
    re.compile(r"ใบลดหนี้", re.IGNORECASE),
]

# 4. Debit Memo Patterns
_DEBIT_MEMO_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bdebit\s*(?:memo|note)\b", re.IGNORECASE),
    re.compile(r"\blastschrift\b", re.IGNORECASE),
    re.compile(r"\bnote\s*de\s*d[eé]bit\b", re.IGNORECASE),
    re.compile(r"\bnota\s*de\s*d[eé]bito\b", re.IGNORECASE),
    re.compile(r"ใบเพิ่มหนี้", re.IGNORECASE),
]

# 5. Purchase Order Document Heading (Standalone Title vs Reference)
_PO_DOC_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*purchase\s*order\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bpurchase\s*order\s*(?:header|summary|form|sheet)\b", re.IGNORECASE),
    re.compile(r"^\s*bestellung\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bbon\s*de\s*commande\b", re.IGNORECASE),
    re.compile(r"\borden\s*de\s*compra\b", re.IGNORECASE),
    re.compile(r"ใบสั่งซื้อ", re.IGNORECASE),
]

# 6. Purchase Order Reference Patterns (commonly found on invoices!)
_PO_REFERENCE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:customer|client|our|your)?\s*(?:po|p\.o\.|purchase\s*order)\s*(?:#|no\.?|number)\b", re.IGNORECASE),
    re.compile(r"\bpo\s*:\s*[A-Z0-9\-/]+", re.IGNORECASE),
    re.compile(r"\bcustomer\s*po\b", re.IGNORECASE),
]

# 7. Supporting Document / Logistics / Packing / Shipping / Customs Patterns
_SUPPORTING_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bpacking\s*list\b", re.IGNORECASE),
    re.compile(r"\bpackliste\b", re.IGNORECASE),
    re.compile(r"\blieferschein\b", re.IGNORECASE),
    re.compile(r"\bdelivery\s*(?:note|slip|docket|receipt)\b", re.IGNORECASE),
    re.compile(r"\bbon\s*de\s*livraison\b", re.IGNORECASE),
    re.compile(r"\bguia\s*de\s*remessa\b", re.IGNORECASE),
    re.compile(r"\bbill\s*of\s*lading\b", re.IGNORECASE),
    re.compile(r"\bair\s*waybill\b", re.IGNORECASE),
    re.compile(r"\bwaybill\b", re.IGNORECASE),
    re.compile(r"\bcertificate\s*of\s*origin\b", re.IGNORECASE),
    re.compile(r"\bshipping\s*(?:marks?|information|manifest)\b", re.IGNORECASE),
    re.compile(r"\b(?:net|gross|ice|tare)\s*weight\b", re.IGNORECASE),
    re.compile(r"\bno\.?\s*of\s*boxes\b", re.IGNORECASE),
    re.compile(r"\b(?:length|width|height|volume\s*m3)\b", re.IGNORECASE),
]

# 8. Receipt Patterns
_RECEIPT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:cash|sales|payment)?\s*receipt\b", re.IGNORECASE),
    re.compile(r"\bquittung\b", re.IGNORECASE),
    re.compile(r"\bkassenbon\b", re.IGNORECASE),
    re.compile(r"\bre[cç]u\b", re.IGNORECASE),
    re.compile(r"\brecibo\s*(?:provisório|de\s*pagamento)?\b", re.IGNORECASE),
    re.compile(r"\bkuitansi\b", re.IGNORECASE),
    re.compile(r"ใบเสร็จรับเงิน", re.IGNORECASE),
]

# 9. Remittance / Payment Confirmation Patterns
_REMITTANCE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bremittance\s*(?:advice|slip|notice)\b", re.IGNORECASE),
    re.compile(r"\bpayment\s*(?:advice|confirmation|voucher|order)\b", re.IGNORECASE),
    re.compile(r"\büberweisungs(?:beleg|auftrag)\b", re.IGNORECASE),
    re.compile(r"\bavis\s*de\s*(?:virement|paiement)\b", re.IGNORECASE),
    re.compile(r"\bcomprovante\s*de\s*(?:pagamento|transfer[eê]ncia)\b", re.IGNORECASE),
]

# 10. Continuation Page Indicators
_CONTINUATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bpage\s*(\d+)\s*(?:of|/)\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bseite\s*(\d+)\s*(?:von|/)\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bfolha\s*(\d+)\s*(?:de|/)\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(?:continued|continuation)\b", re.IGNORECASE),
    re.compile(r"\b(?:fortsetzung|folgeblatt)\b", re.IGNORECASE),
    re.compile(r"\b(?:suite|continua[cç][aã]o)\b", re.IGNORECASE),
    re.compile(r"\b(?:carry\s*forward|übertrag)\b", re.IGNORECASE),
]

# 11. Payable / Actionable Total Patterns
_PAYABLE_TOTAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:total\s+amount\s+due|amount\s+due|balance\s+due)\b", re.IGNORECASE),
    re.compile(r"\b(?:endbetrag|gesamtsumme|zu\s+zahlen)\b", re.IGNORECASE),
    re.compile(r"\b(?:total\s+da\s+fa[ck]tura|total\s+a\s+pagar)\b", re.IGNORECASE),
    re.compile(r"\b(?:montant\s+total|net\s+to\s+pay|grand\s+total)\b", re.IGNORECASE),
    re.compile(r"\btotal\s*payment\b", re.IGNORECASE),
    re.compile(r"\btotal\s*:\s*[\d.,]+\s*(?:eur|usd|gbp|try|thb|chf|€|\$|£|฿)\b", re.IGNORECASE),
    re.compile(r"[\d.,]+\s*(?:eur|usd|gbp|try|thb|chf|€|\$|£|฿)\s*total\b", re.IGNORECASE),
    re.compile(r"จำนวนเงินที่ต้องชำระ", re.IGNORECASE),
]

# 12. Tax & Line Item Structure Patterns
_TAX_STRUCTURE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:vat|mwst|iva|tva|gst|sst|tax|withholding\s*tax)\b", re.IGNORECASE),
    re.compile(r"\breverse\s*charge\b", re.IGNORECASE),
    re.compile(r"\bkäibemaks\b", re.IGNORECASE),
    re.compile(r"ภาษีมูลค่าเพิ่ม", re.IGNORECASE),
    re.compile(r"ภาษีหัก\s*ณ\s*ที่จ่าย", re.IGNORECASE),
]

_LINE_ITEM_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:unit\s*price|quantity|qty|extended\s*total|item\s*description)\b", re.IGNORECASE),
    re.compile(r"\b(?:einzelpreis|menge|gesamtpreis|pos\.?)\b", re.IGNORECASE),
    re.compile(r"\b(?:preço\s*unitário|quantidade|valor\s*ilíquido)\b", re.IGNORECASE),
    re.compile(r"\b(?:prix\s*unitaire|quantité)\b", re.IGNORECASE),
    re.compile(r"\b(?:ühiku\s*hind|kogus|summa)\b", re.IGNORECASE),
    re.compile(r"\bhts\s+(?:us|sg|eu)\b", re.IGNORECASE),
    re.compile(r"\beccn\b", re.IGNORECASE),
]


# ══════════════════════════════════════════════════════════════════════════
# Typed Result Model
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PageUnderstanding:
    """Page-level document understanding result with audit trail and explainability."""
    document_id: str
    page_number: int
    page_role: PageRole
    payable_relevance: PayableRelevance
    confidence: Optional[float] = None  # None for uncalibrated deterministic baseline
    evidence_ids: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    decision_reasons: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to clean JSON-serializable dictionary."""
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "page_role": self.page_role.value,
            "payable_relevance": self.payable_relevance.value,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "signals": self.signals,
            "decision_reasons": self.decision_reasons,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageUnderstanding:
        """Construct PageUnderstanding from dictionary."""
        return cls(
            document_id=data["document_id"],
            page_number=int(data["page_number"]),
            page_role=PageRole(data["page_role"]),
            payable_relevance=PayableRelevance(data["payable_relevance"]),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            evidence_ids=data.get("evidence_ids", []),
            signals=data.get("signals", {}),
            decision_reasons=data.get("decision_reasons", []),
            provenance=data.get("provenance", {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> PageUnderstanding:
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# Deterministic Signal Extractor
# ══════════════════════════════════════════════════════════════════════════

def _scan_evidence_for_patterns(
    items: Sequence[Evidence],
    patterns: Sequence[re.Pattern],
) -> tuple[int, list[str], list[str]]:
    """Scan evidence items against a sequence of regex patterns.
    
    Returns:
        (hit_count, matched_snippets, evidence_ids)
    """
    hits = 0
    snippets: list[str] = []
    ev_ids: list[str] = []
    seen_ids: Set[str] = set()

    for item in items:
        text = item.content
        if not text:
            continue
        for pat in patterns:
            match = pat.search(text)
            if match:
                hits += 1
                snippets.append(match.group(0))
                if item.evidence_id and item.evidence_id not in seen_ids:
                    ev_ids.append(item.evidence_id)
                    seen_ids.add(item.evidence_id)
                break  # Don't double count multiple patterns on same item

    return hits, snippets, ev_ids


def _detect_continuation_markers(
    items: Sequence[Evidence],
) -> tuple[bool, Optional[int], Optional[int], list[str]]:
    """Detect pagination structure (e.g. Page 2 of 2) or explicit continuation words."""
    is_cont = False
    cur_p: Optional[int] = None
    tot_p: Optional[int] = None
    ev_ids: list[str] = []

    for item in items:
        text = item.content
        if not text:
            continue
        for pat in _CONTINUATION_PATTERNS:
            m = pat.search(text)
            if m:
                ev_ids.append(item.evidence_id)
                if m.lastindex and m.lastindex >= 2:
                    try:
                        c_num = int(m.group(1))
                        t_num = int(m.group(2))
                        cur_p = c_num
                        tot_p = t_num
                        if c_num > 1:
                            is_cont = True
                    except (ValueError, IndexError):
                        pass
                else:
                    # Keyword like "Fortsetzung" or "Continued"
                    is_cont = True

    return is_cont, cur_p, tot_p, ev_ids


# ══════════════════════════════════════════════════════════════════════════
# Classifier Engine
# ══════════════════════════════════════════════════════════════════════════

def classify_page(page_evidence: PageEvidence) -> PageUnderstanding:
    """Classify the page role and payable relevance of a single document page.
    
    CRITICAL SCOPE CONSTRAINTS:
    - Purely page-level.
    - Zero dependencies on filenames, document_id, or page_number.
    - Purely deterministic; does not call Qwen or perform external I/O.
    - `confidence` is strictly None (no fabricated pseudo-probabilities).
    - Preserves all supporting evidence_ids for full auditability.
    """
    items = page_evidence.items
    doc_id = page_evidence.document_id
    page_num = page_evidence.page_number

    # ── 1. Scan evidence for semantic signal categories ──
    inv_title_hits, inv_title_snips, inv_title_ids = _scan_evidence_for_patterns(items, _INVOICE_TITLE_PATTERNS)
    inv_meta_hits, inv_meta_snips, inv_meta_ids = _scan_evidence_for_patterns(items, _INVOICE_METADATA_PATTERNS)
    credit_hits, credit_snips, credit_ids = _scan_evidence_for_patterns(items, _CREDIT_MEMO_PATTERNS)
    debit_hits, debit_snips, debit_ids = _scan_evidence_for_patterns(items, _DEBIT_MEMO_PATTERNS)
    po_doc_hits, po_doc_snips, po_doc_ids = _scan_evidence_for_patterns(items, _PO_DOC_PATTERNS)
    po_ref_hits, po_ref_snips, po_ref_ids = _scan_evidence_for_patterns(items, _PO_REFERENCE_PATTERNS)
    supp_hits, supp_snips, supp_ids = _scan_evidence_for_patterns(items, _SUPPORTING_PATTERNS)
    rcpt_hits, rcpt_snips, rcpt_ids = _scan_evidence_for_patterns(items, _RECEIPT_PATTERNS)
    remit_hits, remit_snips, remit_ids = _scan_evidence_for_patterns(items, _REMITTANCE_PATTERNS)
    pay_tot_hits, pay_tot_snips, pay_tot_ids = _scan_evidence_for_patterns(items, _PAYABLE_TOTAL_PATTERNS)
    tax_hits, tax_snips, tax_ids = _scan_evidence_for_patterns(items, _TAX_STRUCTURE_PATTERNS)
    line_hits, line_snips, line_ids = _scan_evidence_for_patterns(items, _LINE_ITEM_PATTERNS)

    is_cont, cur_page_marker, tot_page_marker, cont_ids = _detect_continuation_markers(items)

    signals: dict[str, Any] = {
        "invoice_title_hits": inv_title_hits,
        "invoice_metadata_hits": inv_meta_hits,
        "credit_memo_hits": credit_hits,
        "debit_memo_hits": debit_hits,
        "po_document_hits": po_doc_hits,
        "po_reference_hits": po_ref_hits,
        "supporting_hits": supp_hits,
        "receipt_hits": rcpt_hits,
        "remittance_hits": remit_hits,
        "payable_total_hits": pay_tot_hits,
        "tax_structure_hits": tax_hits,
        "line_item_hits": line_hits,
        "is_continuation_marker": is_cont,
        "detected_pagination": f"{cur_page_marker}/{tot_page_marker}" if cur_page_marker else None,
        "evidence_item_count": len(items),
    }

    # ── 2. Determine PageRole (orthogonal from payable relevance) ──
    decision_reasons: list[str] = []
    supporting_evidence_ids: list[str] = []

    def _add_evidence_ids(ids: list[str]) -> None:
        for ev_id in ids:
            if ev_id and ev_id not in supporting_evidence_ids:
                supporting_evidence_ids.append(ev_id)

    # Disambiguate PO document vs PO reference:
    # A reference like "Customer PO: 123" when invoice or line signals are present is NOT a PO document!
    has_invoice_signals = (inv_title_hits > 0 or inv_meta_hits > 0 or (tax_hits > 0 and line_hits > 0))
    is_genuine_po_doc = (po_doc_hits > 0 and not has_invoice_signals)

    # Role evaluation order based on specificity
    if credit_hits > 0:
        role = PageRole.CREDIT_MEMO
        decision_reasons.append(f"Credit memo terminology observed ({', '.join(set(credit_snips))})")
        _add_evidence_ids(credit_ids)
    elif debit_hits > 0:
        role = PageRole.DEBIT_MEMO
        decision_reasons.append(f"Debit memo terminology observed ({', '.join(set(debit_snips))})")
        _add_evidence_ids(debit_ids)
    elif is_genuine_po_doc:
        role = PageRole.PURCHASE_ORDER
        decision_reasons.append(f"Document-level purchase order heading observed ({', '.join(set(po_doc_snips))}) without invoice indicators")
        _add_evidence_ids(po_doc_ids)
    elif rcpt_hits > 0 and not has_invoice_signals:
        role = PageRole.RECEIPT
        decision_reasons.append(f"Receipt terminology observed ({', '.join(set(rcpt_snips))}) without invoice header")
        _add_evidence_ids(rcpt_ids)
    elif remit_hits > 0 and not has_invoice_signals:
        role = PageRole.REMITTANCE
        decision_reasons.append(f"Remittance advice terminology observed ({', '.join(set(remit_snips))})")
        _add_evidence_ids(remit_ids)
    elif is_cont and not (inv_title_hits > 0 and inv_meta_hits > 0):
        # Continuation page marker (e.g. Page 2 of 2) without primary standalone invoice header block
        role = PageRole.CONTINUATION
        pagination_str = f"Page {cur_page_marker} of {tot_page_marker}" if cur_page_marker else "continuation marker"
        decision_reasons.append(f"Page continuation state detected ({pagination_str})")
        _add_evidence_ids(cont_ids)
        if line_hits > 0:
            _add_evidence_ids(line_ids)
    elif supp_hits > 0 and not has_invoice_signals:
        role = PageRole.SUPPORTING_DOCUMENT
        decision_reasons.append(f"Logistics/packing/freight terminology observed ({', '.join(set(supp_snips))}) without invoice billing headers")
        _add_evidence_ids(supp_ids)
    elif has_invoice_signals:
        role = PageRole.INVOICE
        reasons_list: list[str] = []
        if inv_title_hits > 0:
            reasons_list.append(f"title keywords ({', '.join(set(inv_title_snips))})")
            _add_evidence_ids(inv_title_ids)
        if inv_meta_hits > 0:
            reasons_list.append(f"billing metadata ({', '.join(set(inv_meta_snips))})")
            _add_evidence_ids(inv_meta_ids)
        if tax_hits > 0:
            reasons_list.append(f"tax schedule ({', '.join(set(tax_snips))})")
            _add_evidence_ids(tax_ids)
        if line_hits > 0:
            reasons_list.append("line-item table structure")
            _add_evidence_ids(line_ids)
        decision_reasons.append(f"Invoice document signals detected: {'; '.join(reasons_list)}")
    elif supp_hits > 0:
        # Supporting indicators with weak or no other signals
        role = PageRole.SUPPORTING_DOCUMENT
        decision_reasons.append(f"Supporting logistics terms observed ({', '.join(set(supp_snips))})")
        _add_evidence_ids(supp_ids)
    else:
        role = PageRole.UNKNOWN
        decision_reasons.append("Insufficient or ambiguous document role signals")

    # ── 3. Determine PayableRelevance (COMPLETELY ORTHOGONAL) ──
    # A page is a PAYABLE_CANDIDATE if it presents actionable financial payment obligations:
    # 1. Explicit payable total/amount due
    # 2. Or complete corroborated invoice structure (bill-to + line items + tax structure)
    #
    # A page is SUPPORTING if it provides logistics details, intermediate line items, or packing specs
    # without an actionable payable total.
    #
    # A page is NON_PAYABLE if it represents operational non-payable docs (PO, receipt, remittance).
    #
    # A page is AMBIGUOUS if signals are conflicting, sparse, or incomplete.

    if role in (PageRole.PURCHASE_ORDER, PageRole.RECEIPT, PageRole.REMITTANCE):
        relevance = PayableRelevance.NON_PAYABLE
        decision_reasons.append(f"Document role '{role.value}' is non-payable by definition (order/settled payment/notification)")
    elif pay_tot_hits > 0:
        # Explicit payable total or amount due is present on this page
        relevance = PayableRelevance.PAYABLE_CANDIDATE
        decision_reasons.append(f"Actionable payable total / amount due observed on page ({', '.join(set(pay_tot_snips))})")
        _add_evidence_ids(pay_tot_ids)
    elif role in (PageRole.INVOICE, PageRole.CREDIT_MEMO, PageRole.DEBIT_MEMO) and (tax_hits > 0 and line_hits > 0):
        # Invoice/credit page with complete itemized schedule and tax breakdown even if 'TOTAL' keyword varied
        relevance = PayableRelevance.PAYABLE_CANDIDATE
        decision_reasons.append("Structured payable lines and tax breakdown observed")
        _add_evidence_ids(tax_ids)
        _add_evidence_ids(line_ids)
    elif role == PageRole.SUPPORTING_DOCUMENT:
        relevance = PayableRelevance.SUPPORTING
        decision_reasons.append("Supporting document provides non-payable logistics/packing/delivery specifications")
    elif role == PageRole.CONTINUATION and pay_tot_hits == 0:
        relevance = PayableRelevance.SUPPORTING
        decision_reasons.append("Continuation page contains intermediate lines without final payable total")
    elif role == PageRole.INVOICE and pay_tot_hits == 0 and tax_hits == 0 and line_hits == 0:
        # An invoice page with no amounts, no lines, no taxes is ambiguous
        relevance = PayableRelevance.AMBIGUOUS
        decision_reasons.append("Invoice terminology present but lacks amounts, line items, or payable totals")
    elif role == PageRole.UNKNOWN:
        relevance = PayableRelevance.AMBIGUOUS
        decision_reasons.append("Uncertain role and absence of validated payable totals")
    else:
        relevance = PayableRelevance.AMBIGUOUS
        decision_reasons.append("Signals insufficient to confirm actionable payable status")

    provenance = {
        "classifier": "DeterministicPageClassifier",
        "version": "Phase7B-1.0",
        "method": "multilingual_rule_based_signals",
        "evidence_item_count": len(items),
    }

    return PageUnderstanding(
        document_id=doc_id,
        page_number=page_num,
        page_role=role,
        payable_relevance=relevance,
        confidence=None,  # Strictly None for uncalibrated baseline (Design Correction 3)
        evidence_ids=supporting_evidence_ids,
        signals=signals,
        decision_reasons=decision_reasons,
        provenance=provenance,
    )
