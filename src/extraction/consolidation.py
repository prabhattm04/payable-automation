"""src/extraction/consolidation.py — Candidate to Document Facts Consolidation Layer.

Phase 9B-2: Consolidates unmerged semantic candidates (ExtractionCandidates)
into the immutable document fact model (DocumentFacts) from Phase 9A.

Core Principles:
1. 9B-2 consolidates evidence; 9B-2 does NOT invent accounting facts.
2. Safe consensus merges agreeing observations while retaining all evidence IDs.
3. Genuine conflict preservation: No silent selection between OCR and Vision.
   Neither source wins by default; disagreements remain unresolved.
4. Complementary partial observations: Taxes, discounts, and charges merge
   when identity, placement, and available fields are compatible. Missing rates
   or amounts are NEVER computed.
5. Repeated line preservation: Identical rows are NOT merged solely on values.
   Page, table, and source-row structure preserve separate line items.
6. Deterministic processing: Candidate sort order is strictly for stability
   and NEVER acts as an implicit evidence precedence rule.
7. Supporting document isolation: Supporting groups remain distinct and their
   financial figures are never aggregated into payable totals.
8. Zero master-data matching and zero accounting calculations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Sequence, Set, Tuple, TypeVar, Union

from src.extraction.candidates import (
    ChargeCandidate,
    DiscountCandidate,
    DocumentIdentityCandidate,
    ExtractionCandidates,
    LineCandidate,
    PartyIdentityCandidate,
    POCandidate,
    TaxCandidate,
    TotalCandidate,
)
from src.understanding.document_facts import (
    BuyerIdentityFact,
    ChargeFact,
    DiscountFact,
    DocumentFacts,
    DocumentIdentityFacts,
    FactOrigin,
    FinancialFacts,
    InvoiceType,
    LineFact,
    PartyIdentityFacts,
    Placement,
    POFacts,
    PrintedTotalsFact,
    SemanticRole,
    SupplierIdentityFact,
    TaxFact,
    to_decimal,
)
from src.understanding.page_classifier import PageRole, PayableRelevance
from src.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


# ══════════════════════════════════════════════════════════════════════════
# Diagnostic Status & Field Container
# ══════════════════════════════════════════════════════════════════════════

class FieldStatus(str, Enum):
    """Diagnostic status of a consolidated field."""
    CONFIRMED = "confirmed"          # 1+ agreeing observations
    COMPLEMENTARY = "complementary"  # Multiple observations merged complementarily
    AMBIGUOUS = "ambiguous"          # Multiple observations with unresolved interpretation
    CONFLICTED = "conflicted"        # Genuine conflict between observations
    MISSING = "missing"              # No candidate observations present


@dataclass(frozen=True)
class ConsolidatedField(Generic[T]):
    """Internal diagnostic container for a consolidated semantic field."""
    field_name: str
    value: Optional[T] = None
    raw_value: Optional[str] = None
    status: FieldStatus = FieldStatus.MISSING
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    candidate_sources: Tuple[str, ...] = field(default_factory=tuple)
    conflicting_values: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    reason: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════
# Deterministic Ordering Helper
# ══════════════════════════════════════════════════════════════════════════

def _sort_candidates_stable(candidates: Sequence[Any]) -> List[Any]:
    """Sort candidates for deterministic processing order.
    
    CRITICAL: This sort is SOLELY for deterministic stability.
    It does NOT establish precedence or allow one source to override another
    during genuine conflicts.
    """
    def key_func(c: Any) -> Tuple[int, str, str, str]:
        p = getattr(c, "page_number", 1)
        src = getattr(c, "source", "")
        ev = "".join(getattr(c, "evidence_ids", ()))
        rv = str(getattr(c, "raw_value", getattr(c, "description", "")))
        return (p, src, rv, ev)

    return sorted(candidates, key=key_func)


def _union_evidence(evidence_iter: Sequence[Tuple[str, ...]]) -> Tuple[str, ...]:
    """Return a deterministic, deduplicated tuple of evidence IDs."""
    seen: Set[str] = set()
    result: List[str] = []
    for ev_tuple in evidence_iter:
        for ev in ev_tuple:
            if ev and ev not in seen:
                seen.add(ev)
                result.append(ev)
    return tuple(sorted(result))


# ══════════════════════════════════════════════════════════════════════════
# Safe Normalization Helpers (Generic, No Document-Specific Rules)
# ══════════════════════════════════════════════════════════════════════════

# Date patterns that are unambiguous
_ISO_DATE = re.compile(r"^\s*(\d{4})[./\-](\d{1,2})[./\-](\d{1,2})\s*$")
_EURO_DATE_UNAMBIGUOUS = re.compile(r"^\s*(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})\s*$")


def normalize_date_safe(raw: Optional[str]) -> Tuple[Optional[str], bool]:
    """Safely normalize a date string without guessing ambiguous formats.
    
    Returns (normalized_or_raw, is_normalized).
    Only standardizes when the format is structurally unambiguous:
    - YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
    - DD.MM.YYYY when day > 12
    Ambiguous numeric formats (e.g. 05/06/2026 where both could be day/month)
    are NOT guessed; their raw string is preserved.
    Does NOT hardcode document-specific calendar shifts (e.g. Buddhist calendar).
    """
    if not raw:
        return (None, False)
    
    clean = raw.strip()
    
    # 1. ISO format: YYYY-MM-DD
    m_iso = _ISO_DATE.match(clean)
    if m_iso:
        y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        if 1 <= m <= 12 and 1 <= d <= 31:
            return (f"{y:04d}-{m:02d}-{d:02d}", True)

    # 2. European / Slash format
    m_eur = _EURO_DATE_UNAMBIGUOUS.match(clean)
    if m_eur:
        first, second, year = int(m_eur.group(1)), int(m_eur.group(2)), int(m_eur.group(3))
        # If first > 12, it must be the day (DD.MM.YYYY)
        if first > 12 and 1 <= second <= 12:
            return (f"{year:04d}-{second:02d}-{first:02d}", True)
        # If second > 12, first must be month (MM.DD.YYYY)
        if second > 12 and 1 <= first <= 12:
            return (f"{year:04d}-{first:02d}-{second:02d}", True)
        # Both <= 12: ambiguous! Do NOT guess.
        return (clean, False)

    return (clean, False)


def _normalize_str(val: Optional[str]) -> str:
    """Normalize string for safe comparison (strip, collapse internal whitespace)."""
    if not val:
        return ""
    return " ".join(str(val).strip().split())


# ══════════════════════════════════════════════════════════════════════════
# 1. Identity Consolidation
# ══════════════════════════════════════════════════════════════════════════

def consolidate_identity(
    candidates: Sequence[DocumentIdentityCandidate],
) -> Tuple[DocumentIdentityFacts, List[Dict[str, Any]]]:
    """Consolidate invoice number, dates, invoice type, and currency.
    
    Returns (DocumentIdentityFacts, conflicts_list).
    """
    conflicts: List[Dict[str, Any]] = []
    sorted_cands = _sort_candidates_stable(candidates)

    field_evs: Dict[str, Tuple[str, ...]] = {}
    raw_vals: Dict[str, str] = {}

    # --- A. Invoice Number ---
    inv_cands = [c for c in sorted_cands if c.field_name == "invoice_number"]
    inv_number: Optional[str] = None
    if inv_cands:
        unique_vals: Dict[str, List[DocumentIdentityCandidate]] = {}
        for c in inv_cands:
            norm = _normalize_str(c.normalized_value or c.raw_value)
            if norm:
                unique_vals.setdefault(norm, []).append(c)

        if len(unique_vals) == 1:
            norm_val, c_list = next(iter(unique_vals.items()))
            inv_number = norm_val
            field_evs["invoice_number"] = _union_evidence([c.evidence_ids for c in c_list])
            raw_vals["invoice_number"] = c_list[0].raw_value
        elif len(unique_vals) > 1:
            # Genuine conflict across observations
            inv_number = None
            conflict_entry = {
                "field": "invoice_number",
                "conflict_type": "disagreement",
                "observations": [
                    {"value": k, "sources": [c.source for c in clist], "evidence_ids": list(_union_evidence([c.evidence_ids for c in clist]))}
                    for k, clist in unique_vals.items()
                ],
            }
            conflicts.append(conflict_entry)

    # --- B. Invoice Date ---
    date_cands = [c for c in sorted_cands if c.field_name == "invoice_date"]
    inv_date: Optional[str] = None
    if date_cands:
        unique_dates: Dict[str, List[DocumentIdentityCandidate]] = {}
        for c in date_cands:
            raw_d = c.raw_value or str(c.normalized_value or "")
            norm_d, _ = normalize_date_safe(raw_d)
            if norm_d:
                unique_dates.setdefault(norm_d, []).append(c)

        if len(unique_dates) == 1:
            d_val, c_list = next(iter(unique_dates.items()))
            inv_date = d_val
            field_evs["invoice_date"] = _union_evidence([c.evidence_ids for c in c_list])
            raw_vals["invoice_date"] = c_list[0].raw_value
        elif len(unique_dates) > 1:
            inv_date = None
            conflicts.append({
                "field": "invoice_date",
                "conflict_type": "disagreement",
                "observations": [
                    {"value": k, "sources": [c.source for c in clist], "evidence_ids": list(_union_evidence([c.evidence_ids for c in clist]))}
                    for k, clist in unique_dates.items()
                ],
            })

    # --- C. Due Date ---
    due_cands = [c for c in sorted_cands if c.field_name == "due_date"]
    due_date: Optional[str] = None
    if due_cands:
        unique_dues: Dict[str, List[DocumentIdentityCandidate]] = {}
        for c in due_cands:
            raw_d = c.raw_value or str(c.normalized_value or "")
            norm_d, _ = normalize_date_safe(raw_d)
            if norm_d:
                unique_dues.setdefault(norm_d, []).append(c)

        if len(unique_dues) == 1:
            d_val, c_list = next(iter(unique_dues.items()))
            due_date = d_val
            field_evs["due_date"] = _union_evidence([c.evidence_ids for c in c_list])
            raw_vals["due_date"] = c_list[0].raw_value
        elif len(unique_dues) > 1:
            due_date = None
            conflicts.append({
                "field": "due_date",
                "conflict_type": "disagreement",
                "observations": [
                    {"value": k, "sources": [c.source for c in clist], "evidence_ids": list(_union_evidence([c.evidence_ids for c in clist]))}
                    for k, clist in unique_dues.items()
                ],
            })

    # --- D. Invoice Type ---
    # Header-level candidates take priority over incidental body mentions
    type_cands = [c for c in sorted_cands if c.field_name == "invoice_type" and c.invoice_type_value]
    header_types = [c for c in type_cands if c.placement == Placement.HEADER or c.extraction_notes.get("is_header") is True]
    chosen_type_cands = header_types if header_types else type_cands

    inv_type: InvoiceType = InvoiceType.UNKNOWN
    if chosen_type_cands:
        unique_types: Dict[InvoiceType, List[DocumentIdentityCandidate]] = {}
        for c in chosen_type_cands:
            if c.invoice_type_value:
                unique_types.setdefault(c.invoice_type_value, []).append(c)

        # Ignore UNKNOWN if specific types are observed
        non_unknown = {k: v for k, v in unique_types.items() if k != InvoiceType.UNKNOWN}
        if len(non_unknown) == 1:
            t_val, c_list = next(iter(non_unknown.items()))
            inv_type = t_val
            field_evs["invoice_type"] = _union_evidence([c.evidence_ids for c in c_list])
            raw_vals["invoice_type"] = c_list[0].raw_value
        elif len(non_unknown) > 1:
            # Genuine header-level conflict e.g. INVOICE vs CREDIT MEMO
            inv_type = InvoiceType.UNKNOWN
            conflicts.append({
                "field": "invoice_type",
                "conflict_type": "disagreement",
                "observations": [
                    {"value": k.value, "sources": [c.source for c in clist], "evidence_ids": list(_union_evidence([c.evidence_ids for c in clist]))}
                    for k, clist in non_unknown.items()
                ],
            })
        elif InvoiceType.UNKNOWN in unique_types:
            c_list = unique_types[InvoiceType.UNKNOWN]
            inv_type = InvoiceType.UNKNOWN
            field_evs["invoice_type"] = _union_evidence([c.evidence_ids for c in c_list])

    # --- E. Currency ---
    curr_cands = [c for c in sorted_cands if c.field_name == "currency"]
    currency: Optional[str] = None
    if curr_cands:
        unique_currs: Dict[str, List[DocumentIdentityCandidate]] = {}
        for c in curr_cands:
            val = _normalize_str(c.normalized_value or c.raw_value).upper()
            if val:
                unique_currs.setdefault(val, []).append(c)

        if len(unique_currs) == 1:
            c_val, c_list = next(iter(unique_currs.items()))
            currency = c_val
            field_evs["currency"] = _union_evidence([c.evidence_ids for c in c_list])
            raw_vals["currency"] = c_list[0].raw_value
        elif len(unique_currs) > 1:
            currency = None
            conflicts.append({
                "field": "currency",
                "conflict_type": "disagreement",
                "observations": [
                    {"value": k, "sources": [c.source for c in clist], "evidence_ids": list(_union_evidence([c.evidence_ids for c in clist]))}
                    for k, clist in unique_currs.items()
                ],
            })

    all_evs = _union_evidence(list(field_evs.values()))
    facts = DocumentIdentityFacts(
        invoice_number=inv_number,
        invoice_date=inv_date,
        due_date=due_date,
        invoice_type=inv_type,
        currency=currency,
        evidence_ids=all_evs,
        field_evidence_ids=field_evs,
        origin=FactOrigin.OBSERVED,
        raw_values=raw_vals,
    )
    return (facts, conflicts)


# ══════════════════════════════════════════════════════════════════════════
# 2. Supplier Consolidation
# ══════════════════════════════════════════════════════════════════════════

def consolidate_supplier(
    candidates: Sequence[PartyIdentityCandidate],
) -> Tuple[SupplierIdentityFact, List[Dict[str, Any]]]:
    """Consolidate supplier attributes independently with complementary field support."""
    conflicts: List[Dict[str, Any]] = []
    sup_cands = [c for c in _sort_candidates_stable(candidates) if c.party_role == "supplier"]

    field_evs: Dict[str, Tuple[str, ...]] = {}
    raw_vals: Dict[str, str] = {}

    def _consolidate_single_attribute(
        target_field: str,
        match_field_names: Tuple[str, ...],
        normalizer = lambda x: _normalize_str(x),
    ) -> Optional[str]:
        matching = [c for c in sup_cands if c.field_name in match_field_names]
        if not matching:
            return None

        unique_vals: Dict[str, List[PartyIdentityCandidate]] = {}
        for c in matching:
            val = normalizer(c.normalized_value or c.raw_value)
            if val:
                unique_vals.setdefault(val, []).append(c)

        if len(unique_vals) == 1:
            val, clist = next(iter(unique_vals.items()))
            field_evs[target_field] = _union_evidence([c.evidence_ids for c in clist])
            raw_vals[target_field] = clist[0].raw_value
            return val
        elif len(unique_vals) > 1:
            conflicts.append({
                "field": f"supplier.{target_field}",
                "conflict_type": "disagreement",
                "observations": [
                    {"value": k, "sources": [c.source for c in clist], "evidence_ids": list(_union_evidence([c.evidence_ids for c in clist]))}
                    for k, clist in unique_vals.items()
                ],
            })
            return None
        return None

    name = _consolidate_single_attribute("observed_name", ("name", "observed_name", "supplier_name"))
    vat = _consolidate_single_attribute("vat_id", ("vat_id", "vat", "tax_id"), lambda x: _normalize_str(x).upper())
    country = _consolidate_single_attribute("country", ("country",), lambda x: _normalize_str(x).upper())
    email = _consolidate_single_attribute("email", ("email",), lambda x: _normalize_str(x).lower())
    iban = _consolidate_single_attribute("bank_iban", ("bank_iban", "iban"), lambda x: _normalize_str(x).replace(" ", "").upper())
    address = _consolidate_single_attribute("address", ("address",))

    all_evs = _union_evidence(list(field_evs.values()))
    facts = SupplierIdentityFact(
        observed_name=name,
        vat_id=vat,
        country=country,
        email=email,
        bank_iban=iban,
        address=address,
        evidence_ids=all_evs,
        field_evidence_ids=field_evs,
        origin=FactOrigin.OBSERVED,
        raw_values=raw_vals,
    )
    return (facts, conflicts)


# ══════════════════════════════════════════════════════════════════════════
# 3. Buyer Consolidation
# ══════════════════════════════════════════════════════════════════════════

def consolidate_buyer(
    candidates: Sequence[PartyIdentityCandidate],
) -> Tuple[BuyerIdentityFact, List[Dict[str, Any]]]:
    """Consolidate buyer attributes independently without hierarchy collapse."""
    conflicts: List[Dict[str, Any]] = []
    buy_cands = [c for c in _sort_candidates_stable(candidates) if c.party_role == "buyer"]

    field_evs: Dict[str, Tuple[str, ...]] = {}
    raw_vals: Dict[str, str] = {}

    def _consolidate_single_attribute(
        target_field: str,
        match_field_names: Tuple[str, ...],
        normalizer = lambda x: _normalize_str(x),
    ) -> Optional[str]:
        matching = [c for c in buy_cands if c.field_name in match_field_names]
        if not matching:
            return None

        unique_vals: Dict[str, List[PartyIdentityCandidate]] = {}
        for c in matching:
            val = normalizer(c.normalized_value or c.raw_value)
            if val:
                unique_vals.setdefault(val, []).append(c)

        if len(unique_vals) == 1:
            val, clist = next(iter(unique_vals.items()))
            field_evs[target_field] = _union_evidence([c.evidence_ids for c in clist])
            raw_vals[target_field] = clist[0].raw_value
            return val
        elif len(unique_vals) > 1:
            conflicts.append({
                "field": f"buyer.{target_field}",
                "conflict_type": "disagreement",
                "observations": [
                    {"value": k, "sources": [c.source for c in clist], "evidence_ids": list(_union_evidence([c.evidence_ids for c in clist]))}
                    for k, clist in unique_vals.items()
                ],
            })
            return None
        return None

    company = _consolidate_single_attribute("observed_company", ("company", "company_name", "observed_company", "name"))
    bu = _consolidate_single_attribute("business_unit", ("business_unit", "business_unit_name"))
    loc = _consolidate_single_attribute("location", ("location", "location_name"))
    cc_code = _consolidate_single_attribute("company_code", ("company_code",), lambda x: _normalize_str(x).upper())
    bu_code = _consolidate_single_attribute("business_unit_code", ("business_unit_code",), lambda x: _normalize_str(x).upper())
    loc_code = _consolidate_single_attribute("location_code", ("location_code",), lambda x: _normalize_str(x).upper())
    addr = _consolidate_single_attribute("invoice_to_address", ("address", "invoice_to_address"))

    all_evs = _union_evidence(list(field_evs.values()))
    facts = BuyerIdentityFact(
        observed_company=company,
        business_unit=bu,
        location=loc,
        company_code=cc_code,
        business_unit_code=bu_code,
        location_code=loc_code,
        invoice_to_address=addr,
        evidence_ids=all_evs,
        field_evidence_ids=field_evs,
        origin=FactOrigin.OBSERVED,
        raw_values=raw_vals,
    )
    return (facts, conflicts)


# ══════════════════════════════════════════════════════════════════════════
# 4. PO Consolidation
# ══════════════════════════════════════════════════════════════════════════

def consolidate_po(
    candidates: Sequence[POCandidate],
) -> Tuple[POFacts, List[Dict[str, Any]]]:
    """Consolidate observed PO references without master substitution."""
    conflicts: List[Dict[str, Any]] = []
    po_cands = _sort_candidates_stable(candidates)

    field_evs: Dict[str, Tuple[str, ...]] = {}
    raw_vals: Dict[str, str] = {}
    po_number: Optional[str] = None

    if po_cands:
        unique_pos: Dict[str, List[POCandidate]] = {}
        for c in po_cands:
            norm = _normalize_str(c.normalized_value or c.raw_value)
            # Remove leading common prefix if present e.g. "PO: " or "#"
            norm = re.sub(r"^(?:po\s*[:#\-]?|purchase\s*order\s*[:#\-]?|#)\s*", "", norm, flags=re.IGNORECASE)
            if norm:
                unique_pos.setdefault(norm, []).append(c)

        if len(unique_pos) == 1:
            val, clist = next(iter(unique_pos.items()))
            po_number = val
            field_evs["observed_po_number"] = _union_evidence([c.evidence_ids for c in clist])
            raw_vals["observed_po_number"] = clist[0].raw_value
        elif len(unique_pos) > 1:
            po_number = None
            conflicts.append({
                "field": "observed_po_number",
                "conflict_type": "disagreement",
                "observations": [
                    {"value": k, "sources": [c.source for c in clist], "evidence_ids": list(_union_evidence([c.evidence_ids for c in clist]))}
                    for k, clist in unique_pos.items()
                ],
            })

    all_evs = _union_evidence(list(field_evs.values()))
    facts = POFacts(
        observed_po_number=po_number,
        evidence_ids=all_evs,
        field_evidence_ids=field_evs,
        origin=FactOrigin.OBSERVED,
        raw_values=raw_vals,
    )
    return (facts, conflicts)


# ══════════════════════════════════════════════════════════════════════════
# 5. Line Item Consolidation (Preserving Repeated Identical Lines)
# ══════════════════════════════════════════════════════════════════════════

def consolidate_lines(
    candidates: Sequence[LineCandidate],
) -> Tuple[List[LineFact], List[Dict[str, Any]]]:
    """Consolidate line candidates while strictly preserving repeated identical lines.
    
    Principles:
    - Never merge invoice lines solely because description, quantity, price,
      and amount are identical (Amendment 2).
    - Candidates from the same source (e.g. two OCR lines) represent separate physical rows.
    - Correspondence across OCR and Vision requires page and structural row matching.
    - Missing quantities, prices, or amounts remain None (zero arithmetic).
    - Preserves SemanticRole.COMPONENT_DETAIL vs BILLED_LINE.
    """
    conflicts: List[Dict[str, Any]] = []
    sorted_cands = _sort_candidates_stable(candidates)

    # Separate billed lines from component details
    billed_cands = [c for c in sorted_cands if c.semantic_role != SemanticRole.COMPONENT_DETAIL]
    comp_cands = [c for c in sorted_cands if c.semantic_role == SemanticRole.COMPONENT_DETAIL]

    def _cluster_and_consolidate(cands: List[LineCandidate], default_role: SemanticRole) -> List[LineFact]:
        if not cands:
            return []

        # Partition by source
        by_source: Dict[str, List[LineCandidate]] = {}
        for c in cands:
            by_source.setdefault(c.source, []).append(c)

        # If only one source produced line candidates, every candidate is a distinct row!
        if len(by_source) == 1:
            facts: List[LineFact] = []
            for idx, c in enumerate(cands, start=1):
                f_evs = {"description": c.evidence_ids} if c.description else {}
                if c.amount is not None:
                    f_evs["amount"] = c.evidence_ids
                if c.quantity is not None:
                    f_evs["quantity"] = c.evidence_ids
                if c.unit_price is not None:
                    f_evs["unit_price"] = c.evidence_ids

                facts.append(LineFact(
                    line_number=idx,
                    description=c.description,
                    quantity=c.quantity,
                    unit_price=c.unit_price,
                    amount=c.amount,
                    discount=c.discount,
                    currency=c.currency,
                    evidence_ids=c.evidence_ids,
                    field_evidence_ids=f_evs,
                    source_row_evidence_ids=c.evidence_ids,
                    semantic_role=c.semantic_role if c.semantic_role != SemanticRole.UNKNOWN else default_role,
                    origin=FactOrigin.OBSERVED,
                    raw_values={"raw_row": c.raw_row, **({"item_type": c.extraction_notes["item_type"]} if "item_type" in c.extraction_notes else {})},
                    metadata={"raw_cells": list(c.raw_cells), "source": c.source, "source_row_number": c.source_row_number, **c.extraction_notes},
                ))
            return facts

        # Multiple sources: match OCR line to Vision line structurally
        primary_source = "ocr" if "ocr" in by_source else list(by_source.keys())[0]
        secondary_source = "vision" if "vision" in by_source else [s for s in by_source if s != primary_source][0]

        primaries = by_source[primary_source]
        secondaries = list(by_source[secondary_source])
        matched_secondary_indices: Set[int] = set()

        line_facts: List[LineFact] = []

        for p_idx, p_cand in enumerate(primaries, start=1):
            # Attempt to find structural match in secondaries
            best_sec_idx: Optional[int] = None

            for s_idx, s_cand in enumerate(secondaries):
                if s_idx in matched_secondary_indices:
                    continue

                # 1. Page number must match
                if p_cand.page_number != s_cand.page_number:
                    continue

                # 2. Structural match signal:
                # Same source_row_number AND numbers compatible
                has_same_row = (
                    p_cand.source_row_number is not None
                    and s_cand.source_row_number is not None
                    and p_cand.source_row_number == s_cand.source_row_number
                )

                # Numbers check
                amt_match = (
                    p_cand.amount is not None and s_cand.amount is not None and p_cand.amount == s_cand.amount
                )
                amt_conflict = (
                    p_cand.amount is not None and s_cand.amount is not None and p_cand.amount != s_cand.amount
                )
                qty_match = (
                    p_cand.quantity is not None and s_cand.quantity is not None and p_cand.quantity == s_cand.quantity
                )

                # Never match rows with conflicting non-empty amounts
                if amt_conflict:
                    continue

                if has_same_row or (amt_match and qty_match):
                    best_sec_idx = s_idx
                    break

            if best_sec_idx is not None:
                matched_secondary_indices.add(best_sec_idx)
                s_cand = secondaries[best_sec_idx]

                # Consolidate p_cand and s_cand
                comb_evs = _union_evidence([p_cand.evidence_ids, s_cand.evidence_ids])
                f_evs: Dict[str, Tuple[str, ...]] = {}

                # Description
                desc = p_cand.description or s_cand.description
                if desc:
                    f_evs["description"] = comb_evs

                # Quantity (keep None if both None; if one has it, take it; if conflicting, keep None)
                qty = p_cand.quantity
                if qty is None:
                    qty = s_cand.quantity
                elif s_cand.quantity is not None and qty != s_cand.quantity:
                    qty = None  # Conflicted

                # Unit Price (keep None if both None; NEVER compute)
                u_price = p_cand.unit_price
                if u_price is None:
                    u_price = s_cand.unit_price
                elif s_cand.unit_price is not None and u_price != s_cand.unit_price:
                    u_price = None

                # Amount (keep None if both None; NEVER compute)
                amt = p_cand.amount
                if amt is None:
                    amt = s_cand.amount
                elif s_cand.amount is not None and amt != s_cand.amount:
                    amt = None

                # Discount & Currency
                disc = p_cand.discount if p_cand.discount is not None else s_cand.discount
                curr = p_cand.currency or s_cand.currency

                if amt is not None:
                    f_evs["amount"] = comb_evs
                if qty is not None:
                    f_evs["quantity"] = comb_evs
                if u_price is not None:
                    f_evs["unit_price"] = comb_evs

                line_facts.append(LineFact(
                    line_number=p_idx,
                    description=desc,
                    quantity=qty,
                    unit_price=u_price,
                    amount=amt,
                    discount=disc,
                    currency=curr,
                    evidence_ids=comb_evs,
                    field_evidence_ids=f_evs,
                    source_row_evidence_ids=comb_evs,
                    semantic_role=p_cand.semantic_role if p_cand.semantic_role != SemanticRole.UNKNOWN else default_role,
                    origin=FactOrigin.OBSERVED,
                    raw_values={"raw_row": p_cand.raw_row or s_cand.raw_row, **({"item_type": (p_cand.extraction_notes or {}).get("item_type") or (s_cand.extraction_notes or {}).get("item_type")} if (p_cand.extraction_notes or {}).get("item_type") or (s_cand.extraction_notes or {}).get("item_type") else {})},
                    metadata={
                        "raw_cells": list(p_cand.raw_cells or s_cand.raw_cells),
                        "sources": [primary_source, secondary_source],
                        "source_row_number": p_cand.source_row_number,
                        **(p_cand.extraction_notes or {}),
                    },
                ))
            else:
                # Primary candidate had no unambiguous secondary match -> retain unmerged!
                f_evs = {"description": p_cand.evidence_ids} if p_cand.description else {}
                if p_cand.amount is not None:
                    f_evs["amount"] = p_cand.evidence_ids
                if p_cand.quantity is not None:
                    f_evs["quantity"] = p_cand.evidence_ids
                if p_cand.unit_price is not None:
                    f_evs["unit_price"] = p_cand.evidence_ids

                line_facts.append(LineFact(
                    line_number=p_idx,
                    description=p_cand.description,
                    quantity=p_cand.quantity,
                    unit_price=p_cand.unit_price,
                    amount=p_cand.amount,
                    discount=p_cand.discount,
                    currency=p_cand.currency,
                    evidence_ids=p_cand.evidence_ids,
                    field_evidence_ids=f_evs,
                    source_row_evidence_ids=p_cand.evidence_ids,
                    semantic_role=p_cand.semantic_role if p_cand.semantic_role != SemanticRole.UNKNOWN else default_role,
                    origin=FactOrigin.OBSERVED,
                    raw_values={"raw_row": p_cand.raw_row, **({"item_type": p_cand.extraction_notes["item_type"]} if "item_type" in (p_cand.extraction_notes or {}) else {})},
                    metadata={"raw_cells": list(p_cand.raw_cells), "source": primary_source, "source_row_number": p_cand.source_row_number, **(p_cand.extraction_notes or {})},
                ))

        # Unmatched secondary candidates are also retained as separate lines
        for s_idx, s_cand in enumerate(secondaries):
            if s_idx not in matched_secondary_indices:
                idx = len(line_facts) + 1
                f_evs = {"description": s_cand.evidence_ids} if s_cand.description else {}
                if s_cand.amount is not None:
                    f_evs["amount"] = s_cand.evidence_ids
                if s_cand.quantity is not None:
                    f_evs["quantity"] = s_cand.evidence_ids
                if s_cand.unit_price is not None:
                    f_evs["unit_price"] = s_cand.evidence_ids

                line_facts.append(LineFact(
                    line_number=idx,
                    description=s_cand.description,
                    quantity=s_cand.quantity,
                    unit_price=s_cand.unit_price,
                    amount=s_cand.amount,
                    discount=s_cand.discount,
                    currency=s_cand.currency,
                    evidence_ids=s_cand.evidence_ids,
                    field_evidence_ids=f_evs,
                    source_row_evidence_ids=s_cand.evidence_ids,
                    semantic_role=s_cand.semantic_role if s_cand.semantic_role != SemanticRole.UNKNOWN else default_role,
                    origin=FactOrigin.OBSERVED,
                    raw_values={"raw_row": s_cand.raw_row, **({"item_type": s_cand.extraction_notes["item_type"]} if "item_type" in (s_cand.extraction_notes or {}) else {})},
                    metadata={"raw_cells": list(s_cand.raw_cells), "source": secondary_source, "source_row_number": s_cand.source_row_number, **(s_cand.extraction_notes or {})},
                ))

        return line_facts

    billed_facts = _cluster_and_consolidate(billed_cands, SemanticRole.BILLED_LINE)
    comp_facts = _cluster_and_consolidate(comp_cands, SemanticRole.COMPONENT_DETAIL)

    return (billed_facts + comp_facts, conflicts)


# ══════════════════════════════════════════════════════════════════════════
# 6. Tax Consolidation (Complementary Partial Observations)
# ══════════════════════════════════════════════════════════════════════════

def consolidate_taxes(
    candidates: Sequence[TaxCandidate],
) -> Tuple[List[TaxFact], List[Dict[str, Any]]]:
    """Consolidate tax candidates supporting complementary partial observations (Amendment 1).
    
    Principles:
    - Does NOT require complete tuple equality for consolidation.
    - OCR (VAT / HEADER / 24% / 117.72) and Vision (VAT / HEADER / None / 117.72)
      merge into (VAT / HEADER / 24% / 117.72) when placement and attributes agree.
    - Missing rates or amounts are NEVER computed.
    - Only genuine disagreements (e.g. 24% / 117.72 vs 18% / 88.20) remain conflicted.
    - Placement (HEADER vs LINE) is strictly preserved.
    """
    conflicts: List[Dict[str, Any]] = []
    sorted_cands = _sort_candidates_stable(candidates)
    if not sorted_cands:
        return ([], conflicts)

    # Cluster candidates by placement
    clusters: List[List[TaxCandidate]] = []

    for cand in sorted_cands:
        matched_cluster = None
        for cluster in clusters:
            rep = cluster[0]
            # 1. Placement must be compatible (LINE vs HEADER never merge)
            if rep.placement != cand.placement and rep.placement != Placement.UNKNOWN and cand.placement != Placement.UNKNOWN:
                continue

            # 2. Tax names compatible if present
            name_rep = _normalize_str(rep.tax_name).upper()
            name_cand = _normalize_str(cand.tax_name).upper()
            if name_rep and name_cand and name_rep != name_cand:
                continue

            # 3. Rate compatibility (both present and unequal -> conflict, not same observation)
            if rep.rate is not None and cand.rate is not None and rep.rate != cand.rate:
                continue

            # 4. Amount compatibility (both present and unequal -> conflict)
            if rep.amount is not None and cand.amount is not None and rep.amount != cand.amount:
                continue

            matched_cluster = cluster
            break

        if matched_cluster is not None:
            matched_cluster.append(cand)
        else:
            clusters.append([cand])

    tax_facts: List[TaxFact] = []
    for cluster in clusters:
        ev_ids = _union_evidence([c.evidence_ids for c in cluster])
        field_evs: Dict[str, Tuple[str, ...]] = {}

        # Merge attributes complementarily
        name = next((c.tax_name for c in cluster if c.tax_name), None)
        tax_type = next((c.tax_type for c in cluster if c.tax_type), None)
        rate = next((c.rate for c in cluster if c.rate is not None), None)
        amt = next((c.amount for c in cluster if c.amount is not None), None)

        # Placement priority: HEADER or LINE over UNKNOWN
        plc = Placement.UNKNOWN
        for c in cluster:
            if c.placement in (Placement.HEADER, Placement.LINE):
                plc = c.placement
                break

        if rate is not None:
            field_evs["rate"] = ev_ids
        if amt is not None:
            field_evs["amount"] = ev_ids

        raw_val = cluster[0].raw_value

        tax_facts.append(TaxFact(
            tax_name=name,
            tax_type=tax_type,
            rate=rate,
            amount=amt,
            placement=plc,
            evidence_ids=ev_ids,
            field_evidence_ids=field_evs,
            origin=FactOrigin.OBSERVED,
            raw_values={"raw_value": raw_val} if raw_val else {},
            metadata={"candidate_count": len(cluster), "sources": [c.source for c in cluster]},
        ))

    return (tax_facts, conflicts)


# ══════════════════════════════════════════════════════════════════════════
# 7. Discount & Charge Consolidation (Complementary Partial Observations)
# ══════════════════════════════════════════════════════════════════════════

def consolidate_discounts_and_charges(
    discount_cands: Sequence[DiscountCandidate],
    charge_cands: Sequence[ChargeCandidate],
) -> Tuple[List[DiscountFact], List[ChargeFact], List[Dict[str, Any]]]:
    """Consolidate discounts and charges supporting complementary partial observations."""
    conflicts: List[Dict[str, Any]] = []

    # --- A. Discounts ---
    disc_facts: List[DiscountFact] = []
    disc_clusters: List[List[DiscountCandidate]] = []

    for cand in _sort_candidates_stable(discount_cands):
        matched = None
        for cluster in disc_clusters:
            rep = cluster[0]
            if rep.placement != cand.placement and rep.placement != Placement.UNKNOWN and cand.placement != Placement.UNKNOWN:
                continue
            if rep.rate is not None and cand.rate is not None and rep.rate != cand.rate:
                continue
            if rep.amount is not None and cand.amount is not None and rep.amount != cand.amount:
                continue
            matched = cluster
            break
        if matched is not None:
            matched.append(cand)
        else:
            disc_clusters.append([cand])

    for cluster in disc_clusters:
        ev_ids = _union_evidence([c.evidence_ids for c in cluster])
        name = next((c.label for c in cluster if c.label), None)
        rate = next((c.rate for c in cluster if c.rate is not None), None)
        amt = next((c.amount for c in cluster if c.amount is not None), None)
        plc = next((c.placement for c in cluster if c.placement != Placement.UNKNOWN), Placement.UNKNOWN)

        field_evs: Dict[str, Tuple[str, ...]] = {}
        if rate is not None:
            field_evs["rate"] = ev_ids
        if amt is not None:
            field_evs["amount"] = ev_ids

        disc_facts.append(DiscountFact(
            name=name,
            rate=rate,
            amount=amt,
            placement=plc,
            evidence_ids=ev_ids,
            field_evidence_ids=field_evs,
            origin=FactOrigin.OBSERVED,
            raw_values={"raw_value": cluster[0].raw_value},
        ))

    # --- B. Charges ---
    chg_facts: List[ChargeFact] = []
    chg_clusters: List[List[ChargeCandidate]] = []

    for cand in _sort_candidates_stable(charge_cands):
        matched = None
        for cluster in chg_clusters:
            rep = cluster[0]
            if rep.placement != cand.placement and rep.placement != Placement.UNKNOWN and cand.placement != Placement.UNKNOWN:
                continue
            if rep.rate is not None and cand.rate is not None and rep.rate != cand.rate:
                continue
            if rep.amount is not None and cand.amount is not None and rep.amount != cand.amount:
                continue
            matched = cluster
            break
        if matched is not None:
            matched.append(cand)
        else:
            chg_clusters.append([cand])

    for cluster in chg_clusters:
        ev_ids = _union_evidence([c.evidence_ids for c in cluster])
        name = next((c.label for c in cluster if c.label), None)
        rate = next((c.rate for c in cluster if c.rate is not None), None)
        amt = next((c.amount for c in cluster if c.amount is not None), None)
        plc = next((c.placement for c in cluster if c.placement != Placement.UNKNOWN), Placement.UNKNOWN)

        field_evs: Dict[str, Tuple[str, ...]] = {}
        if rate is not None:
            field_evs["rate"] = ev_ids
        if amt is not None:
            field_evs["amount"] = ev_ids

        chg_facts.append(ChargeFact(
            name=name,
            rate=rate,
            amount=amt,
            placement=plc,
            evidence_ids=ev_ids,
            field_evidence_ids=field_evs,
            origin=FactOrigin.OBSERVED,
            raw_values={"raw_value": cluster[0].raw_value},
        ))

    return (disc_facts, chg_facts, conflicts)


# ══════════════════════════════════════════════════════════════════════════
# 8. Printed Totals Consolidation
# ══════════════════════════════════════════════════════════════════════════

def consolidate_printed_totals(
    candidates: Sequence[TotalCandidate],
) -> Tuple[Optional[PrintedTotalsFact], List[Dict[str, Any]]]:
    """Consolidate printed totals without arithmetic balancing."""
    conflicts: List[Dict[str, Any]] = []
    sorted_cands = _sort_candidates_stable(candidates)
    if not sorted_cands:
        return (None, conflicts)

    field_evs: Dict[str, Tuple[str, ...]] = {}
    raw_vals: Dict[str, str] = {}

    def _consolidate_single_total(total_type: str) -> Optional[Decimal]:
        matching = [c for c in sorted_cands if c.total_type == total_type and c.normalized_value is not None]
        if not matching:
            return None

        unique_amts: Dict[Decimal, List[TotalCandidate]] = {}
        for c in matching:
            if c.normalized_value is not None:
                unique_amts.setdefault(c.normalized_value, []).append(c)

        if len(unique_amts) == 1:
            amt, clist = next(iter(unique_amts.items()))
            field_evs[total_type] = _union_evidence([c.evidence_ids for c in clist])
            raw_vals[total_type] = clist[0].raw_value
            return amt
        elif len(unique_amts) > 1:
            conflicts.append({
                "field": f"printed_totals.{total_type}",
                "conflict_type": "disagreement",
                "observations": [
                    {"value": str(k), "sources": [c.source for c in clist], "evidence_ids": list(_union_evidence([c.evidence_ids for c in clist]))}
                    for k, clist in unique_amts.items()
                ],
            })
            return None
        return None

    subtotal = _consolidate_single_total("subtotal")
    net = _consolidate_single_total("net")
    taxable_base = _consolidate_single_total("taxable_base")
    tax_total = _consolidate_single_total("tax_total")
    gross_total = _consolidate_single_total("gross_total")
    amount_due = _consolidate_single_total("amount_due")
    payment_total = _consolidate_single_total("payment_total")

    all_evs = _union_evidence([c.evidence_ids for c in sorted_cands])
    facts = PrintedTotalsFact(
        subtotal=subtotal,
        net=net,
        taxable_base=taxable_base,
        tax_total=tax_total,
        gross_total=gross_total,
        amount_due=amount_due,
        payment_total=payment_total,
        evidence_ids=all_evs,
        field_evidence_ids=field_evs,
        origin=FactOrigin.OBSERVED,
        raw_values=raw_vals,
    )
    return (facts, conflicts)


# ══════════════════════════════════════════════════════════════════════════
# Top-Level Consolidation Entry Points
# ══════════════════════════════════════════════════════════════════════════

def consolidate_candidates(
    candidates: Union[ExtractionCandidates, Sequence[ExtractionCandidates]],
) -> DocumentFacts:
    """Consolidate semantic candidates into an immutable DocumentFacts representation.
    
    Handles both a single ExtractionCandidates object or a sequence representing
    multi-page documents. Isolates supporting documents and references them
    via supporting_group_ids without financial aggregation.
    """
    if isinstance(candidates, ExtractionCandidates):
        cand_list = [candidates]
    else:
        cand_list = list(candidates)

    if not cand_list:
        raise ValueError("Cannot consolidate an empty candidate collection.")

    # Partition by group_id
    by_group: Dict[Optional[str], List[ExtractionCandidates]] = {}
    for c in cand_list:
        by_group.setdefault(c.group_id, []).append(c)

    # Determine primary payable group vs supporting groups
    # Look for groups with PAYABLE_CANDIDATE relevance
    payable_group_id: Optional[str] = None
    supporting_group_ids: List[str] = []

    for g_id, g_cands in by_group.items():
        relevance = g_cands[0].payable_relevance
        if relevance == PayableRelevance.SUPPORTING:
            if g_id:
                supporting_group_ids.append(g_id)
        elif payable_group_id is None:
            payable_group_id = g_id

    # Fallback if no explicit payable group
    if payable_group_id is None and by_group:
        payable_group_id = next(iter(by_group.keys()))

    target_cands = by_group.get(payable_group_id, cand_list)

    # Aggregate candidate collections for target group
    ident_cands: List[DocumentIdentityCandidate] = []
    party_cands: List[PartyIdentityCandidate] = []
    po_cands: List[POCandidate] = []
    line_cands: List[LineCandidate] = []
    tax_cands: List[TaxCandidate] = []
    disc_cands: List[DiscountCandidate] = []
    chg_cands: List[ChargeCandidate] = []
    tot_cands: List[TotalCandidate] = []
    page_numbers_set: Set[int] = set()

    for ec in target_cands:
        ident_cands.extend(ec.identity_candidates)
        party_cands.extend(ec.party_candidates)
        po_cands.extend(ec.po_candidates)
        line_cands.extend(ec.line_candidates)
        tax_cands.extend(ec.tax_candidates)
        disc_cands.extend(ec.discount_candidates)
        chg_cands.extend(ec.charge_candidates)
        tot_cands.extend(ec.total_candidates)
        page_numbers_set.update(ec.page_numbers)

    doc_id = target_cands[0].document_id
    p_relevance = target_cands[0].payable_relevance
    p_role = target_cands[0].page_role

    all_conflicts: List[Dict[str, Any]] = []

    # 1. Consolidate Identity
    ident_facts, ident_conflicts = consolidate_identity(ident_cands)
    all_conflicts.extend(ident_conflicts)

    # 2. Consolidate Parties (Supplier & Buyer)
    sup_facts, sup_conflicts = consolidate_supplier(party_cands)
    buy_facts, buy_conflicts = consolidate_buyer(party_cands)
    all_conflicts.extend(sup_conflicts)
    all_conflicts.extend(buy_conflicts)
    party_evs = _union_evidence([sup_facts.evidence_ids, buy_facts.evidence_ids])
    parties_facts = PartyIdentityFacts(
        supplier=sup_facts,
        buyer=buy_facts,
        evidence_ids=party_evs,
    )

    # 3. Consolidate PO
    po_facts, po_conflicts = consolidate_po(po_cands)
    all_conflicts.extend(po_conflicts)

    # 4. Consolidate Lines
    lines_facts, line_conflicts = consolidate_lines(line_cands)
    all_conflicts.extend(line_conflicts)

    # 5. Consolidate Taxes
    taxes_facts, tax_conflicts = consolidate_taxes(tax_cands)
    all_conflicts.extend(tax_conflicts)

    # 6. Consolidate Discounts & Charges
    discs_facts, chgs_facts, dc_conflicts = consolidate_discounts_and_charges(disc_cands, chg_cands)
    all_conflicts.extend(dc_conflicts)

    # 7. Consolidate Printed Totals
    totals_facts, tot_conflicts = consolidate_printed_totals(tot_cands)
    all_conflicts.extend(tot_conflicts)

    # Financial Facts Container
    fin_evs = _union_evidence([
        *(ln.evidence_ids for ln in lines_facts),
        *(tx.evidence_ids for tx in taxes_facts),
        *(d.evidence_ids for d in discs_facts),
        *(c.evidence_ids for c in chgs_facts),
        totals_facts.evidence_ids if totals_facts else (),
    ])
    financials = FinancialFacts(
        lines=tuple(lines_facts),
        discounts=tuple(discs_facts),
        charges=tuple(chgs_facts),
        taxes=tuple(taxes_facts),
        printed_totals=totals_facts,
        currency=ident_facts.currency,
        evidence_ids=fin_evs,
    )

    # Top-Level Document Evidence
    doc_evs = _union_evidence([
        ident_facts.evidence_ids,
        parties_facts.evidence_ids,
        po_facts.evidence_ids,
        financials.evidence_ids,
    ])

    return DocumentFacts(
        document_id=doc_id,
        group_id=payable_group_id,
        page_numbers=tuple(sorted(page_numbers_set)),
        document_role=p_role,
        payable_relevance=p_relevance,
        supporting_group_ids=tuple(sorted(supporting_group_ids)),
        identity=ident_facts,
        parties=parties_facts,
        po=po_facts,
        financials=financials,
        conflicting_facts=tuple(all_conflicts),
        evidence_ids=doc_evs,
        provenance={
            "consolidator": "DeterministicCandidateConsolidator",
            "version": "Phase9B-2-1.0",
            "conflict_count": len(all_conflicts),
        },
    )


def consolidate_document_groups(
    candidates: Sequence[ExtractionCandidates],
) -> List[DocumentFacts]:
    """Consolidate candidates partitioned across every logical group in a document."""
    if not candidates:
        return []

    by_group: Dict[Optional[str], List[ExtractionCandidates]] = {}
    for c in candidates:
        by_group.setdefault(c.group_id, []).append(c)

    supporting_ids = [
        g_id for g_id, g_cands in by_group.items()
        if g_id and g_cands[0].payable_relevance == PayableRelevance.SUPPORTING
    ]

    results: List[DocumentFacts] = []
    for g_id, g_cands in by_group.items():
        doc_fact = consolidate_candidates(g_cands)
        # If this is not a supporting doc itself, attach supporting IDs
        if doc_fact.payable_relevance != PayableRelevance.SUPPORTING:
            doc_fact = DocumentFacts(
                document_id=doc_fact.document_id,
                group_id=doc_fact.group_id,
                page_numbers=doc_fact.page_numbers,
                document_role=doc_fact.document_role,
                payable_relevance=doc_fact.payable_relevance,
                supporting_group_ids=tuple(sorted(supporting_ids)),
                identity=doc_fact.identity,
                parties=doc_fact.parties,
                po=doc_fact.po,
                financials=doc_fact.financials,
                conflicting_facts=doc_fact.conflicting_facts,
                evidence_ids=doc_fact.evidence_ids,
                metadata=doc_fact.metadata,
                provenance=doc_fact.provenance,
            )
        results.append(doc_fact)

    return results
