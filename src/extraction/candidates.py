"""src/extraction/candidates.py — Candidate Semantic Extraction Layer.

Phase 9B-1: Converts raw document evidence (Phase 7A PageEvidence, Phase 7B
PageUnderstanding, Phase 7C DocumentGroup, Phase 7D RoutingDecision / Vision)
into typed semantic candidates.

Core Principles:
1. Extract WHAT THE DOCUMENT SHOWS — Do not decide accounting truth.
2. Zero Arithmetic: Missing quantities, unit prices, line amounts, or totals
   are never computed.
3. Conflict Preservation: Disagreeing (and agreeing) OCR vs Qwen candidates
   coexist with separate provenance.
4. Validation Boundary: All Qwen vision observations pass through an explicit
   adapter boundary ensuring valid evidence IDs.
5. Contextual Currency & Type: Currency symbols only map to ISO codes when
   contextual evidence justifies it; invoice types look to header evidence,
   not body references.
6. Row Representation: Raw table-cell and row representations are preserved
   in LineCandidate for later auditing and debugging.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.understanding.document_facts import (
    InvoiceType,
    Placement,
    SemanticRole,
    to_decimal,
)
from src.understanding.evidence import (
    Evidence,
    EvidenceSource,
    PageEvidence,
)
from src.understanding.page_classifier import PageRole, PayableRelevance, PageUnderstanding, classify_page
from src.understanding.document_grouper import DocumentGroup, group_document
from src.understanding.qwen_router import RouterDecisionType, RoutingDecision, route_page, execute_vision_escalation
from src.vision.provider import VisionProvider
from src.utils.logging import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Currency Normalization Helpers (Context-Aware)
# ══════════════════════════════════════════════════════════════════════════

# Eurozone country prefixes / codes
_EUROZONE_COUNTRIES: Set[str] = {
    "AT", "BE", "CY", "DE", "EE", "ES", "FI", "FR", "GR", "HR",
    "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PT", "SI", "SK",
    "GERMANY", "FRANCE", "ESTONIA", "PORTUGAL", "SPAIN", "ITALY",
    "DEUTSCHLAND", "BELGIUM", "AUSTRIA", "FINLAND", "GREECE", "IRELAND",
}


_CURRENCY_NAME_MAP: Dict[str, str] = {
    "BAHT": "THB",
    "บาท": "THB",
    "EURO": "EUR",
    "EUROS": "EUR",
    "DOLLAR": "USD",
    "DOLLARS": "USD",
    "POUND": "GBP",
    "POUNDS": "GBP",
    "CEDI": "GHS",
    "CEDIS": "GHS",
    "LIRA": "TRY",
}

_STANDARD_ISO_CODES: Set[str] = {
    "EUR", "USD", "GBP", "THB", "CHF", "JPY", "CAD", "AUD",
    "SEK", "NOK", "DKK", "PLN", "GHS", "SGD", "TRY",
}


def normalize_currency_with_context(
    raw_curr: Optional[str],
    context_tokens: Sequence[str] = (),
) -> Tuple[Optional[str], bool]:
    """Normalize raw currency text to ISO code only when justified by context.
    
    Returns:
        (normalized_currency, is_context_justified)
    """
    if not raw_curr:
        return (None, False)
    
    clean = raw_curr.strip()
    upper = clean.upper()
    
    # Already a standard 3-letter ISO code
    if upper in _STANDARD_ISO_CODES:
        return (upper, True)

    # Named currency lookup
    for name, iso in _CURRENCY_NAME_MAP.items():
        if name in upper or name in clean:
            return (iso, True)
    
    # Context tokens flattened to uppercase for matching
    upper_context = {tok.upper() for tok in context_tokens if tok}
    
    # Ghanaian Cedi aliases: GH¢, GH₵, GHC, GH$, GH€, GHS
    if any(alias in clean for alias in ("GH¢", "GH₵", "GHC", "GH$", "GH€", "GHS")) or re.search(r"\bGH[C\u00A2\u058F\u20B5c€]\b", clean, re.IGNORECASE):
        return ("GHS", True)

    # Euro symbol: €
    if "€" in clean:
        # Check if eurozone country or VAT ID or IBAN is present in context
        has_eurozone_evidence = any(
            c in upper_context
            or any(tok.startswith(c) and len(tok) >= 5 and tok[len(c):].isalnum() for tok in upper_context)
            for c in _EUROZONE_COUNTRIES
        )
        if has_eurozone_evidence:
            return ("EUR", True)
        return ("€", False)  # Preserve raw symbol when context is absent

    
    # Dollar symbol: $
    if "$" in clean:
        has_us_evidence = any(
            tok in upper_context for tok in {"US", "USA", "UNITED STATES", "USD"}
        )
        if has_us_evidence:
            return ("USD", True)
        # Check if contextual evidence indicates Ghanaian Cedi (e.g. GH$ or GHC in context)
        has_ghana_evidence = any(
            any(alias in tok for alias in ("GH¢", "GH₵", "GHC", "GH$", "GHS", "GHANA", "ACCRA"))
            for tok in upper_context
        )
        if has_ghana_evidence:
            return ("GHS", True)
        # Without explicit US context, could be CAD/AUD/USD -> preserve raw symbol
        return ("$", False)
    
    # Pound symbol: £
    if "£" in clean:
        has_uk_evidence = any(
            tok in upper_context for tok in {"GB", "UK", "UNITED KINGDOM", "GBP"}
        )
        if has_uk_evidence:
            return ("GBP", True)
        return ("£", False)
    
    # Thai Baht symbol: ฿
    if "฿" in clean or "บาท" in clean:
        return ("THB", True)
    
    return (clean, False)


# ══════════════════════════════════════════════════════════════════════════
# Candidate Models (Immutable Typed Data Classes)
# ══════════════════════════════════════════════════════════════════════════

def _validate_candidate_evidence(candidate_type: str, evidence_ids: Tuple[str, ...]) -> None:
    """Ensure every accounting-relevant candidate retains valid evidence provenance."""
    if not evidence_ids:
        raise ValueError(f"Naked candidate in {candidate_type}: must contain at least one valid evidence ID.")


@dataclass(frozen=True)
class DocumentIdentityCandidate:
    """Candidate for invoice identity (invoice number, date, due date, type, currency)."""
    field_name: str
    raw_value: str
    normalized_value: Any
    evidence_ids: Tuple[str, ...]
    source: str = "ocr"  # "ocr" | "vision" | "native_pdf"
    page_number: int = 1
    group_id: Optional[str] = None
    invoice_type_value: Optional[InvoiceType] = None
    semantic_role: Optional[SemanticRole] = None
    placement: Optional[Placement] = None
    extraction_notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ev_tuple = tuple(str(x) for x in self.evidence_ids if x)
        _validate_candidate_evidence("DocumentIdentityCandidate", ev_tuple)
        object.__setattr__(self, "evidence_ids", ev_tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_type": "document_identity",
            "field_name": self.field_name,
            "raw_value": self.raw_value,
            "normalized_value": str(self.normalized_value) if isinstance(self.normalized_value, Decimal) else self.normalized_value,
            "evidence_ids": list(self.evidence_ids),
            "source": self.source,
            "page_number": self.page_number,
            "group_id": self.group_id,
            "invoice_type_value": self.invoice_type_value.value if self.invoice_type_value else None,
            "semantic_role": self.semantic_role.value if self.semantic_role else None,
            "placement": self.placement.value if self.placement else None,
            "extraction_notes": dict(self.extraction_notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentIdentityCandidate:
        inv_val = data.get("invoice_type_value")
        inv_type = InvoiceType(inv_val) if inv_val else None
        sem_val = data.get("semantic_role")
        sem_role = SemanticRole(sem_val) if sem_val else None
        plc_val = data.get("placement")
        plc = Placement(plc_val) if plc_val else None
        return cls(
            field_name=data["field_name"],
            raw_value=data["raw_value"],
            normalized_value=data.get("normalized_value"),
            evidence_ids=tuple(data.get("evidence_ids") or []),
            source=data.get("source", "ocr"),
            page_number=int(data.get("page_number", 1)),
            group_id=data.get("group_id"),
            invoice_type_value=inv_type,
            semantic_role=sem_role,
            placement=plc,
            extraction_notes=dict(data.get("extraction_notes") or {}),
        )


@dataclass(frozen=True)
class PartyIdentityCandidate:
    """Candidate for supplier or buyer identity attributes."""
    party_role: str  # "supplier" | "buyer"
    field_name: str  # "name" | "vat_id" | "country" | "email" | "bank_iban" | "address" | ...
    raw_value: str
    normalized_value: Any
    evidence_ids: Tuple[str, ...]
    source: str = "ocr"
    page_number: int = 1
    group_id: Optional[str] = None
    extraction_notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ev_tuple = tuple(str(x) for x in self.evidence_ids if x)
        _validate_candidate_evidence("PartyIdentityCandidate", ev_tuple)
        object.__setattr__(self, "evidence_ids", ev_tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_type": "party_identity",
            "party_role": self.party_role,
            "field_name": self.field_name,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "evidence_ids": list(self.evidence_ids),
            "source": self.source,
            "page_number": self.page_number,
            "group_id": self.group_id,
            "extraction_notes": dict(self.extraction_notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PartyIdentityCandidate:
        return cls(
            party_role=data["party_role"],
            field_name=data["field_name"],
            raw_value=data["raw_value"],
            normalized_value=data.get("normalized_value"),
            evidence_ids=tuple(data.get("evidence_ids") or []),
            source=data.get("source", "ocr"),
            page_number=int(data.get("page_number", 1)),
            group_id=data.get("group_id"),
            extraction_notes=dict(data.get("extraction_notes") or {}),
        )


@dataclass(frozen=True)
class POCandidate:
    """Candidate for purchase order or order reference numbers."""
    field_name: str
    raw_value: str
    normalized_value: Any
    evidence_ids: Tuple[str, ...]
    source: str = "ocr"
    page_number: int = 1
    group_id: Optional[str] = None
    label: Optional[str] = None
    extraction_notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ev_tuple = tuple(str(x) for x in self.evidence_ids if x)
        _validate_candidate_evidence("POCandidate", ev_tuple)
        object.__setattr__(self, "evidence_ids", ev_tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_type": "po_reference",
            "field_name": self.field_name,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "evidence_ids": list(self.evidence_ids),
            "source": self.source,
            "page_number": self.page_number,
            "group_id": self.group_id,
            "label": self.label,
            "extraction_notes": dict(self.extraction_notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> POCandidate:
        return cls(
            field_name=data.get("field_name", "po_number"),
            raw_value=data["raw_value"],
            normalized_value=data.get("normalized_value"),
            evidence_ids=tuple(data.get("evidence_ids") or []),
            source=data.get("source", "ocr"),
            page_number=int(data.get("page_number", 1)),
            group_id=data.get("group_id"),
            label=data.get("label"),
            extraction_notes=dict(data.get("extraction_notes") or {}),
        )


@dataclass(frozen=True)
class LineCandidate:
    """Candidate for a table row or line item.
    
    Preserves:
    - source_row_number (source table row numbering, not final accounting numbering)
    - raw_cells (original cell tokens for debugging)
    - raw_row (full text representation)
    - strict omission of arithmetic: missing amounts/prices remain None.
    """
    source_row_number: Optional[int] = None
    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    discount: Optional[Decimal] = None
    currency: Optional[str] = None
    raw_row: str = ""
    raw_cells: Tuple[str, ...] = field(default_factory=tuple)
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    source: str = "ocr"
    page_number: int = 1
    group_id: Optional[str] = None
    semantic_role: SemanticRole = SemanticRole.UNKNOWN
    extraction_notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_qty = to_decimal(self.quantity) if self.quantity is not None else None
        dec_price = to_decimal(self.unit_price) if self.unit_price is not None else None
        dec_amt = to_decimal(self.amount) if self.amount is not None else None
        dec_disc = to_decimal(self.discount) if self.discount is not None else None

        role = (
            self.semantic_role
            if isinstance(self.semantic_role, SemanticRole)
            else SemanticRole(str(self.semantic_role).lower())
        )
        ev_tuple = tuple(str(x) for x in self.evidence_ids if x)
        _validate_candidate_evidence("LineCandidate", ev_tuple)

        object.__setattr__(self, "quantity", dec_qty)
        object.__setattr__(self, "unit_price", dec_price)
        object.__setattr__(self, "amount", dec_amt)
        object.__setattr__(self, "discount", dec_disc)
        object.__setattr__(self, "semantic_role", role)
        object.__setattr__(self, "evidence_ids", ev_tuple)
        object.__setattr__(self, "raw_cells", tuple(str(c) for c in self.raw_cells))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_type": "line_item",
            "source_row_number": self.source_row_number,
            "description": self.description,
            "quantity": str(self.quantity) if self.quantity is not None else None,
            "unit_price": str(self.unit_price) if self.unit_price is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "discount": str(self.discount) if self.discount is not None else None,
            "currency": self.currency,
            "raw_row": self.raw_row,
            "raw_cells": list(self.raw_cells),
            "evidence_ids": list(self.evidence_ids),
            "source": self.source,
            "page_number": self.page_number,
            "group_id": self.group_id,
            "semantic_role": self.semantic_role.value,
            "extraction_notes": dict(self.extraction_notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LineCandidate:
        return cls(
            source_row_number=data.get("source_row_number"),
            description=data.get("description"),
            quantity=to_decimal(data.get("quantity")),
            unit_price=to_decimal(data.get("unit_price")),
            amount=to_decimal(data.get("amount")),
            discount=to_decimal(data.get("discount")),
            currency=data.get("currency"),
            raw_row=data.get("raw_row", ""),
            raw_cells=tuple(data.get("raw_cells") or []),
            evidence_ids=tuple(data.get("evidence_ids") or []),
            source=data.get("source", "ocr"),
            page_number=int(data.get("page_number", 1)),
            group_id=data.get("group_id"),
            semantic_role=SemanticRole(data.get("semantic_role", "unknown")),
            extraction_notes=dict(data.get("extraction_notes") or {}),
        )


@dataclass(frozen=True)
class TaxCandidate:
    """Candidate for document tax components (VAT, GST, Sales Tax)."""
    tax_name: Optional[str] = None
    tax_type: Optional[str] = None
    rate: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    raw_value: str = ""
    placement: Placement = Placement.UNKNOWN
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    source: str = "ocr"
    page_number: int = 1
    group_id: Optional[str] = None
    extraction_notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_rate = to_decimal(self.rate) if self.rate is not None else None
        dec_amt = to_decimal(self.amount) if self.amount is not None else None
        plc = (
            self.placement
            if isinstance(self.placement, Placement)
            else Placement(str(self.placement).lower())
        )
        ev_tuple = tuple(str(x) for x in self.evidence_ids if x)
        _validate_candidate_evidence("TaxCandidate", ev_tuple)

        object.__setattr__(self, "rate", dec_rate)
        object.__setattr__(self, "amount", dec_amt)
        object.__setattr__(self, "placement", plc)
        object.__setattr__(self, "evidence_ids", ev_tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_type": "tax",
            "tax_name": self.tax_name,
            "tax_type": self.tax_type,
            "rate": str(self.rate) if self.rate is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "raw_value": self.raw_value,
            "placement": self.placement.value,
            "evidence_ids": list(self.evidence_ids),
            "source": self.source,
            "page_number": self.page_number,
            "group_id": self.group_id,
            "extraction_notes": dict(self.extraction_notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaxCandidate:
        return cls(
            tax_name=data.get("tax_name"),
            tax_type=data.get("tax_type"),
            rate=to_decimal(data.get("rate")),
            amount=to_decimal(data.get("amount")),
            raw_value=data.get("raw_value", ""),
            placement=Placement(data.get("placement", "unknown")),
            evidence_ids=tuple(data.get("evidence_ids") or []),
            source=data.get("source", "ocr"),
            page_number=int(data.get("page_number", 1)),
            group_id=data.get("group_id"),
            extraction_notes=dict(data.get("extraction_notes") or {}),
        )


@dataclass(frozen=True)
class DiscountCandidate:
    """Candidate for early payment or volume discounts."""
    label: Optional[str] = None
    rate: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    raw_value: str = ""
    placement: Placement = Placement.UNKNOWN
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    source: str = "ocr"
    page_number: int = 1
    group_id: Optional[str] = None
    extraction_notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_rate = to_decimal(self.rate) if self.rate is not None else None
        dec_amt = to_decimal(self.amount) if self.amount is not None else None
        plc = (
            self.placement
            if isinstance(self.placement, Placement)
            else Placement(str(self.placement).lower())
        )
        ev_tuple = tuple(str(x) for x in self.evidence_ids if x)
        _validate_candidate_evidence("DiscountCandidate", ev_tuple)

        object.__setattr__(self, "rate", dec_rate)
        object.__setattr__(self, "amount", dec_amt)
        object.__setattr__(self, "placement", plc)
        object.__setattr__(self, "evidence_ids", ev_tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_type": "discount",
            "label": self.label,
            "rate": str(self.rate) if self.rate is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "raw_value": self.raw_value,
            "placement": self.placement.value,
            "evidence_ids": list(self.evidence_ids),
            "source": self.source,
            "page_number": self.page_number,
            "group_id": self.group_id,
            "extraction_notes": dict(self.extraction_notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiscountCandidate:
        return cls(
            label=data.get("label"),
            rate=to_decimal(data.get("rate")),
            amount=to_decimal(data.get("amount")),
            raw_value=data.get("raw_value", ""),
            placement=Placement(data.get("placement", "unknown")),
            evidence_ids=tuple(data.get("evidence_ids") or []),
            source=data.get("source", "ocr"),
            page_number=int(data.get("page_number", 1)),
            group_id=data.get("group_id"),
            extraction_notes=dict(data.get("extraction_notes") or {}),
        )


@dataclass(frozen=True)
class ChargeCandidate:
    """Candidate for ancillary charges (shipping, handling, service fees)."""
    label: Optional[str] = None
    rate: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    raw_value: str = ""
    placement: Placement = Placement.UNKNOWN
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    source: str = "ocr"
    page_number: int = 1
    group_id: Optional[str] = None
    extraction_notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_rate = to_decimal(self.rate) if self.rate is not None else None
        dec_amt = to_decimal(self.amount) if self.amount is not None else None
        plc = (
            self.placement
            if isinstance(self.placement, Placement)
            else Placement(str(self.placement).lower())
        )
        ev_tuple = tuple(str(x) for x in self.evidence_ids if x)
        _validate_candidate_evidence("ChargeCandidate", ev_tuple)

        object.__setattr__(self, "rate", dec_rate)
        object.__setattr__(self, "amount", dec_amt)
        object.__setattr__(self, "placement", plc)
        object.__setattr__(self, "evidence_ids", ev_tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_type": "charge",
            "label": self.label,
            "rate": str(self.rate) if self.rate is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "raw_value": self.raw_value,
            "placement": self.placement.value,
            "evidence_ids": list(self.evidence_ids),
            "source": self.source,
            "page_number": self.page_number,
            "group_id": self.group_id,
            "extraction_notes": dict(self.extraction_notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChargeCandidate:
        return cls(
            label=data.get("label"),
            rate=to_decimal(data.get("rate")),
            amount=to_decimal(data.get("amount")),
            raw_value=data.get("raw_value", ""),
            placement=Placement(data.get("placement", "unknown")),
            evidence_ids=tuple(data.get("evidence_ids") or []),
            source=data.get("source", "ocr"),
            page_number=int(data.get("page_number", 1)),
            group_id=data.get("group_id"),
            extraction_notes=dict(data.get("extraction_notes") or {}),
        )


@dataclass(frozen=True)
class TotalCandidate:
    """Candidate for printed document totals."""
    total_type: str  # "subtotal" | "net" | "taxable_base" | "tax_total" | "gross_total" | "amount_due" | "payment_total" | "other"
    raw_label: str
    raw_value: str
    normalized_value: Optional[Decimal]
    evidence_ids: Tuple[str, ...]
    source: str = "ocr"
    page_number: int = 1
    group_id: Optional[str] = None
    extraction_notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_val = to_decimal(self.normalized_value) if self.normalized_value is not None else None
        ev_tuple = tuple(str(x) for x in self.evidence_ids if x)
        _validate_candidate_evidence("TotalCandidate", ev_tuple)

        object.__setattr__(self, "normalized_value", dec_val)
        object.__setattr__(self, "evidence_ids", ev_tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_type": "printed_total",
            "total_type": self.total_type,
            "raw_label": self.raw_label,
            "raw_value": self.raw_value,
            "normalized_value": str(self.normalized_value) if self.normalized_value is not None else None,
            "evidence_ids": list(self.evidence_ids),
            "source": self.source,
            "page_number": self.page_number,
            "group_id": self.group_id,
            "extraction_notes": dict(self.extraction_notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TotalCandidate:
        return cls(
            total_type=data["total_type"],
            raw_label=data.get("raw_label", ""),
            raw_value=data.get("raw_value", ""),
            normalized_value=to_decimal(data.get("normalized_value")),
            evidence_ids=tuple(data.get("evidence_ids") or []),
            source=data.get("source", "ocr"),
            page_number=int(data.get("page_number", 1)),
            group_id=data.get("group_id"),
            extraction_notes=dict(data.get("extraction_notes") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Candidate Container Model
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExtractionCandidates:
    """Container holding all extracted semantic candidates for a page or document group."""
    document_id: str
    group_id: Optional[str] = None
    page_numbers: Tuple[int, ...] = field(default_factory=tuple)
    payable_relevance: PayableRelevance = PayableRelevance.PAYABLE_CANDIDATE
    page_role: PageRole = PageRole.INVOICE
    identity_candidates: Tuple[DocumentIdentityCandidate, ...] = field(default_factory=tuple)
    party_candidates: Tuple[PartyIdentityCandidate, ...] = field(default_factory=tuple)
    po_candidates: Tuple[POCandidate, ...] = field(default_factory=tuple)
    line_candidates: Tuple[LineCandidate, ...] = field(default_factory=tuple)
    tax_candidates: Tuple[TaxCandidate, ...] = field(default_factory=tuple)
    discount_candidates: Tuple[DiscountCandidate, ...] = field(default_factory=tuple)
    charge_candidates: Tuple[ChargeCandidate, ...] = field(default_factory=tuple)
    total_candidates: Tuple[TotalCandidate, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "group_id": self.group_id,
            "page_numbers": list(self.page_numbers),
            "payable_relevance": self.payable_relevance.value,
            "page_role": self.page_role.value,
            "identity_candidates": [c.to_dict() for c in self.identity_candidates],
            "party_candidates": [c.to_dict() for c in self.party_candidates],
            "po_candidates": [c.to_dict() for c in self.po_candidates],
            "line_candidates": [c.to_dict() for c in self.line_candidates],
            "tax_candidates": [c.to_dict() for c in self.tax_candidates],
            "discount_candidates": [c.to_dict() for c in self.discount_candidates],
            "charge_candidates": [c.to_dict() for c in self.charge_candidates],
            "total_candidates": [c.to_dict() for c in self.total_candidates],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractionCandidates:
        return cls(
            document_id=data["document_id"],
            group_id=data.get("group_id"),
            page_numbers=tuple(data.get("page_numbers") or []),
            payable_relevance=PayableRelevance(data.get("payable_relevance", "payable_candidate")),
            page_role=PageRole(data.get("page_role", "invoice")),
            identity_candidates=tuple(DocumentIdentityCandidate.from_dict(c) for c in (data.get("identity_candidates") or [])),
            party_candidates=tuple(PartyIdentityCandidate.from_dict(c) for c in (data.get("party_candidates") or [])),
            po_candidates=tuple(POCandidate.from_dict(c) for c in (data.get("po_candidates") or [])),
            line_candidates=tuple(LineCandidate.from_dict(c) for c in (data.get("line_candidates") or [])),
            tax_candidates=tuple(TaxCandidate.from_dict(c) for c in (data.get("tax_candidates") or [])),
            discount_candidates=tuple(DiscountCandidate.from_dict(c) for c in (data.get("discount_candidates") or [])),
            charge_candidates=tuple(ChargeCandidate.from_dict(c) for c in (data.get("charge_candidates") or [])),
            total_candidates=tuple(TotalCandidate.from_dict(c) for c in (data.get("total_candidates") or [])),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> ExtractionCandidates:
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# Reusable Structural Extractor
# ══════════════════════════════════════════════════════════════════════════

# Concise patterns for key labels
_INV_NUM_LABEL = re.compile(
    r"^\s*(?:(?:tax\s+)?(?:invoice|rechnung|factura|facture|arve)\s*(?:no\.?|nr\.?|number|#)|(?:invoice|rechnung|factura|facture|arve)\s*[:#]|(?:no\.?|nr\.?|number|#)\s*[:#\-]?)\s*[:#\-]?\s*$",
    re.IGNORECASE,
)
_INV_NUM_INLINE = re.compile(
    r"\b(?:tax\s+)?(?:invoice|rechnung|factura|facture|arve)\s*(?:(?:no\.?|nr\.?|number|#)\s*[:#\-]?:?|[:#\-]|(?=\d))\s*([A-Z0-9\-/]{2,})\b",
    re.IGNORECASE,
)


_DATE_INLINE = re.compile(
    r"\b(?:invoice\s*date|rechnungsdatum|date|datum|data|kuupäev)\s*[:#\-]?\s*(\d{1,4}[./\-]\d{1,2}[./\-]\d{2,4})\b",
    re.IGNORECASE,
)
_DUE_DATE_INLINE = re.compile(
    r"\b(?:due\s*date|fälligkeitsdatum|zahlungsziel|vencimento|maksetähtaeg)\s*[:#\-]?\s*(\d{1,4}[./\-]\d{1,2}[./\-]\d{2,4})\b",
    re.IGNORECASE,
)

_PO_INLINE = re.compile(
    r"\b(?:customer\s*po|client\s*po|po\s*(?:no\.?|#|number)|purchase\s*order\s*(?:no\.?|#|number)|bestellnummer|bestell-nr\.?)\s*[:#\-]?\s*([A-Z0-9\-/]{3,})\b",
    re.IGNORECASE,
)

_VAT_INLINE = re.compile(
    r"\b(?:vat\s*(?:id|no\.?|number)?|ust-idnr\.?|nif|tin|tax\s*id)\s*[:#\-]?\s*([A-Z0-9\-]{5,20})\b",
    re.IGNORECASE,
)

_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IBAN_REGEX = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b")

_TOTAL_LABELS: List[Tuple[str, re.Pattern]] = [
    ("taxable_base", re.compile(r"\b(?:total\s*levy\s*inclusive\s*value|levy\s*inclusive\s*value|taxable\s*base|base\s*imponible|base\s*tributável|steuerpflichtiger\s*betrag)\b", re.IGNORECASE)),
    ("subtotal", re.compile(r"\b(?:total\s*tax\s*exclusive\s*value|tax\s*exclusive\s*value|tax\s*exclusive|subtotal|zwischensumme|sous-total|vahesumma|summa\s*ilma\s*k[aä]ibemaksuta|total\s*il[ií]quido)\b", re.IGNORECASE)),
    ("amount_due", re.compile(r"\b(?:amount\s*due|balance\s*due|net\s*payable|fälliger\s*betrag|tasumata|total\s*payment|จำนวนเงินที่ต้องช่าระ|a\s*pagar)\b", re.IGNORECASE)),
    ("payment_total", re.compile(r"\b(?:payment\s*total|total\s*paid|bezahlt|total\s*pago)\b", re.IGNORECASE)),
    ("gross_total", re.compile(r"\b(?:total\s*tax\s*inclusive\s*value|tax\s*inclusive\s*value|grand\s*total|gross\s*total|endbetrag|gesamtsumme|ยอดรวมทั้งสิ้น|summa\s*koos\s*k[aä]ibemaksuga|total\s*da\s*factura|total\s*(?:amount|ttc)?)\b", re.IGNORECASE)),
    ("net", re.compile(r"\b(?:nettobetrag|net\s*amount|total\s*net|total\s*l[ií]quido)\b", re.IGNORECASE)),
    ("tax_total", re.compile(r"\b(?:tax\s*total|total\s*tax|mwst\.?\s*gesamt|total\s*iva)\b", re.IGNORECASE)),
]

_TAX_INLINE = re.compile(
    r"\b(?:mwst|vat|iva|tax|impuesto)\b(?:\s*(?:zzgl\.?|inkl\.?)?\s*(\d+(?:[.,]\d+)?)\s*%)?(?:\s*[:#\-]?\s*(\d+(?:[.,]\d+)?)(?!\s*%))?",
    re.IGNORECASE,
)

_CURRENCY_SYMBOLS: List[str] = [
    "GH¢", "GH₵", "GHC", "GH$", "GH€", "GHS",
    "EUR", "USD", "GBP", "CHF", "THB", "SGD", "TRY", "PLN",
    "€", "£", "฿", "$", "BAHT", "บาท"
]

_INDEX_HEADER_RE = re.compile(r"^(?:pos|pos\.?|item|nr\.?|no\.?|ลำดับ|item\s*no\.?)$", re.IGNORECASE)
_DESC_HEADER_RE = re.compile(r"\b(?:description|beschreibung|désignation|bezeichnung|artikel|deqgkgdfiyt|slubls|service|product|item\s*desc|product\s*number\s*/\s*description)\b", re.IGNORECASE)
_QTY_HEADER_RE = re.compile(r"\b(?:qty|quantity|menge|quantidades|qtd|kogus|anzahl|nenle|units)\b", re.IGNORECASE)
_PRICE_HEADER_RE = re.compile(r"\b(?:unit\s*price|einzelpreis|preço\s*unitário|unit\s*cost|prix\s*unitaire|hind|benneululs|price|preis|list\s*price(?:\s*per\s*unit)?)\b", re.IGNORECASE)
_TOTAL_HEADER_RE = re.compile(r"\b(?:total|amount|gesamtpreis|extended\s*total|ilíquido|kokku|ngnenle|ext\s*price|montant\s*ht)\b", re.IGNORECASE)
_PROD_CODE_HEADER_RE = re.compile(r"\b(?:product\s*(?:number|no|nr|code)|cust\.?\s*mat-no|part\s*(?:no|number)|art(?:\.|ikel)?(?:-?nr\.?|-?no\.?)?|sku|ean|model)\b", re.IGNORECASE)
_HTS_HEADER_RE = re.compile(r"\b(?:hts|tariff|hs\s*code|customs\s*tariff)\b", re.IGNORECASE)
_TAX_COL_HEADER_RE = re.compile(r"\b(?:tax|vat|iva|mwst|tva|gst)\b", re.IGNORECASE)
_DISC_COL_HEADER_RE = re.compile(r"\b(?:disc(?:ount)?|rabatt|desconto)\b", re.IGNORECASE)
_WEIGHT_HEADER_RE = re.compile(r"\b(?:weight|gewicht|net\s*weight|poids|peso)\b", re.IGNORECASE)
_DURATION_HEADER_RE = re.compile(r"\b(?:duration|dauer|tundi|hours|zeitraum)\b", re.IGNORECASE)

_SUMMARY_START_RE = re.compile(
    r"\b(?:subtotal|zwischensumme|vahesumma|total\s*(?:exclusive|inclusive|due|amount|payment|da\s*factura|ilíquido)|"
    r"grand\s*total|endbetrag|gesamtsumme|amount\s*due|balance\s*due|net\s*payable|fälliger\s*betrag|tasumata|"
    r"summa\s*(?:ilma|koos)?\s*k[aä]ibemaksuta|summa\s*koos\s*k[aä]ibemaksuga|tax\s*exclusive|levy\s*inclusive|tax\s*inclusive|"
    r"total\s*payment|จำนวนเงินที่ต้องช่าระ|total\s*\(including|bitte\s*begleichen|bankverbindung|iban:|swift:|"
    r"processado\s*por\s*programa|approved\s*by|presented\s*by|total\s*:|total\s*il[ií]quido)\b",
    re.IGNORECASE,
)

_METADATA_LINE_RE = re.compile(
    r"\b(?:vat\s*(?:id|no\.?|number)?|ust-idnr\.?|nif|tin|tax\s*id|nipc|eori|เลขประจำตัวผู้เสียภาษี|"
    r"invoice\s*(?:no\.?|#|number|date)|rechnung\s*(?:nr\.?|nummer)|rechnungsdatum|arve\s*#|tellimus\s*#|"
    r"kunden-nr\.?|customer\s*po|sales\s*order|delivery(?:\s*group)?|bestell-nr\.?|n°\s*cliente|"
    r"iban|swift(?:\s*code)?|sort\s*code|account\s*(?:no\.?|number|#)|bank\s*name|branch|kto(?:\.|-nr)?|blz|"
    r"tel|telefon|telephone|phone|fax|เบอร์โทร|โทร|แฟกซ์|"
    r"street|str\.|strasse|straße|avenue|road|boulevard|p\.?o\.?\s*box|postfach|"
    r"harjumaa|tallinn|tartu|kadaka\s*tee|lindenstrasse|avenida|machelen|kagithane|istanbul|kraainem|lisboa|"
    r"ที่อยู่|ตำบล|จังหวัด|ean:\s*\d+|decreto-lei|programa\s*certificado|c\.r\.c\.|sirer\s*#|capital\s*social|"
    r"vendor\s*details|customer\s*details|bank\s*details|prepared\s*for|ship\s*to|ship\s*from|importer\s*of\s*record|"
    r"billing\s*information|shipment\s*information|invoice\s*information|sehr\s*geehrte[rn]?|the\s*detail\s*are\s*as\s*below|"
    r"due\s*date|fälligkeitsdatum|zahlungsziel|vencimento|maksetähtaeg|väljastamine|tagastamine|วันที่|zeitraum\s*von|"
    r"project|ซื่องาน)\b",
    re.IGNORECASE,
)

_DURATION_TOKEN_RE = re.compile(r"^\d+\s*(?:tundi|h|std|hrs|hours|päev|days)$", re.IGNORECASE)
_QTY_PREFIX_RE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*[x×]\b", re.IGNORECASE)
_LEADING_QTY_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:[xX]|\u00D7)(?:\s+|$)", re.IGNORECASE)

class _ColDef:
    def __init__(self, role: str, x_min: float, x_max: float, x_mid: float):
        self.role = role
        self.x_min = x_min
        self.x_max = x_max
        self.x_mid = x_mid



def extract_identity_from_evidence(
    page_evidence: PageEvidence,
    page_understanding: Optional[PageUnderstanding] = None,
    group_id: Optional[str] = None,
    source: str = "ocr",
) -> List[DocumentIdentityCandidate]:
    """Extract document identity candidates (invoice number, dates, type, currency)."""
    candidates: List[DocumentIdentityCandidate] = []
    items = page_evidence.items
    p_num = page_evidence.page_number
    all_text = [item.content for item in items]
    context_tokens = [tok for t in all_text for tok in t.split()]

    # 1. Invoice Number Candidates
    for i, item in enumerate(items):
        text = item.content.strip()
        # Check if line is purely a label for a split block
        if _INV_NUM_LABEL.search(text) and i + 1 < len(items):
            next_val = items[i + 1].content.strip()
            if next_val and len(next_val) >= 2 and next_val.upper() not in {"NUMBER", "DATE", "PAGE", "TOTAL"}:
                ev_ids = tuple(dict.fromkeys([item.evidence_id, items[i + 1].evidence_id]))
                candidates.append(DocumentIdentityCandidate(
                    field_name="invoice_number",
                    raw_value=next_val,
                    normalized_value=next_val,
                    evidence_ids=ev_ids,
                    source=source,
                    page_number=p_num,
                    group_id=group_id,
                    extraction_notes={"match_type": "split_label_val"},
                ))
        else:
            # Inline match
            m_inline = _INV_NUM_INLINE.search(text)
            if m_inline:
                val = m_inline.group(1).strip()
                # Verify not a common stop word
                if val.upper() not in {"NUMBER", "DATE", "TOTAL", "INVOICE", "PAGE"}:
                    candidates.append(DocumentIdentityCandidate(
                        field_name="invoice_number",
                        raw_value=val,
                        normalized_value=val,
                        evidence_ids=(item.evidence_id,),
                        source=source,
                        page_number=p_num,
                        group_id=group_id,
                        extraction_notes={"match_type": "inline_pattern", "line": text},
                    ))

    # 2. Date Candidates (Invoice Date & Due Date)
    for item in items:
        text = item.content.strip()
        m_date = _DATE_INLINE.search(text)
        if m_date:
            raw_d = m_date.group(1).strip()
            candidates.append(DocumentIdentityCandidate(
                field_name="invoice_date",
                raw_value=raw_d,
                normalized_value=raw_d,
                evidence_ids=(item.evidence_id,),
                source=source,
                page_number=p_num,
                group_id=group_id,
                extraction_notes={"match_type": "invoice_date"},
            ))

        m_due = _DUE_DATE_INLINE.search(text)
        if m_due:
            raw_due = m_due.group(1).strip()
            candidates.append(DocumentIdentityCandidate(
                field_name="due_date",
                raw_value=raw_due,
                normalized_value=raw_due,
                evidence_ids=(item.evidence_id,),
                source=source,
                page_number=p_num,
                group_id=group_id,
                extraction_notes={"match_type": "due_date"},
            ))

    # 3. Currency Candidates (Context-Aware)
    found_currencies: Set[str] = set()
    for item in items:
        text = item.content.strip()
        matched_symbols_in_item: Set[str] = set()
        for sym in _CURRENCY_SYMBOLS:
            is_matched = bool(re.search(rf"\b{re.escape(sym)}\b", text, re.IGNORECASE)) if sym.isalpha() else (sym in text)
            if is_matched:
                if sym in ("$", "€") and (
                    any(g in matched_symbols_in_item for g in ("GH$", "GH¢", "GH₵", "GHC", "GH€", "GHS"))
                    or f"GH{sym}" in text
                ):
                    continue
                matched_symbols_in_item.add(sym)
                if sym not in found_currencies:
                    found_currencies.add(sym)
                    norm_curr, justified = normalize_currency_with_context(sym, context_tokens)
                    candidates.append(DocumentIdentityCandidate(
                        field_name="currency",
                        raw_value=sym,
                        normalized_value=norm_curr,
                        evidence_ids=(item.evidence_id,),
                        source=source,
                        page_number=p_num,
                        group_id=group_id,
                        extraction_notes={"context_justified": justified},
                    ))

    # 4. Invoice Type Candidates (Contextual Header Evidence Only)
    # Check top header blocks (e.g. first 8 blocks or top 30% of page)
    header_items = items[:min(10, len(items))]
    type_candidate_emitted = False

    for h_item in header_items:
        h_text = h_item.content.strip()
        
        # Check for Credit Memo in header
        if re.search(r"\b(?:credit\s*(?:memo|note)|gutschrift|note\s*de\s*cr[eé]dit|nota\s*de\s*cr[eé]dito|ใบลดหนี้)\b", h_text, re.IGNORECASE):
            # Verify this is not a contextual reference to a prior invoice/credit memo
            if not re.search(r"\b(?:replaces|referencing|bezug\s*auf|gemäß|ref\.?)\b", h_text, re.IGNORECASE):
                candidates.append(DocumentIdentityCandidate(
                    field_name="invoice_type",
                    raw_value=h_text,
                    normalized_value="credit_memo",
                    invoice_type_value=InvoiceType.CREDIT_MEMO,
                    evidence_ids=(h_item.evidence_id,),
                    source=source,
                    page_number=p_num,
                    group_id=group_id,
                    extraction_notes={"header_title": h_text},
                ))
                type_candidate_emitted = True
                break

        # Check for Debit Memo in header
        elif re.search(r"\b(?:debit\s*(?:memo|note)|lastschrift|note\s*de\s*d[eé]bit|nota\s*de\s*d[eé]bito|ใบเพิ่มหนี้)\b", h_text, re.IGNORECASE):
            if not re.search(r"\b(?:replaces|referencing|bezug|ref\.?)\b", h_text, re.IGNORECASE):
                candidates.append(DocumentIdentityCandidate(
                    field_name="invoice_type",
                    raw_value=h_text,
                    normalized_value="debit_memo",
                    invoice_type_value=InvoiceType.DEBIT_MEMO,
                    evidence_ids=(h_item.evidence_id,),
                    source=source,
                    page_number=p_num,
                    group_id=group_id,
                    extraction_notes={"header_title": h_text},
                ))
                type_candidate_emitted = True
                break

        # Check for Standard Invoice in header
        elif re.search(r"\b(?:(?:tax\s+)?invoice|rechnung|factura|facture|arve|ใบวางบิล|ใบแจ้งหนี้|ใบกำกับภาษี)\b", h_text, re.IGNORECASE):
            candidates.append(DocumentIdentityCandidate(
                field_name="invoice_type",
                raw_value=h_text,
                normalized_value="invoice",
                invoice_type_value=InvoiceType.INVOICE,
                evidence_ids=(h_item.evidence_id,),
                source=source,
                page_number=p_num,
                group_id=group_id,
                extraction_notes={"header_title": h_text},
            ))
            type_candidate_emitted = True
            break

    if not type_candidate_emitted and items:
        inferred_type = InvoiceType.UNKNOWN
        if page_understanding and page_understanding.page_role == PageRole.INVOICE:
            inferred_type = InvoiceType.INVOICE
        elif page_understanding and page_understanding.page_role == PageRole.CREDIT_MEMO:
            inferred_type = InvoiceType.CREDIT_MEMO

        candidates.append(DocumentIdentityCandidate(
            field_name="invoice_type",
            raw_value="",
            normalized_value=inferred_type.value,
            invoice_type_value=inferred_type,
            evidence_ids=(items[0].evidence_id,),
            source=source,
            page_number=p_num,
            group_id=group_id,
            extraction_notes={"page_role": page_understanding.page_role.value if page_understanding else "unknown"},
        ))

    return candidates


def extract_party_and_po_from_evidence(
    page_evidence: PageEvidence,
    group_id: Optional[str] = None,
    source: str = "ocr",
) -> Tuple[List[PartyIdentityCandidate], List[POCandidate]]:
    """Extract observed supplier/buyer fields and purchase order candidates."""
    party_candidates: List[PartyIdentityCandidate] = []
    po_candidates: List[POCandidate] = []
    items = page_evidence.items
    p_num = page_evidence.page_number

    current_scope = "unknown"

    _VENDOR_HEADING_RE = re.compile(
        r"^\s*(?:vendor\s*details|supplier\s*(?:details|info|information)?|seller\s*(?:details|info)?|billing\s*information|company\s*details)\s*[:#\-]?\s*$",
        re.IGNORECASE,
    )
    _BANK_HEADING_RE = re.compile(
        r"^\s*(?:bank\s*details|banking\s*(?:details|info)?|payment\s*details)\s*[:#\-]?\s*$",
        re.IGNORECASE,
    )
    _BUYER_HEADING_RE = re.compile(
        r"^\s*(?:customer\s*(?:details|info)?|buyer\s*(?:details|info)?|client\s*(?:details|info)?|bill\s*to\s*details)\s*[:#\-]?\s*$",
        re.IGNORECASE,
    )
    _BANK_ROW_RE = re.compile(
        r"\b(?:account\s*(?:no\.?|number|#)|bank\s*name|branch|sort\s*code|swift(?:\s*code)?|iban|kto(?:\.|-nr)?|blz)\b",
        re.IGNORECASE,
    )

    for i, item in enumerate(items):
        text = item.content.strip()

        # Update active section scope from headings
        if _VENDOR_HEADING_RE.match(text):
            current_scope = "supplier"
            continue
        elif _BANK_HEADING_RE.match(text):
            current_scope = "banking"
            continue
        elif _BUYER_HEADING_RE.match(text):
            current_scope = "buyer"
            continue

        # 1. VAT / Tax ID
        m_vat = _VAT_INLINE.search(text)
        if m_vat:
            vat_val = m_vat.group(1).strip()
            vat_clean = re.sub(r"[\s\-.]+", "", vat_val).upper()
            party_candidates.append(PartyIdentityCandidate(
                party_role="supplier",
                field_name="vat_id",
                raw_value=vat_val,
                normalized_value=vat_clean,
                evidence_ids=(item.evidence_id,),
                source=source,
                page_number=p_num,
                group_id=group_id,
            ))

        # 2. Email Address
        m_email = _EMAIL_REGEX.search(text)
        if m_email:
            email_val = m_email.group(0).strip().lower()
            party_candidates.append(PartyIdentityCandidate(
                party_role="supplier",
                field_name="email",
                raw_value=m_email.group(0).strip(),
                normalized_value=email_val,
                evidence_ids=(item.evidence_id,),
                source=source,
                page_number=p_num,
                group_id=group_id,
            ))

        # 3. Bank IBAN
        m_iban = _IBAN_REGEX.search(text)
        if m_iban:
            iban_raw = m_iban.group(1).strip()
            iban_clean = re.sub(r"[\s\-]+", "", iban_raw).upper()
            party_candidates.append(PartyIdentityCandidate(
                party_role="supplier",
                field_name="bank_iban",
                raw_value=iban_raw,
                normalized_value=iban_clean,
                evidence_ids=(item.evidence_id,),
                source=source,
                page_number=p_num,
                group_id=group_id,
            ))

        # 4. PO References
        m_po = _PO_INLINE.search(text)
        if m_po:
            po_val = m_po.group(1).strip()
            po_candidates.append(POCandidate(
                field_name="po_number",
                raw_value=po_val,
                normalized_value=po_val,
                evidence_ids=(item.evidence_id,),
                source=source,
                page_number=p_num,
                group_id=group_id,
                label=text,
                extraction_notes={"line": text},
            ))

        # 5. Generic Labelled Field: Name (Supplier or Buyer)
        m_name_label = re.search(
            r"^\s*(?:name|company(?:\s*name)?|supplier(?:\s*name)?|vendor(?:\s*name)?|legal\s*name)\s*[:#\-]\s*(.+)$",
            text,
            re.IGNORECASE,
        )
        if m_name_label:
            val = m_name_label.group(1).strip()
            if len(val) >= 2 and val.upper() not in {"DETAILS", "INFORMATION", "NAME"}:
                role = "buyer" if current_scope == "buyer" else "supplier"
                party_candidates.append(PartyIdentityCandidate(
                    party_role=role,
                    field_name="name",
                    raw_value=val,
                    normalized_value=val,
                    evidence_ids=(item.evidence_id,),
                    source=source,
                    page_number=p_num,
                    group_id=group_id,
                    extraction_notes={"match_type": "labelled_name", "scope": current_scope},
                ))

        # 6. Bill To / Prepared For / Buyer Reference
        m_buyer_inline = re.search(
            r"\b(?:prepared\s*for|bill\s*to|billed\s*to|invoice\s*to|sold\s*to|customer|client|rechnungsempfänger)\s*[:#\-]\s*(.+)$",
            text,
            re.IGNORECASE,
        )
        if m_buyer_inline:
            b_val = m_buyer_inline.group(1).strip()
            if len(b_val) >= 2 and not _BANK_ROW_RE.search(b_val):
                party_candidates.append(PartyIdentityCandidate(
                    party_role="buyer",
                    field_name="name",
                    raw_value=b_val,
                    normalized_value=b_val,
                    evidence_ids=(item.evidence_id,),
                    source=source,
                    page_number=p_num,
                    group_id=group_id,
                    extraction_notes={"match_type": "inline_buyer", "context_label": text},
                ))
        elif re.search(r"^\s*(?:prepared\s*for|bill\s*to|billed\s*to|invoice\s*to|sold\s*to|rechnungsempfänger|kunden-nr\.?)\s*[:#\-]?\s*$", text, re.IGNORECASE):
            # Target buyer company in subsequent blocks
            for next_idx in range(i + 1, min(i + 6, len(items))):
                next_item = items[next_idx]
                next_text = next_item.content.strip()
                if _BANK_ROW_RE.search(next_text) or _VENDOR_HEADING_RE.match(next_text) or _BANK_HEADING_RE.match(next_text):
                    continue
                if re.search(r"^\s*(?:item|qty|price|total|pos|p\.?o\.?)\s*$", next_text, re.IGNORECASE):
                    continue
                if item.bbox and len(item.bbox) >= 4 and next_item.bbox and len(next_item.bbox) >= 4:
                    if abs(next_item.bbox[0] - item.bbox[0]) > 250 or next_item.bbox[1] < item.bbox[1]:
                        continue
                if len(next_text) >= 2:
                    party_candidates.append(PartyIdentityCandidate(
                        party_role="buyer",
                        field_name="name",
                        raw_value=next_text,
                        normalized_value=next_text,
                        evidence_ids=(item.evidence_id, next_item.evidence_id),
                        source=source,
                        page_number=p_num,
                        group_id=group_id,
                        extraction_notes={"match_type": "split_buyer", "context_label": text},
                    ))
                    break

    # Supplier name fallback: inspect top header blocks, skipping headings and banking rows
    if items and not any(p.field_name == "name" and p.party_role == "supplier" for p in party_candidates):
        for top_item in items[:min(6, len(items))]:
            s_name = top_item.content.strip()
            if _VENDOR_HEADING_RE.match(s_name) or _BANK_HEADING_RE.match(s_name) or _BUYER_HEADING_RE.match(s_name):
                continue
            if _BANK_ROW_RE.search(s_name) or _VAT_INLINE.search(s_name):
                continue
            if re.search(r"\b(?:page|seite|invoice|rechnung|tax\s*invoice)\b", s_name, re.IGNORECASE):
                continue
            if len(s_name) > 3:
                party_candidates.append(PartyIdentityCandidate(
                    party_role="supplier",
                    field_name="name",
                    raw_value=s_name,
                    normalized_value=s_name,
                    evidence_ids=(top_item.evidence_id,),
                    source=source,
                    page_number=p_num,
                    group_id=group_id,
                    extraction_notes={"header_top_block": True},
                ))
                break

    return party_candidates, po_candidates


def extract_totals_and_taxes_from_evidence(
    page_evidence: PageEvidence,
    group_id: Optional[str] = None,
    source: str = "ocr",
) -> Tuple[List[TotalCandidate], List[TaxCandidate], List[DiscountCandidate], List[ChargeCandidate]]:
    """Extract printed totals, tax summaries, discounts, and charges."""
    total_candidates: List[TotalCandidate] = []
    tax_candidates: List[TaxCandidate] = []
    discount_candidates: List[DiscountCandidate] = []
    charge_candidates: List[ChargeCandidate] = []
    items = page_evidence.items
    p_num = page_evidence.page_number

    for i, item in enumerate(items):
        text = item.content.strip()

        # A. Totals Extraction
        for t_type, pat in _TOTAL_LABELS:
            if pat.search(text):
                if t_type == "gross_total" and re.search(r"\b(?:exclusive|levy)\b", text, re.IGNORECASE):
                    continue
                # Search for amount inline
                num_match = re.search(r"(\d{1,3}(?:[.,\s]\d{3})+[.,]\d{2}|\d+[.,]\d{2})", text)
                if num_match:
                    raw_amt = num_match.group(1).strip()
                    dec_amt = to_decimal(raw_amt)
                    total_candidates.append(TotalCandidate(
                        total_type=t_type,
                        raw_label=text,
                        raw_value=raw_amt,
                        normalized_value=dec_amt,
                        evidence_ids=(item.evidence_id,),
                        source=source,
                        page_number=p_num,
                        group_id=group_id,
                        extraction_notes={"pattern": pat.pattern},
                    ))
                else:
                    best_match = None
                    if item.bbox and len(item.bbox) >= 4:
                        y_center = (item.bbox[1] + item.bbox[3]) / 2.0
                        min_x_dist = float("inf")
                        for other_it in items:
                            if other_it.evidence_id == item.evidence_id:
                                continue
                            if other_it.bbox and len(other_it.bbox) >= 4:
                                other_y_center = (other_it.bbox[1] + other_it.bbox[3]) / 2.0
                                if abs(other_y_center - y_center) <= 20.0 and other_it.bbox[0] >= item.bbox[0]:
                                    other_amts = re.findall(r"(\d{1,3}(?:[.,\s]\d{3})+[.,]\d{2}|\d+[.,]\d{2})", other_it.content)
                                    if other_amts:
                                        dist = other_it.bbox[0] - item.bbox[0]
                                        if dist < min_x_dist:
                                            min_x_dist = dist
                                            best_match = (other_amts[-1], other_it)
                    if best_match:
                        raw_amt = best_match[0]
                        dec_amt = to_decimal(raw_amt)
                        total_candidates.append(TotalCandidate(
                            total_type=t_type,
                            raw_label=text,
                            raw_value=raw_amt,
                            normalized_value=dec_amt,
                            evidence_ids=(item.evidence_id, best_match[1].evidence_id),
                            source=source,
                            page_number=p_num,
                            group_id=group_id,
                            extraction_notes={"split_total": True, "spatial": True},
                        ))
                    elif i + 1 < len(items):
                        # Next block contains the number
                        next_text = items[i + 1].content.strip()
                        next_num = re.search(r"(\d{1,3}(?:[.,\s]\d{3})+[.,]\d{2}|\d+[.,]\d{2})", next_text)
                        if next_num:
                            raw_amt = next_num.group(1).strip()
                            dec_amt = to_decimal(raw_amt)
                            total_candidates.append(TotalCandidate(
                                total_type=t_type,
                                raw_label=text,
                                raw_value=raw_amt,
                                normalized_value=dec_amt,
                                evidence_ids=(item.evidence_id, items[i + 1].evidence_id),
                                source=source,
                                page_number=p_num,
                                group_id=group_id,
                                extraction_notes={"split_total": True},
                            ))
                break

        # B. Tax Header Summaries (Standard & Generic Statutory Levies)
        m_tax = _TAX_INLINE.search(text)
        if m_tax and not any(t.raw_value == text for t in tax_candidates):
            raw_rate = m_tax.group(1)
            raw_amt = m_tax.group(2)
            if raw_rate or raw_amt:
                dec_rate = to_decimal(raw_rate) if raw_rate else None
                dec_amt = to_decimal(raw_amt) if raw_amt else None
                tax_candidates.append(TaxCandidate(
                    tax_name=text,
                    tax_type="VAT" if "vat" in text.lower() or "mwst" in text.lower() or "iva" in text.lower() else "Tax",
                    rate=dec_rate,
                    amount=dec_amt,
                    raw_value=text,
                    placement=Placement.HEADER,
                    evidence_ids=(item.evidence_id,),
                    source=source,
                    page_number=p_num,
                    group_id=group_id,
                ))

        # Check for generic statutory tax / levy rows with percentage and/or amount
        if not any(t.raw_value == text for t in tax_candidates):
            tax_kw = re.search(r"\b(?:vat|mwst|iva|tax|tva|gst|hst|pst|levy|nhil|getfund|covid(?:-19)?(?:\s*levy)?|duty|impuesto|steuer|moms|k[aä]ibemaks|km)\b", text, re.IGNORECASE)
            is_total_row = any(pat.search(text) for _, pat in _TOTAL_LABELS) or re.search(r"\b(?:subtotal|total\s*il[ií]quido|tax\s*exclusive|taxable\s*base)\b", text, re.IGNORECASE)
            if tax_kw and not is_total_row:
                m_rate = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
                # Strip percentages first so rates (e.g. 19%, 7 %) are never treated as tax amounts
                text_no_rate = re.sub(r"\d+(?:[.,]\d+)?\s*%", "", text)
                amounts_found = re.findall(r"(\d{1,3}(?:[.,\s]\d{3})+[.,]\d{2}|\d+[.,]\d{2})", text_no_rate)
                raw_amt = None
                ev_ids = (item.evidence_id,)

                if amounts_found:
                    raw_amt = amounts_found[-1]
                else:
                    # Generic split/multi-column row: look for horizontally aligned amount on the same row
                    if item.bbox and len(item.bbox) >= 4:
                        y_center = (item.bbox[1] + item.bbox[3]) / 2.0
                        best_match = None
                        min_x_dist = float("inf")
                        for other_it in items:
                            if other_it.evidence_id == item.evidence_id:
                                continue
                            if other_it.bbox and len(other_it.bbox) >= 4:
                                other_y_center = (other_it.bbox[1] + other_it.bbox[3]) / 2.0
                                if abs(other_y_center - y_center) <= 20.0 and other_it.bbox[0] >= item.bbox[0]:
                                    # Ensure no intervening subtotal / total label between tax label and amount
                                    intervening = False
                                    for mid_it in items:
                                        if mid_it.evidence_id in (item.evidence_id, other_it.evidence_id):
                                            continue
                                        if mid_it.bbox and len(mid_it.bbox) >= 4:
                                            mid_y = (mid_it.bbox[1] + mid_it.bbox[3]) / 2.0
                                            if abs(mid_y - y_center) <= 20.0 and item.bbox[0] < mid_it.bbox[0] < other_it.bbox[0]:
                                                mid_txt = mid_it.content.strip()
                                                if any(pat.search(mid_txt) for _, pat in _TOTAL_LABELS) or re.search(r"\b(?:subtotal|total|base|il[ií]quido|net)\b", mid_txt, re.IGNORECASE):
                                                    intervening = True
                                                    break
                                    if intervening:
                                        continue

                                    other_no_rate = re.sub(r"\d+(?:[.,]\d+)?\s*%", "", other_it.content)
                                    other_amts = re.findall(r"(\d{1,3}(?:[.,\s]\d{3})+[.,]\d{2}|\d+[.,]\d{2})", other_no_rate)
                                    if other_amts:
                                        dist = other_it.bbox[0] - item.bbox[0]
                                        if dist < min_x_dist:
                                            min_x_dist = dist
                                            best_match = (other_amts[-1], other_it)
                        if best_match:
                            raw_amt = best_match[0]
                            ev_ids = (item.evidence_id, best_match[1].evidence_id)

                    # Fallback to next item in reading order only if on same line and no intervening label
                    if raw_amt is None and i + 1 < len(items):
                        next_it = items[i + 1]
                        next_txt = next_it.content.strip()
                        if not any(pat.search(next_txt) for _, pat in _TOTAL_LABELS):
                            next_no_rate = re.sub(r"\d+(?:[.,]\d+)?\s*%", "", next_txt)
                            next_amts = re.findall(r"(\d{1,3}(?:[.,\s]\d{3})+[.,]\d{2}|\d+[.,]\d{2})", next_no_rate)
                            if next_amts:
                                raw_amt = next_amts[-1]
                                ev_ids = (item.evidence_id, next_it.evidence_id)

                if raw_amt is not None or m_rate is not None:
                    dec_amt = to_decimal(raw_amt) if raw_amt else None
                    dec_rate = to_decimal(m_rate.group(1)) if m_rate else None

                    clean_name = text
                    clean_name = re.sub(r"^\s*(?:\([ivx\d]+\)|\d+\.)\s*", "", clean_name, flags=re.IGNORECASE)
                    clean_name = re.sub(r"\(\s*\d+(?:[.,]\d+)?\s*%\s*\)|\d+(?:[.,]\d+)?\s*%", "", clean_name)
                    clean_name = re.sub(r"\([ivx\d+ ]+\)", "", clean_name, flags=re.IGNORECASE)
                    clean_name = re.sub(r"(?:GH[¢₵$cC€]|GHS|EUR|USD|GBP|CHF|THB|[€$£฿])\s*[\d.,\s]+$", "", clean_name)
                    clean_name = clean_name.strip(" :-#\t\n")

                    if not clean_name:
                        clean_name = tax_kw.group(0).upper()

                    tax_candidates.append(TaxCandidate(
                        tax_name=clean_name,
                        tax_type="VAT" if re.search(r"\b(?:vat|mwst|iva|tva|nhil|getfund|levy|k[aä]ibemaks|km)\b", clean_name, re.IGNORECASE) else "Tax",
                        rate=dec_rate,
                        amount=dec_amt,
                        raw_value=text,
                        placement=Placement.HEADER,
                        evidence_ids=ev_ids,
                        source=source,
                        page_number=p_num,
                        group_id=group_id,
                    ))

        # C. Discounts
        if re.search(r"\b(?:discount|rabatt|desconto|escompte|allahindlus)\b", text, re.IGNORECASE):
            m_rate = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
            dec_rate = to_decimal(m_rate.group(1)) if m_rate else None
            num_match = re.search(r"(\d{1,3}(?:[.,\s]\d{3})+[.,]\d{2}|\d+[.,]\d{2})", text)
            raw_amt = None
            ev_ids = (item.evidence_id,)
            if num_match:
                raw_amt = num_match.group(1)
            else:
                if item.bbox and len(item.bbox) >= 4:
                    y_center = (item.bbox[1] + item.bbox[3]) / 2.0
                    min_x_dist = float("inf")
                    for other_it in items:
                        if other_it.evidence_id == item.evidence_id:
                            continue
                        if other_it.bbox and len(other_it.bbox) >= 4:
                            other_y_center = (other_it.bbox[1] + other_it.bbox[3]) / 2.0
                            if abs(other_y_center - y_center) <= 20.0 and other_it.bbox[0] >= item.bbox[0]:
                                other_amts = re.findall(r"(\d{1,3}(?:[.,\s]\d{3})+[.,]\d{2}|\d+[.,]\d{2})", other_it.content)
                                if other_amts:
                                    dist = other_it.bbox[0] - item.bbox[0]
                                    if dist < min_x_dist:
                                        min_x_dist = dist
                                        raw_amt = other_amts[-1]
                                        ev_ids = (item.evidence_id, other_it.evidence_id)
                if raw_amt is None and i + 1 < len(items):
                    next_it = items[i + 1]
                    next_amts = re.findall(r"(\d{1,3}(?:[.,\s]\d{3})+[.,]\d{2}|\d+[.,]\d{2})", next_it.content)
                    if next_amts:
                        raw_amt = next_amts[-1]
                        ev_ids = (item.evidence_id, next_it.evidence_id)
            dec_amt = to_decimal(raw_amt) if raw_amt else None
            discount_candidates.append(DiscountCandidate(
                label=text,
                rate=dec_rate,
                amount=dec_amt,
                raw_value=text,
                placement=Placement.HEADER,
                evidence_ids=ev_ids,
                source=source,
                page_number=p_num,
                group_id=group_id,
            ))

        # D. Charges (Freight / Shipping / Fees)
        if re.search(r"\b(?:freight|shipping|porto|versand|service\s*fee|handling)\b", text, re.IGNORECASE):
            num_match = re.search(r"(\d+(?:[.,]\d+)?)", text)
            dec_amt = to_decimal(num_match.group(1)) if num_match else None
            charge_candidates.append(ChargeCandidate(
                label=text,
                rate=None,
                amount=dec_amt,
                raw_value=text,
                placement=Placement.HEADER,
                evidence_ids=(item.evidence_id,),
                source=source,
                page_number=p_num,
                group_id=group_id,
            ))

    return total_candidates, tax_candidates, discount_candidates, charge_candidates


def extract_lines_from_evidence(
    page_evidence: PageEvidence,
    group_id: Optional[str] = None,
    source: str = "ocr",
) -> List[LineCandidate]:
    """Extract table rows and line item candidates structurally without arithmetic."""
    lines: List[LineCandidate] = []
    items = page_evidence.items
    p_num = page_evidence.page_number

    # Cluster blocks by vertical bbox (y-center) if bounding boxes exist
    rows: List[List[Evidence]] = []
    items_with_bbox = [it for it in items if it.bbox and len(it.bbox) >= 4]

    if items_with_bbox:
        # Sort items vertically by y1, then horizontally by x1
        sorted_items = sorted(items_with_bbox, key=lambda it: (it.bbox[1], it.bbox[0]))
        current_row: List[Evidence] = []
        current_y_center: Optional[float] = None
        y_tolerance = 12.0  # Pixels tolerance for same row

        for it in sorted_items:
            y_center = (it.bbox[1] + it.bbox[3]) / 2.0
            if current_y_center is None:
                current_row.append(it)
                current_y_center = y_center
            elif abs(y_center - current_y_center) <= y_tolerance:
                current_row.append(it)
            else:
                # Flush row sorted by x
                current_row.sort(key=lambda x: x.bbox[0])
                rows.append(current_row)
                current_row = [it]
                current_y_center = y_center

        if current_row:
            current_row.sort(key=lambda x: x.bbox[0])
            rows.append(current_row)

    # 1. Detect table column headers if present
    table_header_idx: Optional[int] = None
    col_defs: List[_ColDef] = []
    header_y_max: Optional[float] = None

    for idx, r in enumerate(rows):
        matches = 0
        cols_cand: List[_ColDef] = []
        for it in r:
            t = it.content.strip()
            role = None
            if _INDEX_HEADER_RE.search(t):
                role = "INDEX"
            elif _PROD_CODE_HEADER_RE.search(t):
                role = "PRODUCT_CODE"
            elif _HTS_HEADER_RE.search(t):
                role = "HTS"
            elif _WEIGHT_HEADER_RE.search(t):
                role = "WEIGHT"
            elif _DURATION_HEADER_RE.search(t):
                role = "DURATION"
            elif _QTY_HEADER_RE.search(t):
                role = "QTY"
            elif _PRICE_HEADER_RE.search(t):
                role = "PRICE"
            elif _TOTAL_HEADER_RE.search(t):
                role = "AMOUNT"
            elif _TAX_COL_HEADER_RE.search(t):
                role = "TAX"
            elif _DISC_COL_HEADER_RE.search(t):
                role = "DISCOUNT"
            elif _DESC_HEADER_RE.search(t):
                role = "DESC"
            if role:
                matches += 1
                cols_cand.append(_ColDef(role, it.bbox[0], it.bbox[2], (it.bbox[0] + it.bbox[2]) / 2.0))
        if matches >= 2 or (matches >= 1 and any(c.role in ("PRICE", "AMOUNT") for c in cols_cand) and any(c.role in ("INDEX", "QTY", "DESC") for c in cols_cand)):
            table_header_idx = idx
            col_defs = sorted(cols_cand, key=lambda c: c.x_mid)
            header_y_max = max(it.bbox[3] for it in r)
            break

    # 2. Detect summary/totals section start
    summary_y_min: Optional[float] = None
    for r in rows:
        r_y_min = min(it.bbox[1] for it in r)
        if header_y_max is not None and r_y_min <= header_y_max + 15.0:
            continue
        full_txt = " ".join(it.content.strip() for it in r)
        if _SUMMARY_START_RE.search(full_txt) or re.search(r"^\s*(?:total|grand\s*total)\b", full_txt, re.IGNORECASE):
            summary_y_min = r_y_min
            break

    # 3. Process line rows
    row_counter = 1
    for idx, r in enumerate(rows):
        if table_header_idx is not None and idx <= table_header_idx:
            continue
        r_y_mid = sum((it.bbox[1] + it.bbox[3]) / 2.0 for it in r) / len(r)
        if header_y_max is not None and r_y_mid < header_y_max - 5.0:
            continue
        if summary_y_min is not None and r_y_mid >= summary_y_min - 5.0:
            continue

        full_row_text = " | ".join(it.content.strip() for it in r)
        ev_ids = tuple(it.evidence_id for it in r if it.evidence_id)

        # Skip metadata, summary, total, and tax rows
        if _METADATA_LINE_RE.search(full_row_text):
            continue
        if _SUMMARY_START_RE.search(full_row_text):
            continue
        if any(pat.search(full_row_text) for _, pat in _TOTAL_LABELS):
            continue
        if re.search(r"(\d+(?:[.,]\d+)?)\s*%", full_row_text) and re.search(r"\b(?:vat|tax|levy|nhil|getfund|covid|mwst|iva|tva|gst|duty|impuesto|k[aä]ibemaks)\b", full_row_text, re.IGNORECASE):
            continue
        if re.search(r"^\s*(?:\d+|[0-9]\.|\+|original|zf1)\s*$", full_row_text, re.IGNORECASE):
            continue

        # Inspect semantic role
        role = SemanticRole.BILLED_LINE
        if re.search(r"\b(?:breakdown|detail|sub-item|info\s*only)\b", full_row_text, re.IGNORECASE):
            role = SemanticRole.COMPONENT_DETAIL
        elif re.search(r"\b(?:discount|rabatt|allahindlus)\b", full_row_text, re.IGNORECASE):
            role = SemanticRole.DISCOUNT
        elif re.search(r"\b(?:shipping|freight|versand)\b", full_row_text, re.IGNORECASE):
            role = SemanticRole.CHARGE
        elif re.search(r"\b(?:subtotal|zwischensumme|vahesumma)\b", full_row_text, re.IGNORECASE):
            role = SemanticRole.SUBTOTAL

        cell_texts = [it.content.strip() for it in r]
        qty: Optional[Decimal] = None
        unit_p: Optional[Decimal] = None
        amt: Optional[Decimal] = None
        desc_parts: List[str] = []
        cand_qtys: List[Decimal] = []
        cand_amts: List[Decimal] = []

        # Check for leading quantity in first cell
        if cell_texts:
            m_lead_first = _LEADING_QTY_RE.match(cell_texts[0])
            if m_lead_first:
                qty = to_decimal(m_lead_first.group(1))

        for it in r:
            t = it.content.strip()
            x_mid = (it.bbox[0] + it.bbox[2]) / 2.0

            # Skip duration token from financial parsing
            if _DURATION_TOKEN_RE.match(t):
                continue

            # Match column role via spatial boundary when headers present
            col_role = None
            if col_defs:
                first_col = col_defs[0]
                if first_col.role in ("PRICE", "AMOUNT", "TAX", "DISCOUNT") and x_mid < first_col.x_min - 40.0:
                    col_role = None
                elif len(col_defs) == 1:
                    if abs(col_defs[0].x_mid - x_mid) <= 150.0:
                        col_role = col_defs[0].role
                else:
                    for c_idx in range(len(col_defs)):
                        lb = -float("inf") if c_idx == 0 else (col_defs[c_idx - 1].x_mid + col_defs[c_idx].x_mid) / 2.0
                        rb = float("inf") if c_idx == len(col_defs) - 1 else (col_defs[c_idx].x_mid + col_defs[c_idx + 1].x_mid) / 2.0
                        if lb <= x_mid < rb:
                            col_role = col_defs[c_idx].role
                            break

            if col_role in ("INDEX", "HTS", "WEIGHT", "DURATION", "PRODUCT_CODE", "TAX", "DISCOUNT"):
                # Non-financial or metadata columns: numeric values must never become prices or quantities
                continue
            elif col_role == "QTY":
                m_n = re.search(r"(\d+(?:[.,]\d+)?)", t)
                if m_n:
                    qty = to_decimal(m_n.group(1))
            elif col_role == "PRICE":
                m_n = re.search(r"(\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", t)
                if m_n:
                    unit_p = to_decimal(m_n.group(1))
            elif col_role == "AMOUNT":
                m_n = re.search(r"(\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", t)
                if m_n:
                    dec = to_decimal(m_n.group(1))
                    if amt is not None and unit_p is None:
                        unit_p = amt
                        amt = dec
                    else:
                        amt = dec
            elif col_role == "DESC":
                desc_parts.append(t)
            else:
                # Unmatched column / no headers present
                m_lead = _LEADING_QTY_RE.match(t)
                if m_lead and qty is None:
                    qty = to_decimal(m_lead.group(1))
                    rem = t[m_lead.end():].strip()
                    if rem:
                        desc_parts.append(rem)
                elif re.search(r"(?:GH[¢₵$cC€]|GHS|EUR|USD|GBP|CHF|THB|[€$£฿])\s*\d", t) or re.match(r"^\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})$", t):
                    m_n = re.search(r"(\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", t)
                    dec = to_decimal(m_n.group(1)) if m_n else None
                    if dec is not None:
                        cand_amts.append(dec)
                elif re.match(r"^\d+(?:[.,]\d+)?$", t):
                    # Standalone numeric token in a separate cell without currency
                    cand_qtys.append(to_decimal(t))
                else:
                    desc_parts.append(t)

        # In headerless rows, assign candidate quantities and amounts by order
        if not col_defs:
            if cand_qtys and qty is None:
                qty = cand_qtys[0]
            if len(cand_amts) == 1:
                if amt is None:
                    amt = cand_amts[0]
            elif len(cand_amts) >= 2:
                if unit_p is None:
                    unit_p = cand_amts[0]
                if amt is None:
                    amt = cand_amts[1]

        desc = " ".join(desc_parts).strip()
        if not desc and cell_texts:
            desc = cell_texts[0]

        # Check for leading quantity pattern in description (e.g. '2 x Product', '2 × Product', '10 × Service')
        m_lead_qty = _LEADING_QTY_RE.match(desc)
        if m_lead_qty:
            if qty is None:
                qty = to_decimal(m_lead_qty.group(1))
            desc = desc[m_lead_qty.end():].strip()

        # Preserve None when document does not provide quantity or price. Never invent qty=1!
        if desc and (amt is not None or unit_p is not None or qty is not None):
            item_type = "SERVICE" if re.search(r"\b(?:fee|charge|service|setup|set\s*up|transport|catering|consulting|maintenance|support|meal)\b", desc, re.IGNORECASE) else "GOODS"
            lines.append(LineCandidate(
                source_row_number=row_counter,
                description=desc,
                quantity=qty,
                unit_price=unit_p,
                amount=amt,
                raw_row=full_row_text,
                raw_cells=tuple(cell_texts),
                evidence_ids=ev_ids,
                source=source,
                page_number=p_num,
                group_id=group_id,
                semantic_role=role,
                extraction_notes={"item_type": item_type},
            ))
            row_counter += 1

    return lines


# ══════════════════════════════════════════════════════════════════════════
# Qwen Vision Output Adapter Boundary
# ══════════════════════════════════════════════════════════════════════════

def adapt_vision_evidence_to_candidates(
    vision_evidence: Evidence,
    page_number: int,
    group_id: Optional[str] = None,
) -> Tuple[
    List[DocumentIdentityCandidate],
    List[PartyIdentityCandidate],
    List[POCandidate],
    List[LineCandidate],
    List[TotalCandidate],
    List[TaxCandidate],
    List[DiscountCandidate],
    List[ChargeCandidate],
]:
    """Adapter boundary converting multimodal Qwen Vision evidence into typed candidates.
    
    Guarantees:
    - Every candidate has source="vision" and evidence_ids=(vision_evidence.evidence_id,).
    - Preserves unmerged candidates without forced consensus or deduplication.
    - Generically extracts dates, currency, PO, subtotals, totals, taxes, discounts,
      charges, and table line items.
    """
    ident_cands: List[DocumentIdentityCandidate] = []
    party_cands: List[PartyIdentityCandidate] = []
    po_cands: List[POCandidate] = []
    line_cands: List[LineCandidate] = []
    total_cands: List[TotalCandidate] = []
    tax_cands: List[TaxCandidate] = []
    disc_cands: List[DiscountCandidate] = []
    chg_cands: List[ChargeCandidate] = []

    ev_id = vision_evidence.evidence_id
    if not ev_id:
        raise ValueError("Cannot adapt Vision Evidence without a valid evidence_id")

    content = vision_evidence.content

    # 1. Invoice Number in Vision output
    m_inv = re.search(r"\b(?:invoice\s*(?:number|no\.?|#)|rechnung\s*nr\.?|arve\s*nr\.?|factura\s*no\.?)\s*[:#\-]?\s*([A-Z0-9\-/]{3,})\b", content, re.IGNORECASE)
    if m_inv:
        val = m_inv.group(1).strip()
        ident_cands.append(DocumentIdentityCandidate(
            field_name="invoice_number",
            raw_value=val,
            normalized_value=val,
            evidence_ids=(ev_id,),
            source="vision",
            page_number=page_number,
            group_id=group_id,
            extraction_notes={"vision_adapter": True},
        ))

    # 2. Dates in Vision output
    m_date = re.search(r"\b(?:invoice\s*date|date|datum|rechnungsdatum|kuupäev)\s*[:#\-]?\s*(\d{1,4}[./\-]\d{1,2}[./\-]\d{2,4})\b", content, re.IGNORECASE)
    if m_date:
        d_val = m_date.group(1).strip()
        ident_cands.append(DocumentIdentityCandidate(
            field_name="invoice_date",
            raw_value=d_val,
            normalized_value=d_val,
            evidence_ids=(ev_id,),
            source="vision",
            page_number=page_number,
            group_id=group_id,
            extraction_notes={"vision_adapter": True},
        ))

    m_due = re.search(r"\b(?:due\s*date|fälligkeitsdatum|zahlungsziel|maksetähtaeg)\s*[:#\-]?\s*(\d{1,4}[./\-]\d{1,2}[./\-]\d{2,4})\b", content, re.IGNORECASE)
    if m_due:
        due_val = m_due.group(1).strip()
        ident_cands.append(DocumentIdentityCandidate(
            field_name="due_date",
            raw_value=due_val,
            normalized_value=due_val,
            evidence_ids=(ev_id,),
            source="vision",
            page_number=page_number,
            group_id=group_id,
            extraction_notes={"vision_adapter": True},
        ))

    # 3. Currency in Vision output
    m_curr = re.search(r"\b(?:currency|währung)\s*[:#\-]?\s*([A-Z]{3})\b", content, re.IGNORECASE)
    curr_str = m_curr.group(1).strip() if m_curr else None
    if not curr_str:
        for sym in _CURRENCY_SYMBOLS:
            if sym.isalpha() and re.search(rf"\b{sym}\b", content, re.IGNORECASE):
                curr_str = sym
                break
            elif not sym.isalpha() and sym in content:
                curr_str = sym
                break
    if curr_str:
        norm_c, just = normalize_currency_with_context(curr_str)
        ident_cands.append(DocumentIdentityCandidate(
            field_name="currency",
            raw_value=curr_str,
            normalized_value=norm_c,
            evidence_ids=(ev_id,),
            source="vision",
            page_number=page_number,
            group_id=group_id,
            extraction_notes={"vision_adapter": True, "context_justified": just},
        ))

    # 4. PO / Order Reference in Vision output
    m_po = re.search(r"\b(?:customer\s*po|client\s*po|purchase\s*order(?:\s*(?:no\.?|#|number))?|po(?:\s*(?:no\.?|#|number))?|bestellnummer|order\s*(?:number|no\.?|#))\s*[:#\-]?\s*([A-Z0-9\-/]{3,})\b", content, re.IGNORECASE)
    if m_po:
        po_val = m_po.group(1).strip()
        po_cands.append(POCandidate(
            field_name="po_number",
            raw_value=po_val,
            normalized_value=po_val,
            evidence_ids=(ev_id,),
            source="vision",
            page_number=page_number,
            group_id=group_id,
            extraction_notes={"vision_adapter": True},
        ))

    # 5. Subtotal in Vision output
    m_sub = re.search(r"\b(?:subtotal|net\s*(?:amount|total)?|zwischensumme|tax\s*exclusive(?:\s*value)?|total\s*il[ií]quido)\s*[:#\-]?\s*(?:[A-Z]{3}|[€$£฿])?\s*(\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d{2})|\d+[.,]\d{2})\b", content, re.IGNORECASE)
    if m_sub:
        val = m_sub.group(1).strip()
        total_cands.append(TotalCandidate(
            total_type="subtotal",
            raw_label="vision_subtotal",
            raw_value=val,
            normalized_value=to_decimal(val),
            evidence_ids=(ev_id,),
            source="vision",
            page_number=page_number,
            group_id=group_id,
            extraction_notes={"vision_adapter": True},
        ))

    # 6. Gross Total in Vision output
    m_tot = re.search(r"\b(?:grand\s*total|gross\s*total|gesamtsumme|endbetrag|total\s*amount|total\s*da\s*factura|total)\s*[:#\-]?\s*(?:[A-Z]{3}|[€$£฿])?\s*(\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d{2})|\d+[.,]\d{2})\b", content, re.IGNORECASE)
    if m_tot:
        val = m_tot.group(1).strip()
        dec_amt = to_decimal(val)
        total_cands.append(TotalCandidate(
            total_type="gross_total",
            raw_label="vision_total",
            raw_value=val,
            normalized_value=dec_amt,
            evidence_ids=(ev_id,),
            source="vision",
            page_number=page_number,
            group_id=group_id,
            extraction_notes={"vision_adapter": True},
        ))

    # 7. Amount Due in Vision output
    m_due_amt = re.search(r"\b(?:amount\s*due|balance\s*due|net\s*payable|total\s*payment|fälliger\s*betrag)\s*[:#\-]?\s*(?:[A-Z]{3}|[€$£฿])?\s*(\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d{2})|\d+[.,]\d{2})\b", content, re.IGNORECASE)
    if m_due_amt:
        val = m_due_amt.group(1).strip()
        total_cands.append(TotalCandidate(
            total_type="amount_due",
            raw_label="vision_amount_due",
            raw_value=val,
            normalized_value=to_decimal(val),
            evidence_ids=(ev_id,),
            source="vision",
            page_number=page_number,
            group_id=group_id,
            extraction_notes={"vision_adapter": True},
        ))

    # 8. Taxes in Vision output
    m_taxes = re.finditer(r"\b(vat|mwst|iva|tva|gst|tax)\s*(?:\(?(\d+(?:[.,]\d+)?)\s*%\)?)?\s*[:#\-]?\s*(?:[A-Z]{3}|[€$£฿])?\s*(\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d{2})|\d+[.,]\d{2})\b", content, re.IGNORECASE)
    for mt in m_taxes:
        t_name = mt.group(1).strip()
        t_rate = to_decimal(mt.group(2)) if mt.group(2) else None
        t_amt = to_decimal(mt.group(3)) if mt.group(3) else None
        tax_cands.append(TaxCandidate(
            tax_name=t_name.upper(),
            tax_type="VAT" if re.search(r"\b(?:vat|mwst|iva|tva)\b", t_name, re.IGNORECASE) else "Tax",
            rate=t_rate,
            amount=t_amt,
            raw_value=mt.group(0),
            placement=Placement.HEADER,
            evidence_ids=(ev_id,),
            source="vision",
            page_number=page_number,
            group_id=group_id,
            extraction_notes={"vision_adapter": True},
        ))

    # 9. Discounts & Charges in Vision output
    m_disc = re.search(r"\b(?:discount|rabatt)\s*[:#\-]?\s*(?:[A-Z]{3}|[€$£฿])?\s*(\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d{2})|\d+[.,]\d{2})\b", content, re.IGNORECASE)
    if m_disc:
        disc_val = m_disc.group(1).strip()
        disc_cands.append(DiscountCandidate(
            label="vision_discount",
            amount=to_decimal(disc_val),
            raw_value=disc_val,
            placement=Placement.HEADER,
            evidence_ids=(ev_id,),
            source="vision",
            page_number=page_number,
            group_id=group_id,
            extraction_notes={"vision_adapter": True},
        ))

    m_chg = re.search(r"\b(?:shipping|freight|versand|delivery)\s*[:#\-]?\s*(?:[A-Z]{3}|[€$£฿])?\s*(\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d{2})|\d+[.,]\d{2})\b", content, re.IGNORECASE)
    if m_chg:
        chg_val = m_chg.group(1).strip()
        chg_cands.append(ChargeCandidate(
            label="vision_charge",
            amount=to_decimal(chg_val),
            raw_value=chg_val,
            placement=Placement.HEADER,
            evidence_ids=(ev_id,),
            source="vision",
            page_number=page_number,
            group_id=group_id,
            extraction_notes={"vision_adapter": True},
        ))

    # 10. Lines in Vision markdown tables
    v_line_row = 1
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("|") and line.endswith("|") and not line.startswith("|---") and not line.startswith("| -"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3 and not any(re.search(r"\b(?:description|total|amount|price|qty)\b", p, re.IGNORECASE) for p in parts[:2]):
                desc = parts[0]
                l_qty: Optional[Decimal] = None
                l_unit_p: Optional[Decimal] = None
                l_amt: Optional[Decimal] = None

                # Check leading quantity in desc
                m_lead = _LEADING_QTY_RE.match(desc)
                if m_lead:
                    l_qty = to_decimal(m_lead.group(1))
                    desc = desc[m_lead.end():].strip()

                for p in parts[1:]:
                    if re.match(r"^\d+(?:[.,]\d+)?$", p) and l_qty is None:
                        l_qty = to_decimal(p)
                    elif re.search(r"(\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", p):
                        m_n = re.search(r"(\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", p)
                        dec = to_decimal(m_n.group(1)) if m_n else None
                        if dec is not None:
                            if l_amt is None:
                                l_amt = dec
                            elif l_unit_p is None:
                                l_unit_p = l_amt
                                l_amt = dec

                if desc and (l_amt is not None or l_unit_p is not None or l_qty is not None):
                    line_cands.append(LineCandidate(
                        source_row_number=v_line_row,
                        description=desc,
                        quantity=l_qty,
                        unit_price=l_unit_p,
                        amount=l_amt,
                        raw_row=line,
                        raw_cells=tuple(parts),
                        evidence_ids=(ev_id,),
                        source="vision",
                        page_number=page_number,
                        group_id=group_id,
                        semantic_role=SemanticRole.BILLED_LINE,
                        extraction_notes={"vision_table": True},
                    ))
                    v_line_row += 1

    return ident_cands, party_cands, po_cands, line_cands, total_cands, tax_cands, disc_cands, chg_cands


# ══════════════════════════════════════════════════════════════════════════
# Main Semantic Extraction Contracts
# ══════════════════════════════════════════════════════════════════════════

def extract_candidates_from_page(
    page_evidence: PageEvidence,
    page_understanding: Optional[PageUnderstanding] = None,
    group: Optional[DocumentGroup] = None,
    routing_decision: Optional[RoutingDecision] = None,
    vision_evidence: Optional[Evidence] = None,
    vision_provider: Optional[VisionProvider] = None,
) -> ExtractionCandidates:
    """Extract semantic candidates for an individual document page.
    
    Consumes:
    - Phase 7A PageEvidence (OCR / native PDF)
    - Phase 7B PageUnderstanding (Role, Payable Relevance)
    - Phase 7C DocumentGroup (Group Context)
    - Phase 7D RoutingDecision (Selective Qwen escalation)
    """
    und = page_understanding or classify_page(page_evidence)
    grp_id = group.group_id if group else None
    p_num = page_evidence.page_number
    doc_id = page_evidence.document_id

    # 1. Extract base OCR candidates
    ident_cands = extract_identity_from_evidence(page_evidence, und, group_id=grp_id, source="ocr")
    party_cands, po_cands = extract_party_and_po_from_evidence(page_evidence, group_id=grp_id, source="ocr")
    tot_cands, tax_cands, disc_cands, chg_cands = extract_totals_and_taxes_from_evidence(page_evidence, group_id=grp_id, source="ocr")
    ln_cands = extract_lines_from_evidence(page_evidence, group_id=grp_id, source="ocr")

    # 2. Check Phase 7D Routing Decision
    r_dec = routing_decision or route_page(page_evidence, und, group=group)

    # 3. Handle Qwen Vision escalation if routed or vision_evidence provided
    v_ev = vision_evidence
    if v_ev is None and r_dec.decision == RouterDecisionType.ROUTE_TO_VISION and vision_provider is not None:
        try:
            v_ev = execute_vision_escalation(
                decision=r_dec,
                page_evidence=page_evidence,
                provider=vision_provider,
            )
        except Exception as e:
            log.warning(f"Vision escalation failed on page {p_num}: {e}")
            v_ev = None

    if v_ev is not None:
        v_res = adapt_vision_evidence_to_candidates(
            v_ev, page_number=p_num, group_id=grp_id
        )
        v_ident, v_party, v_po, v_lines, v_totals, v_taxes, v_discounts, v_charges = v_res
        # Separate candidates: retain both OCR and Vision candidates without deduplication
        ident_cands.extend(v_ident)
        party_cands.extend(v_party)
        po_cands.extend(v_po)
        ln_cands.extend(v_lines)
        tot_cands.extend(v_totals)
        tax_cands.extend(v_taxes)
        disc_cands.extend(v_discounts)
        chg_cands.extend(v_charges)


    return ExtractionCandidates(
        document_id=doc_id,
        group_id=grp_id,
        page_numbers=(p_num,),
        payable_relevance=und.payable_relevance,
        page_role=und.page_role,
        identity_candidates=tuple(ident_cands),
        party_candidates=tuple(party_cands),
        po_candidates=tuple(po_cands),
        line_candidates=tuple(ln_cands),
        tax_candidates=tuple(tax_cands),
        discount_candidates=tuple(disc_cands),
        charge_candidates=tuple(chg_cands),
        total_candidates=tuple(tot_cands),
        metadata={
            "routing_decision": r_dec.decision.value,
            "routing_reasons": r_dec.reasons,
        },
    )


def extract_candidates_from_document(
    page_evidences: Sequence[PageEvidence],
    understandings: Optional[Sequence[PageUnderstanding]] = None,
    groups: Optional[Sequence[DocumentGroup]] = None,
    routing_decisions: Optional[Sequence[RoutingDecision]] = None,
    vision_provider: Optional[VisionProvider] = None,
) -> List[ExtractionCandidates]:
    """Extract semantic candidates across an entire physical document with grouping context."""
    if not page_evidences:
        return []

    doc_id = page_evidences[0].document_id
    unds = list(understandings) if understandings else [classify_page(pe) for pe in page_evidences]

    # Resolve grouping
    if groups is None:
        grouping_result = group_document(page_evidences, unds)
        doc_groups = grouping_result.groups
    else:
        doc_groups = list(groups)

    results: List[ExtractionCandidates] = []

    for pe, und in zip(page_evidences, unds):
        # Find group containing this page
        matching_grp = next((g for g in doc_groups if pe.page_number in g.page_numbers), None)
        
        # Find matching routing decision if supplied
        r_dec = None
        if routing_decisions:
            r_dec = next((rd for rd in routing_decisions if rd.page_number == pe.page_number), None)

        p_cands = extract_candidates_from_page(
            page_evidence=pe,
            page_understanding=und,
            group=matching_grp,
            routing_decision=r_dec,
            vision_provider=vision_provider,
        )
        results.append(p_cands)

    return results
