"""src/understanding/document_facts.py — Document Fact Model & Evidence-Aware Semantic Extraction Contract.

Phase 9A: Establishes the intermediate, document-level fact representation
that subsequent accounting reconciliation, ERP validation, and autodraft
generation phases will consume.

Core Principles:
1. Traceability: Every extracted fact retains evidence ID(s) and provenance.
   Never create naked accounting values without traceability.
2. Separation of Fact Origins:
   - OBSERVED: Directly present on the physical document.
   - DERIVED: Computed or inferred from observed values.
   - MATCHED: Mapped to master data by Phase 8 matchers.
   Never silently convert one category into another.
3. Financial Component Separation:
   Lines, discounts, charges, taxes, and printed totals are held strictly
   separate and never flattened into a single amount.
4. Exact Numeric Representation:
   Decimal is enforced for all monetary and rate figures.
5. No Premature Accounting Reasoning:
   No arithmetic calculation (e.g. quantity * price), no ERP validation,
   no total reconciliation, and no automatic correction.
6. Uncertainty Preservation:
   PayableRelevance and PageRole from Phase 7 are preserved; uncertainty
   is never collapsed into a naive boolean.
7. Supporting Document Distinctness:
   Supporting documents are referenced via non-recursive identifiers
   (e.g. `supporting_group_ids`) without merging their facts into payable totals.
8. Non-Inference from Metadata:
   Invoice classification is never inferred from filenames or document IDs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.understanding.page_classifier import PageRole, PayableRelevance
from src.utils.logging import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Controlled Vocabularies & Enums
# ══════════════════════════════════════════════════════════════════════════

class FactOrigin(str, Enum):
    """Categorization of fact provenance."""
    OBSERVED = "observed"  # Directly present on document
    DERIVED = "derived"    # Inferred/computed from observed values
    MATCHED = "matched"    # Mapped to master data by Phase 8


class InvoiceType(str, Enum):
    """Controlled invoice/document identity type."""
    INVOICE = "invoice"
    CREDIT_MEMO = "credit_memo"
    DEBIT_MEMO = "debit_memo"
    UNKNOWN = "unknown"


class SemanticRole(str, Enum):
    """Semantic role of a document row or line item."""
    BILLED_LINE = "billed_line"
    COMPONENT_DETAIL = "component_detail"
    SUBTOTAL = "subtotal"
    DISCOUNT = "discount"
    CHARGE = "charge"
    TAX = "tax"
    TOTAL = "total"
    UNKNOWN = "unknown"


class Placement(str, Enum):
    """Document level where an item (tax, discount, charge) is placed."""
    HEADER = "header"
    LINE = "line"
    UNKNOWN = "unknown"


# ══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════

def to_decimal(val: Any) -> Optional[Decimal]:
    """Convert an observed numeric value or string to a Decimal.
    
    Preserves exact numeric representation without float precision loss.
    Returns None if val is None or empty.
    """
    if val is None or val == "":
        return None
    if isinstance(val, Decimal):
        return val
    # Handle int, float, str
    clean = str(val).strip()
    if not clean:
        return None
    # Strip common currency symbols and percent signs if present in raw string
    clean = clean.replace("$", "").replace("€", "").replace("£", "").replace("%", "").strip()
    # Normalize multiple dots e.g. 6.500.00 -> 6500.00
    if clean.count(".") > 1 and "," not in clean:
        parts = clean.split(".")
        clean = "".join(parts[:-1]) + "." + parts[-1]
    elif clean.count(",") > 1 and "." not in clean:
        parts = clean.split(",")
        clean = "".join(parts[:-1]) + "." + parts[-1]
    # Normalize comma if used as decimal separator without period
    elif "," in clean and "." not in clean:
        clean = clean.replace(",", ".")
    elif "," in clean and "." in clean:
        # standard 1,234.56 format -> remove comma
        if clean.rfind(".") > clean.rfind(","):
            clean = clean.replace(",", "")
        else:
            # European 1.234,56 format -> swap
            clean = clean.replace(".", "").replace(",", ".")
    try:
        return Decimal(clean)
    except (InvalidOperation, ValueError):
        return None



def _normalize_tuple_str(val: Any) -> Tuple[str, ...]:
    """Normalize input to a tuple of strings."""
    if val is None:
        return ()
    if isinstance(val, str):
        s = val.strip()
        return (s,) if s else ()
    if isinstance(val, (list, tuple, set)):
        items = []
        for x in val:
            if x is not None:
                sx = str(x).strip()
                if sx and sx not in items:
                    items.append(sx)
        return tuple(items)
    s = str(val).strip()
    return (s,) if s else ()


def _normalize_field_evidence(val: Any) -> Dict[str, Tuple[str, ...]]:
    """Normalize field_evidence_ids dict to Dict[str, Tuple[str, ...]]."""
    if not val or not isinstance(val, dict):
        return {}
    res: Dict[str, Tuple[str, ...]] = {}
    for k, v in val.items():
        sk = str(k).strip()
        res[sk] = _normalize_tuple_str(v)
    return res


def _validate_evidence_contract(
    class_name: str,
    accounting_values: Dict[str, Any],
    origin: FactOrigin,
    evidence_ids: Tuple[str, ...],
    field_evidence_ids: Dict[str, Tuple[str, ...]],
    derivation_rule: Optional[str] = None,
    derivation_source_fields: Tuple[str, ...] = (),
    matched_result: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate that non-empty accounting fields have appropriate provenance.
    
    Rules:
    - OBSERVED: Requires direct source evidence in evidence_ids or field_evidence_ids.
    - DERIVED: Requires derivation rule, derivation_source_fields, or derivation metadata.
    - MATCHED: Requires match result/provenance AND source observation reference.
    """
    has_non_empty_accounting = any(v is not None and v != "" for v in accounting_values.values())
    if not has_non_empty_accounting:
        return  # Missing optional fields are allowed

    # Flatten all available evidence IDs
    all_ev_ids = set(evidence_ids)
    for fld_evs in field_evidence_ids.values():
        all_ev_ids.update(fld_evs)

    if origin == FactOrigin.OBSERVED:
        if not all_ev_ids:
            fields_with_values = [k for k, v in accounting_values.items() if v is not None and v != ""]
            raise ValueError(
                f"Naked accounting field(s) in {class_name}: {fields_with_values} have observed values "
                f"but no evidence IDs were provided in evidence_ids or field_evidence_ids."
            )

    elif origin == FactOrigin.DERIVED:
        has_derivation_provenance = (
            bool(derivation_rule)
            or bool(derivation_source_fields)
            or bool(all_ev_ids)
            or (metadata and ("derivation_rule" in metadata or "derivation_source" in metadata))
        )
        if not has_derivation_provenance:
            fields_with_values = [k for k, v in accounting_values.items() if v is not None and v != ""]
            raise ValueError(
                f"Derived accounting field(s) in {class_name}: {fields_with_values} with origin DERIVED "
                f"require derivation_rule, derivation_source_fields, or derivation metadata."
            )

    elif origin == FactOrigin.MATCHED:
        has_match_provenance = bool(matched_result) or (metadata and "matched_result" in metadata)
        if not has_match_provenance:
            fields_with_values = [k for k, v in accounting_values.items() if v is not None and v != ""]
            raise ValueError(
                f"Matched accounting field(s) in {class_name}: {fields_with_values} with origin MATCHED "
                f"require matched_result or match provenance."
            )


# ══════════════════════════════════════════════════════════════════════════
# Document Identity Facts
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DocumentIdentityFacts:
    """Observed and typed document identity facts."""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    invoice_type: InvoiceType = InvoiceType.UNKNOWN
    currency: Optional[str] = None
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    origin: FactOrigin = FactOrigin.OBSERVED
    raw_values: Dict[str, str] = field(default_factory=dict)
    derivation_rule: Optional[str] = None
    derivation_source_fields: Tuple[str, ...] = field(default_factory=tuple)
    matched_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ev_ids = _normalize_tuple_str(self.evidence_ids)
        field_evs = _normalize_field_evidence(self.field_evidence_ids)
        inv_type = (
            self.invoice_type
            if isinstance(self.invoice_type, InvoiceType)
            else InvoiceType(str(self.invoice_type).lower())
        )
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        object.__setattr__(self, "evidence_ids", ev_ids)
        object.__setattr__(self, "field_evidence_ids", field_evs)
        object.__setattr__(self, "invoice_type", inv_type)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "derivation_source_fields", _normalize_tuple_str(self.derivation_source_fields))

        # Check evidence contract for accounting fields
        accounting = {
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date,
            "due_date": self.due_date,
            "currency": self.currency,
        }
        _validate_evidence_contract(
            "DocumentIdentityFacts",
            accounting,
            orig,
            ev_ids,
            field_evs,
            self.derivation_rule,
            self.derivation_source_fields,
            self.matched_result,
            self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date,
            "due_date": self.due_date,
            "invoice_type": self.invoice_type.value,
            "currency": self.currency,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "origin": self.origin.value,
            "raw_values": dict(self.raw_values),
            "derivation_rule": self.derivation_rule,
            "derivation_source_fields": list(self.derivation_source_fields),
            "matched_result": self.matched_result,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentIdentityFacts:
        return cls(
            invoice_number=data.get("invoice_number"),
            invoice_date=data.get("invoice_date"),
            due_date=data.get("due_date"),
            invoice_type=InvoiceType(data.get("invoice_type", "unknown")),
            currency=data.get("currency"),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            origin=FactOrigin(data.get("origin", "observed")),
            raw_values=dict(data.get("raw_values") or {}),
            derivation_rule=data.get("derivation_rule"),
            derivation_source_fields=_normalize_tuple_str(data.get("derivation_source_fields")),
            matched_result=data.get("matched_result"),
            metadata=dict(data.get("metadata") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Party Identity Facts
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SupplierIdentityFact:
    """Observed supplier identity facts (kept strictly separate from Phase 8B matches)."""
    observed_name: Optional[str] = None
    vat_id: Optional[str] = None
    country: Optional[str] = None
    email: Optional[str] = None
    bank_iban: Optional[str] = None
    address: Optional[str] = None
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    origin: FactOrigin = FactOrigin.OBSERVED
    raw_values: Dict[str, str] = field(default_factory=dict)
    derivation_rule: Optional[str] = None
    derivation_source_fields: Tuple[str, ...] = field(default_factory=tuple)
    matched_result: Optional[Dict[str, Any]] = None  # Reference to Phase 8B Match result
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ev_ids = _normalize_tuple_str(self.evidence_ids)
        field_evs = _normalize_field_evidence(self.field_evidence_ids)
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        object.__setattr__(self, "evidence_ids", ev_ids)
        object.__setattr__(self, "field_evidence_ids", field_evs)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "derivation_source_fields", _normalize_tuple_str(self.derivation_source_fields))

        accounting = {
            "observed_name": self.observed_name,
            "vat_id": self.vat_id,
            "bank_iban": self.bank_iban,
            "email": self.email,
            "address": self.address,
        }
        _validate_evidence_contract(
            "SupplierIdentityFact",
            accounting,
            orig,
            ev_ids,
            field_evs,
            self.derivation_rule,
            self.derivation_source_fields,
            self.matched_result,
            self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observed_name": self.observed_name,
            "vat_id": self.vat_id,
            "country": self.country,
            "email": self.email,
            "bank_iban": self.bank_iban,
            "address": self.address,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "origin": self.origin.value,
            "raw_values": dict(self.raw_values),
            "derivation_rule": self.derivation_rule,
            "derivation_source_fields": list(self.derivation_source_fields),
            "matched_result": self.matched_result,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SupplierIdentityFact:
        return cls(
            observed_name=data.get("observed_name") or data.get("name"),
            vat_id=data.get("vat_id"),
            country=data.get("country"),
            email=data.get("email"),
            bank_iban=data.get("bank_iban"),
            address=data.get("address"),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            origin=FactOrigin(data.get("origin", "observed")),
            raw_values=dict(data.get("raw_values") or {}),
            derivation_rule=data.get("derivation_rule"),
            derivation_source_fields=_normalize_tuple_str(data.get("derivation_source_fields")),
            matched_result=data.get("matched_result"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class BuyerIdentityFact:
    """Observed buyer identity facts (kept strictly separate from Phase 8B matches)."""
    observed_company: Optional[str] = None
    business_unit: Optional[str] = None
    location: Optional[str] = None
    company_code: Optional[str] = None
    business_unit_code: Optional[str] = None
    location_code: Optional[str] = None
    invoice_to_address: Optional[str] = None
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    origin: FactOrigin = FactOrigin.OBSERVED
    raw_values: Dict[str, str] = field(default_factory=dict)
    derivation_rule: Optional[str] = None
    derivation_source_fields: Tuple[str, ...] = field(default_factory=tuple)
    matched_result: Optional[Dict[str, Any]] = None  # Reference to Phase 8B Match result
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ev_ids = _normalize_tuple_str(self.evidence_ids)
        field_evs = _normalize_field_evidence(self.field_evidence_ids)
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        object.__setattr__(self, "evidence_ids", ev_ids)
        object.__setattr__(self, "field_evidence_ids", field_evs)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "derivation_source_fields", _normalize_tuple_str(self.derivation_source_fields))

        accounting = {
            "observed_company": self.observed_company,
            "company_code": self.company_code,
            "business_unit_code": self.business_unit_code,
            "location_code": self.location_code,
            "invoice_to_address": self.invoice_to_address,
        }
        _validate_evidence_contract(
            "BuyerIdentityFact",
            accounting,
            orig,
            ev_ids,
            field_evs,
            self.derivation_rule,
            self.derivation_source_fields,
            self.matched_result,
            self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observed_company": self.observed_company,
            "business_unit": self.business_unit,
            "location": self.location,
            "company_code": self.company_code,
            "business_unit_code": self.business_unit_code,
            "location_code": self.location_code,
            "invoice_to_address": self.invoice_to_address,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "origin": self.origin.value,
            "raw_values": dict(self.raw_values),
            "derivation_rule": self.derivation_rule,
            "derivation_source_fields": list(self.derivation_source_fields),
            "matched_result": self.matched_result,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BuyerIdentityFact:
        return cls(
            observed_company=data.get("observed_company") or data.get("company_name"),
            business_unit=data.get("business_unit") or data.get("business_unit_name"),
            location=data.get("location") or data.get("location_name"),
            company_code=data.get("company_code"),
            business_unit_code=data.get("business_unit_code"),
            location_code=data.get("location_code"),
            invoice_to_address=data.get("invoice_to_address"),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            origin=FactOrigin(data.get("origin", "observed")),
            raw_values=dict(data.get("raw_values") or {}),
            derivation_rule=data.get("derivation_rule"),
            derivation_source_fields=_normalize_tuple_str(data.get("derivation_source_fields")),
            matched_result=data.get("matched_result"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class PartyIdentityFacts:
    """Container holding observed supplier and buyer facts."""
    supplier: SupplierIdentityFact = field(default_factory=SupplierIdentityFact)
    buyer: BuyerIdentityFact = field(default_factory=BuyerIdentityFact)
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supplier": self.supplier.to_dict(),
            "buyer": self.buyer.to_dict(),
            "evidence_ids": list(self.evidence_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PartyIdentityFacts:
        sup_data = data.get("supplier") or {}
        buy_data = data.get("buyer") or {}
        supplier = (
            sup_data
            if isinstance(sup_data, SupplierIdentityFact)
            else SupplierIdentityFact.from_dict(sup_data)
        )
        buyer = (
            buy_data
            if isinstance(buy_data, BuyerIdentityFact)
            else BuyerIdentityFact.from_dict(buy_data)
        )
        return cls(
            supplier=supplier,
            buyer=buyer,
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            metadata=dict(data.get("metadata") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# PO Facts
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class POFacts:
    """Observed purchase order references (never replaces observed value with match)."""
    observed_po_number: Optional[str] = None
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    origin: FactOrigin = FactOrigin.OBSERVED
    raw_values: Dict[str, str] = field(default_factory=dict)
    derivation_rule: Optional[str] = None
    derivation_source_fields: Tuple[str, ...] = field(default_factory=tuple)
    matched_po_result: Optional[Dict[str, Any]] = None  # Reference to Phase 8D result
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ev_ids = _normalize_tuple_str(self.evidence_ids)
        field_evs = _normalize_field_evidence(self.field_evidence_ids)
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        object.__setattr__(self, "evidence_ids", ev_ids)
        object.__setattr__(self, "field_evidence_ids", field_evs)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "derivation_source_fields", _normalize_tuple_str(self.derivation_source_fields))

        accounting = {
            "observed_po_number": self.observed_po_number,
        }
        _validate_evidence_contract(
            "POFacts",
            accounting,
            orig,
            ev_ids,
            field_evs,
            self.derivation_rule,
            self.derivation_source_fields,
            self.matched_po_result,
            self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observed_po_number": self.observed_po_number,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "origin": self.origin.value,
            "raw_values": dict(self.raw_values),
            "derivation_rule": self.derivation_rule,
            "derivation_source_fields": list(self.derivation_source_fields),
            "matched_po_result": self.matched_po_result,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> POFacts:
        return cls(
            observed_po_number=data.get("observed_po_number") or data.get("po_number"),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            origin=FactOrigin(data.get("origin", "observed")),
            raw_values=dict(data.get("raw_values") or {}),
            derivation_rule=data.get("derivation_rule"),
            derivation_source_fields=_normalize_tuple_str(data.get("derivation_source_fields")),
            matched_po_result=data.get("matched_po_result"),
            metadata=dict(data.get("metadata") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Tax Facts
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TaxFact:
    """Document tax component fact."""
    tax_name: Optional[str] = None
    tax_type: Optional[str] = None
    rate: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    placement: Placement = Placement.UNKNOWN
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    origin: FactOrigin = FactOrigin.OBSERVED
    raw_values: Dict[str, str] = field(default_factory=dict)
    derivation_rule: Optional[str] = None
    derivation_source_fields: Tuple[str, ...] = field(default_factory=tuple)
    matched_tax_result: Optional[Dict[str, Any]] = None  # Reference to Phase 8C match
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_rate = to_decimal(self.rate) if self.rate is not None else None
        dec_amount = to_decimal(self.amount) if self.amount is not None else None
        plc = (
            self.placement
            if isinstance(self.placement, Placement)
            else Placement(str(self.placement).lower())
        )
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        ev_ids = _normalize_tuple_str(self.evidence_ids)
        field_evs = _normalize_field_evidence(self.field_evidence_ids)

        object.__setattr__(self, "rate", dec_rate)
        object.__setattr__(self, "amount", dec_amount)
        object.__setattr__(self, "placement", plc)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", ev_ids)
        object.__setattr__(self, "field_evidence_ids", field_evs)
        object.__setattr__(self, "derivation_source_fields", _normalize_tuple_str(self.derivation_source_fields))

        accounting = {
            "rate": dec_rate,
            "amount": dec_amount,
        }
        _validate_evidence_contract(
            "TaxFact",
            accounting,
            orig,
            ev_ids,
            field_evs,
            self.derivation_rule,
            self.derivation_source_fields,
            self.matched_tax_result,
            self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tax_name": self.tax_name,
            "tax_type": self.tax_type,
            "rate": str(self.rate) if self.rate is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "placement": self.placement.value,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "origin": self.origin.value,
            "raw_values": dict(self.raw_values),
            "derivation_rule": self.derivation_rule,
            "derivation_source_fields": list(self.derivation_source_fields),
            "matched_tax_result": self.matched_tax_result,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaxFact:
        return cls(
            tax_name=data.get("tax_name") or data.get("name"),
            tax_type=data.get("tax_type"),
            rate=to_decimal(data.get("rate")),
            amount=to_decimal(data.get("amount")),
            placement=Placement(data.get("placement", "unknown")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            origin=FactOrigin(data.get("origin", "observed")),
            raw_values=dict(data.get("raw_values") or {}),
            derivation_rule=data.get("derivation_rule"),
            derivation_source_fields=_normalize_tuple_str(data.get("derivation_source_fields")),
            matched_tax_result=data.get("matched_tax_result"),
            metadata=dict(data.get("metadata") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Discount & Charge Facts
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DiscountFact:
    """Document discount component fact."""
    name: Optional[str] = None
    rate: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    placement: Placement = Placement.UNKNOWN
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    origin: FactOrigin = FactOrigin.OBSERVED
    raw_values: Dict[str, str] = field(default_factory=dict)
    derivation_rule: Optional[str] = None
    derivation_source_fields: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_rate = to_decimal(self.rate) if self.rate is not None else None
        dec_amount = to_decimal(self.amount) if self.amount is not None else None
        plc = (
            self.placement
            if isinstance(self.placement, Placement)
            else Placement(str(self.placement).lower())
        )
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        ev_ids = _normalize_tuple_str(self.evidence_ids)
        field_evs = _normalize_field_evidence(self.field_evidence_ids)

        object.__setattr__(self, "rate", dec_rate)
        object.__setattr__(self, "amount", dec_amount)
        object.__setattr__(self, "placement", plc)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", ev_ids)
        object.__setattr__(self, "field_evidence_ids", field_evs)
        object.__setattr__(self, "derivation_source_fields", _normalize_tuple_str(self.derivation_source_fields))

        accounting = {
            "rate": dec_rate,
            "amount": dec_amount,
        }
        _validate_evidence_contract(
            "DiscountFact",
            accounting,
            orig,
            ev_ids,
            field_evs,
            self.derivation_rule,
            self.derivation_source_fields,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rate": str(self.rate) if self.rate is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "placement": self.placement.value,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "origin": self.origin.value,
            "raw_values": dict(self.raw_values),
            "derivation_rule": self.derivation_rule,
            "derivation_source_fields": list(self.derivation_source_fields),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiscountFact:
        return cls(
            name=data.get("name") or data.get("label"),
            rate=to_decimal(data.get("rate")),
            amount=to_decimal(data.get("amount")),
            placement=Placement(data.get("placement", "unknown")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            origin=FactOrigin(data.get("origin", "observed")),
            raw_values=dict(data.get("raw_values") or {}),
            derivation_rule=data.get("derivation_rule"),
            derivation_source_fields=_normalize_tuple_str(data.get("derivation_source_fields")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ChargeFact:
    """Document charge/fee component fact (shipping, handling, service fee)."""
    name: Optional[str] = None
    rate: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    placement: Placement = Placement.UNKNOWN
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    origin: FactOrigin = FactOrigin.OBSERVED
    raw_values: Dict[str, str] = field(default_factory=dict)
    derivation_rule: Optional[str] = None
    derivation_source_fields: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_rate = to_decimal(self.rate) if self.rate is not None else None
        dec_amount = to_decimal(self.amount) if self.amount is not None else None
        plc = (
            self.placement
            if isinstance(self.placement, Placement)
            else Placement(str(self.placement).lower())
        )
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        ev_ids = _normalize_tuple_str(self.evidence_ids)
        field_evs = _normalize_field_evidence(self.field_evidence_ids)

        object.__setattr__(self, "rate", dec_rate)
        object.__setattr__(self, "amount", dec_amount)
        object.__setattr__(self, "placement", plc)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", ev_ids)
        object.__setattr__(self, "field_evidence_ids", field_evs)
        object.__setattr__(self, "derivation_source_fields", _normalize_tuple_str(self.derivation_source_fields))

        accounting = {
            "rate": dec_rate,
            "amount": dec_amount,
        }
        _validate_evidence_contract(
            "ChargeFact",
            accounting,
            orig,
            ev_ids,
            field_evs,
            self.derivation_rule,
            self.derivation_source_fields,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rate": str(self.rate) if self.rate is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "placement": self.placement.value,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "origin": self.origin.value,
            "raw_values": dict(self.raw_values),
            "derivation_rule": self.derivation_rule,
            "derivation_source_fields": list(self.derivation_source_fields),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChargeFact:
        return cls(
            name=data.get("name") or data.get("label"),
            rate=to_decimal(data.get("rate")),
            amount=to_decimal(data.get("amount")),
            placement=Placement(data.get("placement", "unknown")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            origin=FactOrigin(data.get("origin", "observed")),
            raw_values=dict(data.get("raw_values") or {}),
            derivation_rule=data.get("derivation_rule"),
            derivation_source_fields=_normalize_tuple_str(data.get("derivation_source_fields")),
            metadata=dict(data.get("metadata") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Line Facts
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LineFact:
    """Line item fact representing observed line rows or component details.
    
    IMPORTANT:
    Does NOT calculate missing unit_price from amount / quantity.
    Does NOT calculate missing amount from quantity * unit_price.
    Observed values must remain observed.
    """
    line_number: Optional[int] = None
    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    discount: Optional[Decimal] = None
    taxes: Tuple[TaxFact, ...] = field(default_factory=tuple)
    currency: Optional[str] = None
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    source_row_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    semantic_role: SemanticRole = SemanticRole.UNKNOWN
    origin: FactOrigin = FactOrigin.OBSERVED
    raw_values: Dict[str, str] = field(default_factory=dict)
    derivation_rule: Optional[str] = None
    derivation_source_fields: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        ev_ids = _normalize_tuple_str(self.evidence_ids)
        field_evs = _normalize_field_evidence(self.field_evidence_ids)
        row_evs = _normalize_tuple_str(self.source_row_evidence_ids)

        # Normalize line taxes
        tx_list: List[TaxFact] = []
        for tx in self.taxes:
            if isinstance(tx, TaxFact):
                tx_list.append(tx)
            elif isinstance(tx, dict):
                tx_list.append(TaxFact.from_dict(tx))

        object.__setattr__(self, "quantity", dec_qty)
        object.__setattr__(self, "unit_price", dec_price)
        object.__setattr__(self, "amount", dec_amt)
        object.__setattr__(self, "discount", dec_disc)
        object.__setattr__(self, "taxes", tuple(tx_list))
        object.__setattr__(self, "semantic_role", role)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", ev_ids)
        object.__setattr__(self, "field_evidence_ids", field_evs)
        object.__setattr__(self, "source_row_evidence_ids", row_evs)
        object.__setattr__(self, "derivation_source_fields", _normalize_tuple_str(self.derivation_source_fields))

        accounting = {
            "quantity": dec_qty,
            "unit_price": dec_price,
            "amount": dec_amt,
            "discount": dec_disc,
        }
        # source_row_evidence_ids also counts as evidence for observed lines
        combined_ev_ids = set(ev_ids).union(row_evs)
        _validate_evidence_contract(
            "LineFact",
            accounting,
            orig,
            tuple(combined_ev_ids),
            field_evs,
            self.derivation_rule,
            self.derivation_source_fields,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_number": self.line_number,
            "description": self.description,
            "quantity": str(self.quantity) if self.quantity is not None else None,
            "unit_price": str(self.unit_price) if self.unit_price is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "discount": str(self.discount) if self.discount is not None else None,
            "taxes": [tx.to_dict() for tx in self.taxes],
            "currency": self.currency,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "source_row_evidence_ids": list(self.source_row_evidence_ids),
            "semantic_role": self.semantic_role.value,
            "origin": self.origin.value,
            "raw_values": dict(self.raw_values),
            "derivation_rule": self.derivation_rule,
            "derivation_source_fields": list(self.derivation_source_fields),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LineFact:
        raw_taxes = data.get("taxes") or []
        parsed_taxes = tuple(
            tx if isinstance(tx, TaxFact) else TaxFact.from_dict(tx)
            for tx in raw_taxes
        )
        return cls(
            line_number=data.get("line_number"),
            description=data.get("description"),
            quantity=to_decimal(data.get("quantity")),
            unit_price=to_decimal(data.get("unit_price")),
            amount=to_decimal(data.get("amount")),
            discount=to_decimal(data.get("discount")),
            taxes=parsed_taxes,
            currency=data.get("currency"),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            source_row_evidence_ids=_normalize_tuple_str(data.get("source_row_evidence_ids")),
            semantic_role=SemanticRole(data.get("semantic_role", "unknown")),
            origin=FactOrigin(data.get("origin", "observed")),
            raw_values=dict(data.get("raw_values") or {}),
            derivation_rule=data.get("derivation_rule"),
            derivation_source_fields=_normalize_tuple_str(data.get("derivation_source_fields")),
            metadata=dict(data.get("metadata") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Printed Totals Facts
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PrintedTotalsFact:
    """Document-printed totals preserved separately without reconciliation."""
    subtotal: Optional[Decimal] = None
    net: Optional[Decimal] = None
    taxable_base: Optional[Decimal] = None
    tax_total: Optional[Decimal] = None
    gross_total: Optional[Decimal] = None
    amount_due: Optional[Decimal] = None
    payment_total: Optional[Decimal] = None
    additional_totals: Dict[str, Decimal] = field(default_factory=dict)
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    origin: FactOrigin = FactOrigin.OBSERVED
    raw_values: Dict[str, str] = field(default_factory=dict)
    derivation_rule: Optional[str] = None
    derivation_source_fields: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_sub = to_decimal(self.subtotal) if self.subtotal is not None else None
        dec_net = to_decimal(self.net) if self.net is not None else None
        dec_base = to_decimal(self.taxable_base) if self.taxable_base is not None else None
        dec_tax = to_decimal(self.tax_total) if self.tax_total is not None else None
        dec_gross = to_decimal(self.gross_total) if self.gross_total is not None else None
        dec_due = to_decimal(self.amount_due) if self.amount_due is not None else None
        dec_pay = to_decimal(self.payment_total) if self.payment_total is not None else None

        add_totals: Dict[str, Decimal] = {}
        if self.additional_totals:
            for k, v in self.additional_totals.items():
                dv = to_decimal(v)
                if dv is not None:
                    add_totals[str(k)] = dv

        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        ev_ids = _normalize_tuple_str(self.evidence_ids)
        field_evs = _normalize_field_evidence(self.field_evidence_ids)

        object.__setattr__(self, "subtotal", dec_sub)
        object.__setattr__(self, "net", dec_net)
        object.__setattr__(self, "taxable_base", dec_base)
        object.__setattr__(self, "tax_total", dec_tax)
        object.__setattr__(self, "gross_total", dec_gross)
        object.__setattr__(self, "amount_due", dec_due)
        object.__setattr__(self, "payment_total", dec_pay)
        object.__setattr__(self, "additional_totals", add_totals)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", ev_ids)
        object.__setattr__(self, "field_evidence_ids", field_evs)
        object.__setattr__(self, "derivation_source_fields", _normalize_tuple_str(self.derivation_source_fields))

        accounting = {
            "subtotal": dec_sub,
            "net": dec_net,
            "taxable_base": dec_base,
            "tax_total": dec_tax,
            "gross_total": dec_gross,
            "amount_due": dec_due,
            "payment_total": dec_pay,
            **add_totals,
        }
        _validate_evidence_contract(
            "PrintedTotalsFact",
            accounting,
            orig,
            ev_ids,
            field_evs,
            self.derivation_rule,
            self.derivation_source_fields,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtotal": str(self.subtotal) if self.subtotal is not None else None,
            "net": str(self.net) if self.net is not None else None,
            "taxable_base": str(self.taxable_base) if self.taxable_base is not None else None,
            "tax_total": str(self.tax_total) if self.tax_total is not None else None,
            "gross_total": str(self.gross_total) if self.gross_total is not None else None,
            "amount_due": str(self.amount_due) if self.amount_due is not None else None,
            "payment_total": str(self.payment_total) if self.payment_total is not None else None,
            "additional_totals": {k: str(v) for k, v in self.additional_totals.items()},
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "origin": self.origin.value,
            "raw_values": dict(self.raw_values),
            "derivation_rule": self.derivation_rule,
            "derivation_source_fields": list(self.derivation_source_fields),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PrintedTotalsFact:
        raw_add = data.get("additional_totals") or {}
        add_parsed = {k: to_decimal(v) for k, v in raw_add.items() if to_decimal(v) is not None}
        return cls(
            subtotal=to_decimal(data.get("subtotal")),
            net=to_decimal(data.get("net")),
            taxable_base=to_decimal(data.get("taxable_base")),
            tax_total=to_decimal(data.get("tax_total")),
            gross_total=to_decimal(data.get("gross_total")),
            amount_due=to_decimal(data.get("amount_due")),
            payment_total=to_decimal(data.get("payment_total")),
            additional_totals=add_parsed,
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            origin=FactOrigin(data.get("origin", "observed")),
            raw_values=dict(data.get("raw_values") or {}),
            derivation_rule=data.get("derivation_rule"),
            derivation_source_fields=_normalize_tuple_str(data.get("derivation_source_fields")),
            metadata=dict(data.get("metadata") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Financial Facts Container
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FinancialFacts:
    """Container holding strictly separated financial components.
    
    Components:
    - lines: Billed line items and component details
    - discounts: Header or line-associated discounts
    - charges: Additional fees (shipping, handling, service charges)
    - taxes: Header-level or unassigned taxes
    - printed_totals: Document-printed totals
    """
    lines: Tuple[LineFact, ...] = field(default_factory=tuple)
    discounts: Tuple[DiscountFact, ...] = field(default_factory=tuple)
    charges: Tuple[ChargeFact, ...] = field(default_factory=tuple)
    taxes: Tuple[TaxFact, ...] = field(default_factory=tuple)
    printed_totals: Optional[PrintedTotalsFact] = None
    currency: Optional[str] = None
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ln_list: List[LineFact] = []
        for ln in self.lines:
            ln_list.append(ln if isinstance(ln, LineFact) else LineFact.from_dict(ln))

        disc_list: List[DiscountFact] = []
        for disc in self.discounts:
            disc_list.append(disc if isinstance(disc, DiscountFact) else DiscountFact.from_dict(disc))

        chg_list: List[ChargeFact] = []
        for chg in self.charges:
            chg_list.append(chg if isinstance(chg, ChargeFact) else ChargeFact.from_dict(chg))

        tx_list: List[TaxFact] = []
        for tx in self.taxes:
            tx_list.append(tx if isinstance(tx, TaxFact) else TaxFact.from_dict(tx))

        pt = self.printed_totals
        if pt is not None and not isinstance(pt, PrintedTotalsFact):
            pt = PrintedTotalsFact.from_dict(pt)

        object.__setattr__(self, "lines", tuple(ln_list))
        object.__setattr__(self, "discounts", tuple(disc_list))
        object.__setattr__(self, "charges", tuple(chg_list))
        object.__setattr__(self, "taxes", tuple(tx_list))
        object.__setattr__(self, "printed_totals", pt)
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))
        object.__setattr__(self, "field_evidence_ids", _normalize_field_evidence(self.field_evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lines": [ln.to_dict() for ln in self.lines],
            "discounts": [d.to_dict() for d in self.discounts],
            "charges": [c.to_dict() for c in self.charges],
            "taxes": [t.to_dict() for t in self.taxes],
            "printed_totals": self.printed_totals.to_dict() if self.printed_totals is not None else None,
            "currency": self.currency,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FinancialFacts:
        pt_data = data.get("printed_totals")
        pt = (
            pt_data
            if isinstance(pt_data, PrintedTotalsFact) or pt_data is None
            else PrintedTotalsFact.from_dict(pt_data)
        )
        return cls(
            lines=tuple(LineFact.from_dict(x) if isinstance(x, dict) else x for x in (data.get("lines") or [])),
            discounts=tuple(DiscountFact.from_dict(x) if isinstance(x, dict) else x for x in (data.get("discounts") or [])),
            charges=tuple(ChargeFact.from_dict(x) if isinstance(x, dict) else x for x in (data.get("charges") or [])),
            taxes=tuple(TaxFact.from_dict(x) if isinstance(x, dict) else x for x in (data.get("taxes") or [])),
            printed_totals=pt,
            currency=data.get("currency"),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            metadata=dict(data.get("metadata") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Top-Level Document Facts Model
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DocumentFacts:
    """Document-level fact representation.
    
    Represents the intermediate extracted facts for a logical document group
    without premature accounting reconciliation.
    """
    document_id: str
    group_id: Optional[str] = None
    page_numbers: Tuple[int, ...] = field(default_factory=tuple)
    document_role: PageRole = PageRole.INVOICE
    payable_relevance: PayableRelevance = PayableRelevance.PAYABLE_CANDIDATE
    supporting_group_ids: Tuple[str, ...] = field(default_factory=tuple)
    identity: DocumentIdentityFacts = field(default_factory=DocumentIdentityFacts)
    parties: PartyIdentityFacts = field(default_factory=PartyIdentityFacts)
    po: POFacts = field(default_factory=POFacts)
    financials: FinancialFacts = field(default_factory=FinancialFacts)
    conflicting_facts: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        doc_id = str(self.document_id).strip()
        if not doc_id:
            raise ValueError("document_id cannot be empty")

        p_nums = tuple(sorted(list(dict.fromkeys(int(p) for p in self.page_numbers))))
        sup_groups = _normalize_tuple_str(self.supporting_group_ids)

        d_role = (
            self.document_role
            if isinstance(self.document_role, PageRole)
            else PageRole(str(self.document_role).lower())
        )
        p_rel = (
            self.payable_relevance
            if isinstance(self.payable_relevance, PayableRelevance)
            else PayableRelevance(str(self.payable_relevance).lower())
        )

        ident = self.identity if isinstance(self.identity, DocumentIdentityFacts) else DocumentIdentityFacts.from_dict(self.identity)
        parties = self.parties if isinstance(self.parties, PartyIdentityFacts) else PartyIdentityFacts.from_dict(self.parties)
        po = self.po if isinstance(self.po, POFacts) else POFacts.from_dict(self.po)
        fin = self.financials if isinstance(self.financials, FinancialFacts) else FinancialFacts.from_dict(self.financials)

        object.__setattr__(self, "document_id", doc_id)
        object.__setattr__(self, "page_numbers", p_nums)
        object.__setattr__(self, "supporting_group_ids", sup_groups)
        object.__setattr__(self, "document_role", d_role)
        object.__setattr__(self, "payable_relevance", p_rel)
        object.__setattr__(self, "identity", ident)
        object.__setattr__(self, "parties", parties)
        object.__setattr__(self, "po", po)
        object.__setattr__(self, "financials", fin)
        object.__setattr__(self, "conflicting_facts", tuple(dict(cf) for cf in self.conflicting_facts))
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        """Convert DocumentFacts to a JSON-serializable dictionary."""
        return {
            "document_id": self.document_id,
            "group_id": self.group_id,
            "page_numbers": list(self.page_numbers),
            "document_role": self.document_role.value,
            "payable_relevance": self.payable_relevance.value,
            "supporting_group_ids": list(self.supporting_group_ids),
            "identity": self.identity.to_dict(),
            "parties": self.parties.to_dict(),
            "po": self.po.to_dict(),
            "financials": self.financials.to_dict(),
            "conflicting_facts": [dict(cf) for cf in self.conflicting_facts],
            "evidence_ids": list(self.evidence_ids),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentFacts:
        """Reconstruct DocumentFacts from dictionary."""
        return cls(
            document_id=data["document_id"],
            group_id=data.get("group_id"),
            page_numbers=tuple(data.get("page_numbers") or []),
            document_role=PageRole(data.get("document_role", "invoice")),
            payable_relevance=PayableRelevance(data.get("payable_relevance", "payable_candidate")),
            supporting_group_ids=_normalize_tuple_str(data.get("supporting_group_ids")),
            identity=DocumentIdentityFacts.from_dict(data.get("identity") or {}),
            parties=PartyIdentityFacts.from_dict(data.get("parties") or {}),
            po=POFacts.from_dict(data.get("po") or {}),
            financials=FinancialFacts.from_dict(data.get("financials") or {}),
            conflicting_facts=tuple(data.get("conflicting_facts") or []),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            metadata=dict(data.get("metadata") or {}),
            provenance=dict(data.get("provenance") or {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize DocumentFacts to deterministic JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> DocumentFacts:
        """Construct DocumentFacts from JSON string."""
        return cls.from_dict(json.loads(json_str))
