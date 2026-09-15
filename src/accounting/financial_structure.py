"""src/accounting/financial_structure.py — Canonical Financial Structure Normalization Layer.

Phase 9C-1: Normalizes validated FinancialDocumentAssembly objects into a canonical
accounting-oriented representation (FinancialStructure) of what the physical document
explicitly provides, while preserving uncertainty, conflicts, and evidence provenance.

Core Architectural Principles:
1. Normalize Structure, Not Accounting Truth:
   Normalizes representation, classifies components into canonical header vs line scopes,
   and standardizes numeric forms (Decimal, ISO currencies).
   Never calculates missing values, balances invoices, or repairs discrepancies.
2. Zero Accounting Arithmetic:
   No quantity * unit_price, no sum(lines), no tax_rate * base, no subtotal + tax = gross,
   and no discount or charge derivation. Missing amounts remain None.
3. Strict Preservation of Upstream FactOrigin:
   Preserves the exact FactOrigin (OBSERVED, DERIVED, MATCHED) of upstream facts.
   Never blanket-coerces facts to OBSERVED or alters origin semantics.
4. Zero Master Matching:
   No master-data matchers called. Observed parties and PO references remain observational.
   Buyer codes (company_code, business_unit_code, location_code) are only retained if
   explicitly observed upstream.
5. Preservation of Upstream Physical Ordering:
   Line items retain the exact sequence established upstream by Phase 9B-3.
   Lines are never sorted by amount, description, line number, or artificial keys.
6. Preservation of Line Identity (No Invented Line Numbers):
   Line numbers are retained if present upstream; if absent, they remain None.
   Traceability is anchored in source_line_id and source_row_evidence_ids.
7. Absolute Supporting Document Isolation:
   Supporting groups remain references (supporting_group_ids) and provenance metadata only.
   Secondary currencies, lines, charges, and totals (e.g. TRY customs pages in DU-02)
   are never merged into the primary payable's accounting structure.
8. Non-Destructive, Idempotent, and Deterministic:
   Input assemblies and facts are treated as strictly read-only and never mutated.
   normalize(normalize(X)) == normalize(X).
"""
from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.extraction.financial_assembly import FinancialDocumentAssembly
from src.extraction.validation import (
    ExtractionValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
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


# ══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════

def _normalize_tuple_str(val: Any) -> Tuple[str, ...]:
    """Normalize input to a deterministic tuple of unique non-empty strings."""
    if val is None:
        return ()
    if isinstance(val, str):
        s = val.strip()
        return (s,) if s else ()
    if isinstance(val, (list, tuple, set)):
        items: List[str] = []
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


# ══════════════════════════════════════════════════════════════════════════
# Normalization Issues
# ══════════════════════════════════════════════════════════════════════════

class NormalizationSeverity(str, Enum):
    """Severity classification of a financial structure normalization finding."""
    INFO = "INFO"          # Informational finding (e.g. missing line amount with qty & price)
    WARNING = "WARNING"    # Structural uncertainty (e.g. missing currency, conflicting totals)
    ERROR = "ERROR"        # Material structural defect (e.g. conflicting currencies in payable)
    BLOCKING = "BLOCKING"  # Critical failure preventing structural interpretation


@dataclass(frozen=True)
class NormalizationIssue:
    """Individual finding produced during financial structure normalization."""
    code: str
    severity: NormalizationSeverity
    message: str
    field: Optional[str] = None
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        sev = (
            self.severity
            if isinstance(self.severity, NormalizationSeverity)
            else NormalizationSeverity(str(self.severity).upper())
        )
        object.__setattr__(self, "severity", sev)
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field,
            "evidence_ids": list(self.evidence_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizationIssue:
        return cls(
            code=data["code"],
            severity=NormalizationSeverity(data.get("severity", "WARNING")),
            message=data["message"],
            field=data.get("field"),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            metadata=dict(data.get("metadata") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Canonical Component Models
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NormalizedParty:
    """Canonical observed representation of a document party (supplier or buyer).
    
    Zero Master Matching Guarantee:
    - Never populated from master data during 9C-1.
    - Buyer codes (company_code, business_unit_code, location_code) are only retained
      if explicitly present in upstream observed facts.
    """
    name: Optional[str] = None
    vat_id: Optional[str] = None
    country: Optional[str] = None
    email: Optional[str] = None
    bank_iban: Optional[str] = None
    address: Optional[str] = None
    company_code: Optional[str] = None
    business_unit_code: Optional[str] = None
    location_code: Optional[str] = None
    origin: FactOrigin = FactOrigin.OBSERVED
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = dataclasses.field(default_factory=dict)
    raw_values: Dict[str, str] = dataclasses.field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))
        object.__setattr__(self, "field_evidence_ids", _normalize_field_evidence(self.field_evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vat_id": self.vat_id,
            "country": self.country,
            "email": self.email,
            "bank_iban": self.bank_iban,
            "address": self.address,
            "company_code": self.company_code,
            "business_unit_code": self.business_unit_code,
            "location_code": self.location_code,
            "origin": self.origin.value,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "raw_values": dict(self.raw_values),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedParty:
        return cls(
            name=data.get("name"),
            vat_id=data.get("vat_id"),
            country=data.get("country"),
            email=data.get("email"),
            bank_iban=data.get("bank_iban"),
            address=data.get("address"),
            company_code=data.get("company_code"),
            business_unit_code=data.get("business_unit_code"),
            location_code=data.get("location_code"),
            origin=FactOrigin(data.get("origin", "observed")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            raw_values=dict(data.get("raw_values") or {}),
            metadata=dict(data.get("metadata") or {}),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(frozen=True)
class NormalizedPO:
    """Canonical observed purchase order reference.
    
    Zero Master Matching Guarantee:
    - Never mapped to master PO ID during 9C-1.
    - Represents strictly the document's printed PO number.
    """
    po_number: Optional[str] = None
    origin: FactOrigin = FactOrigin.OBSERVED
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = dataclasses.field(default_factory=dict)
    raw_values: Dict[str, str] = dataclasses.field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))
        object.__setattr__(self, "field_evidence_ids", _normalize_field_evidence(self.field_evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "po_number": self.po_number,
            "origin": self.origin.value,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "raw_values": dict(self.raw_values),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedPO:
        return cls(
            po_number=data.get("po_number"),
            origin=FactOrigin(data.get("origin", "observed")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            raw_values=dict(data.get("raw_values") or {}),
            metadata=dict(data.get("metadata") or {}),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(frozen=True)
class NormalizedTax:
    """Canonical tax component with strict scope preservation (HEADER vs LINE).
    
    Zero Accounting Arithmetic:
    - Never calculates tax_amount from rate * base.
    - Never calculates rate from amount / base.
    """
    tax_name: Optional[str] = None
    tax_type: Optional[str] = None
    rate: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    scope: Placement = Placement.UNKNOWN
    line_id: Optional[str] = None
    origin: FactOrigin = FactOrigin.OBSERVED
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = dataclasses.field(default_factory=dict)
    raw_values: Dict[str, str] = dataclasses.field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_rate = to_decimal(self.rate) if self.rate is not None else None
        dec_amount = to_decimal(self.amount) if self.amount is not None else None
        sc = (
            self.scope
            if isinstance(self.scope, Placement)
            else Placement(str(self.scope).lower())
        )
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        object.__setattr__(self, "rate", dec_rate)
        object.__setattr__(self, "amount", dec_amount)
        object.__setattr__(self, "scope", sc)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))
        object.__setattr__(self, "field_evidence_ids", _normalize_field_evidence(self.field_evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tax_name": self.tax_name,
            "tax_type": self.tax_type,
            "rate": str(self.rate) if self.rate is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "scope": self.scope.value,
            "line_id": self.line_id,
            "origin": self.origin.value,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "raw_values": dict(self.raw_values),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedTax:
        return cls(
            tax_name=data.get("tax_name"),
            tax_type=data.get("tax_type"),
            rate=to_decimal(data.get("rate")),
            amount=to_decimal(data.get("amount")),
            scope=Placement(data.get("scope", "unknown")),
            line_id=data.get("line_id"),
            origin=FactOrigin(data.get("origin", "observed")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            raw_values=dict(data.get("raw_values") or {}),
            metadata=dict(data.get("metadata") or {}),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(frozen=True)
class NormalizedDiscount:
    """Canonical discount component with strict scope preservation (HEADER vs LINE).
    
    Zero Accounting Arithmetic:
    - Never computes amount from percentage or vice versa.
    """
    name: Optional[str] = None
    rate: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    scope: Placement = Placement.UNKNOWN
    line_id: Optional[str] = None
    origin: FactOrigin = FactOrigin.OBSERVED
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = dataclasses.field(default_factory=dict)
    raw_values: Dict[str, str] = dataclasses.field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_rate = to_decimal(self.rate) if self.rate is not None else None
        dec_amount = to_decimal(self.amount) if self.amount is not None else None
        sc = (
            self.scope
            if isinstance(self.scope, Placement)
            else Placement(str(self.scope).lower())
        )
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        object.__setattr__(self, "rate", dec_rate)
        object.__setattr__(self, "amount", dec_amount)
        object.__setattr__(self, "scope", sc)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))
        object.__setattr__(self, "field_evidence_ids", _normalize_field_evidence(self.field_evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rate": str(self.rate) if self.rate is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "scope": self.scope.value,
            "line_id": self.line_id,
            "origin": self.origin.value,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "raw_values": dict(self.raw_values),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedDiscount:
        return cls(
            name=data.get("name"),
            rate=to_decimal(data.get("rate")),
            amount=to_decimal(data.get("amount")),
            scope=Placement(data.get("scope", "unknown")),
            line_id=data.get("line_id"),
            origin=FactOrigin(data.get("origin", "observed")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            raw_values=dict(data.get("raw_values") or {}),
            metadata=dict(data.get("metadata") or {}),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(frozen=True)
class NormalizedCharge:
    """Canonical charge component with strict scope preservation (HEADER vs LINE).
    
    Zero Accounting Arithmetic:
    - Preserves charges; never converts charges to discounts or vice versa.
    """
    name: Optional[str] = None
    rate: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    charge_category: Optional[str] = None
    scope: Placement = Placement.UNKNOWN
    line_id: Optional[str] = None
    origin: FactOrigin = FactOrigin.OBSERVED
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = dataclasses.field(default_factory=dict)
    raw_values: Dict[str, str] = dataclasses.field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_rate = to_decimal(self.rate) if self.rate is not None else None
        dec_amount = to_decimal(self.amount) if self.amount is not None else None
        sc = (
            self.scope
            if isinstance(self.scope, Placement)
            else Placement(str(self.scope).lower())
        )
        orig = (
            self.origin
            if isinstance(self.origin, FactOrigin)
            else FactOrigin(str(self.origin).lower())
        )
        object.__setattr__(self, "rate", dec_rate)
        object.__setattr__(self, "amount", dec_amount)
        object.__setattr__(self, "scope", sc)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))
        object.__setattr__(self, "field_evidence_ids", _normalize_field_evidence(self.field_evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rate": str(self.rate) if self.rate is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "charge_category": self.charge_category,
            "scope": self.scope.value,
            "line_id": self.line_id,
            "origin": self.origin.value,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "raw_values": dict(self.raw_values),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedCharge:
        return cls(
            name=data.get("name"),
            rate=to_decimal(data.get("rate")),
            amount=to_decimal(data.get("amount")),
            charge_category=data.get("charge_category"),
            scope=Placement(data.get("scope", "unknown")),
            line_id=data.get("line_id"),
            origin=FactOrigin(data.get("origin", "observed")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            raw_values=dict(data.get("raw_values") or {}),
            metadata=dict(data.get("metadata") or {}),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(frozen=True)
class NormalizedLine:
    """Canonical line item representation preserving upstream physical order and identity.
    
    Zero Accounting Arithmetic:
    - quantity * unit_price is NEVER calculated if amount is missing.
    - amount / quantity is NEVER calculated if unit_price is missing.
    - Missing amounts or quantities remain None.
    
    Zero Invented Line Numbers:
    - line_number is preserved if present upstream; otherwise remains None.
    """
    source_line_id: str
    line_number: Optional[int] = None
    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    currency: Optional[str] = None
    semantic_role: SemanticRole = SemanticRole.BILLED_LINE
    discounts: Tuple[NormalizedDiscount, ...] = dataclasses.field(default_factory=tuple)
    taxes: Tuple[NormalizedTax, ...] = dataclasses.field(default_factory=tuple)
    charges: Tuple[NormalizedCharge, ...] = dataclasses.field(default_factory=tuple)
    origin: FactOrigin = FactOrigin.OBSERVED
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = dataclasses.field(default_factory=dict)
    source_row_evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    raw_values: Dict[str, str] = dataclasses.field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        dec_qty = to_decimal(self.quantity) if self.quantity is not None else None
        dec_price = to_decimal(self.unit_price) if self.unit_price is not None else None
        dec_amount = to_decimal(self.amount) if self.amount is not None else None

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

        dsc_list = [d if isinstance(d, NormalizedDiscount) else NormalizedDiscount.from_dict(d) for d in self.discounts]
        tx_list = [t if isinstance(t, NormalizedTax) else NormalizedTax.from_dict(t) for t in self.taxes]
        chg_list = [c if isinstance(c, NormalizedCharge) else NormalizedCharge.from_dict(c) for c in self.charges]

        object.__setattr__(self, "quantity", dec_qty)
        object.__setattr__(self, "unit_price", dec_price)
        object.__setattr__(self, "amount", dec_amount)
        object.__setattr__(self, "semantic_role", role)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "discounts", tuple(dsc_list))
        object.__setattr__(self, "taxes", tuple(tx_list))
        object.__setattr__(self, "charges", tuple(chg_list))
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))
        object.__setattr__(self, "field_evidence_ids", _normalize_field_evidence(self.field_evidence_ids))
        object.__setattr__(self, "source_row_evidence_ids", _normalize_tuple_str(self.source_row_evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_line_id": self.source_line_id,
            "line_number": self.line_number,
            "description": self.description,
            "quantity": str(self.quantity) if self.quantity is not None else None,
            "unit_price": str(self.unit_price) if self.unit_price is not None else None,
            "amount": str(self.amount) if self.amount is not None else None,
            "currency": self.currency,
            "semantic_role": self.semantic_role.value,
            "discounts": [d.to_dict() for d in self.discounts],
            "taxes": [t.to_dict() for t in self.taxes],
            "charges": [c.to_dict() for c in self.charges],
            "origin": self.origin.value,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "source_row_evidence_ids": list(self.source_row_evidence_ids),
            "raw_values": dict(self.raw_values),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedLine:
        return cls(
            source_line_id=data["source_line_id"],
            line_number=data.get("line_number"),
            description=data.get("description"),
            quantity=to_decimal(data.get("quantity")),
            unit_price=to_decimal(data.get("unit_price")),
            amount=to_decimal(data.get("amount")),
            currency=data.get("currency"),
            semantic_role=SemanticRole(data.get("semantic_role", "billed_line")),
            discounts=tuple(NormalizedDiscount.from_dict(d) if isinstance(d, dict) else d for d in (data.get("discounts") or [])),
            taxes=tuple(NormalizedTax.from_dict(t) if isinstance(t, dict) else t for t in (data.get("taxes") or [])),
            charges=tuple(NormalizedCharge.from_dict(c) if isinstance(c, dict) else c for c in (data.get("charges") or [])),
            origin=FactOrigin(data.get("origin", "observed")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            source_row_evidence_ids=_normalize_tuple_str(data.get("source_row_evidence_ids")),
            raw_values=dict(data.get("raw_values") or {}),
            metadata=dict(data.get("metadata") or {}),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(frozen=True)
class NormalizedPrintedTotals:
    """Canonical document-printed totals preserved as observed values.
    
    Zero Accounting Arithmetic:
    - Never recalculates totals from lines or taxes.
    - Preserves contradictions; never selects values to force ERP balance.
    """
    subtotal: Optional[Decimal] = None
    net: Optional[Decimal] = None
    taxable_base: Optional[Decimal] = None
    tax_total: Optional[Decimal] = None
    gross_total: Optional[Decimal] = None
    amount_due: Optional[Decimal] = None
    payment_total: Optional[Decimal] = None
    additional_totals: Dict[str, Decimal] = dataclasses.field(default_factory=dict)
    origin: FactOrigin = FactOrigin.OBSERVED
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    field_evidence_ids: Dict[str, Tuple[str, ...]] = dataclasses.field(default_factory=dict)
    raw_values: Dict[str, str] = dataclasses.field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

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

        object.__setattr__(self, "subtotal", dec_sub)
        object.__setattr__(self, "net", dec_net)
        object.__setattr__(self, "taxable_base", dec_base)
        object.__setattr__(self, "tax_total", dec_tax)
        object.__setattr__(self, "gross_total", dec_gross)
        object.__setattr__(self, "amount_due", dec_due)
        object.__setattr__(self, "payment_total", dec_pay)
        object.__setattr__(self, "additional_totals", add_totals)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))
        object.__setattr__(self, "field_evidence_ids", _normalize_field_evidence(self.field_evidence_ids))

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
            "origin": self.origin.value,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": {k: list(v) for k, v in self.field_evidence_ids.items()},
            "raw_values": dict(self.raw_values),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedPrintedTotals:
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
            origin=FactOrigin(data.get("origin", "observed")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            field_evidence_ids=_normalize_field_evidence(data.get("field_evidence_ids")),
            raw_values=dict(data.get("raw_values") or {}),
            metadata=dict(data.get("metadata") or {}),
            provenance=dict(data.get("provenance") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Top-Level Canonical Model: FinancialStructure
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FinancialStructure:
    """Canonical accounting-oriented representation of a payable's explicit financial facts.
    
    Represents the output of Phase 9C-1:
    - Pure canonical structure of what the document explicitly provides.
    - Zero accounting calculations or reconciliations.
    - Prepares data for downstream ERP reconstruction (Phase 9C-2) without prematurely
      invoking ERP logic.
    """
    assembly_id: str
    document_id: str
    document_type: InvoiceType = InvoiceType.INVOICE
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    currency: Optional[str] = None

    supplier: Optional[NormalizedParty] = None
    buyer: Optional[NormalizedParty] = None
    purchase_order: Optional[NormalizedPO] = None

    lines: Tuple[NormalizedLine, ...] = dataclasses.field(default_factory=tuple)

    header_discounts: Tuple[NormalizedDiscount, ...] = dataclasses.field(default_factory=tuple)
    line_discounts: Tuple[NormalizedDiscount, ...] = dataclasses.field(default_factory=tuple)

    header_charges: Tuple[NormalizedCharge, ...] = dataclasses.field(default_factory=tuple)
    line_charges: Tuple[NormalizedCharge, ...] = dataclasses.field(default_factory=tuple)

    header_taxes: Tuple[NormalizedTax, ...] = dataclasses.field(default_factory=tuple)
    line_taxes: Tuple[NormalizedTax, ...] = dataclasses.field(default_factory=tuple)

    printed_totals: Optional[NormalizedPrintedTotals] = None

    validation_status: Optional[ValidationStatus] = None
    validation_issues: Tuple[ValidationIssue, ...] = dataclasses.field(default_factory=tuple)
    normalization_issues: Tuple[NormalizationIssue, ...] = dataclasses.field(default_factory=tuple)

    conflicts: Tuple[Dict[str, Any], ...] = dataclasses.field(default_factory=tuple)
    supporting_group_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        inv_type = (
            self.document_type
            if isinstance(self.document_type, InvoiceType)
            else InvoiceType(str(self.document_type).lower())
        )
        val_status = (
            self.validation_status
            if self.validation_status is None or isinstance(self.validation_status, ValidationStatus)
            else ValidationStatus(str(self.validation_status).upper())
        )

        sup = self.supplier if (self.supplier is None or isinstance(self.supplier, NormalizedParty)) else NormalizedParty.from_dict(self.supplier)
        buy = self.buyer if (self.buyer is None or isinstance(self.buyer, NormalizedParty)) else NormalizedParty.from_dict(self.buyer)
        po = self.purchase_order if (self.purchase_order is None or isinstance(self.purchase_order, NormalizedPO)) else NormalizedPO.from_dict(self.purchase_order)
        pt = self.printed_totals if (self.printed_totals is None or isinstance(self.printed_totals, NormalizedPrintedTotals)) else NormalizedPrintedTotals.from_dict(self.printed_totals)

        ln_list = [ln if isinstance(ln, NormalizedLine) else NormalizedLine.from_dict(ln) for ln in self.lines]
        hd_disc = [d if isinstance(d, NormalizedDiscount) else NormalizedDiscount.from_dict(d) for d in self.header_discounts]
        li_disc = [d if isinstance(d, NormalizedDiscount) else NormalizedDiscount.from_dict(d) for d in self.line_discounts]
        hd_chg = [c if isinstance(c, NormalizedCharge) else NormalizedCharge.from_dict(c) for c in self.header_charges]
        li_chg = [c if isinstance(c, NormalizedCharge) else NormalizedCharge.from_dict(c) for c in self.line_charges]
        hd_tx = [t if isinstance(t, NormalizedTax) else NormalizedTax.from_dict(t) for t in self.header_taxes]
        li_tx = [t if isinstance(t, NormalizedTax) else NormalizedTax.from_dict(t) for t in self.line_taxes]

        val_issues = [i if isinstance(i, ValidationIssue) else ValidationIssue.from_dict(i) for i in self.validation_issues]
        norm_issues = [i if isinstance(i, NormalizationIssue) else NormalizationIssue.from_dict(i) for i in self.normalization_issues]

        object.__setattr__(self, "document_type", inv_type)
        object.__setattr__(self, "validation_status", val_status)
        object.__setattr__(self, "supplier", sup)
        object.__setattr__(self, "buyer", buy)
        object.__setattr__(self, "purchase_order", po)
        object.__setattr__(self, "printed_totals", pt)
        object.__setattr__(self, "lines", tuple(ln_list))
        object.__setattr__(self, "header_discounts", tuple(hd_disc))
        object.__setattr__(self, "line_discounts", tuple(li_disc))
        object.__setattr__(self, "header_charges", tuple(hd_chg))
        object.__setattr__(self, "line_charges", tuple(li_chg))
        object.__setattr__(self, "header_taxes", tuple(hd_tx))
        object.__setattr__(self, "line_taxes", tuple(li_tx))
        object.__setattr__(self, "validation_issues", tuple(val_issues))
        object.__setattr__(self, "normalization_issues", tuple(norm_issues))
        object.__setattr__(self, "conflicts", tuple(dict(cf) for cf in self.conflicts))
        object.__setattr__(self, "supporting_group_ids", _normalize_tuple_str(self.supporting_group_ids))
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize FinancialStructure to a deterministic dictionary."""
        return {
            "assembly_id": self.assembly_id,
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date,
            "due_date": self.due_date,
            "currency": self.currency,
            "supplier": self.supplier.to_dict() if self.supplier is not None else None,
            "buyer": self.buyer.to_dict() if self.buyer is not None else None,
            "purchase_order": self.purchase_order.to_dict() if self.purchase_order is not None else None,
            "lines": [ln.to_dict() for ln in self.lines],
            "header_discounts": [d.to_dict() for d in self.header_discounts],
            "line_discounts": [d.to_dict() for d in self.line_discounts],
            "header_charges": [c.to_dict() for c in self.header_charges],
            "line_charges": [c.to_dict() for c in self.line_charges],
            "header_taxes": [t.to_dict() for t in self.header_taxes],
            "line_taxes": [t.to_dict() for t in self.line_taxes],
            "printed_totals": self.printed_totals.to_dict() if self.printed_totals is not None else None,
            "validation_status": self.validation_status.value if self.validation_status is not None else None,
            "validation_issues": [i.to_dict() for i in self.validation_issues],
            "normalization_issues": [i.to_dict() for i in self.normalization_issues],
            "conflicts": [dict(cf) for cf in self.conflicts],
            "supporting_group_ids": list(self.supporting_group_ids),
            "evidence_ids": list(self.evidence_ids),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FinancialStructure:
        """Construct FinancialStructure from dictionary."""
        sup_data = data.get("supplier")
        buy_data = data.get("buyer")
        po_data = data.get("purchase_order")
        pt_data = data.get("printed_totals")
        val_status_data = data.get("validation_status")

        return cls(
            assembly_id=data["assembly_id"],
            document_id=data["document_id"],
            document_type=InvoiceType(data.get("document_type", "invoice")),
            invoice_number=data.get("invoice_number"),
            invoice_date=data.get("invoice_date"),
            due_date=data.get("due_date"),
            currency=data.get("currency"),
            supplier=NormalizedParty.from_dict(sup_data) if sup_data else None,
            buyer=NormalizedParty.from_dict(buy_data) if buy_data else None,
            purchase_order=NormalizedPO.from_dict(po_data) if po_data else None,
            lines=tuple(NormalizedLine.from_dict(ln) for ln in (data.get("lines") or [])),
            header_discounts=tuple(NormalizedDiscount.from_dict(d) for d in (data.get("header_discounts") or [])),
            line_discounts=tuple(NormalizedDiscount.from_dict(d) for d in (data.get("line_discounts") or [])),
            header_charges=tuple(NormalizedCharge.from_dict(c) for c in (data.get("header_charges") or [])),
            line_charges=tuple(NormalizedCharge.from_dict(c) for c in (data.get("line_charges") or [])),
            header_taxes=tuple(NormalizedTax.from_dict(t) for t in (data.get("header_taxes") or [])),
            line_taxes=tuple(NormalizedTax.from_dict(t) for t in (data.get("line_taxes") or [])),
            printed_totals=NormalizedPrintedTotals.from_dict(pt_data) if pt_data else None,
            validation_status=ValidationStatus(val_status_data) if val_status_data else None,
            validation_issues=tuple(ValidationIssue.from_dict(i) for i in (data.get("validation_issues") or [])),
            normalization_issues=tuple(NormalizationIssue.from_dict(i) for i in (data.get("normalization_issues") or [])),
            conflicts=tuple(dict(cf) for cf in (data.get("conflicts") or [])),
            supporting_group_ids=_normalize_tuple_str(data.get("supporting_group_ids")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            provenance=dict(data.get("provenance") or {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to deterministic JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> FinancialStructure:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# Normalization Engine
# ══════════════════════════════════════════════════════════════════════════

def normalize_financial_structure(
    assembly: Union[FinancialDocumentAssembly, FinancialStructure],
    validation: Optional[ExtractionValidationResult] = None,
) -> FinancialStructure:
    """Normalize a validated FinancialDocumentAssembly into a canonical FinancialStructure.
    
    Idempotent:
    If an already-normalized FinancialStructure is passed, returns an equivalent instance.
    
    Args:
        assembly: The assembled financial document (or already normalized structure).
        validation: Optional extraction validation result from Phase 9B-4.
        
    Returns:
        Canonical FinancialStructure representing explicit document facts.
    """
    # ── 1. Idempotence Check ───────────────────────────────────────────────
    if isinstance(assembly, FinancialStructure):
        val_status = validation.status if validation else assembly.validation_status
        val_issues = validation.issues if validation else assembly.validation_issues
        return FinancialStructure(
            assembly_id=assembly.assembly_id,
            document_id=assembly.document_id,
            document_type=assembly.document_type,
            invoice_number=assembly.invoice_number,
            invoice_date=assembly.invoice_date,
            due_date=assembly.due_date,
            currency=assembly.currency,
            supplier=assembly.supplier,
            buyer=assembly.buyer,
            purchase_order=assembly.purchase_order,
            lines=assembly.lines,
            header_discounts=assembly.header_discounts,
            line_discounts=assembly.line_discounts,
            header_charges=assembly.header_charges,
            line_charges=assembly.line_charges,
            header_taxes=assembly.header_taxes,
            line_taxes=assembly.line_taxes,
            printed_totals=assembly.printed_totals,
            validation_status=val_status,
            validation_issues=val_issues,
            normalization_issues=assembly.normalization_issues,
            conflicts=assembly.conflicts,
            supporting_group_ids=assembly.supporting_group_ids,
            evidence_ids=assembly.evidence_ids,
            provenance=dict(assembly.provenance),
        )

    # ── 2. Read Facts from FinancialDocumentAssembly ───────────────────────
    facts: DocumentFacts = assembly.facts
    issues: List[NormalizationIssue] = []

    # ── 3. Identity Facts ──────────────────────────────────────────────────
    doc_type = facts.identity.invoice_type
    inv_num = facts.identity.invoice_number
    inv_date = facts.identity.invoice_date
    due_date = facts.identity.due_date

    # Currency Scope: primary currency only; supporting currencies isolated
    currency = assembly.currency or facts.identity.currency or facts.financials.currency
    if not currency:
        issues.append(
            NormalizationIssue(
                code="MISSING_CURRENCY",
                severity=NormalizationSeverity.WARNING,
                message="Document has no explicit primary currency identified.",
                field="currency",
                evidence_ids=facts.identity.evidence_ids,
            )
        )

    # Check for internal currency conflicts in primary assembly (lines vs header)
    primary_currencies: Set[str] = set()
    if currency:
        primary_currencies.add(currency)
    for ln in facts.financials.lines:
        if ln.currency:
            primary_currencies.add(ln.currency)

    if len(primary_currencies) > 1:
        issues.append(
            NormalizationIssue(
                code="CONFLICTING_CURRENCY",
                severity=NormalizationSeverity.ERROR,
                message=f"Multiple conflicting currencies observed in primary payable: {sorted(primary_currencies)}",
                field="currency",
                evidence_ids=facts.identity.evidence_ids,
            )
        )

    # ── 4. Parties (Preserve FactOrigin & Zero Master Matching) ─────────────
    # Supplier
    supplier_fact: SupplierIdentityFact = facts.parties.supplier
    supplier: Optional[NormalizedParty] = None
    has_supplier_data = any(
        getattr(supplier_fact, fld) for fld in (
            "observed_name", "vat_id", "country", "email", "bank_iban", "address"
        )
    )
    if has_supplier_data or supplier_fact.evidence_ids:
        supplier = NormalizedParty(
            name=supplier_fact.observed_name,
            vat_id=supplier_fact.vat_id,
            country=supplier_fact.country,
            email=supplier_fact.email,
            bank_iban=supplier_fact.bank_iban,
            address=supplier_fact.address,
            company_code=None,
            business_unit_code=None,
            location_code=None,
            origin=supplier_fact.origin,
            evidence_ids=supplier_fact.evidence_ids,
            field_evidence_ids=supplier_fact.field_evidence_ids,
            raw_values=supplier_fact.raw_values,
            metadata=dict(supplier_fact.metadata),
            provenance={"source_fact": "SupplierIdentityFact", **supplier_fact.metadata},
        )

    # Buyer (Buyer codes only retained if explicitly present upstream; never matched)
    buyer_fact: BuyerIdentityFact = facts.parties.buyer
    buyer: Optional[NormalizedParty] = None
    has_buyer_data = any(
        getattr(buyer_fact, fld) for fld in (
            "observed_company", "company_code", "business_unit_code", "location_code", "invoice_to_address"
        )
    )
    if has_buyer_data or buyer_fact.evidence_ids:
        buyer = NormalizedParty(
            name=buyer_fact.observed_company,
            vat_id=None,
            country=None,
            email=None,
            bank_iban=None,
            address=buyer_fact.invoice_to_address,
            company_code=buyer_fact.company_code,
            business_unit_code=buyer_fact.business_unit_code,
            location_code=buyer_fact.location_code,
            origin=buyer_fact.origin,
            evidence_ids=buyer_fact.evidence_ids,
            field_evidence_ids=buyer_fact.field_evidence_ids,
            raw_values=buyer_fact.raw_values,
            metadata=dict(buyer_fact.metadata),
            provenance={"source_fact": "BuyerIdentityFact", **buyer_fact.metadata},
        )

    # ── 5. Purchase Order (Preserve FactOrigin & Zero Master Matching) ───────
    po_fact: POFacts = facts.po
    po: Optional[NormalizedPO] = None
    if po_fact.observed_po_number or po_fact.evidence_ids:
        po = NormalizedPO(
            po_number=po_fact.observed_po_number,
            origin=po_fact.origin,
            evidence_ids=po_fact.evidence_ids,
            field_evidence_ids=po_fact.field_evidence_ids,
            raw_values=po_fact.raw_values,
            metadata=dict(po_fact.metadata),
            provenance={"source_fact": "POFacts", **po_fact.metadata},
        )

    # ── 6. Lines (Preserve Physical Order, Roles, & Origin; Do Not Invent Line Numbers)
    norm_lines: List[NormalizedLine] = []
    all_line_taxes: List[NormalizedTax] = []
    all_line_discounts: List[NormalizedDiscount] = []
    all_line_charges: List[NormalizedCharge] = []

    for idx, ln in enumerate(facts.financials.lines):
        line_id = ln.metadata.get("line_id") or f"{assembly.assembly_id}:line:{idx + 1}"

        # Line-associated taxes
        line_tax_list: List[NormalizedTax] = []
        for tx in ln.taxes:
            norm_tx = NormalizedTax(
                tax_name=tx.tax_name,
                tax_type=tx.tax_type,
                rate=tx.rate,
                amount=tx.amount,
                scope=Placement.LINE,
                line_id=line_id,
                origin=tx.origin,
                evidence_ids=tx.evidence_ids,
                field_evidence_ids=tx.field_evidence_ids,
                raw_values=tx.raw_values,
                metadata=dict(tx.metadata),
                provenance={"source_fact": "LineFact.taxes", "line_id": line_id, **tx.metadata},
            )
            line_tax_list.append(norm_tx)
            all_line_taxes.append(norm_tx)

        # Line-associated discount (if present on LineFact)
        line_disc_list: List[NormalizedDiscount] = []
        if ln.discount is not None:
            norm_disc = NormalizedDiscount(
                name="Line Discount",
                rate=None,
                amount=ln.discount,
                scope=Placement.LINE,
                line_id=line_id,
                origin=ln.origin,
                evidence_ids=ln.evidence_ids,
                field_evidence_ids=ln.field_evidence_ids,
                raw_values=ln.raw_values,
                metadata={"source_field": "discount"},
                provenance={"source_fact": "LineFact.discount", "line_id": line_id},
            )
            line_disc_list.append(norm_disc)
            all_line_discounts.append(norm_disc)

        # Flag missing line amount if quantity & unit price are present (informational state)
        if ln.amount is None and ln.quantity is not None and ln.unit_price is not None:
            issues.append(
                NormalizationIssue(
                    code="MISSING_LINE_AMOUNT",
                    severity=NormalizationSeverity.INFO,
                    message=f"Line '{ln.description or line_id}' has quantity ({ln.quantity}) and unit_price ({ln.unit_price}) but no explicit observed amount.",
                    field="amount",
                    evidence_ids=ln.evidence_ids,
                    metadata={"source_line_id": line_id},
                )
            )

        # Flag unknown semantic role
        if ln.semantic_role == SemanticRole.UNKNOWN:
            issues.append(
                NormalizationIssue(
                    code="UNKNOWN_SEMANTIC_ROLE",
                    severity=NormalizationSeverity.INFO,
                    message=f"Line '{ln.description or line_id}' has UNKNOWN semantic role.",
                    field="semantic_role",
                    evidence_ids=ln.evidence_ids,
                    metadata={"source_line_id": line_id},
                )
            )

        norm_line = NormalizedLine(
            source_line_id=line_id,
            line_number=ln.line_number,  # Preserved if present; never invented
            description=ln.description,
            quantity=ln.quantity,
            unit_price=ln.unit_price,
            amount=ln.amount,            # Preserved strictly; never calculated
            currency=ln.currency or currency,
            semantic_role=ln.semantic_role,
            discounts=tuple(line_disc_list),
            taxes=tuple(line_tax_list),
            charges=(),
            origin=ln.origin,
            evidence_ids=ln.evidence_ids,
            field_evidence_ids=ln.field_evidence_ids,
            source_row_evidence_ids=ln.source_row_evidence_ids,
            raw_values=ln.raw_values,
            metadata=dict(ln.metadata),
            provenance={"source_fact": "LineFact", "physical_index": idx, **ln.metadata},
        )
        norm_lines.append(norm_line)

    # ── 7. Taxes (Strict Scope Preservation: HEADER vs LINE) ───────────────
    header_taxes: List[NormalizedTax] = []
    for tx in facts.financials.taxes:
        if tx.placement == Placement.LINE:
            norm_tx = NormalizedTax(
                tax_name=tx.tax_name,
                tax_type=tx.tax_type,
                rate=tx.rate,
                amount=tx.amount,
                scope=Placement.LINE,
                origin=tx.origin,
                evidence_ids=tx.evidence_ids,
                field_evidence_ids=tx.field_evidence_ids,
                raw_values=tx.raw_values,
                metadata=dict(tx.metadata),
                provenance={"source_fact": "FinancialFacts.taxes", **tx.metadata},
            )
            all_line_taxes.append(norm_tx)
        else:
            # HEADER or UNKNOWN placement at document level becomes HEADER tax
            norm_tx = NormalizedTax(
                tax_name=tx.tax_name,
                tax_type=tx.tax_type,
                rate=tx.rate,
                amount=tx.amount,
                scope=Placement.HEADER,
                origin=tx.origin,
                evidence_ids=tx.evidence_ids,
                field_evidence_ids=tx.field_evidence_ids,
                raw_values=tx.raw_values,
                metadata=dict(tx.metadata),
                provenance={"source_fact": "FinancialFacts.taxes", **tx.metadata},
            )
            header_taxes.append(norm_tx)

    # ── 8. Discounts (Strict Scope Preservation: HEADER vs LINE) ───────────
    header_discounts: List[NormalizedDiscount] = []
    for d in facts.financials.discounts:
        if d.placement == Placement.LINE:
            norm_disc = NormalizedDiscount(
                name=d.name,
                rate=d.rate,
                amount=d.amount,
                scope=Placement.LINE,
                origin=d.origin,
                evidence_ids=d.evidence_ids,
                field_evidence_ids=d.field_evidence_ids,
                raw_values=d.raw_values,
                metadata=dict(d.metadata),
                provenance={"source_fact": "FinancialFacts.discounts", **d.metadata},
            )
            all_line_discounts.append(norm_disc)
        else:
            norm_disc = NormalizedDiscount(
                name=d.name,
                rate=d.rate,
                amount=d.amount,
                scope=Placement.HEADER,
                origin=d.origin,
                evidence_ids=d.evidence_ids,
                field_evidence_ids=d.field_evidence_ids,
                raw_values=d.raw_values,
                metadata=dict(d.metadata),
                provenance={"source_fact": "FinancialFacts.discounts", **d.metadata},
            )
            header_discounts.append(norm_disc)

    # ── 9. Charges (Strict Scope Preservation: HEADER vs LINE) ─────────────
    header_charges: List[NormalizedCharge] = []
    for c in facts.financials.charges:
        if c.placement == Placement.LINE:
            norm_chg = NormalizedCharge(
                name=c.name,
                rate=c.rate,
                amount=c.amount,
                scope=Placement.LINE,
                origin=c.origin,
                evidence_ids=c.evidence_ids,
                field_evidence_ids=c.field_evidence_ids,
                raw_values=c.raw_values,
                metadata=dict(c.metadata),
                provenance={"source_fact": "FinancialFacts.charges", **c.metadata},
            )
            all_line_charges.append(norm_chg)
        else:
            norm_chg = NormalizedCharge(
                name=c.name,
                rate=c.rate,
                amount=c.amount,
                scope=Placement.HEADER,
                origin=c.origin,
                evidence_ids=c.evidence_ids,
                field_evidence_ids=c.field_evidence_ids,
                raw_values=c.raw_values,
                metadata=dict(c.metadata),
                provenance={"source_fact": "FinancialFacts.charges", **c.metadata},
            )
            header_charges.append(norm_chg)

    # ── 10. Printed Totals (Preserve Observed Values; Zero Calculation) ─────
    pt_fact: Optional[PrintedTotalsFact] = facts.financials.printed_totals
    printed_totals: Optional[NormalizedPrintedTotals] = None
    if pt_fact is not None:
        printed_totals = NormalizedPrintedTotals(
            subtotal=pt_fact.subtotal,
            net=pt_fact.net,
            taxable_base=pt_fact.taxable_base,
            tax_total=pt_fact.tax_total,
            gross_total=pt_fact.gross_total,
            amount_due=pt_fact.amount_due,
            payment_total=pt_fact.payment_total,
            additional_totals=dict(pt_fact.additional_totals),
            origin=pt_fact.origin,
            evidence_ids=pt_fact.evidence_ids,
            field_evidence_ids=pt_fact.field_evidence_ids,
            raw_values=pt_fact.raw_values,
            metadata=dict(pt_fact.metadata),
            provenance={"source_fact": "PrintedTotalsFact", **pt_fact.metadata},
        )

    # ── 11. Preserved Conflicts & Validation Gate ─────────────────────────
    conflicts = list(assembly.conflicts)
    if facts.conflicting_facts:
        conflicts.extend(facts.conflicting_facts)

    for cf in conflicts:
        cf_field = cf.get("field") or "unknown"
        issues.append(
            NormalizationIssue(
                code="PRESERVED_CONFLICT",
                severity=NormalizationSeverity.WARNING,
                message=f"Preserved conflict in field '{cf_field}': {cf.get('reason') or cf}",
                field=cf_field,
                evidence_ids=_normalize_tuple_str(cf.get("evidence_ids")),
                metadata=dict(cf),
            )
        )

    val_status = validation.status if validation else None
    val_issues = validation.issues if validation else ()

    # ── 12. Assemble Canonical FinancialStructure ─────────────────────────
    all_evidence = set(assembly.evidence_ids)
    all_evidence.update(facts.evidence_ids)

    return FinancialStructure(
        assembly_id=assembly.assembly_id,
        document_id=assembly.document_id,
        document_type=doc_type,
        invoice_number=inv_num,
        invoice_date=inv_date,
        due_date=due_date,
        currency=currency,
        supplier=supplier,
        buyer=buyer,
        purchase_order=po,
        lines=tuple(norm_lines),
        header_discounts=tuple(header_discounts),
        line_discounts=tuple(all_line_discounts),
        header_charges=tuple(header_charges),
        line_charges=tuple(all_line_charges),
        header_taxes=tuple(header_taxes),
        line_taxes=tuple(all_line_taxes),
        printed_totals=printed_totals,
        validation_status=val_status,
        validation_issues=tuple(val_issues),
        normalization_issues=tuple(issues),
        conflicts=tuple(conflicts),
        supporting_group_ids=assembly.supporting_group_ids,
        evidence_ids=tuple(sorted(all_evidence)),
        provenance={
            "assembly_id": assembly.assembly_id,
            "document_id": assembly.document_id,
            "primary_group_ids": list(assembly.primary_group_ids),
            "page_numbers": list(assembly.page_numbers),
            "supporting_group_ids": list(assembly.supporting_group_ids),
            "assembly_provenance": dict(assembly.assembly_provenance),
        },
    )


def normalize_financial_structures(
    assemblies: Sequence[FinancialDocumentAssembly],
    validations: Optional[Sequence[ExtractionValidationResult]] = None,
) -> List[FinancialStructure]:
    """Batch normalize multiple FinancialDocumentAssembly objects.
    
    Multiple payables remain strictly isolated; no cross-document aggregation.
    """
    validation_map: Dict[str, ExtractionValidationResult] = {}
    if validations:
        for v in validations:
            validation_map[v.assembly_id] = v

    result: List[FinancialStructure] = []
    for asm in assemblies:
        val = validation_map.get(asm.assembly_id)
        norm = normalize_financial_structure(asm, val)
        result.append(norm)

    return result
