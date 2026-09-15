"""src/matching/match_models.py — Controlled data models for master-data matching.

Phase 8B: Supplier + Buyer Master-Data Matching.

Design Principles:
1. Controlled Match Status: Strictly `matched`, `ambiguous`, or `no_match`.
   Never use `best_guess`. A wrong match is worse than no match.
2. No Fabricated Confidence: `confidence` is strictly None.
3. Explicit Method & Provenance: The result describes the exact matching method,
   the matched fields, the supporting evidence IDs, and raw observed values.
4. Separate Buyer Hierarchy: Preserves company, business unit, and location codes
   separately in the result details.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class MatchStatus(str, Enum):
    """Controlled match status. No probabilistic or guess statuses permitted."""
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"


@dataclass
class ObservedSupplierIdentity:
    """Observed supplier identity signals extracted upstream or supplied by caller."""
    name: Optional[str] = None
    vat_id: Optional[str] = None
    country: Optional[str] = None
    email: Optional[str] = None
    bank_iban: Optional[str] = None
    address: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    field_evidence_ids: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservedSupplierIdentity:
        """Create an ObservedSupplierIdentity from a dictionary."""
        ev_ids = list(data.get("evidence_ids") or [])
        field_ev: Dict[str, str] = dict(data.get("field_evidence_ids") or {})

        # Extract per-field evidence ID helpers if present (e.g. name_evidence_id)
        for fld in ("name", "vat_id", "country", "email", "bank_iban", "address"):
            key = f"{fld}_evidence_id"
            if key in data and data[key]:
                field_ev[fld] = str(data[key])
                if str(data[key]) not in ev_ids:
                    ev_ids.append(str(data[key]))

        return cls(
            name=data.get("name"),
            vat_id=data.get("vat_id"),
            country=data.get("country"),
            email=data.get("email"),
            bank_iban=data.get("bank_iban"),
            address=data.get("address"),
            evidence_ids=ev_ids,
            field_evidence_ids=field_ev,
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vat_id": self.vat_id,
            "country": self.country,
            "email": self.email,
            "bank_iban": self.bank_iban,
            "address": self.address,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": dict(self.field_evidence_ids),
            "metadata": dict(self.metadata),
        }


@dataclass
class ObservedBuyerIdentity:
    """Observed buyer identity signals extracted upstream or supplied by caller."""
    company_name: Optional[str] = None
    company_code: Optional[str] = None
    business_unit_name: Optional[str] = None
    business_unit_code: Optional[str] = None
    location_name: Optional[str] = None
    location_code: Optional[str] = None
    invoice_to_address: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    field_evidence_ids: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservedBuyerIdentity:
        """Create an ObservedBuyerIdentity from a dictionary."""
        ev_ids = list(data.get("evidence_ids") or [])
        field_ev: Dict[str, str] = dict(data.get("field_evidence_ids") or {})

        for fld in (
            "company_name",
            "company_code",
            "business_unit_name",
            "business_unit_code",
            "location_name",
            "location_code",
            "invoice_to_address",
        ):
            key = f"{fld}_evidence_id"
            if key in data and data[key]:
                field_ev[fld] = str(data[key])
                if str(data[key]) not in ev_ids:
                    ev_ids.append(str(data[key]))

        return cls(
            company_name=data.get("company_name"),
            company_code=data.get("company_code"),
            business_unit_name=data.get("business_unit_name"),
            business_unit_code=data.get("business_unit_code"),
            location_name=data.get("location_name"),
            location_code=data.get("location_code"),
            invoice_to_address=data.get("invoice_to_address"),
            evidence_ids=ev_ids,
            field_evidence_ids=field_ev,
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_name": self.company_name,
            "company_code": self.company_code,
            "business_unit_name": self.business_unit_name,
            "business_unit_code": self.business_unit_code,
            "location_name": self.location_name,
            "location_code": self.location_code,
            "invoice_to_address": self.invoice_to_address,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": dict(self.field_evidence_ids),
            "metadata": dict(self.metadata),
        }


def to_decimal_rate(val: Any) -> Optional[Decimal]:
    """Convert an observed rate into a normalized Decimal representation.
    Ensures exact numeric equivalence: e.g. Decimal('19') == Decimal('19.0') == Decimal('19.00').
    """
    if val is None or val == "":
        return None
    if isinstance(val, Decimal):
        # normalize removes trailing zeros in exponent, e.g. 19.00 -> 19
        normalized = val.normalize()
        # In Decimal, 0.00.normalize() becomes 0E-2, convert to 0 if zero
        return Decimal(0) if normalized.is_zero() else normalized
    s = str(val).replace("%", "").strip()
    if not s:
        return None
    try:
        d = Decimal(s)
        normalized = d.normalize()
        return Decimal(0) if normalized.is_zero() else normalized
    except (InvalidOperation, ValueError):
        return None


@dataclass
class ObservedTaxIdentity:
    """Observed tax identity signals extracted upstream or supplied by caller."""
    tax_name: Optional[str] = None
    tax_type: Optional[str] = None
    rate: Optional[Union[Decimal, float, int, str]] = None
    country: Optional[str] = None
    tax_code: Optional[str] = None  # ONLY populated when explicitly identified as a tax code
    placement: Optional[str] = None  # Metadata only: "header" | "line" | "unknown"
    evidence_ids: List[str] = field(default_factory=list)
    field_evidence_ids: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservedTaxIdentity:
        """Create an ObservedTaxIdentity from a dictionary."""
        ev_ids = list(data.get("evidence_ids") or [])
        field_ev: Dict[str, str] = dict(data.get("field_evidence_ids") or {})

        for fld in ("tax_name", "tax_type", "rate", "country", "tax_code", "placement"):
            key = f"{fld}_evidence_id"
            if key in data and data[key]:
                field_ev[fld] = str(data[key])
                if str(data[key]) not in ev_ids:
                    ev_ids.append(str(data[key]))

        return cls(
            tax_name=data.get("tax_name") or data.get("name"),
            tax_type=data.get("tax_type"),
            rate=data.get("rate") or data.get("tax_rate"),
            country=data.get("country"),
            tax_code=data.get("tax_code") or data.get("code"),
            placement=data.get("placement"),
            evidence_ids=ev_ids,
            field_evidence_ids=field_ev,
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tax_name": self.tax_name,
            "tax_type": self.tax_type,
            "rate": str(self.rate) if isinstance(self.rate, Decimal) else self.rate,
            "country": self.country,
            "tax_code": self.tax_code,
            "placement": self.placement,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": dict(self.field_evidence_ids),
            "metadata": dict(self.metadata),
        }


@dataclass
class ObservedPaymentTermIdentity:
    """Observed payment term signals extracted upstream or supplied by caller."""
    raw_text: Optional[str] = None  # Explicit text alias / wording
    days: Optional[int] = None  # Explicit printed days count
    payment_term_id: Optional[str] = None  # Explicit master ID if printed
    date_derived_days: Optional[int] = None  # Derived date difference (strictly separated)
    evidence_ids: List[str] = field(default_factory=list)
    field_evidence_ids: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservedPaymentTermIdentity:
        """Create an ObservedPaymentTermIdentity from a dictionary."""
        ev_ids = list(data.get("evidence_ids") or [])
        field_ev: Dict[str, str] = dict(data.get("field_evidence_ids") or {})

        for fld in ("raw_text", "days", "payment_term_id", "date_derived_days"):
            key = f"{fld}_evidence_id"
            if key in data and data[key]:
                field_ev[fld] = str(data[key])
                if str(data[key]) not in ev_ids:
                    ev_ids.append(str(data[key]))

        days_val = data.get("days")
        int_days = int(days_val) if days_val is not None and not isinstance(days_val, bool) and str(days_val).isdigit() else None
        if isinstance(days_val, int) and not isinstance(days_val, bool):
            int_days = days_val

        derived_days_val = data.get("date_derived_days")
        int_derived_days = (
            int(derived_days_val)
            if derived_days_val is not None and not isinstance(derived_days_val, bool) and str(derived_days_val).isdigit()
            else None
        )
        if isinstance(derived_days_val, int) and not isinstance(derived_days_val, bool):
            int_derived_days = derived_days_val

        return cls(
            raw_text=data.get("raw_text") or data.get("text") or data.get("alias"),
            days=int_days,
            payment_term_id=data.get("payment_term_id"),
            date_derived_days=int_derived_days,
            evidence_ids=ev_ids,
            field_evidence_ids=field_ev,
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "days": self.days,
            "payment_term_id": self.payment_term_id,
            "date_derived_days": self.date_derived_days,
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": dict(self.field_evidence_ids),
            "metadata": dict(self.metadata),
        }


@dataclass
class ObservedPOLineEvidence:
    """Observed line evidence for an invoice or purchase order."""
    description: Optional[str] = None
    quantity: Optional[Union[Decimal, float, int]] = None
    unit_price: Optional[Union[Decimal, float, int, str]] = None
    amount: Optional[Union[Decimal, float, int, str]] = None
    uom: Optional[str] = None
    evidence_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservedPOLineEvidence:
        qty_val = data.get("quantity")
        dec_qty = to_decimal_rate(qty_val) if qty_val is not None else None

        price_val = data.get("unit_price")
        dec_price = to_decimal_rate(price_val) if price_val is not None else None

        amt_val = data.get("amount") or data.get("total")
        dec_amt = to_decimal_rate(amt_val) if amt_val is not None else None

        return cls(
            description=data.get("description"),
            quantity=dec_qty,
            unit_price=dec_price,
            amount=dec_amt,
            uom=data.get("uom"),
            evidence_id=data.get("evidence_id"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "quantity": str(self.quantity) if isinstance(self.quantity, Decimal) else self.quantity,
            "unit_price": str(self.unit_price) if isinstance(self.unit_price, Decimal) else self.unit_price,
            "amount": str(self.amount) if isinstance(self.amount, Decimal) else self.amount,
            "uom": self.uom,
            "evidence_id": self.evidence_id,
        }


@dataclass
class ObservedPOIdentity:
    """Observed purchase order signals extracted upstream or supplied by caller."""
    po_number: Optional[str] = None
    supplier_id: Optional[str] = None  # Resolved upstream by SupplierMatcher
    supplier_name: Optional[str] = None
    currency: Optional[str] = None
    gross_amount: Optional[Union[Decimal, float, int, str]] = None  # Observed only, NEVER matched on amount!
    lines: List[ObservedPOLineEvidence] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    field_evidence_ids: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservedPOIdentity:
        ev_ids = list(data.get("evidence_ids") or [])
        field_ev: Dict[str, str] = dict(data.get("field_evidence_ids") or {})

        for fld in ("po_number", "supplier_id", "supplier_name", "currency", "gross_amount"):
            key = f"{fld}_evidence_id"
            if key in data and data[key]:
                field_ev[fld] = str(data[key])
                if str(data[key]) not in ev_ids:
                    ev_ids.append(str(data[key]))

        raw_lines = data.get("lines") or data.get("po_lines") or []
        lines_list: List[ObservedPOLineEvidence] = []
        for line_item in raw_lines:
            if isinstance(line_item, ObservedPOLineEvidence):
                lines_list.append(line_item)
            elif isinstance(line_item, dict):
                lines_list.append(ObservedPOLineEvidence.from_dict(line_item))

        amt_val = data.get("gross_amount") or data.get("gross_total") or data.get("amount")
        dec_amt = to_decimal_rate(amt_val) if amt_val is not None else None

        return cls(
            po_number=data.get("po_number") or data.get("po_id"),
            supplier_id=data.get("supplier_id"),
            supplier_name=data.get("supplier_name"),
            currency=data.get("currency"),
            gross_amount=dec_amt,
            lines=lines_list,
            evidence_ids=ev_ids,
            field_evidence_ids=field_ev,
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "po_number": self.po_number,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "currency": self.currency,
            "gross_amount": str(self.gross_amount) if isinstance(self.gross_amount, Decimal) else self.gross_amount,
            "lines": [ln.to_dict() for ln in self.lines],
            "evidence_ids": list(self.evidence_ids),
            "field_evidence_ids": dict(self.field_evidence_ids),
            "metadata": dict(self.metadata),
        }


@dataclass
class MasterMatchResult:
    """Controlled result of matching an observed identity against master data."""
    entity_type: str  # "supplier" | "buyer"
    status: MatchStatus
    master_id: Optional[str] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "none"
    evidence_ids: List[str] = field(default_factory=list)
    matched_fields: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: None = None  # Never fabricate probabilistic confidence
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to clean JSON-serializable dictionary."""
        return {
            "entity_type": self.entity_type,
            "status": self.status.value,
            "master_id": self.master_id,
            "candidates": list(self.candidates),
            "method": self.method,
            "evidence_ids": list(self.evidence_ids),
            "matched_fields": list(self.matched_fields),
            "details": dict(self.details),
            "confidence": None,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MasterMatchResult:
        """Reconstruct a MasterMatchResult from a serialized dictionary."""
        status_raw = data.get("status", "no_match")
        status = MatchStatus(status_raw) if isinstance(status_raw, str) else status_raw
        return cls(
            entity_type=str(data.get("entity_type", "")),
            status=status,
            master_id=data.get("master_id"),
            candidates=list(data.get("candidates") or []),
            method=str(data.get("method", "none")),
            evidence_ids=list(data.get("evidence_ids") or []),
            matched_fields=list(data.get("matched_fields") or []),
            details=dict(data.get("details") or {}),
            confidence=None,
            provenance=dict(data.get("provenance") or {}),
        )
