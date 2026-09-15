"""src/matching/models.py — Typed internal representations for master-data records.

Phase 8A: Master-Data Loading and Indexing Foundation.

Design Principles:
1. Strict Identity Preservation: Source codes and identifiers are retained exactly as supplied.
2. Raw Provenance: Original raw records are preserved in `raw_record` for downstream auditing.
3. Immutability: Records are defined with frozen dataclasses to prevent accidental mutation.
4. Clean Serialization: Each record provides `.to_dict()` for round-trip serialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SupplierRecord:
    """Master record for an approved vendor/supplier."""
    supplier_id: str
    name: str
    vat_id: str = ""
    country: str = ""
    email: str = ""
    address: str = ""
    bank_iban: str = ""
    raw_record: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supplier_id": self.supplier_id,
            "name": self.name,
            "vat_id": self.vat_id,
            "country": self.country,
            "email": self.email,
            "address": self.address,
            "bank_iban": self.bank_iban,
        }


@dataclass(frozen=True)
class LocationRecord:
    """Leaf organizational unit representing a physical office or billing location."""
    location_code: str
    location_name: str
    invoice_to_address: str = ""
    company_code: str = ""
    business_unit_code: str = ""
    raw_record: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location_code": self.location_code,
            "location_name": self.location_name,
            "invoice_to_address": self.invoice_to_address,
            "company_code": self.company_code,
            "business_unit_code": self.business_unit_code,
        }


@dataclass(frozen=True)
class BusinessUnitRecord:
    """Intermediate organizational unit under a company (e.g. legal operating entity)."""
    business_unit_code: str
    business_unit_name: str
    company_code: str = ""
    locations: Tuple[LocationRecord, ...] = field(default_factory=tuple, hash=False)
    raw_record: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_unit_code": self.business_unit_code,
            "business_unit_name": self.business_unit_name,
            "company_code": self.company_code,
            "locations": [loc.to_dict() for loc in self.locations],
        }


@dataclass(frozen=True)
class CompanyRecord:
    """Top-level legal entity or group in the chart of books."""
    company_code: str
    company_name: str
    business_units: Tuple[BusinessUnitRecord, ...] = field(default_factory=tuple, hash=False)
    raw_record: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_code": self.company_code,
            "company_name": self.company_name,
            "business_units": [bu.to_dict() for bu in self.business_units],
        }


@dataclass(frozen=True)
class TaxRecord:
    """Tax master reference record corresponding to ERP tax_type_code."""
    code: str
    country: str
    tax_type: str
    rate: float
    name: str
    raw_record: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "country": self.country,
            "tax_type": self.tax_type,
            "rate": self.rate,
            "name": self.name,
        }


@dataclass(frozen=True)
class PaymentTermRecord:
    """Payment term reference record mapping days/aliases to payment_term_id."""
    payment_term_id: str
    days: int
    text_aliases: Tuple[str, ...] = field(default_factory=tuple, hash=False)
    raw_record: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "payment_term_id": self.payment_term_id,
            "days": self.days,
            "text_aliases": list(self.text_aliases),
        }


@dataclass(frozen=True)
class POLineRecord:
    """Individual line item in an ERP purchase order."""
    line_id: str
    description: str
    quantity: float
    uom: str
    unit_price: str
    raw_record: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "description": self.description,
            "quantity": self.quantity,
            "uom": self.uom,
            "unit_price": self.unit_price,
        }


@dataclass(frozen=True)
class PurchaseOrderRecord:
    """ERP purchase order master record."""
    po_id: str
    po_number: str
    supplier_id: str
    currency: str
    po_lines: Tuple[POLineRecord, ...] = field(default_factory=tuple, hash=False)
    raw_record: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "po_id": self.po_id,
            "po_number": self.po_number,
            "supplier_id": self.supplier_id,
            "currency": self.currency,
            "po_lines": [line.to_dict() for line in self.po_lines],
        }
