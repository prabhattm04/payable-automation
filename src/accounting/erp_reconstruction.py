"""src/accounting/erp_reconstruction.py — ERP Reconstruction Layer.

Phase 9C-2: Consumes canonical FinancialStructure from Phase 9C-1 and invokes the existing
authoritative ERP implementation in `erp.py` to reconstruct the accounting interpretation
of the document without changing, repairing, or inventing document facts.

Core Architectural Principles:
1. Reconstruct, Don't Repair:
   Calculates ERP-side values using the existing ERP contract.
   Never modifies observed document facts, printed totals, or missing fields.
   Observed values and ERP reconstructed values coexist with explicit provenance.
2. Zero ERP Formula Duplication:
   All accounting calculations (line bases, line taxes, header taxes, and booking gross)
   are performed directly by calling the authoritative functions in `erp.py`.
   9C-2 orchestrates, maps inputs, and records provenance.
3. Correct FactOrigin for ERP Outputs:
   Reconstructed accounting values are explicitly typed as FactOrigin.DERIVED with
   provenance linking to `erp.py`. Observed inputs strictly retain their upstream origin.
4. Semantic Role Filtering:
   Only BILLED_LINE items are posted to the ERP payload. COMPONENT_DETAIL rows are preserved
   in line reconstructions with `is_posting=False` and excluded from ERP line booking.
5. Absolute Supporting Document Isolation:
   Supporting groups remain references only. Secondary currencies (e.g. TRY customs pages
   in DU-02) never enter primary EUR ERP reconstruction.
6. Strict 9C-3 Boundary (No Reconciliation):
   9C-2 exposes both observed and reconstructed values side by side.
   Never performs reconciliation, tolerance checks, or autodraft/decline decisions.
7. Read-Only / Non-Mutation Guarantee:
   The input FinancialStructure is treated as strictly read-only and is never mutated.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import erp
from src.accounting.financial_structure import (
    FinancialStructure,
    NormalizedCharge,
    NormalizedDiscount,
    NormalizedLine,
    NormalizedPrintedTotals,
    NormalizedTax,
)
from src.understanding.document_facts import (
    FactOrigin,
    InvoiceType,
    Placement,
    SemanticRole,
    to_decimal,
)
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


# ══════════════════════════════════════════════════════════════════════════
# Enums & Issue Models
# ══════════════════════════════════════════════════════════════════════════

class ERPSeverity(str, Enum):
    """Severity classification of an ERP reconstruction finding."""
    INFO = "INFO"          # Informational note (e.g. non-posting line excluded)
    WARNING = "WARNING"    # Reconstruction condition (e.g. line has price but missing qty -> base 0.0)
    ERROR = "ERROR"        # Material input issue affecting ERP interpretation
    BLOCKING = "BLOCKING"  # Failure preventing ERP booking execution


@dataclass(frozen=True)
class ERPReconstructionIssue:
    """Individual finding produced during ERP reconstruction."""
    code: str
    severity: ERPSeverity
    message: str
    field: Optional[str] = None
    source_line_id: Optional[str] = None
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        sev = (
            self.severity
            if isinstance(self.severity, ERPSeverity)
            else ERPSeverity(str(self.severity).upper())
        )
        object.__setattr__(self, "severity", sev)
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field,
            "source_line_id": self.source_line_id,
            "evidence_ids": list(self.evidence_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ERPReconstructionIssue:
        return cls(
            code=data["code"],
            severity=ERPSeverity(data.get("severity", "WARNING")),
            message=data["message"],
            field=data.get("field"),
            source_line_id=data.get("source_line_id"),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            metadata=dict(data.get("metadata") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Reconstructed Component Models
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ERPComponentReconstruction:
    """Reconstructed individual component (tax, discount, or charge).
    
    Distinguishes observed values from ERP reconstructed amounts.
    """
    component_id: str
    name: Optional[str] = None
    scope: Placement = Placement.UNKNOWN
    observed_rate: Optional[Decimal] = None
    observed_amount: Optional[Decimal] = None
    reconstructed_amount: Optional[Decimal] = None
    source_line_id: Optional[str] = None
    origin: FactOrigin = FactOrigin.DERIVED
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
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
        object.__setattr__(self, "scope", sc)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "name": self.name,
            "scope": self.scope.value,
            "observed_rate": str(self.observed_rate) if self.observed_rate is not None else None,
            "observed_amount": str(self.observed_amount) if self.observed_amount is not None else None,
            "reconstructed_amount": str(self.reconstructed_amount) if self.reconstructed_amount is not None else None,
            "source_line_id": self.source_line_id,
            "origin": self.origin.value,
            "evidence_ids": list(self.evidence_ids),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ERPComponentReconstruction:
        return cls(
            component_id=data["component_id"],
            name=data.get("name"),
            scope=Placement(data.get("scope", "unknown")),
            observed_rate=to_decimal(data.get("observed_rate")),
            observed_amount=to_decimal(data.get("observed_amount")),
            reconstructed_amount=to_decimal(data.get("reconstructed_amount")),
            source_line_id=data.get("source_line_id"),
            origin=FactOrigin(data.get("origin", "derived")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            metadata=dict(data.get("metadata") or {}),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(frozen=True)
class ERPLineReconstruction:
    """Reconstructed line item preserving observed inputs alongside ERP outputs.
    
    Zero Document Mutation:
    - observed_amount remains strictly what the document provided (or None).
    - reconstructed_base is computed by calling erp._line_base().
    """
    source_line_id: str
    line_number: Optional[int] = None
    description: Optional[str] = None
    semantic_role: SemanticRole = SemanticRole.BILLED_LINE
    is_posting: bool = True
    observed_quantity: Optional[Decimal] = None
    observed_unit_price: Optional[Decimal] = None
    observed_amount: Optional[Decimal] = None
    reconstructed_base: Optional[Decimal] = None
    reconstructed_tax: Optional[Decimal] = None
    discounts: Tuple[ERPComponentReconstruction, ...] = dataclasses.field(default_factory=tuple)
    taxes: Tuple[ERPComponentReconstruction, ...] = dataclasses.field(default_factory=tuple)
    charges: Tuple[ERPComponentReconstruction, ...] = dataclasses.field(default_factory=tuple)
    origin: FactOrigin = FactOrigin.DERIVED
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
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
        dsc_list = [d if isinstance(d, ERPComponentReconstruction) else ERPComponentReconstruction.from_dict(d) for d in self.discounts]
        tx_list = [t if isinstance(t, ERPComponentReconstruction) else ERPComponentReconstruction.from_dict(t) for t in self.taxes]
        chg_list = [c if isinstance(c, ERPComponentReconstruction) else ERPComponentReconstruction.from_dict(c) for c in self.charges]

        object.__setattr__(self, "semantic_role", role)
        object.__setattr__(self, "origin", orig)
        object.__setattr__(self, "discounts", tuple(dsc_list))
        object.__setattr__(self, "taxes", tuple(tx_list))
        object.__setattr__(self, "charges", tuple(chg_list))
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_line_id": self.source_line_id,
            "line_number": self.line_number,
            "description": self.description,
            "semantic_role": self.semantic_role.value,
            "is_posting": self.is_posting,
            "observed_quantity": str(self.observed_quantity) if self.observed_quantity is not None else None,
            "observed_unit_price": str(self.observed_unit_price) if self.observed_unit_price is not None else None,
            "observed_amount": str(self.observed_amount) if self.observed_amount is not None else None,
            "reconstructed_base": str(self.reconstructed_base) if self.reconstructed_base is not None else None,
            "reconstructed_tax": str(self.reconstructed_tax) if self.reconstructed_tax is not None else None,
            "discounts": [d.to_dict() for d in self.discounts],
            "taxes": [t.to_dict() for t in self.taxes],
            "charges": [c.to_dict() for c in self.charges],
            "origin": self.origin.value,
            "evidence_ids": list(self.evidence_ids),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ERPLineReconstruction:
        return cls(
            source_line_id=data["source_line_id"],
            line_number=data.get("line_number"),
            description=data.get("description"),
            semantic_role=SemanticRole(data.get("semantic_role", "billed_line")),
            is_posting=bool(data.get("is_posting", True)),
            observed_quantity=to_decimal(data.get("observed_quantity")),
            observed_unit_price=to_decimal(data.get("observed_unit_price")),
            observed_amount=to_decimal(data.get("observed_amount")),
            reconstructed_base=to_decimal(data.get("reconstructed_base")),
            reconstructed_tax=to_decimal(data.get("reconstructed_tax")),
            discounts=tuple(ERPComponentReconstruction.from_dict(d) if isinstance(d, dict) else d for d in (data.get("discounts") or [])),
            taxes=tuple(ERPComponentReconstruction.from_dict(t) if isinstance(t, dict) else t for t in (data.get("taxes") or [])),
            charges=tuple(ERPComponentReconstruction.from_dict(c) if isinstance(c, dict) else c for c in (data.get("charges") or [])),
            origin=FactOrigin(data.get("origin", "derived")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            metadata=dict(data.get("metadata") or {}),
            provenance=dict(data.get("provenance") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Top-Level ERP Reconstruction Model
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ERPReconstruction:
    """Complete accounting interpretation reconstructed by the ERP contract.
    
    Preserves observed document printed totals alongside ERP reconstructed totals.
    """
    assembly_id: str
    document_id: str
    currency: Optional[str] = None
    document_type: InvoiceType = InvoiceType.INVOICE

    lines: Tuple[ERPLineReconstruction, ...] = dataclasses.field(default_factory=tuple)

    header_discounts: Tuple[ERPComponentReconstruction, ...] = dataclasses.field(default_factory=tuple)
    line_discounts: Tuple[ERPComponentReconstruction, ...] = dataclasses.field(default_factory=tuple)

    header_charges: Tuple[ERPComponentReconstruction, ...] = dataclasses.field(default_factory=tuple)
    line_charges: Tuple[ERPComponentReconstruction, ...] = dataclasses.field(default_factory=tuple)

    header_taxes: Tuple[ERPComponentReconstruction, ...] = dataclasses.field(default_factory=tuple)
    line_taxes: Tuple[ERPComponentReconstruction, ...] = dataclasses.field(default_factory=tuple)

    reconstructed_subtotal: Optional[Decimal] = None
    reconstructed_net: Optional[Decimal] = None
    reconstructed_tax_total: Optional[Decimal] = None
    reconstructed_gross_total: Optional[Decimal] = None

    printed_totals: Optional[NormalizedPrintedTotals] = None
    erp_book_result: Dict[str, Any] = dataclasses.field(default_factory=dict)
    issues: Tuple[ERPReconstructionIssue, ...] = dataclasses.field(default_factory=tuple)

    supporting_group_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        inv_type = (
            self.document_type
            if isinstance(self.document_type, InvoiceType)
            else InvoiceType(str(self.document_type).lower())
        )
        pt = self.printed_totals if (self.printed_totals is None or isinstance(self.printed_totals, NormalizedPrintedTotals)) else NormalizedPrintedTotals.from_dict(self.printed_totals)

        ln_list = [ln if isinstance(ln, ERPLineReconstruction) else ERPLineReconstruction.from_dict(ln) for ln in self.lines]
        hd_disc = [d if isinstance(d, ERPComponentReconstruction) else ERPComponentReconstruction.from_dict(d) for d in self.header_discounts]
        li_disc = [d if isinstance(d, ERPComponentReconstruction) else ERPComponentReconstruction.from_dict(d) for d in self.line_discounts]
        hd_chg = [c if isinstance(c, ERPComponentReconstruction) else ERPComponentReconstruction.from_dict(c) for c in self.header_charges]
        li_chg = [c if isinstance(c, ERPComponentReconstruction) else ERPComponentReconstruction.from_dict(c) for c in self.line_charges]
        hd_tx = [t if isinstance(t, ERPComponentReconstruction) else ERPComponentReconstruction.from_dict(t) for t in self.header_taxes]
        li_tx = [t if isinstance(t, ERPComponentReconstruction) else ERPComponentReconstruction.from_dict(t) for t in self.line_taxes]
        issue_list = [i if isinstance(i, ERPReconstructionIssue) else ERPReconstructionIssue.from_dict(i) for i in self.issues]

        object.__setattr__(self, "document_type", inv_type)
        object.__setattr__(self, "printed_totals", pt)
        object.__setattr__(self, "lines", tuple(ln_list))
        object.__setattr__(self, "header_discounts", tuple(hd_disc))
        object.__setattr__(self, "line_discounts", tuple(li_disc))
        object.__setattr__(self, "header_charges", tuple(hd_chg))
        object.__setattr__(self, "line_charges", tuple(li_chg))
        object.__setattr__(self, "header_taxes", tuple(hd_tx))
        object.__setattr__(self, "line_taxes", tuple(li_tx))
        object.__setattr__(self, "issues", tuple(issue_list))
        object.__setattr__(self, "supporting_group_ids", _normalize_tuple_str(self.supporting_group_ids))
        object.__setattr__(self, "evidence_ids", _normalize_tuple_str(self.evidence_ids))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ERPReconstruction to a deterministic dictionary."""
        return {
            "assembly_id": self.assembly_id,
            "document_id": self.document_id,
            "currency": self.currency,
            "document_type": self.document_type.value,
            "lines": [ln.to_dict() for ln in self.lines],
            "header_discounts": [d.to_dict() for d in self.header_discounts],
            "line_discounts": [d.to_dict() for d in self.line_discounts],
            "header_charges": [c.to_dict() for c in self.header_charges],
            "line_charges": [c.to_dict() for c in self.line_charges],
            "header_taxes": [t.to_dict() for t in self.header_taxes],
            "line_taxes": [t.to_dict() for t in self.line_taxes],
            "reconstructed_subtotal": str(self.reconstructed_subtotal) if self.reconstructed_subtotal is not None else None,
            "reconstructed_net": str(self.reconstructed_net) if self.reconstructed_net is not None else None,
            "reconstructed_tax_total": str(self.reconstructed_tax_total) if self.reconstructed_tax_total is not None else None,
            "reconstructed_gross_total": str(self.reconstructed_gross_total) if self.reconstructed_gross_total is not None else None,
            "printed_totals": self.printed_totals.to_dict() if self.printed_totals is not None else None,
            "erp_book_result": dict(self.erp_book_result),
            "issues": [i.to_dict() for i in self.issues],
            "supporting_group_ids": list(self.supporting_group_ids),
            "evidence_ids": list(self.evidence_ids),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ERPReconstruction:
        pt_data = data.get("printed_totals")
        return cls(
            assembly_id=data["assembly_id"],
            document_id=data["document_id"],
            currency=data.get("currency"),
            document_type=InvoiceType(data.get("document_type", "invoice")),
            lines=tuple(ERPLineReconstruction.from_dict(ln) for ln in (data.get("lines") or [])),
            header_discounts=tuple(ERPComponentReconstruction.from_dict(d) for d in (data.get("header_discounts") or [])),
            line_discounts=tuple(ERPComponentReconstruction.from_dict(d) for d in (data.get("line_discounts") or [])),
            header_charges=tuple(ERPComponentReconstruction.from_dict(c) for c in (data.get("header_charges") or [])),
            line_charges=tuple(ERPComponentReconstruction.from_dict(c) for c in (data.get("line_charges") or [])),
            header_taxes=tuple(ERPComponentReconstruction.from_dict(t) for t in (data.get("header_taxes") or [])),
            line_taxes=tuple(ERPComponentReconstruction.from_dict(t) for t in (data.get("line_taxes") or [])),
            reconstructed_subtotal=to_decimal(data.get("reconstructed_subtotal")),
            reconstructed_net=to_decimal(data.get("reconstructed_net")),
            reconstructed_tax_total=to_decimal(data.get("reconstructed_tax_total")),
            reconstructed_gross_total=to_decimal(data.get("reconstructed_gross_total")),
            printed_totals=NormalizedPrintedTotals.from_dict(pt_data) if pt_data else None,
            erp_book_result=dict(data.get("erp_book_result") or {}),
            issues=tuple(ERPReconstructionIssue.from_dict(i) for i in (data.get("issues") or [])),
            supporting_group_ids=_normalize_tuple_str(data.get("supporting_group_ids")),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            provenance=dict(data.get("provenance") or {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> ERPReconstruction:
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# ERP Input Adaptation & Reconstruction Engine
# ══════════════════════════════════════════════════════════════════════════

def build_erp_payload(financial_structure: FinancialStructure) -> Dict[str, Any]:
    """Adapt canonical FinancialStructure into the input payload expected by erp.py.
    
    Pure structural adaptation:
    - Filters posting lines (BILLED_LINE) from non-posting details.
    - Preserves scopes for taxes, discounts, and charges.
    - Zero formula execution; prepares the payload for erp.erp_book().
    """
    line_items: List[Dict[str, Any]] = []

    for ln in financial_structure.lines:
        # Only BILLED_LINE items post to the ERP ledger
        if ln.semantic_role != SemanticRole.BILLED_LINE:
            continue

        li_dict: Dict[str, Any] = {
            "description": ln.description or "",
            "quantity": str(ln.quantity) if ln.quantity is not None else "",
            "unit_price": str(ln.unit_price) if ln.unit_price is not None else "",
            "discount": "",
            "discount_percentage": "",
            "taxes": [],
        }

        # Line discounts
        for d in ln.discounts:
            if d.rate is not None:
                li_dict["discount_percentage"] = str(d.rate)
            elif d.amount is not None:
                li_dict["discount"] = str(d.amount)

        # Line taxes
        li_taxes: List[Dict[str, Any]] = []
        for tx in ln.taxes:
            li_taxes.append({
                "tax_name": tx.tax_name or "",
                "tax_rate": str(tx.rate) if tx.rate is not None else "",
                "tax_amount": str(tx.amount) if tx.amount is not None else "",
            })
        li_dict["taxes"] = li_taxes

        line_items.append(li_dict)

    # Header discount (sum of explicit header discount amounts)
    header_discount_str = ""
    if financial_structure.header_discounts:
        disc_sum = Decimal("0.00")
        has_val = False
        for d in financial_structure.header_discounts:
            if d.amount is not None:
                disc_sum += d.amount
                has_val = True
        if has_val:
            header_discount_str = str(disc_sum)

    # Header taxes
    header_taxes: List[Dict[str, Any]] = []
    for tx in financial_structure.header_taxes:
        header_taxes.append({
            "tax_name": tx.tax_name or "",
            "tax_rate": str(tx.rate) if tx.rate is not None else "",
            "tax_amount": str(tx.amount) if tx.amount is not None else "",
        })

    # Header charges categorized for ERP
    freight_amt = Decimal("0.00")
    insurance_amt = Decimal("0.00")
    excise_amt = Decimal("0.00")
    extra_amt = Decimal("0.00")

    for chg in financial_structure.header_charges:
        if chg.amount is None:
            continue
        cat = (chg.charge_category or "").lower()
        nm = (chg.name or "").lower()
        if "freight" in cat or "freight" in nm or "shipping" in nm:
            freight_amt += chg.amount
        elif "insurance" in cat or "insurance" in nm:
            insurance_amt += chg.amount
        elif "excise" in cat or "excise" in nm or "duty" in nm:
            excise_amt += chg.amount
        else:
            extra_amt += chg.amount

    payload: Dict[str, Any] = {
        "currency": financial_structure.currency or "",
        "line_items": line_items,
        "discount_amount": header_discount_str,
        "taxes": header_taxes,
        "freight_charges": str(freight_amt) if freight_amt > 0 else "",
        "insurance_charges": str(insurance_amt) if insurance_amt > 0 else "",
        "extra_charges": str(extra_amt) if extra_amt > 0 else "",
        "excise_duties": str(excise_amt) if excise_amt > 0 else "",
    }
    return payload


def reconstruct_erp(
    financial_structure: FinancialStructure,
) -> ERPReconstruction:
    """Execute ERP accounting reconstruction from canonical FinancialStructure.
    
    Zero Formula Duplication:
    Delegates all accounting calculations directly to `erp.py`.
    
    Args:
        financial_structure: The canonical normalized document structure.
        
    Returns:
        ERPReconstruction containing ERP-side outputs alongside preserved document inputs.
    """
    issues: List[ERPReconstructionIssue] = []

    # ── 1. Adapt FinancialStructure to ERP input payload ────────────────────
    payload = build_erp_payload(financial_structure)

    # ── 2. Reconstruct Line Items via erp.py helpers ────────────────────────
    reconstructed_lines: List[ERPLineReconstruction] = []
    reconstructed_line_taxes: List[ERPComponentReconstruction] = []
    reconstructed_line_discounts: List[ERPComponentReconstruction] = []
    reconstructed_line_charges: List[ERPComponentReconstruction] = []

    posting_line_bases: List[float] = []
    posting_line_taxes: List[float] = []

    for ln in financial_structure.lines:
        is_posting = (ln.semantic_role == SemanticRole.BILLED_LINE)

        # Check missing input conditions according to ERP contract
        if is_posting:
            if ln.quantity is None and ln.unit_price is not None:
                issues.append(
                    ERPReconstructionIssue(
                        code="MISSING_QUANTITY",
                        severity=ERPSeverity.WARNING,
                        message=f"Line '{ln.description or ln.source_line_id}' has unit price but missing quantity; ERP calculates base as 0.0.",
                        field="quantity",
                        source_line_id=ln.source_line_id,
                        evidence_ids=ln.evidence_ids,
                    )
                )
            elif ln.quantity is not None and ln.unit_price is None:
                issues.append(
                    ERPReconstructionIssue(
                        code="MISSING_UNIT_PRICE",
                        severity=ERPSeverity.WARNING,
                        message=f"Line '{ln.description or ln.source_line_id}' has quantity but missing unit price; ERP calculates base as 0.0.",
                        field="unit_price",
                        source_line_id=ln.source_line_id,
                        evidence_ids=ln.evidence_ids,
                    )
                )
            elif ln.quantity is None and ln.unit_price is None and ln.amount is not None:
                issues.append(
                    ERPReconstructionIssue(
                        code="MISSING_QUANTITY_AND_PRICE",
                        severity=ERPSeverity.WARNING,
                        message=f"Line '{ln.description or ln.source_line_id}' has explicit amount ({ln.amount}) but missing quantity and unit price; ERP calculates base from components as 0.0.",
                        field="quantity",
                        source_line_id=ln.source_line_id,
                        evidence_ids=ln.evidence_ids,
                    )
                )

        # Reconstruct line components
        line_disc_recons: List[ERPComponentReconstruction] = []
        for idx, d in enumerate(ln.discounts):
            comp_id = f"{ln.source_line_id}:disc:{idx + 1}"
            r_comp = ERPComponentReconstruction(
                component_id=comp_id,
                name=d.name,
                scope=Placement.LINE,
                observed_rate=d.rate,
                observed_amount=d.amount,
                reconstructed_amount=d.amount,  # discount magnitude preserved for line base
                source_line_id=ln.source_line_id,
                origin=FactOrigin.DERIVED,
                evidence_ids=d.evidence_ids,
                provenance={"source": "NormalizedLine.discounts", "contract": "erp.py"},
            )
            line_disc_recons.append(r_comp)
            reconstructed_line_discounts.append(r_comp)

        line_tax_recons: List[ERPComponentReconstruction] = []
        for idx, tx in enumerate(ln.taxes):
            comp_id = f"{ln.source_line_id}:tax:{idx + 1}"
            r_comp = ERPComponentReconstruction(
                component_id=comp_id,
                name=tx.tax_name,
                scope=Placement.LINE,
                observed_rate=tx.rate,
                observed_amount=tx.amount,
                reconstructed_amount=tx.amount,
                source_line_id=ln.source_line_id,
                origin=FactOrigin.DERIVED,
                evidence_ids=tx.evidence_ids,
                provenance={"source": "NormalizedLine.taxes", "contract": "erp.py"},
            )
            line_tax_recons.append(r_comp)
            reconstructed_line_taxes.append(r_comp)

        if is_posting:
            # Build individual line payload and call existing erp._line_base & erp._line_taxes
            li_dict: Dict[str, Any] = {
                "quantity": str(ln.quantity) if ln.quantity is not None else "",
                "unit_price": str(ln.unit_price) if ln.unit_price is not None else "",
                "discount": "",
                "discount_percentage": "",
                "taxes": [
                    {
                        "tax_name": tx.tax_name or "",
                        "tax_rate": str(tx.rate) if tx.rate is not None else "",
                        "tax_amount": str(tx.amount) if tx.amount is not None else "",
                    }
                    for tx in ln.taxes
                ],
            }
            for d in ln.discounts:
                if d.rate is not None:
                    li_dict["discount_percentage"] = str(d.rate)
                elif d.amount is not None:
                    li_dict["discount"] = str(d.amount)

            # Authoritative ERP calculation via existing erp.py helper
            base_float = erp._line_base(li_dict)
            line_tax_float = erp._line_taxes(li_dict, base_float)

            posting_line_bases.append(base_float)
            posting_line_taxes.append(line_tax_float)

            rec_base = Decimal(str(base_float))
            rec_tax = Decimal(str(line_tax_float))
            line_prov = {
                "erp_function": "erp._line_base / erp._line_taxes",
                "is_posting": True,
            }
        else:
            # Non-posting detail row: preserved without ERP ledger booking
            rec_base = None
            rec_tax = None
            line_prov = {
                "is_posting": False,
                "reason": f"Semantic role '{ln.semantic_role.value}' excluded from posting line items",
            }
            issues.append(
                ERPReconstructionIssue(
                    code="NON_POSTING_LINE_EXCLUDED",
                    severity=ERPSeverity.INFO,
                    message=f"Line '{ln.description or ln.source_line_id}' with role '{ln.semantic_role.value}' is preserved as non-posting detail.",
                    source_line_id=ln.source_line_id,
                    evidence_ids=ln.evidence_ids,
                )
            )

        line_recon = ERPLineReconstruction(
            source_line_id=ln.source_line_id,
            line_number=ln.line_number,
            description=ln.description,
            semantic_role=ln.semantic_role,
            is_posting=is_posting,
            observed_quantity=ln.quantity,
            observed_unit_price=ln.unit_price,
            observed_amount=ln.amount,
            reconstructed_base=rec_base,
            reconstructed_tax=rec_tax,
            discounts=tuple(line_disc_recons),
            taxes=tuple(line_tax_recons),
            charges=(),
            origin=FactOrigin.DERIVED,
            evidence_ids=ln.evidence_ids,
            metadata=dict(ln.metadata),
            provenance=line_prov,
        )
        reconstructed_lines.append(line_recon)

    # ── 3. Reconstruct Header Discounts ────────────────────────────────────
    reconstructed_header_discounts: List[ERPComponentReconstruction] = []
    for idx, d in enumerate(financial_structure.header_discounts):
        comp_id = f"header:discount:{idx + 1}"
        r_comp = ERPComponentReconstruction(
            component_id=comp_id,
            name=d.name,
            scope=Placement.HEADER,
            observed_rate=d.rate,
            observed_amount=d.amount,
            reconstructed_amount=d.amount,
            origin=FactOrigin.DERIVED,
            evidence_ids=d.evidence_ids,
            provenance={"source": "FinancialStructure.header_discounts", "contract": "erp.py"},
        )
        reconstructed_header_discounts.append(r_comp)

    # ── 4. Reconstruct Header Taxes via erp._header_taxes ──────────────────
    item_discounted_total_float = sum(posting_line_bases)
    header_discount_float = abs(erp.num(payload.get("discount_amount")))
    net_base_float = item_discounted_total_float - header_discount_float

    reconstructed_header_taxes: List[ERPComponentReconstruction] = []
    for idx, tx in enumerate(financial_structure.header_taxes):
        comp_id = f"header:tax:{idx + 1}"
        tx_dict = {
            "tax_name": tx.tax_name or "",
            "tax_rate": str(tx.rate) if tx.rate is not None else "",
            "tax_amount": str(tx.amount) if tx.amount is not None else "",
        }
        # Call authoritative erp._header_taxes
        tx_amt_float = erp._header_taxes([tx_dict], net_base_float)
        rec_amt = Decimal(str(tx_amt_float))

        r_comp = ERPComponentReconstruction(
            component_id=comp_id,
            name=tx.tax_name,
            scope=Placement.HEADER,
            observed_rate=tx.rate,
            observed_amount=tx.amount,
            reconstructed_amount=rec_amt,
            origin=FactOrigin.DERIVED,
            evidence_ids=tx.evidence_ids,
            provenance={"source": "FinancialStructure.header_taxes", "erp_function": "erp._header_taxes"},
        )
        reconstructed_header_taxes.append(r_comp)

    # ── 5. Reconstruct Header Charges ──────────────────────────────────────
    reconstructed_header_charges: List[ERPComponentReconstruction] = []
    for idx, chg in enumerate(financial_structure.header_charges):
        comp_id = f"header:charge:{idx + 1}"
        rec_amt = chg.amount
        r_comp = ERPComponentReconstruction(
            component_id=comp_id,
            name=chg.name,
            scope=Placement.HEADER,
            observed_rate=chg.rate,
            observed_amount=chg.amount,
            reconstructed_amount=rec_amt,
            origin=FactOrigin.DERIVED,
            evidence_ids=chg.evidence_ids,
            provenance={"source": "FinancialStructure.header_charges", "contract": "erp.py"},
        )
        reconstructed_header_charges.append(r_comp)

    # ── 6. Execute erp.erp_book() for authoritative gross total ────────────
    erp_result = erp.erp_book(payload)
    reconstructed_gross = Decimal(str(erp_result.get("will_book_gross", 0.0)))

    # Compute intermediate totals directly from ERP helper results without local formula recreation
    rec_subtotal = Decimal(str(erp.round2(item_discounted_total_float)))
    rec_net = Decimal(str(erp.round2(net_base_float)))
    total_tax_float = sum(posting_line_taxes) + erp._header_taxes(payload.get("taxes"), net_base_float)
    rec_tax_total = Decimal(str(erp.round2(total_tax_float)))

    # ── 7. Assemble Complete ERPReconstruction ─────────────────────────────
    return ERPReconstruction(
        assembly_id=financial_structure.assembly_id,
        document_id=financial_structure.document_id,
        currency=financial_structure.currency,
        document_type=financial_structure.document_type,
        lines=tuple(reconstructed_lines),
        header_discounts=tuple(reconstructed_header_discounts),
        line_discounts=tuple(reconstructed_line_discounts),
        header_charges=tuple(reconstructed_header_charges),
        line_charges=tuple(reconstructed_line_charges),
        header_taxes=tuple(reconstructed_header_taxes),
        line_taxes=tuple(reconstructed_line_taxes),
        reconstructed_subtotal=rec_subtotal,
        reconstructed_net=rec_net,
        reconstructed_tax_total=rec_tax_total,
        reconstructed_gross_total=reconstructed_gross,
        printed_totals=financial_structure.printed_totals,
        erp_book_result=dict(erp_result),
        issues=tuple(issues),
        supporting_group_ids=financial_structure.supporting_group_ids,
        evidence_ids=financial_structure.evidence_ids,
        provenance={
            "erp_engine": "erp.py",
            "erp_version": "authoritative",
            "erp_input_payload": dict(payload),
            "source_assembly_id": financial_structure.assembly_id,
        },
    )


def reconstruct_erp_batch(
    financial_structures: Sequence[FinancialStructure],
) -> List[ERPReconstruction]:
    """Batch reconstruct multiple FinancialStructure objects into ERP interpretations.
    
    Independent payables remain strictly isolated; no cross-document aggregation.
    """
    return [reconstruct_erp(fs) for fs in financial_structures]
