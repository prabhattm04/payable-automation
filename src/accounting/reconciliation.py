"""src/accounting/reconciliation.py — Reconciliation Layer.

Phase 9C-3: Compares the document-derived financial facts / observed printed totals
against the ERP reconstruction produced by Phase 9C-2.

Core Architectural Principles:
1. Validate, Do Not Repair:
   Compares observed document facts against ERP reconstructed values.
   Never invents discounts, charges, taxes, or balancing lines to force an equation.
   Any variance is reported as an unexplained discrepancy.
2. Zero ERP Formula Duplication:
   Consumes ERP reconstruction outputs directly from ERPReconstruction.
   Never recalculates line bases, taxes, subtotals, or gross totals locally.
3. Strict Fact Provenance & Separation:
   Document-observed facts retain their upstream origin and document evidence IDs.
   ERP reconstructed values retain their DERIVED origin and ERP provenance.
   Never overwrites observed values with derived values.
4. Intra-Record Line Correspondence:
   Guaranteed by evaluating line records where source_line_id pairs document-observed
   amounts and ERP-reconstructed bases. No arbitrary index matching.
5. Neutral Diagnostic Language:
   Reports numerical variances and correlations without asserting unproven causation.
   (e.g. notes that gross variance equals tax variance without claiming tax caused it).
6. Supporting Document Isolation:
   Supporting groups (e.g. TRY customs pages in DU-02) remain references only and
   never contaminate the primary payable reconciliation.
7. Component Details Separated:
   Non-posting detail rows (COMPONENT_DETAIL) are marked NOT_COMPARABLE at the component
   level and do NOT cause the payable document to become NOT_COMPARABLE.
8. Exact Decimal Arithmetic & Explicit Tolerance:
   Monetary comparisons use Decimal only. Tolerance is an explicit technical input
   defaulting to Decimal("0.00").
9. Immutability & Determinism:
   Frozen dataclasses with defensive dictionary copying. Zero side-effects.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.accounting.erp_reconstruction import (
    ERPComponentReconstruction,
    ERPLineReconstruction,
    ERPReconstruction,
    ERPReconstructionIssue,
    ERPSeverity,
)
from src.accounting.financial_structure import NormalizedPrintedTotals
from src.understanding.document_facts import (
    FactOrigin,
    InvoiceType,
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
# Enums
# ══════════════════════════════════════════════════════════════════════════

class ComponentStatus(str, Enum):
    """Evaluation status for an individual financial component comparison."""
    MATCH = "MATCH"                                     # Values agree within tolerance
    MISMATCH = "MISMATCH"                               # Numerical difference exceeds tolerance
    MISSING_DOCUMENT_VALUE = "MISSING_DOCUMENT_VALUE"   # Document did not print this component
    MISSING_ERP_VALUE = "MISSING_ERP_VALUE"             # ERP did not reconstruct this component
    NOT_COMPARABLE = "NOT_COMPARABLE"                   # Non-posting component detail row


class ReconciliationStatus(str, Enum):
    """Document-level reconciliation status."""
    MATCH = "MATCH"                                     # Gross and all printed intermediate components match
    MATCH_PARTIAL = "MATCH_PARTIAL"                     # Gross matches, but intermediate components were unprinted
    MISMATCH = "MISMATCH"                               # Disagreement exceeding tolerance on gross or lines
    INCOMPLETE = "INCOMPLETE"                           # Critical gross total missing from document or ERP side
    CONFLICT = "CONFLICT"                               # Upstream fatal issue prevents safe reconciliation
    NOT_COMPARABLE = "NOT_COMPARABLE"                   # Document is not an accounting payable


class DiscrepancyClassification(str, Enum):
    """Neutral diagnostic classification of a comparison result."""
    EXACT_MATCH = "EXACT_MATCH"                         # diff == Decimal("0.00")
    WITHIN_TOLERANCE = "WITHIN_TOLERANCE"               # 0 < abs(diff) <= tolerance
    NUMERICAL_VARIANCE = "NUMERICAL_VARIANCE"           # abs(diff) > tolerance
    UNAVAILABLE_DOCUMENT_VALUE = "UNAVAILABLE_DOCUMENT_VALUE" # Document value not provided
    UNAVAILABLE_ERP_VALUE = "UNAVAILABLE_ERP_VALUE"     # ERP value not reconstructed
    NON_POSTING_DETAIL = "NON_POSTING_DETAIL"           # Informational detail excluded from ledger


# ══════════════════════════════════════════════════════════════════════════
# Comparison & Discrepancy Models
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ReconciliationComparison:
    """Detailed comparison for a specific financial component."""
    component_name: str
    status: ComponentStatus
    classification: DiscrepancyClassification
    document_value: Optional[Decimal]
    document_origin: Optional[FactOrigin] = None
    reconstructed_value: Optional[Decimal] = None
    reconstructed_origin: FactOrigin = FactOrigin.DERIVED
    difference: Optional[Decimal] = None
    tolerance_applied: Decimal = Decimal("0.00")
    currency: Optional[str] = None
    document_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    erp_provenance: Dict[str, Any] = field(default_factory=dict)
    diagnostic_notes: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        st = (
            self.status
            if isinstance(self.status, ComponentStatus)
            else ComponentStatus(str(self.status).upper())
        )
        cl = (
            self.classification
            if isinstance(self.classification, DiscrepancyClassification)
            else DiscrepancyClassification(str(self.classification).upper())
        )
        d_orig = (
            self.document_origin
            if (self.document_origin is None or isinstance(self.document_origin, FactOrigin))
            else FactOrigin(str(self.document_origin).lower())
        )
        r_orig = (
            self.reconstructed_origin
            if isinstance(self.reconstructed_origin, FactOrigin)
            else FactOrigin(str(self.reconstructed_origin).lower())
        )
        object.__setattr__(self, "status", st)
        object.__setattr__(self, "classification", cl)
        object.__setattr__(self, "document_origin", d_orig)
        object.__setattr__(self, "reconstructed_origin", r_orig)
        object.__setattr__(self, "document_evidence_ids", _normalize_tuple_str(self.document_evidence_ids))
        object.__setattr__(self, "diagnostic_notes", _normalize_tuple_str(self.diagnostic_notes))
        object.__setattr__(self, "erp_provenance", dict(self.erp_provenance))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_name": self.component_name,
            "status": self.status.value,
            "classification": self.classification.value,
            "document_value": str(self.document_value) if self.document_value is not None else None,
            "document_origin": self.document_origin.value if self.document_origin is not None else None,
            "reconstructed_value": str(self.reconstructed_value) if self.reconstructed_value is not None else None,
            "reconstructed_origin": self.reconstructed_origin.value,
            "difference": str(self.difference) if self.difference is not None else None,
            "tolerance_applied": str(self.tolerance_applied),
            "currency": self.currency,
            "document_evidence_ids": list(self.document_evidence_ids),
            "erp_provenance": dict(self.erp_provenance),
            "diagnostic_notes": list(self.diagnostic_notes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReconciliationComparison:
        d_orig = data.get("document_origin")
        r_orig = data.get("reconstructed_origin")
        return cls(
            component_name=data["component_name"],
            status=ComponentStatus(data["status"]),
            classification=DiscrepancyClassification(data["classification"]),
            document_value=to_decimal(data.get("document_value")),
            document_origin=FactOrigin(d_orig) if d_orig else None,
            reconstructed_value=to_decimal(data.get("reconstructed_value")),
            reconstructed_origin=FactOrigin(r_orig) if r_orig else FactOrigin.DERIVED,
            difference=to_decimal(data.get("difference")),
            tolerance_applied=to_decimal(data.get("tolerance_applied")) or Decimal("0.00"),
            currency=data.get("currency"),
            document_evidence_ids=_normalize_tuple_str(data.get("document_evidence_ids")),
            erp_provenance=dict(data.get("erp_provenance") or {}),
            diagnostic_notes=_normalize_tuple_str(data.get("diagnostic_notes")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ReconciliationDiscrepancy:
    """Structured diagnostic issue when a comparison fails."""
    field: str
    code: str
    message: str
    document_amount: Optional[Decimal]
    reconstructed_amount: Optional[Decimal]
    difference: Optional[Decimal]
    source_line_id: Optional[str] = None
    document_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_evidence_ids", _normalize_tuple_str(self.document_evidence_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "code": self.code,
            "message": self.message,
            "document_amount": str(self.document_amount) if self.document_amount is not None else None,
            "reconstructed_amount": str(self.reconstructed_amount) if self.reconstructed_amount is not None else None,
            "difference": str(self.difference) if self.difference is not None else None,
            "source_line_id": self.source_line_id,
            "document_evidence_ids": list(self.document_evidence_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReconciliationDiscrepancy:
        return cls(
            field=data["field"],
            code=data["code"],
            message=data["message"],
            document_amount=to_decimal(data.get("document_amount")),
            reconstructed_amount=to_decimal(data.get("reconstructed_amount")),
            difference=to_decimal(data.get("difference")),
            source_line_id=data.get("source_line_id"),
            document_evidence_ids=_normalize_tuple_str(data.get("document_evidence_ids")),
            metadata=dict(data.get("metadata") or {}),
        )


# ══════════════════════════════════════════════════════════════════════════
# Top-Level Reconciliation Result Model
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ReconciliationResult:
    """Complete document-level reconciliation result."""
    assembly_id: str
    document_id: str
    currency: Optional[str] = None
    document_type: InvoiceType = InvoiceType.INVOICE
    status: ReconciliationStatus = ReconciliationStatus.INCOMPLETE

    gross_comparison: Optional[ReconciliationComparison] = None
    subtotal_comparison: Optional[ReconciliationComparison] = None
    tax_comparison: Optional[ReconciliationComparison] = None
    line_comparisons: Tuple[ReconciliationComparison, ...] = field(default_factory=tuple)

    all_comparisons: Tuple[ReconciliationComparison, ...] = field(default_factory=tuple)
    discrepancies: Tuple[ReconciliationDiscrepancy, ...] = field(default_factory=tuple)

    tolerance_used: Decimal = Decimal("0.00")
    supporting_group_ids: Tuple[str, ...] = field(default_factory=tuple)
    document_evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        inv_type = (
            self.document_type
            if isinstance(self.document_type, InvoiceType)
            else InvoiceType(str(self.document_type).lower())
        )
        st = (
            self.status
            if isinstance(self.status, ReconciliationStatus)
            else ReconciliationStatus(str(self.status).upper())
        )

        gc = (
            self.gross_comparison
            if (self.gross_comparison is None or isinstance(self.gross_comparison, ReconciliationComparison))
            else ReconciliationComparison.from_dict(self.gross_comparison)
        )
        sc = (
            self.subtotal_comparison
            if (self.subtotal_comparison is None or isinstance(self.subtotal_comparison, ReconciliationComparison))
            else ReconciliationComparison.from_dict(self.subtotal_comparison)
        )
        tc = (
            self.tax_comparison
            if (self.tax_comparison is None or isinstance(self.tax_comparison, ReconciliationComparison))
            else ReconciliationComparison.from_dict(self.tax_comparison)
        )

        ln_comps = [
            lc if isinstance(lc, ReconciliationComparison) else ReconciliationComparison.from_dict(lc)
            for lc in self.line_comparisons
        ]
        all_c = [
            ac if isinstance(ac, ReconciliationComparison) else ReconciliationComparison.from_dict(ac)
            for ac in self.all_comparisons
        ]
        disc_list = [
            d if isinstance(d, ReconciliationDiscrepancy) else ReconciliationDiscrepancy.from_dict(d)
            for d in self.discrepancies
        ]

        object.__setattr__(self, "document_type", inv_type)
        object.__setattr__(self, "status", st)
        object.__setattr__(self, "gross_comparison", gc)
        object.__setattr__(self, "subtotal_comparison", sc)
        object.__setattr__(self, "tax_comparison", tc)
        object.__setattr__(self, "line_comparisons", tuple(ln_comps))
        object.__setattr__(self, "all_comparisons", tuple(all_c))
        object.__setattr__(self, "discrepancies", tuple(disc_list))
        object.__setattr__(self, "supporting_group_ids", _normalize_tuple_str(self.supporting_group_ids))
        object.__setattr__(self, "document_evidence_ids", _normalize_tuple_str(self.document_evidence_ids))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def is_match(self) -> bool:
        """True if gross total matches (exact, within tolerance, or partial)."""
        return self.status in (ReconciliationStatus.MATCH, ReconciliationStatus.MATCH_PARTIAL)

    @property
    def is_exact_match(self) -> bool:
        """True if document status is MATCH and all evaluated components agree exactly."""
        if self.status != ReconciliationStatus.MATCH:
            return False
        return all(
            c.classification == DiscrepancyClassification.EXACT_MATCH
            for c in self.all_comparisons
            if c.status == ComponentStatus.MATCH
        )

    @property
    def is_partial(self) -> bool:
        """True if gross matched but intermediate components were omitted on the document."""
        return self.status == ReconciliationStatus.MATCH_PARTIAL

    @property
    def has_discrepancy(self) -> bool:
        """True if document has an explicit numerical mismatch or discrepancy."""
        return self.status == ReconciliationStatus.MISMATCH or len(self.discrepancies) > 0

    @property
    def is_incomplete(self) -> bool:
        """True if critical values are missing such that reconciliation could not complete."""
        return self.status == ReconciliationStatus.INCOMPLETE

    @property
    def is_conflict(self) -> bool:
        """True if unresolved conflicts prevented safe reconciliation."""
        return self.status == ReconciliationStatus.CONFLICT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assembly_id": self.assembly_id,
            "document_id": self.document_id,
            "currency": self.currency,
            "document_type": self.document_type.value,
            "status": self.status.value,
            "gross_comparison": self.gross_comparison.to_dict() if self.gross_comparison else None,
            "subtotal_comparison": self.subtotal_comparison.to_dict() if self.subtotal_comparison else None,
            "tax_comparison": self.tax_comparison.to_dict() if self.tax_comparison else None,
            "line_comparisons": [lc.to_dict() for lc in self.line_comparisons],
            "all_comparisons": [ac.to_dict() for ac in self.all_comparisons],
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "tolerance_used": str(self.tolerance_used),
            "supporting_group_ids": list(self.supporting_group_ids),
            "document_evidence_ids": list(self.document_evidence_ids),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReconciliationResult:
        gc_data = data.get("gross_comparison")
        sc_data = data.get("subtotal_comparison")
        tc_data = data.get("tax_comparison")
        return cls(
            assembly_id=data["assembly_id"],
            document_id=data["document_id"],
            currency=data.get("currency"),
            document_type=InvoiceType(data.get("document_type", "invoice")),
            status=ReconciliationStatus(data.get("status", "incomplete")),
            gross_comparison=ReconciliationComparison.from_dict(gc_data) if gc_data else None,
            subtotal_comparison=ReconciliationComparison.from_dict(sc_data) if sc_data else None,
            tax_comparison=ReconciliationComparison.from_dict(tc_data) if tc_data else None,
            line_comparisons=tuple(
                ReconciliationComparison.from_dict(lc) for lc in (data.get("line_comparisons") or [])
            ),
            all_comparisons=tuple(
                ReconciliationComparison.from_dict(ac) for ac in (data.get("all_comparisons") or [])
            ),
            discrepancies=tuple(
                ReconciliationDiscrepancy.from_dict(d) for d in (data.get("discrepancies") or [])
            ),
            tolerance_used=to_decimal(data.get("tolerance_used")) or Decimal("0.00"),
            supporting_group_ids=_normalize_tuple_str(data.get("supporting_group_ids")),
            document_evidence_ids=_normalize_tuple_str(data.get("document_evidence_ids")),
            provenance=dict(data.get("provenance") or {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> ReconciliationResult:
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# Reconciliation Engine
# ══════════════════════════════════════════════════════════════════════════

def reconcile(
    reconstruction: ERPReconstruction,
    tolerance: Union[Decimal, str, float, int] = Decimal("0.00"),
) -> ReconciliationResult:
    """Compare document-derived financial facts against ERP reconstruction.

    Zero Formula Duplication:
    Consumes outputs from ERPReconstruction directly without calculating lines,
    taxes, or gross totals locally.

    Zero Accounting Repair:
    Never creates balancing facts, discounts, charges, or lines to force agreement.

    Args:
        reconstruction: The ERPReconstruction from Phase 9C-2.
        tolerance: Explicit technical tolerance parameter (non-negative Decimal).

    Returns:
        ReconciliationResult with component comparisons and discrepancy analysis.
    """
    tol = to_decimal(tolerance)
    if tol is None or tol < Decimal("0.00"):
        raise ValueError(f"tolerance must be a non-negative Decimal; got '{tolerance}'")

    currency = reconstruction.currency
    discrepancies: List[ReconciliationDiscrepancy] = []
    all_comparisons: List[ReconciliationComparison] = []

    # ── 1. Check Fatal / Blocking Upstream Conditions ───────────────────────
    has_blocking_issue = any(
        issue.severity == ERPSeverity.BLOCKING for issue in reconstruction.issues
    )

    # ── 2. Gross Total Comparison ──────────────────────────────────────────
    doc_gross: Optional[Decimal] = None
    doc_gross_evidence: Tuple[str, ...] = ()
    doc_gross_origin: Optional[FactOrigin] = None

    if reconstruction.printed_totals is not None:
        doc_gross = reconstruction.printed_totals.gross_total
        doc_gross_evidence = reconstruction.printed_totals.evidence_ids
        doc_gross_origin = (
            reconstruction.printed_totals.metadata.get("origin")
            if isinstance(reconstruction.printed_totals.metadata.get("origin"), FactOrigin)
            else FactOrigin.OBSERVED
        )

    erp_gross = reconstruction.reconstructed_gross_total

    gross_status: ComponentStatus
    gross_class: DiscrepancyClassification
    gross_diff: Optional[Decimal] = None
    gross_notes: List[str] = []

    if not currency:
        gross_notes.append("Document primary currency is missing or empty")

    if doc_gross is None and erp_gross is None:
        gross_status = ComponentStatus.MISSING_DOCUMENT_VALUE
        gross_class = DiscrepancyClassification.UNAVAILABLE_DOCUMENT_VALUE
        gross_notes.append("Neither document gross nor ERP gross is available")
    elif doc_gross is None:
        gross_status = ComponentStatus.MISSING_DOCUMENT_VALUE
        gross_class = DiscrepancyClassification.UNAVAILABLE_DOCUMENT_VALUE
        gross_notes.append("Document did not print a gross total")
    elif erp_gross is None:
        gross_status = ComponentStatus.MISSING_ERP_VALUE
        gross_class = DiscrepancyClassification.UNAVAILABLE_ERP_VALUE
        gross_notes.append("ERP did not reconstruct a gross total")
    else:
        gross_diff = erp_gross - doc_gross
        if gross_diff == Decimal("0.00"):
            gross_status = ComponentStatus.MATCH
            gross_class = DiscrepancyClassification.EXACT_MATCH
        elif abs(gross_diff) <= tol:
            gross_status = ComponentStatus.MATCH
            gross_class = DiscrepancyClassification.WITHIN_TOLERANCE
            gross_notes.append(f"Difference of {gross_diff} is within tolerance {tol}")
        else:
            gross_status = ComponentStatus.MISMATCH
            gross_class = DiscrepancyClassification.NUMERICAL_VARIANCE
            discrepancies.append(
                ReconciliationDiscrepancy(
                    field="gross_total",
                    code="GROSS_TOTAL_MISMATCH",
                    message=f"Gross total variance of {gross_diff} exceeds tolerance {tol} (reconstructed: {erp_gross}, document: {doc_gross}).",
                    document_amount=doc_gross,
                    reconstructed_amount=erp_gross,
                    difference=gross_diff,
                    document_evidence_ids=doc_gross_evidence,
                )
            )

    gross_comp = ReconciliationComparison(
        component_name="gross_total",
        status=gross_status,
        classification=gross_class,
        document_value=doc_gross,
        document_origin=doc_gross_origin,
        reconstructed_value=erp_gross,
        reconstructed_origin=FactOrigin.DERIVED,
        difference=gross_diff,
        tolerance_applied=tol,
        currency=currency,
        document_evidence_ids=doc_gross_evidence,
        erp_provenance=dict(reconstruction.provenance),
        diagnostic_notes=tuple(gross_notes),
    )
    all_comparisons.append(gross_comp)

    # ── 3. Subtotal Comparison ─────────────────────────────────────────────
    doc_subtotal: Optional[Decimal] = None
    doc_subtotal_evidence: Tuple[str, ...] = ()
    doc_subtotal_origin: Optional[FactOrigin] = None

    if reconstruction.printed_totals is not None:
        doc_subtotal = reconstruction.printed_totals.subtotal
        doc_subtotal_evidence = reconstruction.printed_totals.evidence_ids
        doc_subtotal_origin = (
            reconstruction.printed_totals.metadata.get("origin")
            if isinstance(reconstruction.printed_totals.metadata.get("origin"), FactOrigin)
            else FactOrigin.OBSERVED
        )

    erp_subtotal = reconstruction.reconstructed_subtotal

    subtotal_status: ComponentStatus
    subtotal_class: DiscrepancyClassification
    subtotal_diff: Optional[Decimal] = None
    subtotal_notes: List[str] = []

    if doc_subtotal is None and erp_subtotal is None:
        subtotal_status = ComponentStatus.MISSING_DOCUMENT_VALUE
        subtotal_class = DiscrepancyClassification.UNAVAILABLE_DOCUMENT_VALUE
        subtotal_notes.append("Subtotal not available on document or ERP side")
    elif doc_subtotal is None:
        subtotal_status = ComponentStatus.MISSING_DOCUMENT_VALUE
        subtotal_class = DiscrepancyClassification.UNAVAILABLE_DOCUMENT_VALUE
        subtotal_notes.append("Document did not print subtotal")
    elif erp_subtotal is None:
        subtotal_status = ComponentStatus.MISSING_ERP_VALUE
        subtotal_class = DiscrepancyClassification.UNAVAILABLE_ERP_VALUE
        subtotal_notes.append("ERP did not reconstruct subtotal")
    else:
        subtotal_diff = erp_subtotal - doc_subtotal
        if subtotal_diff == Decimal("0.00"):
            subtotal_status = ComponentStatus.MATCH
            subtotal_class = DiscrepancyClassification.EXACT_MATCH
        elif abs(subtotal_diff) <= tol:
            subtotal_status = ComponentStatus.MATCH
            subtotal_class = DiscrepancyClassification.WITHIN_TOLERANCE
            subtotal_notes.append(f"Difference of {subtotal_diff} is within tolerance {tol}")
        else:
            subtotal_status = ComponentStatus.MISMATCH
            subtotal_class = DiscrepancyClassification.NUMERICAL_VARIANCE
            discrepancies.append(
                ReconciliationDiscrepancy(
                    field="subtotal",
                    code="SUBTOTAL_MISMATCH",
                    message=f"Subtotal variance of {subtotal_diff} exceeds tolerance {tol} (reconstructed: {erp_subtotal}, document: {doc_subtotal}).",
                    document_amount=doc_subtotal,
                    reconstructed_amount=erp_subtotal,
                    difference=subtotal_diff,
                    document_evidence_ids=doc_subtotal_evidence,
                )
            )

    subtotal_comp = ReconciliationComparison(
        component_name="subtotal",
        status=subtotal_status,
        classification=subtotal_class,
        document_value=doc_subtotal,
        document_origin=doc_subtotal_origin,
        reconstructed_value=erp_subtotal,
        reconstructed_origin=FactOrigin.DERIVED,
        difference=subtotal_diff,
        tolerance_applied=tol,
        currency=currency,
        document_evidence_ids=doc_subtotal_evidence,
        erp_provenance=dict(reconstruction.provenance),
        diagnostic_notes=tuple(subtotal_notes),
    )
    all_comparisons.append(subtotal_comp)

    # ── 4. Tax Total Comparison ────────────────────────────────────────────
    doc_tax: Optional[Decimal] = None
    doc_tax_evidence: Tuple[str, ...] = ()
    doc_tax_origin: Optional[FactOrigin] = None

    if reconstruction.printed_totals is not None:
        doc_tax = reconstruction.printed_totals.tax_total
        doc_tax_evidence = reconstruction.printed_totals.evidence_ids
        doc_tax_origin = (
            reconstruction.printed_totals.metadata.get("origin")
            if isinstance(reconstruction.printed_totals.metadata.get("origin"), FactOrigin)
            else FactOrigin.OBSERVED
        )

    erp_tax = reconstruction.reconstructed_tax_total

    tax_status: ComponentStatus
    tax_class: DiscrepancyClassification
    tax_diff: Optional[Decimal] = None
    tax_notes: List[str] = []

    if doc_tax is None and erp_tax is None:
        tax_status = ComponentStatus.MISSING_DOCUMENT_VALUE
        tax_class = DiscrepancyClassification.UNAVAILABLE_DOCUMENT_VALUE
        tax_notes.append("Tax total not available on document or ERP side")
    elif doc_tax is None:
        tax_status = ComponentStatus.MISSING_DOCUMENT_VALUE
        tax_class = DiscrepancyClassification.UNAVAILABLE_DOCUMENT_VALUE
        tax_notes.append("Document did not print tax total")
    elif erp_tax is None:
        tax_status = ComponentStatus.MISSING_ERP_VALUE
        tax_class = DiscrepancyClassification.UNAVAILABLE_ERP_VALUE
        tax_notes.append("ERP did not reconstruct tax total")
    else:
        tax_diff = erp_tax - doc_tax
        if tax_diff == Decimal("0.00"):
            tax_status = ComponentStatus.MATCH
            tax_class = DiscrepancyClassification.EXACT_MATCH
        elif abs(tax_diff) <= tol:
            tax_status = ComponentStatus.MATCH
            tax_class = DiscrepancyClassification.WITHIN_TOLERANCE
            tax_notes.append(f"Difference of {tax_diff} is within tolerance {tol}")
        else:
            tax_status = ComponentStatus.MISMATCH
            tax_class = DiscrepancyClassification.NUMERICAL_VARIANCE
            discrepancies.append(
                ReconciliationDiscrepancy(
                    field="tax_total",
                    code="TAX_TOTAL_MISMATCH",
                    message=f"Tax total variance of {tax_diff} exceeds tolerance {tol} (reconstructed: {erp_tax}, document: {doc_tax}).",
                    document_amount=doc_tax,
                    reconstructed_amount=erp_tax,
                    difference=tax_diff,
                    document_evidence_ids=doc_tax_evidence,
                )
            )

    tax_comp = ReconciliationComparison(
        component_name="tax_total",
        status=tax_status,
        classification=tax_class,
        document_value=doc_tax,
        document_origin=doc_tax_origin,
        reconstructed_value=erp_tax,
        reconstructed_origin=FactOrigin.DERIVED,
        difference=tax_diff,
        tolerance_applied=tol,
        currency=currency,
        document_evidence_ids=doc_tax_evidence,
        erp_provenance=dict(reconstruction.provenance),
        diagnostic_notes=tuple(tax_notes),
    )
    all_comparisons.append(tax_comp)

    # ── 5. Neutral Diagnostic Correlation (No Causal Claim) ────────────────
    # Check if numerical variances coincide without asserting causation
    if gross_diff is not None and gross_diff != Decimal("0.00"):
        extra_gross_notes = list(gross_comp.diagnostic_notes)
        if tax_diff is not None and gross_diff == tax_diff:
            extra_gross_notes.append(
                f"Gross variance of {gross_diff} numerically equals tax component variance of {tax_diff}"
            )
        if subtotal_diff is not None and gross_diff == subtotal_diff:
            extra_gross_notes.append(
                f"Gross variance of {gross_diff} numerically equals subtotal component variance of {subtotal_diff}"
            )
        if len(extra_gross_notes) != len(gross_comp.diagnostic_notes):
            gross_comp = dataclasses.replace(gross_comp, diagnostic_notes=tuple(extra_gross_notes))
            all_comparisons[0] = gross_comp

    # ── 6. Intra-Record Line Correspondence & Comparison ───────────────────
    line_comparisons: List[ReconciliationComparison] = []

    for ln in reconstruction.lines:
        line_name = ln.source_line_id or f"line_{ln.line_number or 'unidentified'}"

        if not ln.is_posting:
            # Non-posting detail row (e.g. COMPONENT_DETAIL)
            l_comp = ReconciliationComparison(
                component_name=line_name,
                status=ComponentStatus.NOT_COMPARABLE,
                classification=DiscrepancyClassification.NON_POSTING_DETAIL,
                document_value=ln.observed_amount,
                document_origin=ln.origin,
                reconstructed_value=None,
                reconstructed_origin=FactOrigin.DERIVED,
                difference=None,
                tolerance_applied=tol,
                currency=currency,
                document_evidence_ids=ln.evidence_ids,
                erp_provenance=dict(ln.provenance),
                diagnostic_notes=("Non-posting detail row excluded from ERP ledger booking",),
            )
            line_comparisons.append(l_comp)
            all_comparisons.append(l_comp)
            continue

        # Posting line (BILLED_LINE)
        l_doc = ln.observed_amount
        l_erp = ln.reconstructed_base
        l_status: ComponentStatus
        l_class: DiscrepancyClassification
        l_diff: Optional[Decimal] = None
        l_notes: List[str] = []

        if l_doc is None and l_erp is None:
            l_status = ComponentStatus.MISSING_DOCUMENT_VALUE
            l_class = DiscrepancyClassification.UNAVAILABLE_DOCUMENT_VALUE
            l_notes.append("Line observed amount and reconstructed base both missing")
        elif l_doc is None:
            l_status = ComponentStatus.MISSING_DOCUMENT_VALUE
            l_class = DiscrepancyClassification.UNAVAILABLE_DOCUMENT_VALUE
            l_notes.append("Document did not print line amount")
        elif l_erp is None:
            l_status = ComponentStatus.MISSING_ERP_VALUE
            l_class = DiscrepancyClassification.UNAVAILABLE_ERP_VALUE
            l_notes.append("ERP did not reconstruct line base")
        else:
            l_diff = l_erp - l_doc
            if l_diff == Decimal("0.00"):
                l_status = ComponentStatus.MATCH
                l_class = DiscrepancyClassification.EXACT_MATCH
            elif abs(l_diff) <= tol:
                l_status = ComponentStatus.MATCH
                l_class = DiscrepancyClassification.WITHIN_TOLERANCE
                l_notes.append(f"Line difference of {l_diff} is within tolerance {tol}")
            else:
                l_status = ComponentStatus.MISMATCH
                l_class = DiscrepancyClassification.NUMERICAL_VARIANCE
                discrepancies.append(
                    ReconciliationDiscrepancy(
                        field=f"lines[{line_name}].base",
                        code="LINE_BASE_MISMATCH",
                        message=f"Line '{line_name}' base variance of {l_diff} exceeds tolerance {tol} (reconstructed: {l_erp}, document: {l_doc}).",
                        document_amount=l_doc,
                        reconstructed_amount=l_erp,
                        difference=l_diff,
                        source_line_id=ln.source_line_id,
                        document_evidence_ids=ln.evidence_ids,
                    )
                )

        l_comp = ReconciliationComparison(
            component_name=line_name,
            status=l_status,
            classification=l_class,
            document_value=l_doc,
            document_origin=ln.origin,
            reconstructed_value=l_erp,
            reconstructed_origin=FactOrigin.DERIVED,
            difference=l_diff,
            tolerance_applied=tol,
            currency=currency,
            document_evidence_ids=ln.evidence_ids,
            erp_provenance=dict(ln.provenance),
            diagnostic_notes=tuple(l_notes),
        )
        line_comparisons.append(l_comp)
        all_comparisons.append(l_comp)

    # ── 7. Document-Level Status Resolution ────────────────────────────────
    # Priority hierarchy:
    # 1. CONFLICT: Upstream blocking issue or fatal contradiction
    # 2. INCOMPLETE: Critical gross total missing or document missing currency
    # 3. MISMATCH: Discrepancy on gross, subtotal, tax, or lines
    # 4. MATCH_PARTIAL: Gross matches, but subtotal or tax was not printed
    # 5. MATCH: Gross matches and all printed intermediate components match
    doc_status: ReconciliationStatus

    if has_blocking_issue:
        doc_status = ReconciliationStatus.CONFLICT
    elif not currency:
        doc_status = ReconciliationStatus.INCOMPLETE
    elif (
        gross_comp.status == ComponentStatus.MISSING_DOCUMENT_VALUE
        or gross_comp.status == ComponentStatus.MISSING_ERP_VALUE
    ):
        doc_status = ReconciliationStatus.INCOMPLETE
    elif (
        gross_comp.status == ComponentStatus.MISMATCH
        or subtotal_comp.status == ComponentStatus.MISMATCH
        or tax_comp.status == ComponentStatus.MISMATCH
        or any(lc.status == ComponentStatus.MISMATCH for lc in line_comparisons)
    ):
        doc_status = ReconciliationStatus.MISMATCH
    elif gross_comp.status == ComponentStatus.MATCH:
        # Check if document printed intermediate components
        has_unprinted_intermediate = (
            subtotal_comp.status == ComponentStatus.MISSING_DOCUMENT_VALUE
            or tax_comp.status == ComponentStatus.MISSING_DOCUMENT_VALUE
        )
        if has_unprinted_intermediate:
            doc_status = ReconciliationStatus.MATCH_PARTIAL
        else:
            doc_status = ReconciliationStatus.MATCH
    else:
        doc_status = ReconciliationStatus.INCOMPLETE

    # ── 8. Assemble Complete ReconciliationResult ──────────────────────────
    all_evidence: List[str] = list(reconstruction.evidence_ids)
    for c in all_comparisons:
        for ev in c.document_evidence_ids:
            if ev and ev not in all_evidence:
                all_evidence.append(ev)

    prov = {
        "stage": "9C-3_reconciliation",
        "tolerance": str(tol),
        "source_assembly_id": reconstruction.assembly_id,
        "source_document_id": reconstruction.document_id,
    }

    return ReconciliationResult(
        assembly_id=reconstruction.assembly_id,
        document_id=reconstruction.document_id,
        currency=currency,
        document_type=reconstruction.document_type,
        status=doc_status,
        gross_comparison=gross_comp,
        subtotal_comparison=subtotal_comp,
        tax_comparison=tax_comp,
        line_comparisons=tuple(line_comparisons),
        all_comparisons=tuple(all_comparisons),
        discrepancies=tuple(discrepancies),
        tolerance_used=tol,
        supporting_group_ids=reconstruction.supporting_group_ids,
        document_evidence_ids=tuple(all_evidence),
        provenance=prov,
    )


def reconcile_batch(
    reconstructions: Sequence[ERPReconstruction],
    tolerance: Union[Decimal, str, float, int] = Decimal("0.00"),
) -> List[ReconciliationResult]:
    """Batch reconcile multiple ERPReconstruction objects.

    Payables remain strictly isolated; no cross-document financial aggregation.
    """
    return [reconcile(r, tolerance=tolerance) for r in reconstructions]
