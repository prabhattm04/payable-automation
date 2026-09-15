"""src/extraction/validation.py — Extraction Validation & Structural Gating Layer.

Phase 9B-4: Validates the structural and semantic integrity of assembled
FinancialDocumentAssembly objects before downstream 9C normalization.

Core Architectural Principles:
1. Validate, Don't Repair:
   If an extraction is incomplete, contradictory, ambiguous, or structurally unsafe,
   flag it, preserve the evidence, explain the reason, and do not mutate or invent values.
2. Zero Accounting Arithmetic:
   No quantity * unit_price, no sum(lines), no tax_rate * base, no subtotal + tax = gross,
   and no derivation of missing amounts. Decimal monetary representation is supported
   for comparison only; no new accounting calculations are performed.
3. Zero Master Matching:
   No master-data matchers called and no master IDs injected. Unmatched master data
   is never an extraction validation failure.
4. Structural Metadata Only:
   Semantic roles and tax placements are evaluated against upstream enum assignments
   (SemanticRole, Placement), NEVER raw text heuristics or description string searches.
5. Evidence & Provenance Preservation:
   Observed facts must retain traceable evidence IDs. Missing evidence is flagged.
6. Supporting Document Isolation:
   Supporting documents (e.g. customs sheets in DU-02) remain isolated in supporting_group_ids;
   different currencies or totals in supporting groups never contaminate the primary payable.
7. Conservative Status Semantics:
   - VALID: Structurally coherent, safe to proceed downstream.
   - WARNING: Usable with non-blocking uncertainty or incompleteness.
   - INVALID: Contains material structural inconsistencies or contradictions.
   - BLOCKED: Empty or structurally unsafe representation that cannot proceed without review.
8. 100% Deterministic, Offline, and Non-Destructive:
   Input assemblies are treated as read-only. No OCR, Vision, Qwen, network, or ERP calls.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.extraction.financial_assembly import FinancialDocumentAssembly
from src.understanding.document_facts import (
    FactOrigin,
    InvoiceType,
    LineFact,
    Placement,
    SemanticRole,
    TaxFact,
)
from src.understanding.page_classifier import PageRole, PayableRelevance
from src.utils.logging import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Enums & Value Models
# ══════════════════════════════════════════════════════════════════════════

class ValidationSeverity(str, Enum):
    """Severity classification of a validation issue."""
    INFO = "INFO"          # Informational finding; fully valid
    WARNING = "WARNING"    # Non-blocking uncertainty or optional field missing
    ERROR = "ERROR"        # Material structural inconsistency or contradiction
    BLOCKING = "BLOCKING"  # Critical structural failure preventing downstream processing


class ValidationStatus(str, Enum):
    """Overall validation status of a FinancialDocumentAssembly."""
    VALID = "VALID"        # Structurally coherent and safe to proceed downstream
    WARNING = "WARNING"    # Usable with non-blocking uncertainty
    INVALID = "INVALID"    # Material structural issues detected; inspectable
    BLOCKED = "BLOCKED"    # Cannot proceed downstream without human review or evidence


# ══════════════════════════════════════════════════════════════════════════
# Validation Issue & Result Models
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ValidationIssue:
    """Individual structural or semantic validation finding."""
    code: str
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    evidence_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    group_ids: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        sev = (
            self.severity
            if isinstance(self.severity, ValidationSeverity)
            else ValidationSeverity(str(self.severity).upper())
        )
        ev_tuple = _normalize_tuple_str(self.evidence_ids)
        grp_tuple = _normalize_tuple_str(self.group_ids)
        object.__setattr__(self, "severity", sev)
        object.__setattr__(self, "evidence_ids", ev_tuple)
        object.__setattr__(self, "group_ids", grp_tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field,
            "evidence_ids": list(self.evidence_ids),
            "group_ids": list(self.group_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationIssue:
        return cls(
            code=data["code"],
            severity=ValidationSeverity(data.get("severity", "WARNING")),
            message=data["message"],
            field=data.get("field"),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            group_ids=_normalize_tuple_str(data.get("group_ids")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ExtractionValidationResult:
    """Immutable validation result associated with a FinancialDocumentAssembly."""
    assembly_id: str
    status: ValidationStatus
    issues: Tuple[ValidationIssue, ...] = dataclasses.field(default_factory=tuple)
    validated_fact_count: int = 0
    validated_evidence_count: int = 0
    validation_provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        st = (
            self.status
            if isinstance(self.status, ValidationStatus)
            else ValidationStatus(str(self.status).upper())
        )
        object.__setattr__(self, "status", st)
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def is_valid(self) -> bool:
        """True if validation status is VALID."""
        return self.status == ValidationStatus.VALID

    @property
    def is_usable(self) -> bool:
        """True if validation status is VALID or WARNING (usable downstream)."""
        return self.status in (ValidationStatus.VALID, ValidationStatus.WARNING)

    @property
    def has_blocking(self) -> bool:
        """True if validation status is BLOCKED."""
        return self.status == ValidationStatus.BLOCKED

    @property
    def has_errors(self) -> bool:
        """True if any ERROR or BLOCKING issues are present."""
        return any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.BLOCKING) for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """True if any WARNING issues are present."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)

    def get_issues_by_severity(self, severity: ValidationSeverity) -> Tuple[ValidationIssue, ...]:
        """Filter issues by severity."""
        sev = severity if isinstance(severity, ValidationSeverity) else ValidationSeverity(str(severity).upper())
        return tuple(i for i in self.issues if i.severity == sev)

    def get_issues_by_field(self, field_name: str) -> Tuple[ValidationIssue, ...]:
        """Filter issues by field name."""
        return tuple(i for i in self.issues if i.field == field_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assembly_id": self.assembly_id,
            "status": self.status.value,
            "issues": [i.to_dict() for i in self.issues],
            "validated_fact_count": self.validated_fact_count,
            "validated_evidence_count": self.validated_evidence_count,
            "validation_provenance": dict(self.validation_provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractionValidationResult:
        raw_issues = data.get("issues") or []
        parsed_issues = tuple(
            i if isinstance(i, ValidationIssue) else ValidationIssue.from_dict(i)
            for i in raw_issues
        )
        return cls(
            assembly_id=data["assembly_id"],
            status=ValidationStatus(data.get("status", "VALID")),
            issues=parsed_issues,
            validated_fact_count=int(data.get("validated_fact_count", 0)),
            validated_evidence_count=int(data.get("validated_evidence_count", 0)),
            validation_provenance=dict(data.get("validation_provenance") or {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> ExtractionValidationResult:
        return cls.from_dict(json.loads(json_str))


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


def _sort_issues_stable(issues: Sequence[ValidationIssue]) -> Tuple[ValidationIssue, ...]:
    """Sort validation issues deterministically."""
    def key_func(issue: ValidationIssue) -> Tuple[str, str, str, str, str, str]:
        grp = issue.group_ids[0] if issue.group_ids else ""
        fld = issue.field or ""
        sev_order = {"BLOCKING": "1", "ERROR": "2", "WARNING": "3", "INFO": "4"}.get(issue.severity.value, "9")
        evs = "".join(issue.evidence_ids)
        return (grp, fld, sev_order, issue.code, issue.message, evs)

    return tuple(sorted(issues, key=key_func))


def _resolve_status(issues: Sequence[ValidationIssue]) -> ValidationStatus:
    """Resolve overall ValidationStatus based on highest issue severity."""
    has_blocking = any(i.severity == ValidationSeverity.BLOCKING for i in issues)
    if has_blocking:
        return ValidationStatus.BLOCKED
    has_error = any(i.severity == ValidationSeverity.ERROR for i in issues)
    if has_error:
        return ValidationStatus.INVALID
    has_warning = any(i.severity == ValidationSeverity.WARNING for i in issues)
    if has_warning:
        return ValidationStatus.WARNING
    return ValidationStatus.VALID


# ══════════════════════════════════════════════════════════════════════════
# Domain-Specific Validation Logic
# ══════════════════════════════════════════════════════════════════════════

def _validate_identity_and_relevance(
    assembly: FinancialDocumentAssembly,
    issues: List[ValidationIssue],
) -> None:
    """Validate document identity and declared payable relevance."""
    primary_groups = assembly.primary_group_ids

    # 1. Relevance classification check
    if assembly.payable_relevance == PayableRelevance.SUPPORTING:
        issues.append(ValidationIssue(
            code="RELEVANCE_SUPPORTING_DOCUMENT",
            severity=ValidationSeverity.INFO,
            message="Assembly represents a supporting document (safely isolated from primary payables).",
            field="payable_relevance",
            group_ids=primary_groups,
            evidence_ids=assembly.evidence_ids,
        ))
        return

    if assembly.payable_relevance == PayableRelevance.NON_PAYABLE:
        issues.append(ValidationIssue(
            code="RELEVANCE_NON_PAYABLE",
            severity=ValidationSeverity.INFO,
            message="Assembly represents a non-payable document.",
            field="payable_relevance",
            group_ids=primary_groups,
            evidence_ids=assembly.evidence_ids,
        ))
        return

    if assembly.payable_relevance == PayableRelevance.AMBIGUOUS:
        issues.append(ValidationIssue(
            code="RELEVANCE_AMBIGUOUS",
            severity=ValidationSeverity.WARNING,
            message="Assembly has ambiguous payable relevance.",
            field="payable_relevance",
            group_ids=primary_groups,
            evidence_ids=assembly.evidence_ids,
        ))

    # 2. For PAYABLE_CANDIDATE: Check if assembly has any meaningful identity or financial evidence
    has_lines = bool(assembly.lines)
    has_totals = assembly.printed_totals is not None and (
        assembly.printed_totals.gross_total is not None
        or assembly.printed_totals.amount_due is not None
        or assembly.printed_totals.net is not None
    )
    has_inv_num = bool(assembly.invoice_number)
    has_supplier = bool(assembly.supplier.observed_name or assembly.supplier.vat_id)
    has_evidence = bool(assembly.evidence_ids)

    if not has_lines and not has_totals and not has_inv_num and not has_supplier and not has_evidence:
        issues.append(ValidationIssue(
            code="IDENTITY_NO_EVIDENCE",
            severity=ValidationSeverity.BLOCKING,
            message="Payable candidate assembly has no financial evidence, identity, lines, or totals.",
            field="document_id",
            group_ids=primary_groups,
        ))
        return

    # 3. Invoice Number presence
    if not has_inv_num:
        issues.append(ValidationIssue(
            code="IDENTITY_MISSING_INVOICE_NUMBER",
            severity=ValidationSeverity.WARNING,
            message="Payable candidate has no observed invoice number.",
            field="invoice_number",
            group_ids=primary_groups,
        ))

    # 4. Invoice Date presence (non-blocking informational)
    if not assembly.facts.identity.invoice_date:
        issues.append(ValidationIssue(
            code="IDENTITY_MISSING_DATE",
            severity=ValidationSeverity.INFO,
            message="No invoice date observed.",
            field="invoice_date",
            group_ids=primary_groups,
        ))


def _validate_parties(
    assembly: FinancialDocumentAssembly,
    issues: List[ValidationIssue],
) -> None:
    """Validate observed parties for payable candidate."""
    if assembly.payable_relevance != PayableRelevance.PAYABLE_CANDIDATE:
        return

    primary_groups = assembly.primary_group_ids

    # Supplier validation (master-match is deferred; check observed presence only)
    has_supplier = bool(assembly.supplier.observed_name or assembly.supplier.vat_id)
    if not has_supplier:
        issues.append(ValidationIssue(
            code="PARTY_SUPPLIER_MISSING",
            severity=ValidationSeverity.WARNING,
            message="No supplier name or VAT ID observed for payable candidate.",
            field="supplier",
            group_ids=primary_groups,
        ))

    # Buyer validation (informational)
    has_buyer = bool(
        assembly.buyer.observed_company
        or assembly.buyer.company_code
        or assembly.buyer.business_unit_code
    )
    if not has_buyer:
        issues.append(ValidationIssue(
            code="PARTY_BUYER_MISSING",
            severity=ValidationSeverity.INFO,
            message="No buyer organization observed on document.",
            field="buyer",
            group_ids=primary_groups,
        ))


def _validate_currency(
    assembly: FinancialDocumentAssembly,
    issues: List[ValidationIssue],
) -> None:
    """Validate currency for payable candidate."""
    if assembly.payable_relevance != PayableRelevance.PAYABLE_CANDIDATE:
        return

    primary_groups = assembly.primary_group_ids
    if not assembly.currency:
        issues.append(ValidationIssue(
            code="CURRENCY_MISSING",
            severity=ValidationSeverity.WARNING,
            message="No currency observed for payable candidate.",
            field="currency",
            group_ids=primary_groups,
        ))


def _validate_financial_structure_presence(
    assembly: FinancialDocumentAssembly,
    issues: List[ValidationIssue],
) -> None:
    """Validate presence and structural coherence of financial components."""
    if assembly.payable_relevance != PayableRelevance.PAYABLE_CANDIDATE:
        return

    primary_groups = assembly.primary_group_ids
    has_lines = bool(assembly.lines)
    has_totals = assembly.printed_totals is not None and (
        assembly.printed_totals.gross_total is not None
        or assembly.printed_totals.amount_due is not None
        or assembly.printed_totals.net is not None
        or assembly.printed_totals.subtotal is not None
    )

    if not has_lines and not has_totals:
        # Avoid duplicate BLOCKING issue if already emitted by _validate_identity_and_relevance
        if not any(i.code == "IDENTITY_NO_EVIDENCE" for i in issues):
            issues.append(ValidationIssue(
                code="FINANCIALS_EMPTY",
                severity=ValidationSeverity.BLOCKING,
                message="Payable candidate contains neither billed line items nor printed totals.",
                field="financials",
                group_ids=primary_groups,
            ))
    elif has_totals and not has_lines:
        issues.append(ValidationIssue(
            code="FINANCIALS_NO_LINES_OBSERVED",
            severity=ValidationSeverity.INFO,
            message="Printed totals are observed but no individual billed line items are present.",
            field="lines",
            group_ids=primary_groups,
        ))
    elif has_lines and not has_totals:
        issues.append(ValidationIssue(
            code="FINANCIALS_NO_PRINTED_TOTAL",
            severity=ValidationSeverity.WARNING,
            message="Billed line items are present but no printed document totals were observed.",
            field="printed_totals",
            group_ids=primary_groups,
        ))


def _validate_semantic_roles(
    assembly: FinancialDocumentAssembly,
    issues: List[ValidationIssue],
) -> None:
    """Validate semantic roles using explicit upstream structural metadata only.
    
    CRITICAL: Does NOT inspect raw text descriptions or perform heuristic keyword matching.
    """
    primary_groups = assembly.primary_group_ids

    # 1. Validate lines container: should not contain facts explicitly classified as totals/subtotals
    for idx, line in enumerate(assembly.lines):
        role = line.semantic_role
        if role in (SemanticRole.TOTAL, SemanticRole.SUBTOTAL):
            issues.append(ValidationIssue(
                code="SEMANTIC_ROLE_STRUCTURAL_MISMATCH",
                severity=ValidationSeverity.WARNING,
                message=(
                    f"Line item at index {idx} has explicit semantic role '{role.value}' "
                    f"but is placed inside the billed lines container."
                ),
                field=f"lines[{idx}].semantic_role",
                group_ids=primary_groups,
                evidence_ids=line.evidence_ids,
            ))
        elif role == SemanticRole.UNKNOWN:
            issues.append(ValidationIssue(
                code="SEMANTIC_ROLE_UNKNOWN",
                severity=ValidationSeverity.INFO,
                message=f"Line item at index {idx} has UNKNOWN semantic role.",
                field=f"lines[{idx}].semantic_role",
                group_ids=primary_groups,
                evidence_ids=line.evidence_ids,
            ))


def _validate_tax_placement(
    assembly: FinancialDocumentAssembly,
    issues: List[ValidationIssue],
) -> None:
    """Validate tax placement invariants using Placement metadata only.
    
    Header taxes must have placement HEADER.
    Line taxes must have placement LINE.
    Placement UNKNOWN produces a warning.
    Contradictory placement produces an error.
    Taxes are NEVER moved or normalized.
    """
    primary_groups = assembly.primary_group_ids

    # 1. Header taxes
    for idx, tax in enumerate(assembly.taxes):
        if tax.placement == Placement.LINE:
            issues.append(ValidationIssue(
                code="TAX_PLACEMENT_MISMATCH",
                severity=ValidationSeverity.ERROR,
                message=(
                    f"Tax at header index {idx} ({tax.tax_name or 'unnamed'}) has Placement.LINE "
                    f"but is positioned in header taxes."
                ),
                field=f"taxes[{idx}].placement",
                group_ids=primary_groups,
                evidence_ids=tax.evidence_ids,
            ))
        elif tax.placement == Placement.UNKNOWN:
            issues.append(ValidationIssue(
                code="TAX_PLACEMENT_UNKNOWN",
                severity=ValidationSeverity.WARNING,
                message=f"Tax at header index {idx} ({tax.tax_name or 'unnamed'}) has Placement.UNKNOWN.",
                field=f"taxes[{idx}].placement",
                group_ids=primary_groups,
                evidence_ids=tax.evidence_ids,
            ))

        if tax.rate is None and tax.amount is None:
            issues.append(ValidationIssue(
                code="TAX_EMPTY_FACT",
                severity=ValidationSeverity.WARNING,
                message=f"Tax at header index {idx} ({tax.tax_name or 'unnamed'}) has neither rate nor amount.",
                field=f"taxes[{idx}]",
                group_ids=primary_groups,
                evidence_ids=tax.evidence_ids,
            ))

    # 2. Line-level taxes
    for line_idx, line in enumerate(assembly.lines):
        for tax_idx, tax in enumerate(line.taxes):
            if tax.placement == Placement.HEADER:
                issues.append(ValidationIssue(
                    code="TAX_PLACEMENT_MISMATCH",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Line tax on line index {line_idx} has Placement.HEADER "
                        f"but is positioned at line level."
                    ),
                    field=f"lines[{line_idx}].taxes[{tax_idx}].placement",
                    group_ids=primary_groups,
                    evidence_ids=tax.evidence_ids,
                ))
            elif tax.placement == Placement.UNKNOWN:
                issues.append(ValidationIssue(
                    code="TAX_PLACEMENT_UNKNOWN",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Line tax on line index {line_idx} has Placement.UNKNOWN."
                    ),
                    field=f"lines[{line_idx}].taxes[{tax_idx}].placement",
                    group_ids=primary_groups,
                    evidence_ids=tax.evidence_ids,
                ))

            if tax.rate is None and tax.amount is None:
                issues.append(ValidationIssue(
                    code="TAX_EMPTY_FACT",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Line tax on line index {line_idx} has neither rate nor amount."
                    ),
                    field=f"lines[{line_idx}].taxes[{tax_idx}]",
                    group_ids=primary_groups,
                    evidence_ids=tax.evidence_ids,
                ))


def _validate_provenance_and_traceability(
    assembly: FinancialDocumentAssembly,
    issues: List[ValidationIssue],
) -> None:
    """Validate that financially relevant observed facts maintain evidence traceability."""
    primary_groups = assembly.primary_group_ids

    # 1. Line facts
    for idx, line in enumerate(assembly.lines):
        if line.origin == FactOrigin.OBSERVED:
            has_line_ev = bool(
                line.evidence_ids
                or line.source_row_evidence_ids
                or any(line.field_evidence_ids.values())
            )
            if not has_line_ev:
                issues.append(ValidationIssue(
                    code="PROVENANCE_MISSING_LINE_EVIDENCE",
                    severity=ValidationSeverity.ERROR,
                    message=f"Observed line item at index {idx} has no traceable evidence IDs.",
                    field=f"lines[{idx}]",
                    group_ids=primary_groups,
                ))

    # 2. Header tax facts
    for idx, tax in enumerate(assembly.taxes):
        if tax.origin == FactOrigin.OBSERVED:
            has_tax_ev = bool(tax.evidence_ids or any(tax.field_evidence_ids.values()))
            if not has_tax_ev:
                issues.append(ValidationIssue(
                    code="PROVENANCE_MISSING_TAX_EVIDENCE",
                    severity=ValidationSeverity.ERROR,
                    message=f"Observed header tax at index {idx} has no traceable evidence IDs.",
                    field=f"taxes[{idx}]",
                    group_ids=primary_groups,
                ))

    # 3. Discount facts
    for idx, disc in enumerate(assembly.discounts):
        if disc.origin == FactOrigin.OBSERVED:
            has_disc_ev = bool(disc.evidence_ids or any(disc.field_evidence_ids.values()))
            if not has_disc_ev:
                issues.append(ValidationIssue(
                    code="PROVENANCE_MISSING_DISCOUNT_CHARGE_EVIDENCE",
                    severity=ValidationSeverity.ERROR,
                    message=f"Observed discount at index {idx} has no traceable evidence IDs.",
                    field=f"discounts[{idx}]",
                    group_ids=primary_groups,
                ))

    # 4. Charge facts
    for idx, charge in enumerate(assembly.charges):
        if charge.origin == FactOrigin.OBSERVED:
            has_charge_ev = bool(charge.evidence_ids or any(charge.field_evidence_ids.values()))
            if not has_charge_ev:
                issues.append(ValidationIssue(
                    code="PROVENANCE_MISSING_DISCOUNT_CHARGE_EVIDENCE",
                    severity=ValidationSeverity.ERROR,
                    message=f"Observed charge at index {idx} has no traceable evidence IDs.",
                    field=f"charges[{idx}]",
                    group_ids=primary_groups,
                ))

    # 5. Printed totals fact
    pt = assembly.printed_totals
    if pt is not None and pt.origin == FactOrigin.OBSERVED:
        has_pt_ev = bool(pt.evidence_ids or any(pt.field_evidence_ids.values()))
        if not has_pt_ev:
            issues.append(ValidationIssue(
                code="PROVENANCE_MISSING_TOTALS_EVIDENCE",
                severity=ValidationSeverity.ERROR,
                message="Observed printed totals fact has no traceable evidence IDs.",
                field="printed_totals",
                group_ids=primary_groups,
            ))


def _validate_conflicts(
    assembly: FinancialDocumentAssembly,
    issues: List[ValidationIssue],
) -> None:
    """Inspect and report pre-existing conflicts from assembly and document facts.
    
    Safely consumes conflicts through existing contracts (assembly.conflicts and
    getattr(assembly.facts, 'conflicting_facts', ())) without assuming unverified fields.
    """
    primary_groups = assembly.primary_group_ids

    # Collect conflicts safely from assembly.conflicts and facts
    raw_conflicts: List[Dict[str, Any]] = list(assembly.conflicts)
    facts_conflicts = getattr(assembly.facts, "conflicting_facts", ())
    if facts_conflicts and isinstance(facts_conflicts, (list, tuple)):
        for fc in facts_conflicts:
            if isinstance(fc, dict) and fc not in raw_conflicts:
                raw_conflicts.append(fc)

    for cf in raw_conflicts:
        field_name = str(cf.get("field", "unknown"))
        c_type = str(cf.get("conflict_type", "conflict"))
        observations = cf.get("observations") or []
        
        # Extract evidence IDs from conflict structure safely
        ev_list: List[str] = []
        if isinstance(observations, list):
            for obs in observations:
                if isinstance(obs, dict):
                    ev_list.extend(obs.get("evidence_ids") or [])
        ev_list.extend(cf.get("evidence_ids") or [])
        ev_tuple = _normalize_tuple_str(ev_list)

        # Classify severity and issue code by affected field
        if field_name == "invoice_number":
            issues.append(ValidationIssue(
                code="IDENTITY_CONTRADICTORY_INVOICE_NUMBER",
                severity=ValidationSeverity.ERROR,
                message=f"Contradictory invoice numbers observed: {c_type}.",
                field="invoice_number",
                group_ids=primary_groups,
                evidence_ids=ev_tuple,
                metadata=cf,
            ))
        elif field_name in ("invoice_date", "due_date"):
            issues.append(ValidationIssue(
                code="IDENTITY_CONTRADICTORY_DATE",
                severity=ValidationSeverity.WARNING,
                message=f"Contradictory {field_name} observations detected: {c_type}.",
                field=field_name,
                group_ids=primary_groups,
                evidence_ids=ev_tuple,
                metadata=cf,
            ))
        elif field_name == "invoice_type":
            issues.append(ValidationIssue(
                code="IDENTITY_CONTRADICTORY_INVOICE_TYPE",
                severity=ValidationSeverity.ERROR,
                message=f"Contradictory invoice types observed: {c_type}.",
                field="invoice_type",
                group_ids=primary_groups,
                evidence_ids=ev_tuple,
                metadata=cf,
            ))
        elif field_name == "currency":
            issues.append(ValidationIssue(
                code="CURRENCY_CONFLICT",
                severity=ValidationSeverity.ERROR,
                message=f"Contradictory currency observations detected in primary assembly: {c_type}.",
                field="currency",
                group_ids=primary_groups,
                evidence_ids=ev_tuple,
                metadata=cf,
            ))
        elif field_name in ("gross_total", "amount_due", "net", "subtotal", "tax_total"):
            issues.append(ValidationIssue(
                code="FINANCIALS_CONTRADICTORY_TOTAL",
                severity=ValidationSeverity.ERROR,
                message=f"Contradictory {field_name} totals observed: {c_type}.",
                field=field_name,
                group_ids=primary_groups,
                evidence_ids=ev_tuple,
                metadata=cf,
            ))
        elif field_name == "placement":
            issues.append(ValidationIssue(
                code="TAX_PLACEMENT_CONFLICT",
                severity=ValidationSeverity.ERROR,
                message=f"Contradictory tax placement observations detected: {c_type}.",
                field="placement",
                group_ids=primary_groups,
                evidence_ids=ev_tuple,
                metadata=cf,
            ))
        elif "supplier" in field_name:
            issues.append(ValidationIssue(
                code="PARTY_SUPPLIER_CONFLICT",
                severity=ValidationSeverity.WARNING,
                message=f"Contradictory supplier identity observations detected: {c_type}.",
                field=field_name,
                group_ids=primary_groups,
                evidence_ids=ev_tuple,
                metadata=cf,
            ))
        elif "buyer" in field_name:
            issues.append(ValidationIssue(
                code="PARTY_BUYER_CONFLICT",
                severity=ValidationSeverity.WARNING,
                message=f"Contradictory buyer identity observations detected: {c_type}.",
                field=field_name,
                group_ids=primary_groups,
                evidence_ids=ev_tuple,
                metadata=cf,
            ))
        else:
            issues.append(ValidationIssue(
                code="CONFLICT_UNRESOLVED",
                severity=ValidationSeverity.WARNING,
                message=f"Unresolved extraction conflict on field '{field_name}': {c_type}.",
                field=field_name,
                group_ids=primary_groups,
                evidence_ids=ev_tuple,
                metadata=cf,
            ))


def _count_facts_and_evidence(
    assembly: FinancialDocumentAssembly,
) -> Tuple[int, int]:
    """Deterministically count validated facts and unique evidence IDs."""
    fact_count = 0
    unique_evidence: Set[str] = set()

    # Identity facts
    if assembly.facts.identity.invoice_number or assembly.facts.identity.invoice_date:
        fact_count += 1
    unique_evidence.update(assembly.facts.identity.evidence_ids)
    for fe in assembly.facts.identity.field_evidence_ids.values():
        unique_evidence.update(fe)

    # Parties
    if assembly.supplier.observed_name or assembly.supplier.vat_id:
        fact_count += 1
    unique_evidence.update(assembly.supplier.evidence_ids)
    for fe in assembly.supplier.field_evidence_ids.values():
        unique_evidence.update(fe)

    if assembly.buyer.observed_company or assembly.buyer.company_code:
        fact_count += 1
    unique_evidence.update(assembly.buyer.evidence_ids)
    for fe in assembly.buyer.field_evidence_ids.values():
        unique_evidence.update(fe)

    # PO
    if assembly.po.observed_po_number:
        fact_count += 1
    unique_evidence.update(assembly.po.evidence_ids)
    for fe in assembly.po.field_evidence_ids.values():
        unique_evidence.update(fe)

    # Lines
    fact_count += len(assembly.lines)
    for ln in assembly.lines:
        unique_evidence.update(ln.evidence_ids)
        unique_evidence.update(ln.source_row_evidence_ids)
        for fe in ln.field_evidence_ids.values():
            unique_evidence.update(fe)
        for tx in ln.taxes:
            fact_count += 1
            unique_evidence.update(tx.evidence_ids)
            for fe in tx.field_evidence_ids.values():
                unique_evidence.update(fe)

    # Header taxes
    fact_count += len(assembly.taxes)
    for tx in assembly.taxes:
        unique_evidence.update(tx.evidence_ids)
        for fe in tx.field_evidence_ids.values():
            unique_evidence.update(fe)

    # Discounts
    fact_count += len(assembly.discounts)
    for d in assembly.discounts:
        unique_evidence.update(d.evidence_ids)
        for fe in d.field_evidence_ids.values():
            unique_evidence.update(fe)

    # Charges
    fact_count += len(assembly.charges)
    for c in assembly.charges:
        unique_evidence.update(c.evidence_ids)
        for fe in c.field_evidence_ids.values():
            unique_evidence.update(fe)

    # Printed totals
    if assembly.printed_totals is not None:
        fact_count += 1
        unique_evidence.update(assembly.printed_totals.evidence_ids)
        for fe in assembly.printed_totals.field_evidence_ids.values():
            unique_evidence.update(fe)

    # Assembly-level evidence
    unique_evidence.update(assembly.evidence_ids)

    # Clean empty strings
    unique_evidence.discard("")

    return fact_count, len(unique_evidence)


# ══════════════════════════════════════════════════════════════════════════
# Public Validation API
# ══════════════════════════════════════════════════════════════════════════

def validate_financial_document_assembly(
    assembly: FinancialDocumentAssembly,
) -> ExtractionValidationResult:
    """Validate the structural and semantic integrity of a FinancialDocumentAssembly.
    
    Non-destructive: The input assembly is strictly unmodified.
    Deterministic: Issue ordering and status resolution are permuted-invariant.
    Offline & Isolated: No OCR, Vision, Qwen, master matching, network, or ERP arithmetic.
    """
    if not isinstance(assembly, FinancialDocumentAssembly):
        raise TypeError(f"Expected FinancialDocumentAssembly, got {type(assembly).__name__}")

    issues: List[ValidationIssue] = []

    # 1. Identity & Relevance
    _validate_identity_and_relevance(assembly, issues)

    # 2. Parties
    _validate_parties(assembly, issues)

    # 3. Currency (with supporting isolation)
    _validate_currency(assembly, issues)

    # 4. Financial Structure Presence
    _validate_financial_structure_presence(assembly, issues)

    # 5. Semantic Roles (structural metadata only, no text heuristics)
    _validate_semantic_roles(assembly, issues)

    # 6. Tax Placement Invariants
    _validate_tax_placement(assembly, issues)

    # 7. Provenance & Traceability
    _validate_provenance_and_traceability(assembly, issues)

    # 8. Pre-existing Conflicts
    _validate_conflicts(assembly, issues)

    # Deterministic sorting & resolution
    sorted_issues = _sort_issues_stable(issues)
    status = _resolve_status(sorted_issues)

    # Fact and evidence counting
    fact_count, evidence_count = _count_facts_and_evidence(assembly)

    # Provenance metadata
    sev_counts = {
        ValidationSeverity.BLOCKING.value: 0,
        ValidationSeverity.ERROR.value: 0,
        ValidationSeverity.WARNING.value: 0,
        ValidationSeverity.INFO.value: 0,
    }
    for iss in sorted_issues:
        sev_counts[iss.severity.value] = sev_counts.get(iss.severity.value, 0) + 1

    provenance = {
        "validator": "DeterministicExtractionValidator",
        "version": "Phase9B-4-1.0",
        "issue_count": len(sorted_issues),
        "issue_counts_by_severity": sev_counts,
        "has_blocking": status == ValidationStatus.BLOCKED,
        "has_errors": any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.BLOCKING) for i in sorted_issues),
        "has_warnings": any(i.severity == ValidationSeverity.WARNING for i in sorted_issues),
    }

    return ExtractionValidationResult(
        assembly_id=assembly.assembly_id,
        status=status,
        issues=sorted_issues,
        validated_fact_count=fact_count,
        validated_evidence_count=evidence_count,
        validation_provenance=provenance,
    )


def validate_financial_document_assemblies(
    assemblies: Sequence[FinancialDocumentAssembly],
) -> List[ExtractionValidationResult]:
    """Batch-validate a sequence of FinancialDocumentAssembly objects independently."""
    return [validate_financial_document_assembly(a) for a in assemblies]
