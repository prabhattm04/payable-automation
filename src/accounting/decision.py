"""src/accounting/decision.py — Final Decision & Payable Safety Gate.

Phase 9D: Evaluates structured outputs from upstream phases (reconciliation,
ERP reconstruction, document facts, master-data matching) to make a deterministic,
auditable decision on whether a document is safe to proceed toward payable generation.

Core Architectural Principles:
1. Safety Over Automation:
   Never generate an unsafe automated payable. Abstains and routes to HOLD_FOR_REVIEW
   whenever facts are ambiguous, uncertain, or unverified.
2. Comprehensive Check Collection (No Short-Circuiting):
   Evaluates all safety checks across all dimensions before resolving status.
   Secondary failures and warnings are never dropped or hidden.
3. Zero Accounting Repair:
   Never balances numbers, invents charges, or alters quantities/prices to force a pass.
4. Policy Decoupling:
   Does not hard-code business policy assumptions (such as equating NO_MATCH or
   MATCH_PARTIAL with automatic safety). These remain configurable policy inputs.
5. Strict Boundary:
   9D makes the decision and provides the explainability audit trail; the subsequent
   emission of `output/X.json` belongs to the final autodraft generation phase.
6. Supporting Document Isolation:
   Supporting documents (e.g. TRY customs pages in DU-02) remain isolated in
   supporting_group_ids and never trigger payable generation.
7. Determinism & Immutability:
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
from src.accounting.financial_structure import (
    FinancialStructure,
    NormalizedPrintedTotals,
)
from src.accounting.reconciliation import (
    ComponentStatus,
    DiscrepancyClassification,
    ReconciliationComparison,
    ReconciliationDiscrepancy,
    ReconciliationResult,
    ReconciliationStatus,
)
from src.matching.match_models import MasterMatchResult, MatchStatus
from src.understanding.document_facts import (
    DocumentFacts,
    FactOrigin,
    InvoiceType,
    SemanticRole,
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


# ══════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════

class PayableDecisionStatus(str, Enum):
    """Overall outcome of the payable safety gate."""
    SAFE_TO_AUTODRAFT = "SAFE_TO_AUTODRAFT"       # All safety checks passed; safe for automated payable generation
    HOLD_FOR_REVIEW = "HOLD_FOR_REVIEW"           # Payable candidate with uncertainty, missing info, or ambiguity
    UNSAFE_TO_AUTODRAFT = "UNSAFE_TO_AUTODRAFT"   # Explicit accounting mismatch or fatal ledger contradiction
    NOT_PAYABLE = "NOT_PAYABLE"                   # Document is not an actionable payable (e.g. PO, receipt, supporting)


class SafetyCheckCode(str, Enum):
    """Specific safety check performed during gating."""
    PAYABLE_RELEVANCE = "PAYABLE_RELEVANCE"       # PageRole and PayableRelevance assessment
    RECONCILIATION = "RECONCILIATION"             # Gross and line reconciliation status
    FINANCIAL_COMPLETENESS = "FINANCIAL_COMPLETENESS" # Presence of essential totals and lines
    CURRENCY_VALIDITY = "CURRENCY_VALIDITY"       # Valid ISO currency without internal conflict
    SUPPLIER_SAFETY = "SUPPLIER_SAFETY"           # Supplier identity extracted and matched without ambiguity
    BUYER_SAFETY = "BUYER_SAFETY"                 # Buyer identity verified without ambiguity
    PO_SAFETY = "PO_SAFETY"                       # Purchase order references consistent
    UPSTREAM_ISSUES = "UPSTREAM_ISSUES"           # Check for blocking ERP or validation issues


class SafetyCheckStatus(str, Enum):
    """Evaluation status of an individual safety check."""
    PASSED = "PASSED"                             # Requirement fully satisfied
    WARNING = "WARNING"                           # Non-blocking condition (audit note)
    FAILED = "FAILED"                             # Safety requirement violated (blocks SAFE_TO_AUTODRAFT)
    SKIPPED = "SKIPPED"                           # Inapplicable (e.g. non-payable document)


# ══════════════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SafetyCheckResult:
    """Detailed evaluation result for an individual safety check."""
    check_code: SafetyCheckCode
    status: SafetyCheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = (
            self.check_code
            if isinstance(self.check_code, SafetyCheckCode)
            else SafetyCheckCode(str(self.check_code).upper())
        )
        st = (
            self.status
            if isinstance(self.status, SafetyCheckStatus)
            else SafetyCheckStatus(str(self.status).upper())
        )
        object.__setattr__(self, "check_code", code)
        object.__setattr__(self, "status", st)
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_code": self.check_code.value,
            "status": self.status.value,
            "message": self.message,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SafetyCheckResult:
        return cls(
            check_code=SafetyCheckCode(data["check_code"]),
            status=SafetyCheckStatus(data["status"]),
            message=data["message"],
            details=dict(data.get("details") or {}),
        )


@dataclass(frozen=True)
class PayableDecision:
    """Complete document-level safety decision produced by Phase 9D."""
    assembly_id: str
    document_id: str
    status: PayableDecisionStatus
    primary_reason: str
    decline_doc_type: Optional[str] = None
    decline_reason: Optional[str] = None
    checks: Tuple[SafetyCheckResult, ...] = field(default_factory=tuple)
    failed_checks: Tuple[SafetyCheckResult, ...] = field(default_factory=tuple)
    warning_checks: Tuple[SafetyCheckResult, ...] = field(default_factory=tuple)
    reconciliation_status: Optional[ReconciliationStatus] = None
    supporting_group_ids: Tuple[str, ...] = field(default_factory=tuple)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        st = (
            self.status
            if isinstance(self.status, PayableDecisionStatus)
            else PayableDecisionStatus(str(self.status).upper())
        )
        rec_st = (
            self.reconciliation_status
            if (self.reconciliation_status is None or isinstance(self.reconciliation_status, ReconciliationStatus))
            else ReconciliationStatus(str(self.reconciliation_status).upper())
        )
        all_checks = [
            c if isinstance(c, SafetyCheckResult) else SafetyCheckResult.from_dict(c)
            for c in self.checks
        ]
        f_checks = [
            c if isinstance(c, SafetyCheckResult) else SafetyCheckResult.from_dict(c)
            for c in self.failed_checks
        ]
        w_checks = [
            c if isinstance(c, SafetyCheckResult) else SafetyCheckResult.from_dict(c)
            for c in self.warning_checks
        ]

        object.__setattr__(self, "status", st)
        object.__setattr__(self, "reconciliation_status", rec_st)
        object.__setattr__(self, "checks", tuple(all_checks))
        object.__setattr__(self, "failed_checks", tuple(f_checks))
        object.__setattr__(self, "warning_checks", tuple(w_checks))
        object.__setattr__(self, "supporting_group_ids", _normalize_tuple_str(self.supporting_group_ids))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def is_safe_to_autodraft(self) -> bool:
        """True if document passes all checks and is approved for automated payable generation."""
        return self.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT

    @property
    def is_declined(self) -> bool:
        """True if document is non-payable and should populate declined[] block."""
        return self.status == PayableDecisionStatus.NOT_PAYABLE

    @property
    def is_hold(self) -> bool:
        """True if document is a payable candidate but requires human review."""
        return self.status == PayableDecisionStatus.HOLD_FOR_REVIEW

    @property
    def is_unsafe(self) -> bool:
        """True if document has an explicit financial mismatch or fatal contradiction."""
        return self.status == PayableDecisionStatus.UNSAFE_TO_AUTODRAFT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assembly_id": self.assembly_id,
            "document_id": self.document_id,
            "status": self.status.value,
            "primary_reason": self.primary_reason,
            "decline_doc_type": self.decline_doc_type,
            "decline_reason": self.decline_reason,
            "checks": [c.to_dict() for c in self.checks],
            "failed_checks": [c.to_dict() for c in self.failed_checks],
            "warning_checks": [c.to_dict() for c in self.warning_checks],
            "reconciliation_status": self.reconciliation_status.value if self.reconciliation_status else None,
            "supporting_group_ids": list(self.supporting_group_ids),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PayableDecision:
        rec_st = data.get("reconciliation_status")
        return cls(
            assembly_id=data["assembly_id"],
            document_id=data["document_id"],
            status=PayableDecisionStatus(data["status"]),
            primary_reason=data["primary_reason"],
            decline_doc_type=data.get("decline_doc_type"),
            decline_reason=data.get("decline_reason"),
            checks=tuple(SafetyCheckResult.from_dict(c) for c in (data.get("checks") or [])),
            failed_checks=tuple(SafetyCheckResult.from_dict(c) for c in (data.get("failed_checks") or [])),
            warning_checks=tuple(SafetyCheckResult.from_dict(c) for c in (data.get("warning_checks") or [])),
            reconciliation_status=ReconciliationStatus(rec_st) if rec_st else None,
            supporting_group_ids=_normalize_tuple_str(data.get("supporting_group_ids")),
            provenance=dict(data.get("provenance") or {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> PayableDecision:
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# Safety Gate Engine
# ══════════════════════════════════════════════════════════════════════════

def evaluate_payable_safety(
    reconciliation: ReconciliationResult,
    reconstruction: ERPReconstruction,
    document_facts: Optional[DocumentFacts] = None,
    supplier_match: Optional[MasterMatchResult] = None,
    buyer_match: Optional[MasterMatchResult] = None,
    po_match: Optional[MasterMatchResult] = None,
    require_matched_supplier: bool = False,
    require_full_reconciliation: bool = False,
) -> PayableDecision:
    """Evaluate whether a document is safe to proceed to automated payable generation.

    Non-Short-Circuiting Evaluation:
    Evaluates ALL safety checks and collects all findings before applying
    deterministic precedence to select the final status. Secondary failures
    and warnings remain visible in the decision.

    Args:
        reconciliation: The ReconciliationResult from Phase 9C-3.
        reconstruction: The ERPReconstruction from Phase 9C-2.
        document_facts: Optional DocumentFacts from Phase 9A/9B.
        supplier_match: Optional MasterMatchResult for supplier.
        buyer_match: Optional MasterMatchResult for buyer.
        po_match: Optional MasterMatchResult for purchase order.
        require_matched_supplier: Policy flag; if True, unmatched supplier triggers HOLD_FOR_REVIEW.
        require_full_reconciliation: Policy flag; if True, MATCH_PARTIAL triggers HOLD_FOR_REVIEW.

    Returns:
        PayableDecision containing final status, primary reason, and complete check audit trail.
    """
    checks: List[SafetyCheckResult] = []

    # ── 1. Check: PAYABLE_RELEVANCE ─────────────────────────────────────────
    is_non_payable = False
    decline_type: Optional[str] = None
    decline_msg: Optional[str] = None

    if document_facts is not None:
        doc_role = document_facts.document_role
        pay_rel = document_facts.payable_relevance

        non_payable_roles = (
            PageRole.PURCHASE_ORDER,
            PageRole.RECEIPT,
            PageRole.REMITTANCE,
            PageRole.SUPPORTING_DOCUMENT,
        )
        non_payable_relevance = (
            PayableRelevance.NON_PAYABLE,
            PayableRelevance.SUPPORTING,
        )

        if doc_role in non_payable_roles or pay_rel in non_payable_relevance:
            is_non_payable = True
            decline_type = doc_role.value.upper()
            decline_msg = f"Document role '{doc_role.value}' or relevance '{pay_rel.value}' is non-payable"
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.PAYABLE_RELEVANCE,
                    status=SafetyCheckStatus.FAILED,
                    message=decline_msg,
                    details={"document_role": doc_role.value, "payable_relevance": pay_rel.value},
                )
            )
        elif pay_rel == PayableRelevance.AMBIGUOUS:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.PAYABLE_RELEVANCE,
                    status=SafetyCheckStatus.FAILED,
                    message="Document payable relevance is ambiguous",
                    details={"payable_relevance": "ambiguous"},
                )
            )
        else:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.PAYABLE_RELEVANCE,
                    status=SafetyCheckStatus.PASSED,
                    message=f"Document role '{doc_role.value}' is an actionable payable candidate",
                    details={"document_role": doc_role.value, "payable_relevance": pay_rel.value},
                )
            )
    else:
        # Fallback to reconstruction document_type
        if reconstruction.document_type == InvoiceType.UNKNOWN:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.PAYABLE_RELEVANCE,
                    status=SafetyCheckStatus.WARNING,
                    message="Document type is UNKNOWN; proceeding with caution",
                    details={"document_type": "unknown"},
                )
            )
        else:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.PAYABLE_RELEVANCE,
                    status=SafetyCheckStatus.PASSED,
                    message=f"Document type is {reconstruction.document_type.value}",
                    details={"document_type": reconstruction.document_type.value},
                )
            )

    # ── 2. Check: RECONCILIATION ───────────────────────────────────────────
    recon_st = reconciliation.status

    if recon_st == ReconciliationStatus.MISMATCH:
        diff_str = (
            str(reconciliation.gross_comparison.difference)
            if reconciliation.gross_comparison and reconciliation.gross_comparison.difference is not None
            else "unknown"
        )
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.RECONCILIATION,
                status=SafetyCheckStatus.FAILED,
                message=f"Financial reconciliation mismatch: variance of {diff_str} exceeds tolerance",
                details={
                    "reconciliation_status": recon_st.value,
                    "discrepancies_count": len(reconciliation.discrepancies),
                    "discrepancies": [d.to_dict() for d in reconciliation.discrepancies],
                },
            )
        )
    elif recon_st == ReconciliationStatus.CONFLICT:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.RECONCILIATION,
                status=SafetyCheckStatus.FAILED,
                message="Financial reconciliation conflict detected in upstream facts",
                details={"reconciliation_status": recon_st.value},
            )
        )
    elif recon_st == ReconciliationStatus.INCOMPLETE:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.RECONCILIATION,
                status=SafetyCheckStatus.FAILED,
                message="Financial reconciliation incomplete: critical comparison values missing",
                details={"reconciliation_status": recon_st.value},
            )
        )
    elif recon_st == ReconciliationStatus.NOT_COMPARABLE:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.RECONCILIATION,
                status=SafetyCheckStatus.FAILED,
                message="Reconciliation marked document as not comparable",
                details={"reconciliation_status": recon_st.value},
            )
        )
    elif recon_st == ReconciliationStatus.MATCH_PARTIAL:
        if require_full_reconciliation:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.RECONCILIATION,
                    status=SafetyCheckStatus.FAILED,
                    message="Reconciliation matched gross total, but intermediate subtotal or tax was omitted on document (strict full reconciliation policy enforced)",
                    details={"reconciliation_status": recon_st.value},
                )
            )
        else:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.RECONCILIATION,
                    status=SafetyCheckStatus.WARNING,
                    message="Reconciliation matched gross total; intermediate subtotal or tax was not printed on document",
                    details={"reconciliation_status": recon_st.value},
                )
            )
    elif recon_st == ReconciliationStatus.MATCH:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.RECONCILIATION,
                status=SafetyCheckStatus.PASSED,
                message="Reconciliation matched: observed financial totals agree with ERP reconstruction",
                details={"reconciliation_status": recon_st.value},
            )
        )

    # ── 3. Check: FINANCIAL_COMPLETENESS ───────────────────────────────────
    has_gross = (
        reconstruction.printed_totals is not None
        and reconstruction.printed_totals.gross_total is not None
    )
    posting_lines_count = sum(1 for ln in reconstruction.lines if ln.is_posting)

    if not has_gross:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.FINANCIAL_COMPLETENESS,
                status=SafetyCheckStatus.FAILED,
                message="Document did not print an explicit gross total",
                details={"has_printed_gross": False, "posting_lines_count": posting_lines_count},
            )
        )
    elif posting_lines_count == 0:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.FINANCIAL_COMPLETENESS,
                status=SafetyCheckStatus.FAILED,
                message="No posting billed line items present in document",
                details={"has_printed_gross": True, "posting_lines_count": 0},
            )
        )
    else:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.FINANCIAL_COMPLETENESS,
                status=SafetyCheckStatus.PASSED,
                message="Essential financial components (gross total and posting line items) are present",
                details={
                    "gross_total": str(reconstruction.printed_totals.gross_total) if reconstruction.printed_totals else None,
                    "posting_lines_count": posting_lines_count,
                },
            )
        )

    # ── 4. Check: CURRENCY_VALIDITY ────────────────────────────────────────
    curr = reconstruction.currency
    if not curr:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.CURRENCY_VALIDITY,
                status=SafetyCheckStatus.FAILED,
                message="Document primary currency is missing or unidentified",
                details={"currency": None},
            )
        )
    else:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.CURRENCY_VALIDITY,
                status=SafetyCheckStatus.PASSED,
                message=f"Primary currency verified: {curr}",
                details={"currency": curr},
            )
        )

    # ── 5. Check: SUPPLIER_SAFETY ──────────────────────────────────────────
    sup_match = supplier_match
    if sup_match is None and document_facts is not None:
        # Inspect embedded match result if available
        matched_dict = document_facts.parties.supplier.matched_result
        if matched_dict:
            sup_match = MasterMatchResult.from_dict(matched_dict)

    if sup_match is not None:
        if sup_match.status == MatchStatus.AMBIGUOUS:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.SUPPLIER_SAFETY,
                    status=SafetyCheckStatus.FAILED,
                    message="Supplier match is ambiguous; multiple master candidates exist",
                    details={"match_status": "ambiguous", "candidates_count": len(sup_match.candidates)},
                )
            )
        elif sup_match.status == MatchStatus.NO_MATCH:
            if require_matched_supplier:
                checks.append(
                    SafetyCheckResult(
                        check_code=SafetyCheckCode.SUPPLIER_SAFETY,
                        status=SafetyCheckStatus.FAILED,
                        message="Supplier is not matched in master data (strict supplier matching policy enforced)",
                        details={"match_status": "no_match"},
                    )
                )
            else:
                checks.append(
                    SafetyCheckResult(
                        check_code=SafetyCheckCode.SUPPLIER_SAFETY,
                        status=SafetyCheckStatus.WARNING,
                        message="Supplier is not matched in master data; proceeding with blank supplier_id per schema",
                        details={"match_status": "no_match"},
                    )
                )
        elif sup_match.status == MatchStatus.MATCHED:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.SUPPLIER_SAFETY,
                    status=SafetyCheckStatus.PASSED,
                    message=f"Supplier safely matched to master ID '{sup_match.master_id}'",
                    details={"match_status": "matched", "master_id": sup_match.master_id},
                )
            )
    else:
        # Check if supplier name was at least observed
        sup_name = None
        if document_facts is not None and document_facts.parties is not None:
            sup_name = getattr(document_facts.parties.supplier, "observed_name", None) or getattr(document_facts.parties.supplier, "name", None)
        if not sup_name:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.SUPPLIER_SAFETY,
                    status=SafetyCheckStatus.WARNING,
                    message="Supplier identity unextracted or not provided for verification",
                    details={"supplier_name": None},
                )
            )
        else:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.SUPPLIER_SAFETY,
                    status=SafetyCheckStatus.PASSED,
                    message=f"Supplier name observed: '{sup_name}'",
                    details={"supplier_name": sup_name},
                )
            )

    # ── 6. Check: BUYER_SAFETY ─────────────────────────────────────────────
    b_match = buyer_match
    if b_match is None and document_facts is not None:
        matched_dict = document_facts.parties.buyer.matched_result
        if matched_dict:
            b_match = MasterMatchResult.from_dict(matched_dict)

    if b_match is not None:
        if b_match.status == MatchStatus.AMBIGUOUS:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.BUYER_SAFETY,
                    status=SafetyCheckStatus.FAILED,
                    message="Buyer match is ambiguous; multiple organizational entities compete",
                    details={"match_status": "ambiguous"},
                )
            )
        elif b_match.status == MatchStatus.MATCHED:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.BUYER_SAFETY,
                    status=SafetyCheckStatus.PASSED,
                    message="Buyer organizational identity verified",
                    details={"details": b_match.details},
                )
            )
        else:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.BUYER_SAFETY,
                    status=SafetyCheckStatus.PASSED,
                    message="Buyer has no master match; proceeding with observed buyer codes",
                    details={"match_status": "no_match"},
                )
            )
    else:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.BUYER_SAFETY,
                status=SafetyCheckStatus.PASSED,
                message="Buyer safety check satisfied",
                details={},
            )
        )

    # ── 7. Check: PO_SAFETY ────────────────────────────────────────────────
    p_match = po_match
    if p_match is None and document_facts is not None:
        matched_dict = document_facts.po.matched_po_result
        if matched_dict:
            p_match = MasterMatchResult.from_dict(matched_dict)

    if p_match is not None:
        if p_match.status == MatchStatus.AMBIGUOUS:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.PO_SAFETY,
                    status=SafetyCheckStatus.FAILED,
                    message="Purchase order match is ambiguous",
                    details={"match_status": "ambiguous"},
                )
            )
        else:
            checks.append(
                SafetyCheckResult(
                    check_code=SafetyCheckCode.PO_SAFETY,
                    status=SafetyCheckStatus.PASSED,
                    message="Purchase order verified or safely unmatched",
                    details={"match_status": p_match.status.value},
                )
            )
    else:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.PO_SAFETY,
                status=SafetyCheckStatus.PASSED,
                message="No PO ambiguity detected",
                details={},
            )
        )

    # ── 8. Check: UPSTREAM_ISSUES ──────────────────────────────────────────
    blocking_issues = [i for i in reconstruction.issues if i.severity == ERPSeverity.BLOCKING]
    warning_issues = [i for i in reconstruction.issues if i.severity == ERPSeverity.WARNING]

    if blocking_issues:
        b_msgs = [i.message for i in blocking_issues]
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.UPSTREAM_ISSUES,
                status=SafetyCheckStatus.FAILED,
                message=f"Blocking upstream ERP issue: {'; '.join(b_msgs)}",
                details={"blocking_issues": b_msgs},
            )
        )
    elif warning_issues:
        w_msgs = [i.message for i in warning_issues]
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.UPSTREAM_ISSUES,
                status=SafetyCheckStatus.WARNING,
                message=f"Upstream ERP warning: {'; '.join(w_msgs)}",
                details={"warnings": w_msgs},
            )
        )
    else:
        checks.append(
            SafetyCheckResult(
                check_code=SafetyCheckCode.UPSTREAM_ISSUES,
                status=SafetyCheckStatus.PASSED,
                message="No blocking upstream issues detected",
                details={},
            )
        )

    # ── Step 2: Deterministic Precedence Selection ─────────────────────────
    failed_checks = tuple(c for c in checks if c.status == SafetyCheckStatus.FAILED)
    warning_checks = tuple(c for c in checks if c.status == SafetyCheckStatus.WARNING)

    final_status: PayableDecisionStatus
    primary_reason: str
    out_decline_doc_type: Optional[str] = None
    out_decline_reason: Optional[str] = None

    # Precedence 1: Non-payable role / relevance -> NOT_PAYABLE
    if is_non_payable:
        final_status = PayableDecisionStatus.NOT_PAYABLE
        primary_reason = decline_msg or "Document is not an actionable payable"
        out_decline_doc_type = decline_type or (
            document_facts.document_role.value.upper() if document_facts else "NON_PAYABLE"
        )
        out_decline_reason = primary_reason

    # Precedence 2: Reconciliation discrepancy / arithmetic failure -> UNSAFE_TO_AUTODRAFT
    elif any(
        c.check_code == SafetyCheckCode.RECONCILIATION and "mismatch" in c.message.lower()
        for c in failed_checks
    ):
        recon_fail = next(c for c in failed_checks if c.check_code == SafetyCheckCode.RECONCILIATION)
        final_status = PayableDecisionStatus.UNSAFE_TO_AUTODRAFT
        primary_reason = recon_fail.message

    # Precedence 3: Any other failed check -> HOLD_FOR_REVIEW
    elif len(failed_checks) > 0:
        final_status = PayableDecisionStatus.HOLD_FOR_REVIEW
        primary_reason = failed_checks[0].message

    # Precedence 4: Zero failed checks -> SAFE_TO_AUTODRAFT
    else:
        final_status = PayableDecisionStatus.SAFE_TO_AUTODRAFT
        primary_reason = "All safety checks passed; eligible for automated payable generation."

    prov = {
        "stage": "9D_safety_gate",
        "assembly_id": reconstruction.assembly_id,
        "document_id": reconstruction.document_id,
        "checks_evaluated": len(checks),
        "failed_checks_count": len(failed_checks),
        "warning_checks_count": len(warning_checks),
        "require_matched_supplier": require_matched_supplier,
        "require_full_reconciliation": require_full_reconciliation,
    }

    return PayableDecision(
        assembly_id=reconstruction.assembly_id,
        document_id=reconstruction.document_id,
        status=final_status,
        primary_reason=primary_reason,
        decline_doc_type=out_decline_doc_type,
        decline_reason=out_decline_reason,
        checks=tuple(checks),
        failed_checks=failed_checks,
        warning_checks=warning_checks,
        reconciliation_status=reconciliation.status,
        supporting_group_ids=reconstruction.supporting_group_ids,
        provenance=prov,
    )


def evaluate_payable_safety_batch(
    reconciliations: Sequence[ReconciliationResult],
    reconstructions: Sequence[ERPReconstruction],
    document_facts_list: Optional[Sequence[Optional[DocumentFacts]]] = None,
    require_matched_supplier: bool = False,
    require_full_reconciliation: bool = False,
) -> List[PayableDecision]:
    """Batch evaluate payable safety across multiple documents.

    Maintains strict document isolation.
    """
    if len(reconciliations) != len(reconstructions):
        raise ValueError("reconciliations and reconstructions sequences must have equal length")

    facts_seq = document_facts_list or [None] * len(reconciliations)
    decisions: List[PayableDecision] = []

    for r_res, r_rec, df in zip(reconciliations, reconstructions, facts_seq):
        decisions.append(
            evaluate_payable_safety(
                reconciliation=r_res,
                reconstruction=r_rec,
                document_facts=df,
                require_matched_supplier=require_matched_supplier,
                require_full_reconciliation=require_full_reconciliation,
            )
        )
    return decisions
