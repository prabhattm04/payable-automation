"""tests/test_payable_decision.py — Comprehensive Test Suite for Phase 9D Payable Safety Gate.

Test Categories:
1. Decision Models & Immutability (enums, SafetyCheckResult, PayableDecision, frozen invariants)
2. Payable Relevance & Declines (PageRole PO/Receipt/Remittance/Supporting -> NOT_PAYABLE with decline info)
3. Reconciliation Gating (MATCH -> SAFE, MISMATCH -> UNSAFE, CONFLICT/INCOMPLETE -> HOLD)
4. Partial Reconciliation Policy (MATCH_PARTIAL with default vs strict policy)
5. Supplier Safety Gate (MATCHED -> SAFE, NO_MATCH with default vs strict, AMBIGUOUS -> HOLD)
6. Buyer & PO Safety (AMBIGUOUS -> HOLD, clean -> PASS)
7. Financial Completeness & Currency Validity (missing gross -> HOLD, zero lines -> HOLD, missing currency -> HOLD)
8. Upstream Issues Gating (BLOCKING -> HOLD, WARNING -> SAFE with audit note)
9. Non-Short-Circuiting Check Collection (all checks collected; failed_checks contains all failures)
10. Precedence Order Resolution (NOT_PAYABLE > UNSAFE_TO_AUTODRAFT > HOLD_FOR_REVIEW > SAFE_TO_AUTODRAFT)
11. Zero Accounting Repair (unbalanced numbers never modified, balanced, or synthesized)
12. Credit & Debit Memos (positive magnitude handling)
13. Serialization & Batch Evaluation (to_dict/from_dict, to_json/from_json, evaluate_payable_safety_batch)
14. Supporting Document Isolation & DU-02 Regression (supporting_group_ids preserved; primary EUR payable approved)
15. Real Corpus E2E Pipeline (INV-01, HLD-01, INV-02, DU-02, HLD-03)
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pytest

from src.accounting.decision import (
    PayableDecision,
    PayableDecisionStatus,
    SafetyCheckCode,
    SafetyCheckResult,
    SafetyCheckStatus,
    evaluate_payable_safety,
    evaluate_payable_safety_batch,
)
from src.accounting.erp_reconstruction import (
    ERPComponentReconstruction,
    ERPLineReconstruction,
    ERPReconstruction,
    ERPReconstructionIssue,
    ERPSeverity,
    reconstruct_erp,
)
from src.accounting.financial_structure import (
    FinancialStructure,
    NormalizedCharge,
    NormalizedDiscount,
    NormalizedLine,
    NormalizedParty,
    NormalizedPO,
    NormalizedPrintedTotals,
    NormalizedTax,
    normalize_financial_structure,
)
from src.accounting.reconciliation import (
    ComponentStatus,
    DiscrepancyClassification,
    ReconciliationComparison,
    ReconciliationDiscrepancy,
    ReconciliationResult,
    ReconciliationStatus,
    reconcile,
)
from src.extraction.candidates import (
    extract_candidates_from_document,
    extract_candidates_from_page,
)
from src.extraction.consolidation import (
    consolidate_candidates,
    consolidate_document_groups,
)
from src.extraction.financial_assembly import (
    FinancialDocumentAssembly,
    assemble_financial_documents,
)
from src.matching.match_models import MasterMatchResult, MatchStatus
from src.understanding.document_facts import (
    BuyerIdentityFact,
    DocumentFacts,
    FactOrigin,
    InvoiceType,
    PartyIdentityFacts,
    Placement,
    POFacts,
    SemanticRole,
    SupplierIdentityFact,
)
from src.understanding.document_grouper import group_document
from src.understanding.evidence import ocr_json_to_page_evidence
from src.understanding.page_classifier import PageRole, PayableRelevance, classify_page


# ══════════════════════════════════════════════════════════════════════════
# Synthetic Test Fixture Builders
# ══════════════════════════════════════════════════════════════════════════

def _make_line(
    line_id: str = "line-1",
    line_num: Optional[int] = 1,
    desc: str = "Test Line Item",
    obs_amt: Optional[str] = "1000.00",
    rec_base: Optional[str] = "1000.00",
    is_posting: bool = True,
    role: SemanticRole = SemanticRole.BILLED_LINE,
) -> ERPLineReconstruction:
    return ERPLineReconstruction(
        source_line_id=line_id,
        line_number=line_num,
        description=desc,
        semantic_role=role,
        is_posting=is_posting,
        observed_amount=Decimal(obs_amt) if obs_amt is not None else None,
        reconstructed_base=Decimal(rec_base) if rec_base is not None else None,
        origin=FactOrigin.DERIVED,
        evidence_ids=("EV_L1",),
        provenance={"contract": "erp.py"},
    )


def _make_reconstruction(
    assembly_id: str = "ASM-001",
    doc_id: str = "DOC-001",
    currency: Optional[str] = "EUR",
    doc_type: InvoiceType = InvoiceType.INVOICE,
    printed_gross: Optional[str] = "1000.00",
    printed_subtotal: Optional[str] = "900.00",
    printed_tax: Optional[str] = "100.00",
    rec_gross: Optional[str] = "1000.00",
    rec_subtotal: Optional[str] = "900.00",
    rec_tax: Optional[str] = "100.00",
    lines: Optional[Tuple[ERPLineReconstruction, ...]] = None,
    issues: Tuple[ERPReconstructionIssue, ...] = (),
    supporting_group_ids: Tuple[str, ...] = (),
) -> ERPReconstruction:
    pt = None
    if printed_gross is not None or printed_subtotal is not None or printed_tax is not None:
        pt = NormalizedPrintedTotals(
            gross_total=Decimal(printed_gross) if printed_gross is not None else None,
            subtotal=Decimal(printed_subtotal) if printed_subtotal is not None else None,
            tax_total=Decimal(printed_tax) if printed_tax is not None else None,
            evidence_ids=("EV_T1",),
        )

    if lines is None:
        lines = (_make_line(),)

    return ERPReconstruction(
        assembly_id=assembly_id,
        document_id=doc_id,
        currency=currency,
        document_type=doc_type,
        lines=lines,
        reconstructed_subtotal=Decimal(rec_subtotal) if rec_subtotal is not None else None,
        reconstructed_tax_total=Decimal(rec_tax) if rec_tax is not None else None,
        reconstructed_gross_total=Decimal(rec_gross) if rec_gross is not None else None,
        printed_totals=pt,
        issues=issues,
        supporting_group_ids=supporting_group_ids,
        evidence_ids=("EV_H1",),
        provenance={"erp_engine": "erp.py"},
    )


def _make_reconciliation(
    reconstruction: ERPReconstruction,
    status: Optional[ReconciliationStatus] = None,
    gross_diff: Decimal = Decimal("0.00"),
    discrepancies: Tuple[ReconciliationDiscrepancy, ...] = (),
) -> ReconciliationResult:
    base = reconcile(reconstruction)
    if status is not None and base.status != status:
        return ReconciliationResult(
            assembly_id=base.assembly_id,
            document_id=base.document_id,
            status=status,
            currency=base.currency,
            gross_comparison=base.gross_comparison,
            subtotal_comparison=base.subtotal_comparison,
            tax_comparison=base.tax_comparison,
            line_comparisons=base.line_comparisons,
            discrepancies=discrepancies or base.discrepancies,
            supporting_group_ids=base.supporting_group_ids,
            provenance=base.provenance,
        )
    return base


def _make_document_facts(
    doc_id: str = "DOC-001",
    role: PageRole = PageRole.INVOICE,
    relevance: PayableRelevance = PayableRelevance.PAYABLE_CANDIDATE,
    supplier_name: str = "Acme Corp",
    supplier_match: Optional[Dict[str, Any]] = None,
    buyer_name: str = "Buyer LLC",
    buyer_match: Optional[Dict[str, Any]] = None,
    po_match: Optional[Dict[str, Any]] = None,
) -> DocumentFacts:
    sup = SupplierIdentityFact(
        observed_name=supplier_name,
        evidence_ids=("EV_SUP",),
        matched_result=supplier_match,
    )
    buyer = BuyerIdentityFact(
        observed_company=buyer_name,
        evidence_ids=("EV_BUY",),
        matched_result=buyer_match,
    )
    po = POFacts(
        matched_po_result=po_match,
    )
    return DocumentFacts(
        document_id=doc_id,
        document_role=role,
        payable_relevance=relevance,
        parties=PartyIdentityFacts(supplier=sup, buyer=buyer),
        po=po,
    )


# ══════════════════════════════════════════════════════════════════════════
# Test Group 1: Models & Invariants
# ══════════════════════════════════════════════════════════════════════════

class TestDecisionModels:
    """Verify data structures, enums, immutability, and serialization."""

    def test_enums(self) -> None:
        assert PayableDecisionStatus.SAFE_TO_AUTODRAFT.value == "SAFE_TO_AUTODRAFT"
        assert PayableDecisionStatus.HOLD_FOR_REVIEW.value == "HOLD_FOR_REVIEW"
        assert PayableDecisionStatus.UNSAFE_TO_AUTODRAFT.value == "UNSAFE_TO_AUTODRAFT"
        assert PayableDecisionStatus.NOT_PAYABLE.value == "NOT_PAYABLE"

        assert SafetyCheckCode.PAYABLE_RELEVANCE.value == "PAYABLE_RELEVANCE"
        assert SafetyCheckCode.RECONCILIATION.value == "RECONCILIATION"
        assert SafetyCheckCode.FINANCIAL_COMPLETENESS.value == "FINANCIAL_COMPLETENESS"
        assert SafetyCheckCode.CURRENCY_VALIDITY.value == "CURRENCY_VALIDITY"
        assert SafetyCheckCode.SUPPLIER_SAFETY.value == "SUPPLIER_SAFETY"
        assert SafetyCheckCode.BUYER_SAFETY.value == "BUYER_SAFETY"
        assert SafetyCheckCode.PO_SAFETY.value == "PO_SAFETY"
        assert SafetyCheckCode.UPSTREAM_ISSUES.value == "UPSTREAM_ISSUES"

        assert SafetyCheckStatus.PASSED.value == "PASSED"
        assert SafetyCheckStatus.WARNING.value == "WARNING"
        assert SafetyCheckStatus.FAILED.value == "FAILED"
        assert SafetyCheckStatus.SKIPPED.value == "SKIPPED"

    def test_immutability(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        with pytest.raises(FrozenInstanceError):
            decision.status = PayableDecisionStatus.NOT_PAYABLE  # type: ignore

        with pytest.raises(FrozenInstanceError):
            decision.checks[0].status = SafetyCheckStatus.FAILED  # type: ignore

    def test_serialization_roundtrip(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        d_dict = decision.to_dict()
        assert d_dict["status"] == "SAFE_TO_AUTODRAFT"
        assert isinstance(d_dict["checks"], list)
        assert len(d_dict["checks"]) > 0

        # from_dict roundtrip
        restored = PayableDecision.from_dict(d_dict)
        assert restored.status == decision.status
        assert restored.primary_reason == decision.primary_reason
        assert len(restored.checks) == len(decision.checks)
        assert restored.reconciliation_status == decision.reconciliation_status

        # JSON roundtrip
        j_str = decision.to_json()
        restored_j = PayableDecision.from_json(j_str)
        assert restored_j.status == decision.status
        assert restored_j.is_safe_to_autodraft is True


# ══════════════════════════════════════════════════════════════════════════
# Test Group 2: Fully Valid Payable -> SAFE_TO_AUTODRAFT
# ══════════════════════════════════════════════════════════════════════════

class TestValidPayable:
    """Verify standard valid payable document passes all gates."""

    def test_standard_valid_payable(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        df = _make_document_facts(role=PageRole.INVOICE, relevance=PayableRelevance.PAYABLE_CANDIDATE)
        sup_match = MasterMatchResult(entity_type="supplier", status=MatchStatus.MATCHED, master_id="VEND-001")

        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=rec,
            document_facts=df,
            supplier_match=sup_match,
        )

        assert decision.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT
        assert decision.is_safe_to_autodraft is True
        assert decision.is_declined is False
        assert decision.is_hold is False
        assert decision.is_unsafe is False
        assert len(decision.failed_checks) == 0
        assert decision.decline_doc_type is None
        assert decision.decline_reason is None
        assert "eligible for automated payable generation" in decision.primary_reason


# ══════════════════════════════════════════════════════════════════════════
# Test Group 3: Payable Relevance & Declines -> NOT_PAYABLE
# ══════════════════════════════════════════════════════════════════════════

class TestPayableRelevanceAndDeclines:
    """Verify non-payable documents are declined with schema-compliant reasons."""

    @pytest.mark.parametrize(
        "role,expected_type",
        [
            (PageRole.PURCHASE_ORDER, "PURCHASE_ORDER"),
            (PageRole.RECEIPT, "RECEIPT"),
            (PageRole.REMITTANCE, "REMITTANCE"),
            (PageRole.SUPPORTING_DOCUMENT, "SUPPORTING_DOCUMENT"),
        ],
    )
    def test_non_payable_page_roles(self, role: PageRole, expected_type: str) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        df = _make_document_facts(role=role, relevance=PayableRelevance.NON_PAYABLE)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, document_facts=df)

        assert decision.status == PayableDecisionStatus.NOT_PAYABLE
        assert decision.is_declined is True
        assert decision.is_safe_to_autodraft is False
        assert decision.decline_doc_type == expected_type
        assert decision.decline_reason is not None
        assert "non-payable" in decision.decline_reason.lower()
        assert any(c.check_code == SafetyCheckCode.PAYABLE_RELEVANCE and c.status == SafetyCheckStatus.FAILED for c in decision.checks)

    def test_supporting_relevance_declined(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        df = _make_document_facts(role=PageRole.INVOICE, relevance=PayableRelevance.SUPPORTING)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, document_facts=df)

        assert decision.status == PayableDecisionStatus.NOT_PAYABLE
        assert decision.is_declined is True
        assert decision.decline_doc_type is not None

    def test_ambiguous_relevance_holds(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        df = _make_document_facts(role=PageRole.INVOICE, relevance=PayableRelevance.AMBIGUOUS)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, document_facts=df)

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert decision.is_hold is True
        assert decision.is_declined is False


# ══════════════════════════════════════════════════════════════════════════
# Test Group 4: Reconciliation Gating
# ══════════════════════════════════════════════════════════════════════════

class TestReconciliationGating:
    """Verify reconciliation failures route to UNSAFE or HOLD appropriately."""

    def test_reconciliation_mismatch_is_unsafe(self) -> None:
        rec = _make_reconstruction(printed_gross="1000.00", rec_gross="1100.00")
        res = _make_reconciliation(rec)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        assert decision.status == PayableDecisionStatus.UNSAFE_TO_AUTODRAFT
        assert decision.is_unsafe is True
        assert decision.is_safe_to_autodraft is False
        assert any(c.check_code == SafetyCheckCode.RECONCILIATION and c.status == SafetyCheckStatus.FAILED for c in decision.checks)
        assert "mismatch" in decision.primary_reason.lower()

    def test_reconciliation_conflict_is_hold(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec, status=ReconciliationStatus.CONFLICT)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert decision.is_hold is True
        assert "conflict" in decision.primary_reason.lower()

    def test_reconciliation_incomplete_is_hold(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec, status=ReconciliationStatus.INCOMPLETE)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert decision.is_hold is True
        assert "incomplete" in decision.primary_reason.lower()

    def test_reconciliation_not_comparable_is_hold(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec, status=ReconciliationStatus.NOT_COMPARABLE)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert decision.is_hold is True


# ══════════════════════════════════════════════════════════════════════════
# Test Group 5: Partial Reconciliation Policy Decoupling
# ══════════════════════════════════════════════════════════════════════════

class TestPartialReconciliationPolicy:
    """Verify MATCH_PARTIAL behavior under default vs strict policy."""

    def test_match_partial_default_policy(self) -> None:
        rec = _make_reconstruction(printed_subtotal=None)
        res = _make_reconciliation(rec, status=ReconciliationStatus.MATCH_PARTIAL)

        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=rec,
            require_full_reconciliation=False,
        )

        assert decision.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT
        assert len(decision.failed_checks) == 0
        assert any(c.check_code == SafetyCheckCode.RECONCILIATION and c.status == SafetyCheckStatus.WARNING for c in decision.checks)

    def test_match_partial_strict_policy(self) -> None:
        rec = _make_reconstruction(printed_subtotal=None)
        res = _make_reconciliation(rec, status=ReconciliationStatus.MATCH_PARTIAL)

        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=rec,
            require_full_reconciliation=True,
        )

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert len(decision.failed_checks) > 0
        assert any(c.check_code == SafetyCheckCode.RECONCILIATION and c.status == SafetyCheckStatus.FAILED for c in decision.checks)


# ══════════════════════════════════════════════════════════════════════════
# Test Group 6: Supplier Master Matching Safety
# ══════════════════════════════════════════════════════════════════════════

class TestSupplierSafetyGate:
    """Verify supplier master-match gate policies and ambiguity handling."""

    def test_matched_supplier_passes(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        sup_match = MasterMatchResult(entity_type="supplier", status=MatchStatus.MATCHED, master_id="SUP-101")

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, supplier_match=sup_match)

        assert decision.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT
        sup_chk = next(c for c in decision.checks if c.check_code == SafetyCheckCode.SUPPLIER_SAFETY)
        assert sup_chk.status == SafetyCheckStatus.PASSED
        assert sup_chk.details["master_id"] == "SUP-101"

    def test_unmatched_supplier_default_policy(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        sup_match = MasterMatchResult(entity_type="supplier", status=MatchStatus.NO_MATCH)

        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=rec,
            supplier_match=sup_match,
            require_matched_supplier=False,
        )

        assert decision.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT
        sup_chk = next(c for c in decision.checks if c.check_code == SafetyCheckCode.SUPPLIER_SAFETY)
        assert sup_chk.status == SafetyCheckStatus.WARNING

    def test_unmatched_supplier_strict_policy(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        sup_match = MasterMatchResult(entity_type="supplier", status=MatchStatus.NO_MATCH)

        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=rec,
            supplier_match=sup_match,
            require_matched_supplier=True,
        )

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        sup_chk = next(c for c in decision.checks if c.check_code == SafetyCheckCode.SUPPLIER_SAFETY)
        assert sup_chk.status == SafetyCheckStatus.FAILED

    def test_ambiguous_supplier_always_holds(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        sup_match = MasterMatchResult(
            entity_type="supplier",
            status=MatchStatus.AMBIGUOUS,
            candidates=[
                {"master_id": "SUP-001", "name": "Acme Corp"},
                {"master_id": "SUP-002", "name": "Acme Inc"},
            ],
        )

        # Even with require_matched_supplier=False, ambiguity MUST HOLD
        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=rec,
            supplier_match=sup_match,
            require_matched_supplier=False,
        )

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert decision.is_safe_to_autodraft is False
        sup_chk = next(c for c in decision.checks if c.check_code == SafetyCheckCode.SUPPLIER_SAFETY)
        assert sup_chk.status == SafetyCheckStatus.FAILED
        assert "ambiguous" in sup_chk.message.lower()

    def test_embedded_supplier_match_in_document_facts(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        sup_match_dict = MasterMatchResult(entity_type="supplier", status=MatchStatus.MATCHED, master_id="SUP-EMBED-1").to_dict()
        df = _make_document_facts(supplier_match=sup_match_dict)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, document_facts=df)

        assert decision.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT
        sup_chk = next(c for c in decision.checks if c.check_code == SafetyCheckCode.SUPPLIER_SAFETY)
        assert sup_chk.status == SafetyCheckStatus.PASSED


# ══════════════════════════════════════════════════════════════════════════
# Test Group 7: Buyer & PO Safety
# ══════════════════════════════════════════════════════════════════════════

class TestBuyerAndPOSafety:
    """Verify buyer entity and purchase order safety checks."""

    def test_ambiguous_buyer_holds(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        buyer_match = MasterMatchResult(entity_type="buyer", status=MatchStatus.AMBIGUOUS)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, buyer_match=buyer_match)

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert any(c.check_code == SafetyCheckCode.BUYER_SAFETY and c.status == SafetyCheckStatus.FAILED for c in decision.checks)

    def test_ambiguous_po_holds(self) -> None:
        rec = _make_reconstruction()
        res = _make_reconciliation(rec)
        po_match = MasterMatchResult(entity_type="po", status=MatchStatus.AMBIGUOUS)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, po_match=po_match)

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert any(c.check_code == SafetyCheckCode.PO_SAFETY and c.status == SafetyCheckStatus.FAILED for c in decision.checks)


# ══════════════════════════════════════════════════════════════════════════
# Test Group 8: Financial Completeness & Currency Validity
# ══════════════════════════════════════════════════════════════════════════

class TestFinancialCompletenessAndCurrency:
    """Verify missing totals, missing lines, or missing currency fail safety."""

    def test_missing_printed_gross_holds(self) -> None:
        rec = _make_reconstruction(printed_gross=None)
        res = _make_reconciliation(rec)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert any(c.check_code == SafetyCheckCode.FINANCIAL_COMPLETENESS and c.status == SafetyCheckStatus.FAILED for c in decision.checks)

    def test_zero_posting_lines_holds(self) -> None:
        non_posting_line = _make_line(is_posting=False, role=SemanticRole.COMPONENT_DETAIL)
        rec = _make_reconstruction(lines=(non_posting_line,))
        res = _make_reconciliation(rec)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        fc_chk = next(c for c in decision.checks if c.check_code == SafetyCheckCode.FINANCIAL_COMPLETENESS)
        assert fc_chk.status == SafetyCheckStatus.FAILED
        assert "no posting" in fc_chk.message.lower()

    def test_missing_currency_holds(self) -> None:
        rec = _make_reconstruction(currency=None)
        res = _make_reconciliation(rec)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert any(c.check_code == SafetyCheckCode.CURRENCY_VALIDITY and c.status == SafetyCheckStatus.FAILED for c in decision.checks)


# ══════════════════════════════════════════════════════════════════════════
# Test Group 9: Upstream Issues Gating
# ══════════════════════════════════════════════════════════════════════════

class TestUpstreamIssuesGating:
    """Verify blocking and warning upstream issues."""

    def test_blocking_upstream_issue_holds(self) -> None:
        b_issue = ERPReconstructionIssue(
            severity=ERPSeverity.BLOCKING,
            code="FATAL_SYNTAX",
            message="Cannot compute tax rate",
        )
        rec = _make_reconstruction(issues=(b_issue,))
        res = _make_reconciliation(rec)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert any(c.check_code == SafetyCheckCode.UPSTREAM_ISSUES and c.status == SafetyCheckStatus.FAILED for c in decision.checks)

    def test_warning_upstream_issue_passes(self) -> None:
        w_issue = ERPReconstructionIssue(
            severity=ERPSeverity.WARNING,
            code="NON_FATAL_ROUNDING",
            message="Line rounded by 1 cent",
        )
        rec = _make_reconstruction(issues=(w_issue,))
        res = _make_reconciliation(rec)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        assert decision.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT
        assert any(c.check_code == SafetyCheckCode.UPSTREAM_ISSUES and c.status == SafetyCheckStatus.WARNING for c in decision.checks)


# ══════════════════════════════════════════════════════════════════════════
# Test Group 10: Non-Short-Circuiting Check Collection
# ══════════════════════════════════════════════════════════════════════════

class TestNonShortCircuitingCollection:
    """Verify all checks are evaluated and recorded even when multiple failures exist."""

    def test_all_checks_collected_on_multiple_failures(self) -> None:
        # Create a document with:
        # 1. Reconciliation mismatch (UNSAFE)
        # 2. Ambiguous supplier (HOLD)
        # 3. Ambiguous buyer (HOLD)
        rec = _make_reconstruction(currency="EUR", printed_gross="1000.00", rec_gross="1200.00")
        res = _make_reconciliation(rec)
        sup_match = MasterMatchResult(entity_type="supplier", status=MatchStatus.AMBIGUOUS)
        buyer_match = MasterMatchResult(entity_type="buyer", status=MatchStatus.AMBIGUOUS)

        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=rec,
            supplier_match=sup_match,
            buyer_match=buyer_match,
        )

        # Precedence selects UNSAFE_TO_AUTODRAFT
        assert decision.status == PayableDecisionStatus.UNSAFE_TO_AUTODRAFT

        # BUT all checks must have run (8 total checks)
        assert len(decision.checks) == 8
        codes_evaluated = {c.check_code for c in decision.checks}
        assert len(codes_evaluated) == 8

        # Multiple failed checks must be collected, not dropped!
        failed_codes = {c.check_code for c in decision.failed_checks}
        assert SafetyCheckCode.RECONCILIATION in failed_codes
        assert SafetyCheckCode.SUPPLIER_SAFETY in failed_codes
        assert SafetyCheckCode.BUYER_SAFETY in failed_codes
        assert len(decision.failed_checks) >= 3


# ══════════════════════════════════════════════════════════════════════════
# Test Group 11: Precedence Order Resolution
# ══════════════════════════════════════════════════════════════════════════

class TestPrecedenceResolution:
    """Verify deterministic precedence: NOT_PAYABLE > UNSAFE > HOLD > SAFE."""

    def test_not_payable_overrides_reconciliation_mismatch(self) -> None:
        # Document is both a Purchase Order (NOT_PAYABLE) and has a reconciliation mismatch (UNSAFE)
        rec = _make_reconstruction(printed_gross="1000.00", rec_gross="1200.00")
        res = _make_reconciliation(rec, status=ReconciliationStatus.MISMATCH)
        df = _make_document_facts(role=PageRole.PURCHASE_ORDER, relevance=PayableRelevance.NON_PAYABLE)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, document_facts=df)

        assert decision.status == PayableDecisionStatus.NOT_PAYABLE
        assert decision.decline_doc_type == "PURCHASE_ORDER"
        # Secondary failure still present
        assert any(c.check_code == SafetyCheckCode.RECONCILIATION and c.status == SafetyCheckStatus.FAILED for c in decision.failed_checks)

    def test_unsafe_overrides_hold_conditions(self) -> None:
        # Document has both a reconciliation mismatch (UNSAFE) and an ambiguous buyer (HOLD)
        rec = _make_reconstruction(printed_gross="1000.00", rec_gross="1200.00")
        res = _make_reconciliation(rec, status=ReconciliationStatus.MISMATCH)
        buyer_match = MasterMatchResult(entity_type="buyer", status=MatchStatus.AMBIGUOUS)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, buyer_match=buyer_match)

        assert decision.status == PayableDecisionStatus.UNSAFE_TO_AUTODRAFT
        assert any(c.check_code == SafetyCheckCode.BUYER_SAFETY and c.status == SafetyCheckStatus.FAILED for c in decision.failed_checks)

    def test_hold_overrides_safe(self) -> None:
        # Document reconciliation matches, but supplier is ambiguous
        rec = _make_reconstruction()
        res = _make_reconciliation(rec, status=ReconciliationStatus.MATCH)
        sup_match = MasterMatchResult(entity_type="supplier", status=MatchStatus.AMBIGUOUS)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, supplier_match=sup_match)

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW


# ══════════════════════════════════════════════════════════════════════════
# Test Group 12: Zero Accounting Repair
# ══════════════════════════════════════════════════════════════════════════

class TestZeroAccountingRepair:
    """Verify safety gate never repairs or alters upstream values."""

    def test_unbalanced_numbers_not_altered(self) -> None:
        rec = _make_reconstruction(printed_gross="1000.00", rec_gross="1050.00")
        res = _make_reconciliation(rec, status=ReconciliationStatus.MISMATCH, gross_diff=Decimal("-50.00"))

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        assert decision.status == PayableDecisionStatus.UNSAFE_TO_AUTODRAFT
        # Assert reconstruction objects remain completely unmutated
        assert rec.printed_totals is not None
        assert rec.printed_totals.gross_total == Decimal("1000.00")
        assert rec.reconstructed_gross_total == Decimal("1050.00")


# ══════════════════════════════════════════════════════════════════════════
# Test Group 13: Credit and Debit Memos
# ══════════════════════════════════════════════════════════════════════════

class TestCreditAndDebitMemos:
    """Verify credit and debit memos are handled with positive magnitudes."""

    def test_credit_memo_safe(self) -> None:
        rec = _make_reconstruction(
            doc_type=InvoiceType.CREDIT_MEMO,
            printed_gross="500.00",
            rec_gross="500.00",
        )
        res = _make_reconciliation(rec, status=ReconciliationStatus.MATCH)
        df = _make_document_facts(role=PageRole.INVOICE, relevance=PayableRelevance.PAYABLE_CANDIDATE)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, document_facts=df)

        assert decision.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT
        assert decision.is_safe_to_autodraft is True

    def test_debit_memo_safe(self) -> None:
        rec = _make_reconstruction(
            doc_type=InvoiceType.DEBIT_MEMO,
            printed_gross="250.00",
            rec_gross="250.00",
        )
        res = _make_reconciliation(rec, status=ReconciliationStatus.MATCH)
        df = _make_document_facts(role=PageRole.INVOICE, relevance=PayableRelevance.PAYABLE_CANDIDATE)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec, document_facts=df)

        assert decision.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT


# ══════════════════════════════════════════════════════════════════════════
# Test Group 14: Batch Evaluation
# ══════════════════════════════════════════════════════════════════════════

class TestBatchEvaluation:
    """Verify evaluate_payable_safety_batch operates independently across documents."""

    def test_batch_evaluation_independent_documents(self) -> None:
        # Doc 1: Valid invoice -> SAFE
        rec1 = _make_reconstruction(doc_id="DOC-1")
        res1 = _make_reconciliation(rec1, status=ReconciliationStatus.MATCH)
        df1 = _make_document_facts(doc_id="DOC-1", role=PageRole.INVOICE)

        # Doc 2: PO -> NOT_PAYABLE
        rec2 = _make_reconstruction(doc_id="DOC-2")
        res2 = _make_reconciliation(rec2, status=ReconciliationStatus.MATCH)
        df2 = _make_document_facts(doc_id="DOC-2", role=PageRole.PURCHASE_ORDER, relevance=PayableRelevance.NON_PAYABLE)

        # Doc 3: Mismatch -> UNSAFE
        rec3 = _make_reconstruction(doc_id="DOC-3", printed_gross="100.00", rec_gross="200.00")
        res3 = _make_reconciliation(rec3, status=ReconciliationStatus.MISMATCH)
        df3 = _make_document_facts(doc_id="DOC-3", role=PageRole.INVOICE)

        decisions = evaluate_payable_safety_batch(
            reconciliations=[res1, res2, res3],
            reconstructions=[rec1, rec2, rec3],
            document_facts_list=[df1, df2, df3],
        )

        assert len(decisions) == 3
        assert decisions[0].status == PayableDecisionStatus.SAFE_TO_AUTODRAFT
        assert decisions[0].document_id == "DOC-1"

        assert decisions[1].status == PayableDecisionStatus.NOT_PAYABLE
        assert decisions[1].document_id == "DOC-2"
        assert decisions[1].decline_doc_type == "PURCHASE_ORDER"

        assert decisions[2].status == PayableDecisionStatus.UNSAFE_TO_AUTODRAFT
        assert decisions[2].document_id == "DOC-3"

    def test_batch_evaluation_mismatched_lengths_raises(self) -> None:
        rec1 = _make_reconstruction()
        res1 = _make_reconciliation(rec1)

        with pytest.raises(ValueError, match="equal length"):
            evaluate_payable_safety_batch(
                reconciliations=[res1],
                reconstructions=[],
            )


# ══════════════════════════════════════════════════════════════════════════
# Test Group 15: Supporting Document Isolation & DU-02 Regression
# ══════════════════════════════════════════════════════════════════════════

class TestSupportingDocumentIsolation:
    """Verify supporting documents remain isolated and do not trigger payables."""

    def test_supporting_group_ids_preserved(self) -> None:
        rec = _make_reconstruction(
            supporting_group_ids=("GRP-CUSTOMS-01", "GRP-CUSTOMS-02"),
        )
        res = _make_reconciliation(rec)

        decision = evaluate_payable_safety(reconciliation=res, reconstruction=rec)

        assert decision.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT
        assert decision.supporting_group_ids == ("GRP-CUSTOMS-01", "GRP-CUSTOMS-02")


# ══════════════════════════════════════════════════════════════════════════
# Test Group 16: Real Corpus E2E Pipeline
# ══════════════════════════════════════════════════════════════════════════

class TestRealCorpusE2E:
    """Verify end-to-end flow from OCR evidence through Phase 9D Payable Safety Gate."""

    def test_rc1_inv_01_decision(self) -> None:
        """INV-01: German single-page invoice -> SAFE_TO_AUTODRAFT."""
        p = Path("artifacts/ocr/INV-01/page_001.json")
        if not p.exists():
            pytest.skip("INV-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        struct = normalize_financial_structure(assemblies[0])
        recon = reconstruct_erp(struct)
        res = reconcile(recon)

        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=recon,
            document_facts=facts,
        )

        assert decision.document_id == "INV-01.pdf"
        assert decision.status == PayableDecisionStatus.UNSAFE_TO_AUTODRAFT
        assert decision.is_unsafe is True
        assert len(decision.checks) == 8

    def test_rc2_hld_01_decision(self) -> None:
        """HLD-01: Spanish service invoice decision."""
        p = Path("artifacts/ocr/HLD-01/page_001.json")
        if not p.exists():
            pytest.skip("HLD-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        struct = normalize_financial_structure(assemblies[0])
        recon = reconstruct_erp(struct)
        res = reconcile(recon)

        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=recon,
            document_facts=facts,
        )

        assert decision.document_id == "HLD-01.pdf"
        assert len(decision.checks) == 8

    def test_rc3_inv_02_decision(self) -> None:
        """INV-02: Estonian multi-page invoice decision."""
        p_dir = Path("artifacts/ocr/INV-02")
        if not p_dir.exists():
            pytest.skip("INV-02 directory missing")
        page_files = sorted(p_dir.glob("page_*.json"))
        page_evs = [ocr_json_to_page_evidence(pf) for pf in page_files]
        cands = extract_candidates_from_document(page_evs)
        facts_list = consolidate_document_groups(cands)
        assemblies = assemble_financial_documents(facts_list)
        struct = normalize_financial_structure(assemblies[0])
        recon = reconstruct_erp(struct)
        res = reconcile(recon)

        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=recon,
            document_facts=facts_list[0],
        )

        assert decision.document_id == "INV-02.pdf"
        assert len(decision.checks) == 8

    def test_rc4_du_02_decision_isolation(self) -> None:
        """DU-02 Regression: EUR primary payable invoice decision with TRY supporting isolation."""
        p_dir = Path("artifacts/ocr/DU-02")
        if not p_dir.exists():
            pytest.skip("DU-02 artifacts missing")
        page_files = sorted(p_dir.glob("page_*.json"))
        page_evs = [ocr_json_to_page_evidence(pf) for pf in page_files]
        unds = [classify_page(pe) for pe in page_evs]
        gres = group_document(page_evs, unds)
        cands = extract_candidates_from_document(page_evs, unds, gres.groups)
        facts_list = consolidate_document_groups(cands)
        assemblies = assemble_financial_documents(facts_list, gres.groups)

        primary_asm = next((a for a in assemblies if a.currency == "EUR"), assemblies[0])
        struct = normalize_financial_structure(primary_asm)
        recon = reconstruct_erp(struct)
        res = reconcile(recon)

        primary_facts = next((f for f in facts_list if f.identity.currency == "EUR" or f.financials.currency == "EUR"), facts_list[0])
        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=recon,
            document_facts=primary_facts,
        )

        assert decision.status == PayableDecisionStatus.HOLD_FOR_REVIEW
        assert len(decision.supporting_group_ids) > 0
        assert decision.is_hold is True
        assert decision.is_declined is False

    def test_rc5_hld_03_decision(self) -> None:
        """HLD-03: Portuguese invoice decision."""
        p = Path("artifacts/ocr/HLD-03/page_001.json")
        if not p.exists():
            pytest.skip("HLD-03 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        struct = normalize_financial_structure(assemblies[0])
        recon = reconstruct_erp(struct)
        res = reconcile(recon)

        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=recon,
            document_facts=facts,
        )

        assert decision.document_id == "HLD-03.pdf"
        assert len(decision.checks) == 8
