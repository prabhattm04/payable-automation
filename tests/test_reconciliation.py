"""tests/test_reconciliation.py — Comprehensive Test Suite for Phase 9C-3 Reconciliation.

Test Categories:
A — Exact match (Document == ERP reconstructed -> MATCH, EXACT_MATCH)
B — Discrepancy within tolerance (diff <= tolerance -> MATCH, WITHIN_TOLERANCE)
C — Discrepancy outside tolerance (diff > tolerance -> MISMATCH, NUMERICAL_VARIANCE)
D — Missing document gross (printed gross is None -> INCOMPLETE, MISSING_DOCUMENT_VALUE)
E — Missing ERP gross (reconstructed gross is None -> INCOMPLETE, MISSING_ERP_VALUE)
F — Upstream issues & conflicts (blocking issue -> CONFLICT)
G — Subtotal comparison (independent subtotal evaluation)
H — Tax total comparison (independent tax total evaluation)
I — Partial comparison status (gross matches, but subtotal/tax unprinted -> MATCH_PARTIAL)
J — Intra-record line correspondence (line matching via source_line_id, no index assumptions)
K — Non-posting component detail lines (COMPONENT_DETAIL is NOT_COMPARABLE; payable remains MATCH)
L — Supporting document isolation (supporting group IDs isolated; secondary currencies excluded)
M — DU-02 regression (EUR primary payable invoice reconciled; TRY customs pages isolated)
N — Credit memo (positive magnitude comparison; zero sign-flipping)
O — Debit memo (positive magnitude comparison)
P — Zero accounting repair (unexplained discrepancy recorded; zero balancing facts invented)
Q — Provenance separation (document_evidence_ids separated from erp_provenance)
R — Input immutability (ERPReconstruction unmutated before/after reconcile)
S — Deterministic output (repeated runs produce identical to_dict output)
T — Serialization (round-trip to_dict/from_dict and to_json/from_json)
U — Anti-formula-duplication (monkeypatch erp.erp_book, verify 9C-3 consumes mocked output)
V — Tolerance parameter validation (negative tolerance raises ValueError)
W — Batch reconciliation (reconcile_batch processes independent payables)
RC — Real corpus verification (INV-01, HLD-01, INV-02, DU-02, HLD-03)
"""
from __future__ import annotations

import copy
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pytest

import erp
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
from src.accounting.erp_reconstruction import (
    ERPComponentReconstruction,
    ERPLineReconstruction,
    ERPReconstruction,
    ERPReconstructionIssue,
    ERPSeverity,
    reconstruct_erp,
)
from src.accounting.reconciliation import (
    ComponentStatus,
    DiscrepancyClassification,
    ReconciliationComparison,
    ReconciliationDiscrepancy,
    ReconciliationResult,
    ReconciliationStatus,
    reconcile,
    reconcile_batch,
)
from src.extraction.financial_assembly import (
    FinancialDocumentAssembly,
    assemble_financial_documents,
)
from src.extraction.candidates import (
    extract_candidates_from_document,
    extract_candidates_from_page,
)
from src.extraction.consolidation import (
    consolidate_candidates,
    consolidate_document_groups,
)
from src.understanding.document_facts import (
    FactOrigin,
    InvoiceType,
    Placement,
    SemanticRole,
)
from src.understanding.document_grouper import group_document
from src.understanding.evidence import ocr_json_to_page_evidence
from src.understanding.page_classifier import classify_page


# ══════════════════════════════════════════════════════════════════════════
# Helper Builders for Synthetic Reconstructions
# ══════════════════════════════════════════════════════════════════════════

def _make_line(
    line_id: str = "line-1",
    line_num: Optional[int] = 1,
    desc: str = "Consulting Service",
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


# ══════════════════════════════════════════════════════════════════════════
# Test Suite
# ══════════════════════════════════════════════════════════════════════════

class TestReconciliationBasics:
    """Core exact match and discrepancy tests."""

    def test_exact_match(self) -> None:
        """Exact agreement between observed and reconstructed values."""
        recon = _make_reconstruction(
            printed_gross="1000.00",
            printed_subtotal="900.00",
            printed_tax="100.00",
            rec_gross="1000.00",
            rec_subtotal="900.00",
            rec_tax="100.00",
        )
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.MATCH
        assert res.is_match is True
        assert res.is_exact_match is True
        assert res.has_discrepancy is False
        assert len(res.discrepancies) == 0

        assert res.gross_comparison is not None
        assert res.gross_comparison.status == ComponentStatus.MATCH
        assert res.gross_comparison.classification == DiscrepancyClassification.EXACT_MATCH
        assert res.gross_comparison.difference == Decimal("0.00")

    def test_discrepancy_within_tolerance(self) -> None:
        """Discrepancy within explicit tolerance parameter produces MATCH / WITHIN_TOLERANCE."""
        recon = _make_reconstruction(
            printed_gross="1000.01",
            rec_gross="1000.00",
        )
        # Default tolerance is 0.00 -> mismatch
        res_strict = reconcile(recon, tolerance=Decimal("0.00"))
        assert res_strict.status == ReconciliationStatus.MISMATCH
        assert res_strict.gross_comparison.classification == DiscrepancyClassification.NUMERICAL_VARIANCE

        # With explicit tolerance 0.02 -> match within tolerance
        res_tol = reconcile(recon, tolerance=Decimal("0.02"))
        assert res_tol.status == ReconciliationStatus.MATCH
        assert res_tol.is_match is True
        assert res_tol.is_exact_match is False
        assert res_tol.gross_comparison.status == ComponentStatus.MATCH
        assert res_tol.gross_comparison.classification == DiscrepancyClassification.WITHIN_TOLERANCE
        assert res_tol.gross_comparison.difference == Decimal("-0.01")
        assert len(res_tol.discrepancies) == 0

    def test_large_discrepancy_mismatch(self) -> None:
        """Explicit numerical variance outside tolerance produces MISMATCH."""
        recon = _make_reconstruction(
            printed_gross="1000.00",
            rec_gross="1050.00",
        )
        res = reconcile(recon, tolerance=Decimal("0.05"))

        assert res.status == ReconciliationStatus.MISMATCH
        assert res.is_match is False
        assert res.has_discrepancy is True
        assert len(res.discrepancies) == 1

        disc = res.discrepancies[0]
        assert disc.field == "gross_total"
        assert disc.code == "GROSS_TOTAL_MISMATCH"
        assert disc.difference == Decimal("50.00")
        assert disc.document_amount == Decimal("1000.00")
        assert disc.reconstructed_amount == Decimal("1050.00")

    def test_negative_tolerance_raises(self) -> None:
        """Negative tolerance is rejected with ValueError."""
        recon = _make_reconstruction()
        with pytest.raises(ValueError, match="tolerance must be a non-negative Decimal"):
            reconcile(recon, tolerance=Decimal("-0.01"))


class TestMissingAndIncomplete:
    """Handling missing document values, missing ERP values, and partial states."""

    def test_missing_document_gross(self) -> None:
        """Document omitted printed gross total -> status INCOMPLETE."""
        recon = _make_reconstruction(
            printed_gross=None,
            rec_gross="1000.00",
        )
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.INCOMPLETE
        assert res.is_incomplete is True
        assert res.is_match is False
        assert res.gross_comparison is not None
        assert res.gross_comparison.status == ComponentStatus.MISSING_DOCUMENT_VALUE
        assert res.gross_comparison.classification == DiscrepancyClassification.UNAVAILABLE_DOCUMENT_VALUE

    def test_missing_erp_gross(self) -> None:
        """ERP failed to reconstruct gross -> status INCOMPLETE."""
        recon = _make_reconstruction(
            printed_gross="1000.00",
            rec_gross=None,
        )
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.INCOMPLETE
        assert res.is_incomplete is True
        assert res.gross_comparison.status == ComponentStatus.MISSING_ERP_VALUE
        assert res.gross_comparison.classification == DiscrepancyClassification.UNAVAILABLE_ERP_VALUE

    def test_missing_currency(self) -> None:
        """Missing currency flags INCOMPLETE."""
        recon = _make_reconstruction(currency=None)
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.INCOMPLETE
        assert res.is_incomplete is True
        assert any("currency" in note.lower() for note in res.gross_comparison.diagnostic_notes)

    def test_partial_comparison_status(self) -> None:
        """Gross matches, but subtotal and tax were not printed -> MATCH_PARTIAL."""
        recon = _make_reconstruction(
            printed_gross="1000.00",
            printed_subtotal=None,
            printed_tax=None,
            rec_gross="1000.00",
            rec_subtotal="900.00",
            rec_tax="100.00",
        )
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.MATCH_PARTIAL
        assert res.is_match is True
        assert res.is_partial is True
        assert res.subtotal_comparison.status == ComponentStatus.MISSING_DOCUMENT_VALUE
        assert res.tax_comparison.status == ComponentStatus.MISSING_DOCUMENT_VALUE

    def test_missing_line_amount(self) -> None:
        """Document omitted line extension -> line status MISSING_DOCUMENT_VALUE."""
        line = _make_line(obs_amt=None, rec_base="200.00")
        recon = _make_reconstruction(lines=(line,))
        res = reconcile(recon)

        assert len(res.line_comparisons) == 1
        lc = res.line_comparisons[0]
        assert lc.status == ComponentStatus.MISSING_DOCUMENT_VALUE
        assert lc.classification == DiscrepancyClassification.UNAVAILABLE_DOCUMENT_VALUE
        assert lc.difference is None


class TestUpstreamIssuesAndConflicts:
    """Interplay with ERPReconstructionIssue and blocking conditions."""

    def test_blocking_upstream_issue_yields_conflict(self) -> None:
        """Blocking ERP issue yields document-level CONFLICT."""
        issue = ERPReconstructionIssue(
            code="FATAL_INPUT_ERROR",
            severity=ERPSeverity.BLOCKING,
            message="Critical unresolvable input error",
        )
        recon = _make_reconstruction(issues=(issue,))
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.CONFLICT
        assert res.is_conflict is True
        assert res.is_match is False

    def test_warning_upstream_issue_permits_match(self) -> None:
        """Warning ERP issue (e.g. MISSING_QUANTITY) does not artificially force CONFLICT."""
        issue = ERPReconstructionIssue(
            code="MISSING_QUANTITY",
            severity=ERPSeverity.WARNING,
            message="Line missing quantity; ERP base is 0.0",
        )
        recon = _make_reconstruction(issues=(issue,))
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.MATCH
        assert res.is_match is True


class TestSubtotalAndTaxComparisons:
    """Component-level subtotal and tax comparisons."""

    def test_subtotal_mismatch(self) -> None:
        """Subtotal discrepancy generates discrepancy and causes MISMATCH."""
        recon = _make_reconstruction(
            printed_gross="1000.00",
            printed_subtotal="850.00",
            rec_gross="1000.00",
            rec_subtotal="900.00",
        )
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.MISMATCH
        assert res.subtotal_comparison.status == ComponentStatus.MISMATCH
        assert res.subtotal_comparison.difference == Decimal("50.00")
        assert any(d.field == "subtotal" for d in res.discrepancies)

    def test_tax_mismatch(self) -> None:
        """Tax discrepancy generates discrepancy and causes MISMATCH."""
        recon = _make_reconstruction(
            printed_gross="1000.00",
            printed_tax="120.00",
            rec_gross="1000.00",
            rec_tax="100.00",
        )
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.MISMATCH
        assert res.tax_comparison.status == ComponentStatus.MISMATCH
        assert res.tax_comparison.difference == Decimal("-20.00")
        assert any(d.field == "tax_total" for d in res.discrepancies)


class TestNeutralDiagnosticCorrelation:
    """Diagnostic correlation notes without causal attribution."""

    def test_neutral_diagnostic_notes_tax(self) -> None:
        """When gross diff equals tax diff, reports neutral correlation note without asserting causation."""
        recon = _make_reconstruction(
            printed_gross="1000.00",
            rec_gross="1020.00",  # diff = +20
            printed_tax="100.00",
            rec_tax="120.00",    # diff = +20
        )
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.MISMATCH
        notes = res.gross_comparison.diagnostic_notes
        assert any("numerically equals tax component variance" in note for note in notes)
        # Verify classification is strictly neutral NUMERICAL_VARIANCE, not causal TAX_DISCREPANCY
        assert res.gross_comparison.classification == DiscrepancyClassification.NUMERICAL_VARIANCE

    def test_neutral_diagnostic_notes_subtotal(self) -> None:
        """When gross diff equals subtotal diff, reports neutral correlation note."""
        recon = _make_reconstruction(
            printed_gross="1000.00",
            rec_gross="1050.00",      # diff = +50
            printed_subtotal="900.00",
            rec_subtotal="950.00",    # diff = +50
        )
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.MISMATCH
        notes = res.gross_comparison.diagnostic_notes
        assert any("numerically equals subtotal component variance" in note for note in notes)


class TestLineCorrespondenceAndNonPosting:
    """Line matching by source_line_id and isolation of non-posting rows."""

    def test_intra_record_line_correspondence(self) -> None:
        """Line items are evaluated via source_line_id intra-record pairing."""
        line1 = _make_line(line_id="item-A", obs_amt="400.00", rec_base="400.00")
        line2 = _make_line(line_id="item-B", obs_amt="600.00", rec_base="650.00")  # diff +50

        recon = _make_reconstruction(lines=(line1, line2), printed_gross="1050.00", rec_gross="1050.00")
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.MISMATCH
        assert len(res.line_comparisons) == 2
        lc1 = res.line_comparisons[0]
        assert lc1.component_name == "item-A"
        assert lc1.status == ComponentStatus.MATCH

        lc2 = res.line_comparisons[1]
        assert lc2.component_name == "item-B"
        assert lc2.status == ComponentStatus.MISMATCH
        assert lc2.difference == Decimal("50.00")

    def test_non_posting_component_details(self) -> None:
        """Non-posting COMPONENT_DETAIL lines are NOT_COMPARABLE and do not fail payable match."""
        billed_line = _make_line(line_id="item-1", obs_amt="1000.00", rec_base="1000.00", is_posting=True)
        detail_row = _make_line(
            line_id="detail-1",
            desc="Included hardware bundle",
            obs_amt="0.00",
            rec_base=None,
            is_posting=False,
            role=SemanticRole.COMPONENT_DETAIL,
        )

        recon = _make_reconstruction(
            lines=(billed_line, detail_row),
            printed_gross="1000.00",
            rec_gross="1000.00",
        )
        res = reconcile(recon)

        # Document itself matches!
        assert res.status == ReconciliationStatus.MATCH
        assert len(res.line_comparisons) == 2

        lc_detail = res.line_comparisons[1]
        assert lc_detail.status == ComponentStatus.NOT_COMPARABLE
        assert lc_detail.classification == DiscrepancyClassification.NON_POSTING_DETAIL
        assert lc_detail.reconstructed_value is None


class TestSupportingIsolation:
    """Supporting document groups are completely isolated from payable reconciliation."""

    def test_supporting_document_isolation(self) -> None:
        """Supporting group IDs remain references only."""
        recon = _make_reconstruction(
            supporting_group_ids=("group_customs_try_001", "group_packing_002"),
        )
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.MATCH
        assert res.supporting_group_ids == ("group_customs_try_001", "group_packing_002")


class TestCreditAndDebitMemos:
    """Credit and debit memos compared as positive magnitudes per erp contract."""

    def test_credit_memo_positive_magnitude(self) -> None:
        """Credit memo compared as positive numbers without sign-flipping."""
        recon = _make_reconstruction(
            doc_type=InvoiceType.CREDIT_MEMO,
            printed_gross="500.00",
            rec_gross="500.00",
            lines=(_make_line(obs_amt="500.00", rec_base="500.00"),),
        )
        res = reconcile(recon)

        assert res.document_type == InvoiceType.CREDIT_MEMO
        assert res.status == ReconciliationStatus.MATCH
        assert res.gross_comparison.document_value == Decimal("500.00")
        assert res.gross_comparison.reconstructed_value == Decimal("500.00")
        assert res.gross_comparison.difference == Decimal("0.00")

    def test_debit_memo_positive_magnitude(self) -> None:
        """Debit memo compared as positive numbers."""
        recon = _make_reconstruction(
            doc_type=InvoiceType.DEBIT_MEMO,
            printed_gross="250.00",
            rec_gross="250.00",
        )
        res = reconcile(recon)

        assert res.document_type == InvoiceType.DEBIT_MEMO
        assert res.status == ReconciliationStatus.MATCH


class TestNoAccountingRepair:
    """Non-negotiable rule: Validate, do NOT repair or invent document facts."""

    def test_no_accounting_repair(self) -> None:
        """Discrepancy of €50 must NOT generate a balancing discount or charge."""
        recon = _make_reconstruction(
            printed_gross="1000.00",
            rec_gross="1050.00",
        )
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.MISMATCH
        assert res.gross_comparison.difference == Decimal("50.00")

        # Zero synthetic comparisons or balancing facts created
        comp_names = [c.component_name for c in res.all_comparisons]
        assert "balancing_discount" not in comp_names
        assert "balancing_charge" not in comp_names
        assert "tax_adjustment" not in comp_names


class TestProvenanceAndImmutability:
    """Strict provenance separation, input immutability, determinism, and serialization."""

    def test_provenance_separation(self) -> None:
        """Document evidence IDs separated from ERP provenance."""
        recon = _make_reconstruction()
        res = reconcile(recon)

        gc = res.gross_comparison
        assert gc.document_evidence_ids == ("EV_T1",)
        assert gc.document_origin == FactOrigin.OBSERVED
        assert gc.reconstructed_origin == FactOrigin.DERIVED
        assert gc.erp_provenance.get("erp_engine") == "erp.py"

    def test_input_immutability(self) -> None:
        """ERPReconstruction is not mutated during reconciliation."""
        recon = _make_reconstruction()
        recon_copy = copy.deepcopy(recon)

        res = reconcile(recon)

        assert recon.to_dict() == recon_copy.to_dict()

    def test_deterministic_output(self) -> None:
        """Repeated reconciliation produces bit-identical dictionary output."""
        recon = _make_reconstruction()

        res1 = reconcile(recon, tolerance=Decimal("0.01"))
        res2 = reconcile(recon, tolerance=Decimal("0.01"))

        assert res1.to_dict() == res2.to_dict()
        assert res1.to_json() == res2.to_json()

    def test_serialization_round_trip(self) -> None:
        """Round-trip to_dict/from_dict and to_json/from_json."""
        recon = _make_reconstruction()
        res = reconcile(recon)

        data = res.to_dict()
        res_from_dict = ReconciliationResult.from_dict(data)
        assert res_from_dict.to_dict() == data

        json_str = res.to_json()
        res_from_json = ReconciliationResult.from_json(json_str)
        assert res_from_json.to_dict() == data


class TestAntiFormulaDuplication:
    """Monkeypatch test: Proves 9C-3 consumes ERPReconstruction output and does NOT duplicate formulas."""

    def test_anti_formula_duplication_monkeypatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Monkeypatch erp.erp_book to return €9,999.99 and verify reconciliation compares against that value."""
        # Create a real financial structure
        line = NormalizedLine(
            source_line_id="line-1",
            line_number=1,
            description="Widget",
            quantity=Decimal("10"),
            unit_price=Decimal("10.00"),
            amount=Decimal("100.00"),
            currency="EUR",
            semantic_role=SemanticRole.BILLED_LINE,
            evidence_ids=("EV1",),
        )
        pt = NormalizedPrintedTotals(
            gross_total=Decimal("100.00"),
            subtotal=Decimal("100.00"),
        )
        fs = FinancialStructure(
            assembly_id="ASM-MOCK",
            document_id="DOC-MOCK",
            currency="EUR",
            lines=(line,),
            printed_totals=pt,
        )

        # Monkeypatch erp.erp_book in erp module to return an artificial number
        def mock_erp_book(payload: dict) -> dict:
            return {"will_book_gross": 9999.99, "currency": "EUR"}

        monkeypatch.setattr(erp, "erp_book", mock_erp_book)

        # Reconstruct ERP with the mocked gross
        recon = reconstruct_erp(fs)
        assert recon.reconstructed_gross_total == Decimal("9999.99")

        # Reconcile must consume this 9999.99 rather than recalculating 10 * 10 = 100
        res = reconcile(recon)

        assert res.status == ReconciliationStatus.MISMATCH
        assert res.gross_comparison.reconstructed_value == Decimal("9999.99")
        assert res.gross_comparison.difference == Decimal("9899.99")


class TestBatchReconciliation:
    """reconcile_batch processes multiple independent payables."""

    def test_reconcile_batch(self) -> None:
        recon1 = _make_reconstruction(doc_id="DOC-1", printed_gross="100.00", rec_gross="100.00")
        recon2 = _make_reconstruction(doc_id="DOC-2", printed_gross="200.00", rec_gross="250.00")

        results = reconcile_batch([recon1, recon2])

        assert len(results) == 2
        assert results[0].document_id == "DOC-1"
        assert results[0].status == ReconciliationStatus.MATCH
        assert results[1].document_id == "DOC-2"
        assert results[1].status == ReconciliationStatus.MISMATCH


# ══════════════════════════════════════════════════════════════════════════
# Real Corpus End-to-End Verification
# ══════════════════════════════════════════════════════════════════════════

class TestRealCorpusReconciliation:
    """Verify complete pipeline (OCR -> Extraction -> Consolidation -> Assembly -> FS -> ERP -> Reconciliation) on real corpus documents."""

    def test_rc1_inv_01_reconciliation(self) -> None:
        """INV-01: German project management invoice reconciliation."""
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

        assert res.document_id == "INV-01.pdf"
        assert res.currency == "EUR"
        assert res.gross_comparison is not None
        assert res.gross_comparison.document_value == Decimal("438.00")
        assert res.gross_comparison.reconstructed_value is not None

    def test_rc2_hld_01_reconciliation(self) -> None:
        """HLD-01: Thai invoice reconciliation."""
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

        assert res.document_id == "HLD-01.pdf"
        assert res.gross_comparison is not None
        assert res.gross_comparison.reconstructed_value is not None

    def test_rc3_inv_02_reconciliation(self) -> None:
        """INV-02: Estonian multi-page invoice reconciliation."""
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

        assert res.document_id == "INV-02.pdf"
        assert res.currency == "EUR"
        assert len(res.line_comparisons) > 0

    def test_rc4_du_02_reconciliation_isolation(self) -> None:
        """DU-02 Regression: EUR primary payable invoice is reconciled; TRY customs pages remain isolated."""
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

        assert res.currency == "EUR"
        assert len(res.supporting_group_ids) > 0
        assert res.gross_comparison.currency == "EUR"
        # Verify no Turkish Lira contamination in comparisons
        for comp in res.all_comparisons:
            assert comp.currency in ("EUR", None)

    def test_rc5_hld_03_reconciliation(self) -> None:
        """HLD-03: Portuguese invoice reconciliation."""
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

        assert res.document_id == "HLD-03.pdf"
        assert res.currency == "EUR"
        assert res.gross_comparison is not None
