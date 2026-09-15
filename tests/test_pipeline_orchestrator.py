"""tests/test_pipeline_orchestrator.py — Unit test suite for End-to-End Pipeline Orchestrator.

Validates the 14 non-negotiable orchestrator requirements:
1. One PDF produces exactly one output JSON file.
2. Deterministic ordering: repeated runs produce byte-equivalent envelopes.
3. SAFE_TO_AUTODRAFT decision reaches build_payable and emits payables[].
4. HOLD_FOR_REVIEW produces empty payables and empty declined.
5. UNSAFE_TO_AUTODRAFT produces empty payables and empty declined.
6. NOT_PAYABLE produces declined entry via build_declined_entry.
7. DU-02 isolation: secondary TRY amounts never contaminate primary EUR payable.
8. Multiple assemblies: multiple safe assemblies emit multiple payables.
9. One-document failure isolation: exception in document N does not stop document N+1.
10. No ERP bypass: orchestrator delegates to reconstruct_erp() without local formula calculation.
11. No filename semantics: pipeline behavior is invariant to filename strings.
12. Output schema: all emitted envelopes conform to AUTODRAFT_SCHEMA / README contract.
13. Vision failure graceful fallback: failing Vision provider does not abort candidate extraction.
14. Failure accounting vs safe decision: unexpected failures increment processing_failures,
    not successful_executions, and emit empty fallback envelope.
"""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import pytest

from src.pdf.loader import PdfFile
from src.matching.store import MasterDataStore
from src.matching.match_models import MasterMatchResult, MatchStatus
from src.accounting.decision import PayableDecision, PayableDecisionStatus
from src.accounting.financial_structure import FinancialStructure
from src.accounting.erp_reconstruction import ERPReconstruction
from src.accounting.payable_builder import (
    PayableContractError,
    validate_file_output,
)
from src.vision.provider import VisionProvider, VisionResponse
from run_pipeline import (
    MasterMatchers,
    PipelineRunSummary,
    acquire_document_evidence,
    process_document,
    run_pipeline,
)


MASTER_DATA_DIR = Path("master_data")
ARTIFACTS_DIR = Path("artifacts")


@pytest.fixture(scope="module")
def matchers() -> MasterMatchers:
    return MasterMatchers.load(MASTER_DATA_DIR)


# ══════════════════════════════════════════════════════════════════════════
# Synthetic Test Fixtures
# ══════════════════════════════════════════════════════════════════════════

def _create_mock_pdf_file(name: str = "INV-01.pdf", path: Optional[Path] = None) -> PdfFile:
    p = path or Path("documents") / name
    return PdfFile(
        path=p,
        relative_path=name,
        filename=name,
        stem=Path(name).stem,
        file_size_bytes=1024,
    )


def _make_valid_payable(inv_num: str = "INV-001", gross: str = "100.00") -> Dict[str, Any]:
    return {
        "invoice_number": inv_num,
        "invoice_date": "2026-02-02",
        "due_date": "2026-02-12",
        "invoice_type": "INVOICE",
        "currency": "EUR",
        "supplier": {"name": "Test Supplier", "supplier_id": "", "address": "", "vat_id": ""},
        "buyer": {"company_code": "", "business_unit_code": "", "location_code": ""},
        "payment_term_id": "",
        "po_number": "",
        "po_id": "",
        "gross_total": gross,
        "subtotal": gross,
        "total_tax_amount": "0.00",
        "discount_amount": "",
        "freight_charges": "",
        "insurance_charges": "",
        "extra_charges": "",
        "excise_duties": "",
        "taxes": [],
        "line_items": [{
            "description": "Item 1",
            "item_type": "SERVICE",
            "uom": "Hr",
            "quantity": "1",
            "unit_price": gross,
            "total": gross,
            "discount": "",
            "discount_percentage": "",
            "tax_rate": "",
            "tax_amount": "",
            "taxes": [],
        }],
    }


# ══════════════════════════════════════════════════════════════════════════
# Test 1 — One PDF Produces One Output
# ══════════════════════════════════════════════════════════════════════════

def test_1_one_pdf_produces_one_output(matchers: MasterMatchers, tmp_path: Path):
    """Verify that processing a valid PDF produces exactly one output JSON file."""
    pdf = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    if not pdf.path.exists():
        pytest.skip("documents/INV-01.pdf not found")

    res = process_document(
        pdf_file=pdf,
        matchers=matchers,
        output_dir=tmp_path,
        artifacts_dir=ARTIFACTS_DIR,
    )

    assert res.success is True
    assert res.output_path is not None
    assert res.output_path.exists()
    assert res.output_path.name == "INV-01.json"

    # Verify JSON content envelope
    with open(res.output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["file"] == "INV-01.pdf"
    assert isinstance(data["payables"], list)
    assert isinstance(data["declined"], list)


# ══════════════════════════════════════════════════════════════════════════
# Test 2 — Deterministic Ordering
# ══════════════════════════════════════════════════════════════════════════

def test_2_deterministic_ordering(matchers: MasterMatchers, tmp_path: Path):
    """Verify repeated runs on the same document yield byte-equivalent envelope outputs."""
    pdf = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    if not pdf.path.exists():
        pytest.skip("documents/INV-01.pdf not found")

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    res1 = process_document(pdf, matchers, output_dir=out1, artifacts_dir=ARTIFACTS_DIR)
    res2 = process_document(pdf, matchers, output_dir=out2, artifacts_dir=ARTIFACTS_DIR)

    assert res1.output_path is not None and res2.output_path is not None
    content1 = res1.output_path.read_text(encoding="utf-8")
    content2 = res2.output_path.read_text(encoding="utf-8")
    assert content1 == content2


# ══════════════════════════════════════════════════════════════════════════
# Test 3 — SAFE Payable Emits Payable
# ══════════════════════════════════════════════════════════════════════════

def test_3_safe_payable(matchers: MasterMatchers, tmp_path: Path):
    """Verify that an approved SAFE_TO_AUTODRAFT decision reaches build_payable."""
    pdf = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    if not pdf.path.exists():
        pytest.skip("documents/INV-01.pdf not found")

    safe_decision = PayableDecision(
        assembly_id="asm-001",
        document_id="INV-01.pdf",
        status=PayableDecisionStatus.SAFE_TO_AUTODRAFT,
        primary_reason="All checks passed",
    )

    with patch("run_pipeline.evaluate_payable_safety", return_value=safe_decision):
        with patch("run_pipeline.build_payable") as mock_build_payable:
            mock_build_payable.return_value = {
                "invoice_number": "852566",
                "invoice_date": "2026-02-02",
                "due_date": "2026-02-12",
                "invoice_type": "INVOICE",
                "currency": "EUR",
                "supplier": {"name": "Test", "supplier_id": "", "address": "", "vat_id": ""},
                "buyer": {"company_code": "", "business_unit_code": "", "location_code": ""},
                "payment_term_id": "",
                "po_number": "",
                "po_id": "",
                "gross_total": "438.00",
                "subtotal": "438.00",
                "total_tax_amount": "0.00",
                "discount_amount": "",
                "freight_charges": "",
                "insurance_charges": "",
                "extra_charges": "",
                "excise_duties": "",
                "taxes": [],
                "line_items": [{
                    "description": "Item 1",
                    "item_type": "SERVICE",
                    "uom": "Hr",
                    "quantity": "4",
                    "unit_price": "73.00",
                    "total": "292.00",
                    "discount": "",
                    "discount_percentage": "",
                    "tax_rate": "",
                    "tax_amount": "",
                    "taxes": [],
                }],
            }
            res = process_document(pdf, matchers, output_dir=tmp_path, artifacts_dir=ARTIFACTS_DIR)

            assert mock_build_payable.called
            assert res.payables_count == 1
            assert res.output_path is not None
            with open(res.output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert len(data["payables"]) == 1
            assert data["payables"][0]["invoice_number"] == "852566"


# ══════════════════════════════════════════════════════════════════════════
# Test 4 — HOLD Produces No Payable
# ══════════════════════════════════════════════════════════════════════════

def test_4_hold(matchers: MasterMatchers, tmp_path: Path):
    """Verify that HOLD_FOR_REVIEW never produces a payable and never enters declined."""
    pdf = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    if not pdf.path.exists():
        pytest.skip("documents/INV-01.pdf not found")

    hold_decision = PayableDecision(
        assembly_id="asm-001",
        document_id="INV-01.pdf",
        status=PayableDecisionStatus.HOLD_FOR_REVIEW,
        primary_reason="Manual review required",
    )

    with patch("run_pipeline.evaluate_payable_safety", return_value=hold_decision):
        res = process_document(pdf, matchers, output_dir=tmp_path, artifacts_dir=ARTIFACTS_DIR)
        assert res.payables_count == 0
        assert res.declined_count == 0
        with open(res.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["payables"] == []
        assert data["declined"] == []


# ══════════════════════════════════════════════════════════════════════════
# Test 5 — UNSAFE Produces No Payable
# ══════════════════════════════════════════════════════════════════════════

def test_5_unsafe(matchers: MasterMatchers, tmp_path: Path):
    """Verify that UNSAFE_TO_AUTODRAFT produces empty payables and empty declined."""
    pdf = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    if not pdf.path.exists():
        pytest.skip("documents/INV-01.pdf not found")

    res = process_document(pdf, matchers, output_dir=tmp_path, artifacts_dir=ARTIFACTS_DIR)
    assert "UNSAFE_TO_AUTODRAFT" in res.decisions
    assert res.payables_count == 0
    assert res.declined_count == 0
    with open(res.output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["payables"] == []
    assert data["declined"] == []


# ══════════════════════════════════════════════════════════════════════════
# Test 6 — NOT_PAYABLE Produces Declined Entry
# ══════════════════════════════════════════════════════════════════════════

def test_6_not_payable(matchers: MasterMatchers, tmp_path: Path):
    """Verify that NOT_PAYABLE produces a declined entry via build_declined_entry."""
    pdf = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    if not pdf.path.exists():
        pytest.skip("documents/INV-01.pdf not found")

    declined_decision = PayableDecision(
        assembly_id="asm-001",
        document_id="INV-01.pdf",
        status=PayableDecisionStatus.NOT_PAYABLE,
        primary_reason="Purchase order document is not payable",
        decline_doc_type="PURCHASE_ORDER",
        decline_reason="Purchase order document is not payable",
    )

    with patch("run_pipeline.evaluate_payable_safety", return_value=declined_decision):
        res = process_document(pdf, matchers, output_dir=tmp_path, artifacts_dir=ARTIFACTS_DIR)
        assert res.payables_count == 0
        assert res.declined_count == 1
        with open(res.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["payables"] == []
        assert len(data["declined"]) == 1
        assert data["declined"][0]["doc_type"] == "PURCHASE_ORDER"
        assert "not payable" in data["declined"][0]["reason"]


# ══════════════════════════════════════════════════════════════════════════
# Test 7 — DU-02 Isolation
# ══════════════════════════════════════════════════════════════════════════

def test_7_du_02_isolation(matchers: MasterMatchers, tmp_path: Path):
    """Verify that supporting TRY documents in DU-02 never contaminate EUR payable."""
    pdf = _create_mock_pdf_file("DU-02.pdf", Path("documents/DU-02.pdf"))
    if not pdf.path.exists():
        pytest.skip("documents/DU-02.pdf not found")

    res = process_document(pdf, matchers, output_dir=tmp_path, artifacts_dir=ARTIFACTS_DIR)
    assert res.success is True
    # DU-02 evaluates to HOLD_FOR_REVIEW
    assert "HOLD_FOR_REVIEW" in res.decisions
    assert res.payables_count == 0

    with open(res.output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["file"] == "DU-02.pdf"
    assert data["payables"] == []


# ══════════════════════════════════════════════════════════════════════════
# Test 8 — Multiple Assemblies
# ══════════════════════════════════════════════════════════════════════════

def test_8_multiple_assemblies(matchers: MasterMatchers, tmp_path: Path):
    """Verify that multiple safe assemblies produce multiple payables in the same file envelope."""
    pdf = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    if not pdf.path.exists():
        pytest.skip("documents/INV-01.pdf not found")

    from src.extraction.financial_assembly import assemble_financial_documents as real_assemble
    with patch("run_pipeline.assemble_financial_documents") as mock_assemble:
        mock_assemble.side_effect = lambda facts_list, groups: real_assemble(facts_list, groups) * 2

        safe_decision = PayableDecision(
            assembly_id="asm-x",
            document_id="INV-01.pdf",
            status=PayableDecisionStatus.SAFE_TO_AUTODRAFT,
            primary_reason="Safe",
        )

        with patch("run_pipeline.evaluate_payable_safety", return_value=safe_decision):
            with patch("run_pipeline.build_payable", side_effect=[
                _make_valid_payable("INV-101", "100.00"),
                _make_valid_payable("INV-102", "200.00"),
            ]):
                res = process_document(pdf, matchers, output_dir=tmp_path, artifacts_dir=ARTIFACTS_DIR)
                assert res.assembly_count == 2
                assert res.payables_count == 2
                with open(res.output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                assert len(data["payables"]) == 2
                assert data["payables"][0]["invoice_number"] == "INV-101"
                assert data["payables"][1]["invoice_number"] == "INV-102"


# ══════════════════════════════════════════════════════════════════════════
# Test 9 — One-Document Failure Isolation
# ══════════════════════════════════════════════════════════════════════════

def test_9_one_document_failure_isolation(matchers: MasterMatchers, tmp_path: Path):
    """Verify that an exception in document N does not halt processing of document N+1."""
    pdf1 = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    pdf2 = _create_mock_pdf_file("INV-02.pdf", Path("documents/INV-02.pdf"))
    if not pdf1.path.exists() or not pdf2.path.exists():
        pytest.skip("documents missing")

    call_count = 0

    def mock_classify(pe):
        nonlocal call_count
        call_count += 1
        if "INV-01" in pe.document_id:
            raise RuntimeError("Simulated crash on INV-01")
        from src.understanding.page_classifier import classify_page as real_classify
        return real_classify(pe)

    with patch("run_pipeline.classify_page", side_effect=mock_classify):
        res1 = process_document(pdf1, matchers, output_dir=tmp_path, artifacts_dir=ARTIFACTS_DIR)
        res2 = process_document(pdf2, matchers, output_dir=tmp_path, artifacts_dir=ARTIFACTS_DIR)

    assert res1.success is False
    assert "Simulated crash" in (res1.error or "")
    assert res1.output_path is not None and res1.output_path.exists()

    assert res2.success is True
    assert res2.output_path is not None and res2.output_path.exists()


# ══════════════════════════════════════════════════════════════════════════
# Test 10 — No ERP Bypass
# ══════════════════════════════════════════════════════════════════════════

def test_10_no_erp_bypass(matchers: MasterMatchers, tmp_path: Path):
    """Verify that reconstruct_erp() is invoked rather than computing ERP locally."""
    pdf = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    if not pdf.path.exists():
        pytest.skip("documents/INV-01.pdf not found")

    with patch("run_pipeline.reconstruct_erp") as mock_reconstruct:
        from src.accounting.erp_reconstruction import ERPReconstruction, reconstruct_erp as real_reconstruct
        def side_effect(struct):
            return real_reconstruct(struct)
        mock_reconstruct.side_effect = side_effect

        res = process_document(pdf, matchers, output_dir=tmp_path, artifacts_dir=ARTIFACTS_DIR)
        assert mock_reconstruct.called
        assert res.success is True


# ══════════════════════════════════════════════════════════════════════════
# Test 11 — No Filename Semantics
# ══════════════════════════════════════════════════════════════════════════

def test_11_no_filename_semantics(matchers: MasterMatchers, tmp_path: Path):
    """Verify that renaming a file does not change its classification or grouping."""
    pdf_orig = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    if not pdf_orig.path.exists():
        pytest.skip("documents/INV-01.pdf not found")

    pdf_renamed = PdfFile(
        path=pdf_orig.path,
        relative_path="NON_INVOICE_XYZ.pdf",
        filename="NON_INVOICE_XYZ.pdf",
        stem="INV-01",  # pointing to same OCR cache
        file_size_bytes=pdf_orig.file_size_bytes,
    )

    res_orig = process_document(pdf_orig, matchers, output_dir=tmp_path / "orig", artifacts_dir=ARTIFACTS_DIR)
    res_renamed = process_document(pdf_renamed, matchers, output_dir=tmp_path / "renamed", artifacts_dir=ARTIFACTS_DIR)

    assert res_orig.decisions == res_renamed.decisions
    assert res_orig.assembly_count == res_renamed.assembly_count


# ══════════════════════════════════════════════════════════════════════════
# Test 12 — Output Schema
# ══════════════════════════════════════════════════════════════════════════

def test_12_output_schema(matchers: MasterMatchers, tmp_path: Path):
    """Verify that emitted envelopes pass validate_file_output contract."""
    pdf = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    if not pdf.path.exists():
        pytest.skip("documents/INV-01.pdf not found")

    res = process_document(pdf, matchers, output_dir=tmp_path, artifacts_dir=ARTIFACTS_DIR)
    assert res.output_path is not None
    data = json.loads(res.output_path.read_text(encoding="utf-8"))
    errors = validate_file_output(data)
    assert not errors, f"Envelope validation errors: {errors}"


# ══════════════════════════════════════════════════════════════════════════
# Test 13 — Vision Graceful Fallback
# ══════════════════════════════════════════════════════════════════════════

def test_13_vision_graceful_fallback(matchers: MasterMatchers, tmp_path: Path):
    """Verify that a failing VisionProvider gracefully falls back to OCR candidates."""
    pdf = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))
    if not pdf.path.exists():
        pytest.skip("documents/INV-01.pdf not found")

    # Mock route_page to demand ROUTE_TO_VISION
    from src.understanding.qwen_router import RoutingDecision, RouterDecisionType
    mock_routing_decision = RoutingDecision(
        document_id="INV-01.pdf",
        page_number=1,
        decision=RouterDecisionType.ROUTE_TO_VISION,
        reasons=["test_forced_vision"],
        signals={},
    )

    # Broken mock VisionProvider
    failing_provider = MagicMock(spec=VisionProvider)
    failing_provider.analyze_image.side_effect = RuntimeError("Network timeout connecting to Vision API")

    with patch("run_pipeline.route_page", return_value=mock_routing_decision):
        res = process_document(
            pdf_file=pdf,
            matchers=matchers,
            output_dir=tmp_path,
            artifacts_dir=ARTIFACTS_DIR,
            vision_provider_getter=lambda: failing_provider,
        )

    # Extraction must survive the vision failure and complete successfully
    assert res.success is True
    assert res.vision_escalation_count >= 1
    assert res.output_path is not None and res.output_path.exists()


# ══════════════════════════════════════════════════════════════════════════
# Test 14 — Failure Accounting vs Safe Decision
# ══════════════════════════════════════════════════════════════════════════

def test_14_failure_accounting_vs_safe_decision(matchers: MasterMatchers, tmp_path: Path):
    """Verify that unexpected failures increment processing_failures and are not counted as safe."""
    pdf_fail = _create_mock_pdf_file("FAIL-01.pdf", Path("documents/FAIL-01.pdf"))
    pdf_ok = _create_mock_pdf_file("INV-01.pdf", Path("documents/INV-01.pdf"))

    with patch("run_pipeline.discover_pdfs", return_value=[pdf_fail, pdf_ok]):
        with patch("run_pipeline.acquire_document_evidence") as mock_acquire:
            def side_acquire(pdf, artifacts_dir):
                if pdf.filename == "FAIL-01.pdf":
                    raise ValueError("Simulated unexpected crash on FAIL-01")
                return acquire_document_evidence(pdf, artifacts_dir)
            mock_acquire.side_effect = side_acquire

            summary = run_pipeline(
                documents_dir="documents",
                output_dir=tmp_path,
                master_data_dir=MASTER_DATA_DIR,
                artifacts_dir=ARTIFACTS_DIR,
            )

    assert summary.total_documents == 2
    assert summary.processing_failures == 1
    assert summary.successful_executions == 1
    # Both output files were written (one fallback, one real)
    assert summary.output_files_written == 2
    assert (tmp_path / "FAIL-01.json").exists()
    assert (tmp_path / "INV-01.json").exists()

    # The failed document envelope is empty
    fail_data = json.loads((tmp_path / "FAIL-01.json").read_text(encoding="utf-8"))
    assert fail_data["payables"] == []
    assert fail_data["declined"] == []
