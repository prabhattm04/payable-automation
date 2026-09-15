"""tests/test_qwen_router.py — Tests for Phase 7D Selective Qwen Routing."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.understanding.evidence import (
    Evidence,
    EvidenceSource,
    PageEvidence,
    ocr_json_to_page_evidence,
)
from src.understanding.page_classifier import (
    PageRole,
    PayableRelevance,
    PageUnderstanding,
    classify_page,
)
from src.understanding.document_grouper import (
    DocumentGroup,
    group_document,
)
from src.understanding.qwen_router import (
    RouterDecisionType,
    RoutingDecision,
    execute_vision_escalation,
    route_page,
)
from src.vision.provider import VisionProvider, VisionResponse


def _make_page(
    lines: list[str],
    confidence: float = 0.98,
    doc_id: str = "TEST-DOC",
    page_num: int = 1,
    boxes: list[list[float]] | None = None,
) -> PageEvidence:
    """Helper to build a synthetic PageEvidence with controlled confidence and layout."""
    page_ev = PageEvidence(document_id=doc_id, page_number=page_num, image_path="test_img.png")
    for idx, text in enumerate(lines):
        if boxes and idx < len(boxes):
            bbox = boxes[idx]
        else:
            bbox = [10.0, float(idx * 40), 300.0, float(idx * 40 + 30)]
        ev = Evidence.create(
            content=text,
            document_id=doc_id,
            page_number=page_num,
            source=EvidenceSource.OCR,
            extraction_method="MockOCR",
            confidence=confidence,
            bbox=bbox,
        )
        page_ev.add_item(ev)
    return page_ev


# ══════════════════════════════════════════════════════════════════════════
# 1. Core Routing Scenarios (A through P)
# ══════════════════════════════════════════════════════════════════════════

class TestRoutingScenarios:
    def test_a_clear_high_quality_invoice(self) -> None:
        """A. Clear high-quality invoice with standard layout -> ocr_sufficient."""
        pe = _make_page([
            "COMMERCIAL INVOICE",
            "Invoice No: INV-100",
            "Bill To: Customer Corp",
            "Item: Standard Software License",
            "Total Amount Due: 100.00 EUR",
        ], confidence=0.99)

        und = classify_page(pe)
        decision = route_page(pe, und)

        assert decision.decision == RouterDecisionType.OCR_SUFFICIENT
        assert "ocr_sufficient" in decision.reasons
        assert decision.confidence is None

    def test_b_high_quality_simple_supporting_page(self) -> None:
        """B. High-quality simple supporting page -> ocr_sufficient."""
        pe = _make_page([
            "PACKING LIST",
            "Net Weight: 10 KG",
            "Gross Weight: 12 KG",
            "Box Count: 1",
        ], confidence=0.99)

        und = classify_page(pe)
        decision = route_page(pe, und)

        assert decision.decision == RouterDecisionType.OCR_SUFFICIENT
        assert "ocr_sufficient" in decision.reasons

    def test_c_low_ocr_quality(self) -> None:
        """C. Low OCR quality triggers route_to_vision."""
        pe = _make_page([
            "RECHNUNG",
            "Kunden-Nr: 12",
            "Endbetrag: 500 EUR",
        ], confidence=0.55)

        und = classify_page(pe)
        decision = route_page(pe, und)

        assert decision.decision == RouterDecisionType.ROUTE_TO_VISION
        assert "low_ocr_quality" in decision.reasons
        assert decision.signals["is_low_ocr_quality"] is True

    def test_d_ambiguous_page_classification(self) -> None:
        """D. Ambiguous page classification triggers route_to_vision."""
        pe = _make_page([
            "Confidential Notice: This memorandum discusses warehouse policy.",
            "Contact operations for details.",
        ], confidence=0.98)

        und = classify_page(pe)
        assert und.page_role == PageRole.UNKNOWN
        decision = route_page(pe, und)

        assert decision.decision == RouterDecisionType.ROUTE_TO_VISION
        assert "classification_ambiguous" in decision.reasons

    def test_e_complex_table_layout(self) -> None:
        """E. Complex dense table layout triggers route_to_vision."""
        # Create 110 blocks across multiple distinct columns and rows
        lines: list[str] = []
        boxes: list[list[float]] = []
        for r in range(25):
            for c in range(5):
                lines.append(f"Cell_{r}_{c}")
                boxes.append([float(c * 200 + 10), float(r * 30 + 10), float(c * 200 + 150), float(r * 30 + 35)])

        pe = _make_page(lines, confidence=0.99, boxes=boxes)
        und = classify_page(pe)
        decision = route_page(pe, und)

        assert decision.decision == RouterDecisionType.ROUTE_TO_VISION
        assert "complex_layout" in decision.reasons

    def test_f_ambiguous_grouping(self) -> None:
        """F. Ambiguous grouping (e.g. orphaned continuation page) triggers route_to_vision."""
        pe = _make_page([
            "Page 2 of 4",
            "Line item continued from previous",
            "Carry forward balance: 250 EUR",
        ], confidence=0.98)

        und = classify_page(pe)
        # Orphaned group: continuation page isolated in a 1-page group
        isolated_grp = DocumentGroup(
            group_id="DOC:group:p2",
            document_id="DOC",
            page_numbers=[2],
        )

        decision = route_page(pe, und, group=isolated_grp)
        assert decision.decision == RouterDecisionType.ROUTE_TO_VISION
        assert "grouping_ambiguous" in decision.reasons

    def test_g_multiple_financial_structure_signals(self) -> None:
        """G. Multiple competing financial adjustments (discounts + excise/charges + taxes) -> route_to_vision."""
        pe = _make_page([
            "INVOICE",
            "Subtotal: 100.00 EUR",
            "Special promotional discount: 20.00-",
            "Excise duty / IEC charge: 15.00",
            "Withholding tax deduction 3%: 2.50-",
            "VAT 19%: 18.00",
            "Total Amount Due: 110.50 EUR",
        ], confidence=0.99)

        und = classify_page(pe)
        decision = route_page(pe, und)

        assert decision.decision == RouterDecisionType.ROUTE_TO_VISION
        assert "complex_financial_structure" in decision.reasons

    def test_h_language_alone_does_not_route(self) -> None:
        """H. Language alone (Thai or Portuguese) does NOT automatically route to Qwen."""
        pe_thai = _make_page([
            "ใบแจ้งหนี้",  # Invoice
            "Invoice No: TH-99",
            "Total Payment: 1,000.00 THB",
        ], confidence=0.98)
        und_thai = classify_page(pe_thai)
        dec_thai = route_page(pe_thai, und_thai)
        assert dec_thai.decision == RouterDecisionType.OCR_SUFFICIENT

        pe_pt = _make_page([
            "FACTURA",
            "N° da Factura: PT-1234",
            "Total da factura: 50.00 EUR",
        ], confidence=0.98)
        und_pt = classify_page(pe_pt)
        dec_pt = route_page(pe_pt, und_pt)
        assert dec_pt.decision == RouterDecisionType.OCR_SUFFICIENT

    def test_i_multipage_alone_does_not_route(self) -> None:
        """I. Multi-page context alone does NOT force Qwen routing."""
        pe = _make_page([
            "Page 1 of 2",
            "INVOICE",
            "Invoice No: 101",
            "Item 1: 50 EUR",
            "Total Amount Due: 50.00 EUR",
        ], confidence=0.99)
        und = classify_page(pe)
        grp = DocumentGroup(group_id="D:group:p1", document_id="D", page_numbers=[1, 2])
        decision = route_page(pe, und, group=grp)
        assert decision.decision == RouterDecisionType.OCR_SUFFICIENT

    def test_j_high_ocr_confidence_and_clear_classification(self) -> None:
        """J. High OCR confidence + clear classification -> ocr_sufficient."""
        pe = _make_page([
            "RECHNUNG",
            "Rechnungsdatum: 01.02.2026",
            "Gesamtsumme: 200,00 EUR",
            "Endbetrag: 200,00 EUR",
        ], confidence=0.99)
        und = classify_page(pe)
        decision = route_page(pe, und)
        assert decision.decision == RouterDecisionType.OCR_SUFFICIENT

    def test_k_evidence_id_traceability(self) -> None:
        """K. Evidence IDs in RoutingDecision trace directly to page items."""
        pe = _make_page([
            "INVOICE",
            "Promotional Discount: 10-",
            "Withholding Tax: 5-",
            "Excise Duty IEC: 2",
            "VAT: 15",
        ], confidence=0.99)
        und = classify_page(pe)
        decision = route_page(pe, und)

        assert len(decision.evidence_ids) > 0
        valid_ids = {item.evidence_id for item in pe.items}
        for eid in decision.evidence_ids:
            assert eid in valid_ids

    def test_l_deterministic_output(self) -> None:
        """L. Identical inputs yield identical decisions, reasons, and signals."""
        pe = _make_page(["TAX INVOICE", "Total: 100 EUR"], confidence=0.98)
        und = classify_page(pe)

        dec1 = route_page(pe, und)
        dec2 = route_page(pe, und)

        assert dec1.decision == dec2.decision
        assert dec1.reasons == dec2.reasons
        assert dec1.signals == dec2.signals

    def test_m_document_id_invariance(self) -> None:
        """M. Changing document_id does not change routing decisions."""
        lines = ["TAX INVOICE", "Invoice No: 12", "Total Amount Due: 100 EUR"]
        pe_a = _make_page(lines, confidence=0.98, doc_id="DOC_A")
        pe_b = _make_page(lines, confidence=0.98, doc_id="DOC_B_RANDOM_UUID")

        dec_a = route_page(pe_a, classify_page(pe_a))
        dec_b = route_page(pe_b, classify_page(pe_b))

        assert dec_a.decision == dec_b.decision
        assert dec_a.reasons == dec_b.reasons

    def test_n_no_fabricated_confidence(self) -> None:
        """N. Deterministic routing confidence is strictly None."""
        pe = _make_page(["COMMERCIAL INVOICE", "Total: 100 EUR"], confidence=0.98)
        dec = route_page(pe, classify_page(pe))
        assert dec.confidence is None

    def test_o_serialization_round_trip(self) -> None:
        """O. RoutingDecision round trips cleanly through JSON."""
        dec = RoutingDecision(
            document_id="DOC-1",
            page_number=1,
            decision=RouterDecisionType.ROUTE_TO_VISION,
            reasons=["low_ocr_quality"],
            signals={"ocr_mean_confidence": 0.65},
            evidence_ids=["ev_1"],
            confidence=None,
            provenance={"router": "DeterministicQwenRouter"},
        )
        json_str = dec.to_json()
        loaded = RoutingDecision.from_json(json_str)

        assert loaded.document_id == dec.document_id
        assert loaded.page_number == dec.page_number
        assert loaded.decision == dec.decision
        assert loaded.reasons == dec.reasons
        assert loaded.confidence is None

    def test_p_vision_runner_mock_integration(self) -> None:
        """P. execute_vision_escalation converts Qwen VisionResponse to Phase 7A Evidence with confidence=None."""
        pe = _make_page(["Blurry text"], confidence=0.5, doc_id="TEST-VISION", page_num=1)
        und = classify_page(pe)
        decision = route_page(pe, und)
        assert decision.decision == RouterDecisionType.ROUTE_TO_VISION

        mock_provider = MagicMock(spec=VisionProvider)
        mock_provider.analyze_image.return_value = VisionResponse(
            success=True,
            content="Visual Analysis: Commercial invoice for 100 EUR",
            model="qwen3-vl-plus",
            provider="Puter",
            latency_seconds=1.23,
        )

        ev = execute_vision_escalation(decision, pe, provider=mock_provider, image_path="test_img.png")
        assert ev is not None
        assert ev.source == EvidenceSource.VISION
        assert ev.confidence is None  # Never fabricated
        assert ev.content == "Visual Analysis: Commercial invoice for 100 EUR"
        assert ev.extraction_method == "Puter/qwen3-vl-plus"
        assert mock_provider.analyze_image.called

    def test_single_weak_signal_does_not_over_route(self) -> None:
        """Design Adjustment 6: A single weak/irrelevant signal does not trigger escalation."""
        pe = _make_page([
            "COMMERCIAL INVOICE",
            "Invoice No: INV-100",
            "Bill To: Customer Corp",
            "Item: Standard Software License",
            "Shipping terms: Standard delivery",  # mentions delivery/shipping casually
            "Total Amount Due: 100.00 EUR",
        ], confidence=0.99)
        decision = route_page(pe, classify_page(pe))
        assert decision.decision == RouterDecisionType.OCR_SUFFICIENT

    def test_high_ocr_confidence_does_not_override_layout_complexity(self) -> None:
        """Design Adjustment 2 & 7: High OCR confidence does NOT suppress layout complexity."""
        # 120 blocks in complex 5-column table layout with 1.0 confidence
        lines: list[str] = []
        boxes: list[list[float]] = []
        for r in range(24):
            for c in range(5):
                lines.append(f"Grid_Item_{r}_{c}")
                boxes.append([float(c * 150 + 10), float(r * 25 + 10), float(c * 150 + 120), float(r * 25 + 25)])

        pe = _make_page(lines, confidence=1.0, boxes=boxes)
        decision = route_page(pe, classify_page(pe))

        # Even with confidence=1.0, complex table layout forces routing!
        assert decision.decision == RouterDecisionType.ROUTE_TO_VISION
        assert "complex_layout" in decision.reasons


# ══════════════════════════════════════════════════════════════════════════
# 2. Representative Real Artifact Routing Tests
# ══════════════════════════════════════════════════════════════════════════

class TestRealArtifactRouting:
    def test_inv_01_page_1_is_ocr_sufficient(self) -> None:
        """INV-01 (clean single-page invoice, high OCR confidence, clear layout) -> ocr_sufficient."""
        p = Path("artifacts/ocr/INV-01/page_001.json")
        if not p.exists():
            pytest.skip("INV-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        und = classify_page(pe)
        decision = route_page(pe, und)

        assert decision.decision == RouterDecisionType.OCR_SUFFICIENT
        assert "ocr_sufficient" in decision.reasons

    def test_du_02_page_5_is_ocr_sufficient(self) -> None:
        """DU-02 page 5 (simple clean packing/weight sheet, high confidence) -> ocr_sufficient."""
        p = Path("artifacts/ocr/DU-02/page_005.json")
        if not p.exists():
            pytest.skip("DU-02 page 5 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        und = classify_page(pe)
        decision = route_page(pe, und)

        assert decision.decision == RouterDecisionType.OCR_SUFFICIENT
        assert "ocr_sufficient" in decision.reasons

    def test_du_02_page_1_routed_due_to_layout_complexity(self) -> None:
        """DU-02 page 1 (201 text blocks, dense HTS multi-column schedule) -> route_to_vision."""
        p = Path("artifacts/ocr/DU-02/page_001.json")
        if not p.exists():
            pytest.skip("DU-02 page 1 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        und = classify_page(pe)
        decision = route_page(pe, und)

        assert decision.decision == RouterDecisionType.ROUTE_TO_VISION
        assert "complex_layout" in decision.reasons

    def test_hld_03_page_1_routed_due_to_financial_complexity(self) -> None:
        """HLD-03 page 1 (promotional discounts + IEC excise duty + VAT) -> route_to_vision."""
        p = Path("artifacts/ocr/HLD-03/page_001.json")
        if not p.exists():
            pytest.skip("HLD-03 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        und = classify_page(pe)
        decision = route_page(pe, und)

        assert decision.decision == RouterDecisionType.ROUTE_TO_VISION
        assert "complex_financial_structure" in decision.reasons
