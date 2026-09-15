"""tests/test_page_classifier.py — Tests for Phase 7B Page-Level Understanding."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.understanding.evidence import (
    Evidence,
    EvidenceProvenance,
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


def _make_page(
    lines: list[str],
    doc_id: str = "TEST-DOC",
    page_num: int = 1,
) -> PageEvidence:
    """Helper to build a synthetic PageEvidence with realistic Evidence items."""
    page_ev = PageEvidence(document_id=doc_id, page_number=page_num)
    for idx, text in enumerate(lines):
        ev = Evidence.create(
            content=text,
            document_id=doc_id,
            page_number=page_num,
            source=EvidenceSource.OCR,
            extraction_method="MockOCR",
            confidence=0.98,
            bbox=[10.0, float(idx * 50), 400.0, float(idx * 50 + 40)],
        )
        page_ev.add_item(ev)
    return page_ev


# ══════════════════════════════════════════════════════════════════════════
# 1. Real Artifact Representative Tests (INV-01, HLD-01, DU-02 p1, DU-02 p5, HLD-03)
# ══════════════════════════════════════════════════════════════════════════

class TestRealArtifactClassification:
    def test_inv_01_page_1_is_invoice_payable_candidate(self) -> None:
        """INV-01 page 1 (German invoice) is classified as invoice and payable_candidate."""
        p = Path("artifacts/ocr/INV-01/page_001.json")
        if not p.exists():
            pytest.skip("INV-01 OCR artifact not found")
        page_ev = ocr_json_to_page_evidence(p)
        result = classify_page(page_ev)

        assert result.page_role == PageRole.INVOICE
        assert result.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE
        assert result.confidence is None  # Never fabricated
        assert len(result.evidence_ids) > 0
        assert any("invoice" in r.lower() for r in result.decision_reasons)
        assert any("payable" in r.lower() for r in result.decision_reasons)

    def test_hld_01_page_1_is_invoice_payable_candidate(self) -> None:
        """HLD-01 page 1 (Thai invoice with VAT & withholding) is invoice + payable_candidate."""
        p = Path("artifacts/ocr/HLD-01/page_001.json")
        if not p.exists():
            pytest.skip("HLD-01 OCR artifact not found")
        page_ev = ocr_json_to_page_evidence(p)
        result = classify_page(page_ev)

        assert result.page_role == PageRole.INVOICE
        assert result.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE
        assert result.confidence is None
        assert len(result.evidence_ids) > 0

    def test_du_02_page_1_is_invoice_payable_candidate(self) -> None:
        """DU-02 page 1 (Customs Consolidated Invoice) is invoice + payable_candidate."""
        p = Path("artifacts/ocr/DU-02/page_001.json")
        if not p.exists():
            pytest.skip("DU-02 page 1 OCR artifact not found")
        page_ev = ocr_json_to_page_evidence(p)
        result = classify_page(page_ev)

        assert result.page_role == PageRole.INVOICE
        assert result.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE
        assert result.confidence is None
        assert len(result.evidence_ids) > 0

    def test_du_02_page_5_is_supporting_document(self) -> None:
        """DU-02 page 5 (weights/dimensions/boxes, no totals) is supporting_document + supporting."""
        p = Path("artifacts/ocr/DU-02/page_005.json")
        if not p.exists():
            pytest.skip("DU-02 page 5 OCR artifact not found")
        page_ev = ocr_json_to_page_evidence(p)
        result = classify_page(page_ev)

        assert result.page_role == PageRole.SUPPORTING_DOCUMENT
        assert result.payable_relevance == PayableRelevance.SUPPORTING
        assert result.confidence is None
        assert len(result.evidence_ids) > 0

    def test_hld_03_page_1_is_invoice_payable_candidate(self) -> None:
        """HLD-03 page 1 (Portuguese invoice with 'Total da factura') is invoice + payable_candidate."""
        p = Path("artifacts/ocr/HLD-03/page_001.json")
        if not p.exists():
            pytest.skip("HLD-03 OCR artifact not found")
        page_ev = ocr_json_to_page_evidence(p)
        result = classify_page(page_ev)

        assert result.page_role == PageRole.INVOICE
        assert result.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE
        assert result.confidence is None
        assert len(result.evidence_ids) > 0


# ══════════════════════════════════════════════════════════════════════════
# 2. Orthogonality: PageRole vs PayableRelevance
# ══════════════════════════════════════════════════════════════════════════

class TestOrthogonalRoleAndRelevance:
    def test_invoice_with_no_amounts_is_ambiguous(self) -> None:
        """An invoice page with no amounts or payable totals produces invoice + ambiguous."""
        page_ev = _make_page([
            "Company Logo",
            "COMMERCIAL INVOICE",
            "Invoice No: INV-9901",
            "Customer Account: ACCT-12",
            "Please refer questions to accounting.",
        ])
        res = classify_page(page_ev)
        assert res.page_role == PageRole.INVOICE
        assert res.payable_relevance == PayableRelevance.AMBIGUOUS
        assert any("lacks amounts" in r.lower() for r in res.decision_reasons)

    def test_continuation_page_with_payable_total_is_payable_candidate(self) -> None:
        """Continuation page containing the grand total is continuation + payable_candidate."""
        page_ev = _make_page([
            "Page 2 of 2",
            "Item 15: Laboratory Sensor",
            "Item 16: Optical Cable",
            "TOTAL AMOUNT DUE: 37,534.94 EUR",
        ])
        res = classify_page(page_ev)
        assert res.page_role == PageRole.CONTINUATION
        assert res.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE

    def test_continuation_page_without_payable_total_is_supporting(self) -> None:
        """Continuation page with intermediate items and no total is continuation + supporting."""
        page_ev = _make_page([
            "Page 2 of 4",
            "Continued from previous page",
            "Item 15: Laboratory Sensor 100 EUR",
            "Item 16: Optical Cable 50 EUR",
            "Carry forward: 150 EUR",
        ])
        res = classify_page(page_ev)
        assert res.page_role == PageRole.CONTINUATION
        assert res.payable_relevance == PayableRelevance.SUPPORTING


# ══════════════════════════════════════════════════════════════════════════
# 3. PO Document vs PO Reference Disambiguation
# ══════════════════════════════════════════════════════════════════════════

class TestPurchaseOrderDisambiguation:
    def test_po_reference_on_invoice_does_not_make_page_a_po(self) -> None:
        """PO Number: 2287 on an invoice does NOT classify the page as a purchase order."""
        page_ev = _make_page([
            "TAX INVOICE",
            "Invoice No: 88124",
            "Customer PO Number: PO-2287-XYZ",
            "Item: Consulting Services",
            "Total Amount Due: 1,500.00 USD",
        ])
        res = classify_page(page_ev)
        assert res.page_role == PageRole.INVOICE
        assert res.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE
        assert res.signals["po_reference_hits"] > 0

    def test_genuine_purchase_order_document(self) -> None:
        """Standalone Purchase Order document is purchase_order + non_payable."""
        page_ev = _make_page([
            "PURCHASE ORDER",
            "Vendor: Acme Supplies Ltd",
            "Deliver To: Warehouse 4B",
            "PO Date: 12/01/2026",
            "Authorized Signature: J. Doe",
        ])
        res = classify_page(page_ev)
        assert res.page_role == PageRole.PURCHASE_ORDER
        assert res.payable_relevance == PayableRelevance.NON_PAYABLE


# ══════════════════════════════════════════════════════════════════════════
# 4. Credit Memo, Debit Memo, Receipt, and Remittance
# ══════════════════════════════════════════════════════════════════════════

class TestSpecializedDocumentRoles:
    def test_credit_memo_with_amount_is_payable_candidate(self) -> None:
        """Credit note with amount due/refund is credit_memo + payable_candidate."""
        page_ev = _make_page([
            "CREDIT NOTE",
            "Credit Note No: CN-5512",
            "Refund for damaged goods",
            "Total Amount Due: 250.00 EUR",
        ])
        res = classify_page(page_ev)
        assert res.page_role == PageRole.CREDIT_MEMO
        assert res.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE

    def test_debit_memo_is_debit_memo(self) -> None:
        """Debit memo page classifies as debit_memo."""
        page_ev = _make_page([
            "DEBIT MEMO",
            "Adjustment for price increase",
            "Total Amount Due: 120.00 USD",
        ])
        res = classify_page(page_ev)
        assert res.page_role == PageRole.DEBIT_MEMO
        assert res.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE

    def test_payment_receipt_is_non_payable(self) -> None:
        """Receipt confirming payment is receipt + non_payable."""
        page_ev = _make_page([
            "CASH RECEIPT",
            "Receipt No: R-10928",
            "Paid in full by credit card",
            "Amount: 45.00 EUR",
        ])
        res = classify_page(page_ev)
        assert res.page_role == PageRole.RECEIPT
        assert res.payable_relevance == PayableRelevance.NON_PAYABLE

    def test_remittance_advice_is_non_payable(self) -> None:
        """Remittance advice is remittance + non_payable."""
        page_ev = _make_page([
            "REMITTANCE ADVICE",
            "We have transferred the following funds to your bank account",
            "Payment Reference: WIRE-99120",
        ])
        res = classify_page(page_ev)
        assert res.page_role == PageRole.REMITTANCE
        assert res.payable_relevance == PayableRelevance.NON_PAYABLE


# ══════════════════════════════════════════════════════════════════════════
# 5. Ambiguity & Noise Robustness
# ══════════════════════════════════════════════════════════════════════════

class TestAmbiguityAndNoise:
    def test_realistic_shipping_text_with_no_invoice_or_payment(self) -> None:
        """Design Correction 7:
        Realistic shipping/support text with no invoice or payment evidence
        must NOT be forced into invoice or non_payable.
        """
        page_ev = _make_page([
            "Container Identification: MSCU1234567",
            "Seal Number: 981245",
            "Port of Loading: Rotterdam",
            "Vessel Name: Nordic Explorer",
            "Temperature Setting: +4C Controlled Atmosphere",
            "Handling Instructions: Keep Dry and Away from Sunlight",
        ])
        res = classify_page(page_ev)
        # Should classify as supporting or unknown with ambiguous relevance, not invoice!
        assert res.page_role in (PageRole.SUPPORTING_DOCUMENT, PageRole.UNKNOWN)
        assert res.payable_relevance in (PayableRelevance.SUPPORTING, PayableRelevance.AMBIGUOUS)
        assert res.page_role != PageRole.INVOICE

    def test_sparse_or_blank_page_is_unknown_ambiguous(self) -> None:
        """Blank or minimal non-committal text produces unknown + ambiguous."""
        page_ev = _make_page(["Confidentiality Notice: This document contains proprietary data."])
        res = classify_page(page_ev)
        assert res.page_role == PageRole.UNKNOWN
        assert res.payable_relevance == PayableRelevance.AMBIGUOUS

    def test_conflicting_signals_do_not_crash(self) -> None:
        """Conflicting keywords (e.g. mentions of receipt, delivery, and invoice together) do not crash."""
        page_ev = _make_page([
            "Delivery slip receipt acknowledgement",
            "Please send invoice later",
            "Packing list details attached",
        ])
        res = classify_page(page_ev)
        assert isinstance(res.page_role, PageRole)
        assert isinstance(res.payable_relevance, PayableRelevance)


# ══════════════════════════════════════════════════════════════════════════
# 6. Traceability, Audit Trail & Serialization
# ══════════════════════════════════════════════════════════════════════════

class TestTraceabilityAndSerialization:
    def test_evidence_ids_trace_directly_to_page_items(self) -> None:
        """Every evidence ID returned in PageUnderstanding belongs to an Evidence item on the page."""
        page_ev = _make_page([
            "COMMERCIAL INVOICE",
            "Bill To: Tenant Corp",
            "Total Amount Due: 500.00 EUR",
        ])
        res = classify_page(page_ev)
        assert len(res.evidence_ids) > 0

        valid_page_ids = {item.evidence_id for item in page_ev.items}
        for ev_id in res.evidence_ids:
            assert ev_id in valid_page_ids

    def test_decision_reasons_and_provenance_populated(self) -> None:
        """Decision reasons explain both role and payable relevance decisions."""
        page_ev = _make_page([
            "RECHNUNG",
            "Gesamtsumme 1.000,00 EUR",
            "MwSt 19%: 190,00 EUR",
            "Endbetrag 1.190,00 EUR",
        ])
        res = classify_page(page_ev)
        assert len(res.decision_reasons) >= 2
        assert res.provenance["classifier"] == "DeterministicPageClassifier"
        assert res.provenance["evidence_item_count"] == 4

    def test_json_serialization_round_trip(self) -> None:
        """PageUnderstanding round-trips cleanly through to_dict / to_json / from_dict / from_json."""
        res = PageUnderstanding(
            document_id="DOC-99",
            page_number=1,
            page_role=PageRole.INVOICE,
            payable_relevance=PayableRelevance.PAYABLE_CANDIDATE,
            confidence=None,
            evidence_ids=["DOC-99:p1:ocr:abc123"],
            signals={"invoice_title_hits": 1},
            decision_reasons=["Invoice heading observed"],
            provenance={"classifier": "DeterministicPageClassifier"},
        )
        json_str = res.to_json()
        loaded = PageUnderstanding.from_json(json_str)

        assert loaded.document_id == res.document_id
        assert loaded.page_number == res.page_number
        assert loaded.page_role == res.page_role
        assert loaded.payable_relevance == res.payable_relevance
        assert loaded.confidence is None
        assert loaded.evidence_ids == res.evidence_ids
        assert loaded.signals == res.signals
        assert loaded.decision_reasons == res.decision_reasons


# ══════════════════════════════════════════════════════════════════════════
# 7. Absence of Document-Specific Rules
# ══════════════════════════════════════════════════════════════════════════

class TestZeroDocumentSpecificRules:
    def test_classification_invariant_to_document_id_and_filename(self) -> None:
        """Changing the document_id produces identical role, relevance, reasons, and signals."""
        content = [
            "Rechnung Nr.: 852566",
            "Northwind Operations OÜ",
            "Gesamtsumme 438,00 €",
            "Endbetrag 438,00 €",
        ]
        page_a = _make_page(content, doc_id="INV-01.pdf", page_num=1)
        page_b = _make_page(content, doc_id="UNKNOWN_RANDOM_UUID_12345.pdf", page_num=1)

        res_a = classify_page(page_a)
        res_b = classify_page(page_b)

        assert res_a.page_role == res_b.page_role
        assert res_a.payable_relevance == res_b.payable_relevance
        assert res_a.signals["invoice_title_hits"] == res_b.signals["invoice_title_hits"]
        assert res_a.signals["payable_total_hits"] == res_b.signals["payable_total_hits"]
        assert res_a.decision_reasons == res_b.decision_reasons
