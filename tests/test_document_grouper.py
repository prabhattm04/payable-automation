"""tests/test_document_grouper.py — Tests for Phase 7C Multi-Page Document Grouping."""
from __future__ import annotations

import json
from pathlib import Path
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
    GroupingResult,
    group_document,
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
# 1. Core Grouping Scenarios (A through I)
# ══════════════════════════════════════════════════════════════════════════

class TestGroupingScenarios:
    def test_a_simple_two_page_invoice(self) -> None:
        """A. Simple two-page invoice: page 1 invoice, page 2 continuation -> one group [1, 2]."""
        p1 = _make_page([
            "COMMERCIAL INVOICE",
            "Invoice Number: INV-1001",
            "Page 1 of 2",
            "Item 1: Consulting 100 EUR",
        ], page_num=1)
        p2 = _make_page([
            "Page 2 of 2",
            "Item 2: Software 200 EUR",
            "TOTAL AMOUNT DUE: 300 EUR",
        ], page_num=2)

        res = group_document([p1, p2])
        assert len(res.groups) == 1
        assert res.groups[0].page_numbers == [1, 2]
        assert res.groups[0].confidence is None  # Deterministic baseline: None

    def test_b_three_page_invoice(self) -> None:
        """B. Three-page invoice: page 1 invoice, page 2 continuation, page 3 continuation -> [1, 2, 3]."""
        p1 = _make_page(["INVOICE", "Invoice No: 8852", "Page 1 of 3"], page_num=1)
        p2 = _make_page(["Page 2 of 3", "Item continued"], page_num=2)
        p3 = _make_page(["Page 3 of 3", "Total: 500 EUR"], page_num=3)

        res = group_document([p1, p2, p3])
        assert len(res.groups) == 1
        assert res.groups[0].page_numbers == [1, 2, 3]

    def test_c_invoice_followed_by_supporting_document(self) -> None:
        """C. Invoice followed by independent supporting document -> separate groups [1], [2]."""
        p1 = _make_page([
            "COMMERCIAL INVOICE",
            "Invoice No: INV-4401",
            "Total Amount Due: 1,200.00 EUR",
        ], page_num=1)
        p2 = _make_page([
            "PACKING LIST",
            "Net Weight: 45 KG",
            "Gross Weight: 50 KG",
            "No. of Boxes: 5",
        ], page_num=2)

        res = group_document([p1, p2])
        assert len(res.groups) == 2
        assert res.groups[0].page_numbers == [1]
        assert res.groups[1].page_numbers == [2]

    def test_d_supporting_document_followed_by_invoice(self) -> None:
        """D. Supporting document followed by invoice -> separate groups [1], [2]."""
        p1 = _make_page([
            "DELIVERY NOTE",
            "Carrier: Freight Express",
            "Bill of Lading: BOL-9912",
        ], page_num=1)
        p2 = _make_page([
            "TAX INVOICE",
            "Invoice No: INV-5502",
            "Total Amount Due: 800.00 USD",
        ], page_num=2)

        res = group_document([p1, p2])
        assert len(res.groups) == 2
        assert res.groups[0].page_numbers == [1]
        assert res.groups[1].page_numbers == [2]

    def test_e_explicit_same_invoice_number_across_pages(self) -> None:
        """E. Explicit same invoice number across pages -> same group [1, 2]."""
        p1 = _make_page(["TAX INVOICE", "Invoice Number: 998812", "Bill To: Acme Corp"], page_num=1)
        p2 = _make_page(["Continued", "Invoice Number: 998812", "Total Amount Due: 450.00 EUR"], page_num=2)

        res = group_document([p1, p2])
        assert len(res.groups) == 1
        assert res.groups[0].page_numbers == [1, 2]
        assert "998812" in res.groups[0].grouping_signals.get("shared_document_numbers", [])

    def test_f_different_invoice_numbers(self) -> None:
        """F. Different invoice/document numbers -> separate groups [1], [2]."""
        p1 = _make_page(["INVOICE", "Invoice No: INV-1001", "Total: 100 EUR"], page_num=1)
        p2 = _make_page(["INVOICE", "Invoice No: INV-2002", "Total: 200 EUR"], page_num=2)

        res = group_document([p1, p2])
        assert len(res.groups) == 2
        assert res.groups[0].page_numbers == [1]
        assert res.groups[1].page_numbers == [2]

    def test_g_page_number_continuation(self) -> None:
        """G. Page-number continuation ("Page 1 of 2", "Page 2 of 2") -> same group [1, 2]."""
        p1 = _make_page(["Heading A", "Page 1 of 2"], page_num=1)
        p2 = _make_page(["Heading B", "Page 2 of 2"], page_num=2)

        res = group_document([p1, p2])
        assert len(res.groups) == 1
        assert res.groups[0].page_numbers == [1, 2]
        assert res.groups[0].grouping_signals.get("sequential_pagination") is True

    def test_h_interleaved_logical_documents(self) -> None:
        """H. Interleaved logical documents:
        Page 1: Doc A (Inv A1, Page 1 of 2)
        Page 2: Doc B (Inv B1, Page 1 of 1)
        Page 3: Doc A (Inv A1, Page 2 of 2)
        Must NOT blindly attach Page 3 to Page 2! Page 3 joins Doc A -> [1, 3] and [2].
        """
        p1 = _make_page(["INVOICE", "Invoice No: DOC-A1", "Page 1 of 2"], page_num=1)
        p2 = _make_page(["PURCHASE ORDER", "PO Number: DOC-B1", "Page 1 of 1"], page_num=2)
        p3 = _make_page(["Invoice No: DOC-A1", "Page 2 of 2", "Total: 500 EUR"], page_num=3)

        res = group_document([p1, p2, p3])
        assert len(res.groups) == 2

        # Group 1 has Doc A pages [1, 3] strictly ordered
        g1 = res.groups[0]
        assert g1.page_numbers == [1, 3]
        assert "DOC-A1" in g1.grouping_signals.get("shared_document_numbers", [])

        # Group 2 has Doc B page [2]
        g2 = res.groups[1]
        assert g2.page_numbers == [2]

    def test_i_ambiguous_pages_safe_no_forced_merge(self) -> None:
        """I. Ambiguous pages without clear join evidence are not forcibly merged."""
        p1 = _make_page(["COMMERCIAL INVOICE", "Invoice No: 123", "Total: 50 EUR"], page_num=1)
        p2 = _make_page(["Confidentiality Notice: This is general correspondence."], page_num=2)

        res = group_document([p1, p2])
        # Safe separation: p2 remains separate
        assert len(res.groups) == 2
        assert res.groups[0].page_numbers == [1]
        assert res.groups[1].page_numbers == [2]


# ══════════════════════════════════════════════════════════════════════════
# 2. Invariance, Traceability, and Serialization (J through M)
# ══════════════════════════════════════════════════════════════════════════

class TestTraceabilityAndInvariance:
    def test_j_evidence_id_traceability(self) -> None:
        """J. Every evidence ID in DocumentGroup belongs to a valid page item in that group."""
        p1 = _make_page(["INVOICE", "Invoice Number: 77123", "Page 1 of 2"], page_num=1)
        p2 = _make_page(["Invoice Number: 77123", "Page 2 of 2"], page_num=2)

        res = group_document([p1, p2])
        grp = res.groups[0]
        assert len(grp.evidence_ids) > 0

        valid_ids = {e.evidence_id for e in p1.items} | {e.evidence_id for e in p2.items}
        for eid in grp.evidence_ids:
            assert eid in valid_ids

    def test_k_deterministic_output(self) -> None:
        """K. Identical inputs produce identical groups and group IDs."""
        pages = [
            _make_page(["INVOICE", "Invoice No: 111", "Page 1 of 2"], page_num=1),
            _make_page(["Page 2 of 2", "Total: 200 EUR"], page_num=2),
        ]
        res1 = group_document(pages)
        res2 = group_document(pages)

        assert len(res1) == len(res2)
        assert res1.groups[0].group_id == res2.groups[0].group_id
        assert res1.groups[0].page_numbers == res2.groups[0].page_numbers
        assert res1.groups[0].grouping_signals == res2.groups[0].grouping_signals

    def test_l_document_id_invariance(self) -> None:
        """L. Changing the document_id produces identical grouping decisions."""
        lines_1 = ["COMMERCIAL INVOICE", "Invoice No: 505", "Page 1 of 2"]
        lines_2 = ["Page 2 of 2", "Total: 100 EUR"]

        doc_a_p1 = _make_page(lines_1, doc_id="DOC_ALPHA.pdf", page_num=1)
        doc_a_p2 = _make_page(lines_2, doc_id="DOC_ALPHA.pdf", page_num=2)

        doc_b_p1 = _make_page(lines_1, doc_id="UNKNOWN_RANDOM_UUID.pdf", page_num=1)
        doc_b_p2 = _make_page(lines_2, doc_id="UNKNOWN_RANDOM_UUID.pdf", page_num=2)

        res_a = group_document([doc_a_p1, doc_a_p2])
        res_b = group_document([doc_b_p1, doc_b_p2])

        assert len(res_a.groups) == len(res_b.groups)
        assert res_a.groups[0].page_numbers == res_b.groups[0].page_numbers
        assert res_a.groups[0].grouping_signals == res_b.groups[0].grouping_signals

    def test_m_serialization_round_trip(self) -> None:
        """M. GroupingResult and DocumentGroup serialize and deserialize without loss."""
        grp = DocumentGroup(
            group_id="TEST:group:p1",
            document_id="TEST",
            page_numbers=[1, 2],
            evidence_ids=["ev_1", "ev_2"],
            grouping_signals={"shared_document_numbers": ["INV-99"]},
            confidence=None,
            provenance={"grouper": "DeterministicDocumentGrouper"},
        )
        res = GroupingResult(document_id="TEST", groups=[grp], provenance={"version": "1.0"})

        json_str = res.to_json()
        loaded = GroupingResult.from_json(json_str)

        assert loaded.document_id == res.document_id
        assert len(loaded.groups) == 1
        assert loaded.groups[0].group_id == grp.group_id
        assert loaded.groups[0].page_numbers == [1, 2]
        assert loaded.groups[0].confidence is None
        assert loaded.groups[0].evidence_ids == ["ev_1", "ev_2"]
        assert loaded.get_group_for_page(2) is not None
        assert loaded.get_group_for_page(3) is None


# ══════════════════════════════════════════════════════════════════════════
# 3. Real Representative Artifact Testing
# ══════════════════════════════════════════════════════════════════════════

class TestRealArtifactGrouping:
    def test_inv_01_single_page_group(self) -> None:
        """INV-01 (single-page invoice) groups into [1]."""
        p = Path("artifacts/ocr/INV-01/page_001.json")
        if not p.exists():
            pytest.skip("INV-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        res = group_document([pe])

        assert len(res.groups) == 1
        assert res.groups[0].page_numbers == [1]

    def test_hld_01_single_page_group(self) -> None:
        """HLD-01 (single-page Thai invoice) groups into [1]."""
        p = Path("artifacts/ocr/HLD-01/page_001.json")
        if not p.exists():
            pytest.skip("HLD-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        res = group_document([pe])

        assert len(res.groups) == 1
        assert res.groups[0].page_numbers == [1]

    def test_hld_03_single_page_group(self) -> None:
        """HLD-03 (single-page Portuguese invoice) groups into [1]."""
        p = Path("artifacts/ocr/HLD-03/page_001.json")
        if not p.exists():
            pytest.skip("HLD-03 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        res = group_document([pe])

        assert len(res.groups) == 1
        assert res.groups[0].page_numbers == [1]

    def test_du_02_pages_1_and_2_group_together(self) -> None:
        """DU-02 page 1 and page 2 (Consolidated Invoice, Page 1 of 2 & Page 2 of 2) group into [1, 2]."""
        p1 = Path("artifacts/ocr/DU-02/page_001.json")
        p2 = Path("artifacts/ocr/DU-02/page_002.json")
        if not (p1.exists() and p2.exists()):
            pytest.skip("DU-02 artifacts missing")

        pe1 = ocr_json_to_page_evidence(p1)
        pe2 = ocr_json_to_page_evidence(p2)

        res = group_document([pe1, pe2])
        assert len(res.groups) == 1
        assert res.groups[0].page_numbers == [1, 2]
        assert "554701215" in res.groups[0].grouping_signals.get("shared_document_numbers", [])

    def test_du_02_invoice_and_packing_separate(self) -> None:
        """DU-02 page 1 (Consolidated Invoice) and page 5 (Packing/weight sheet) form separate groups."""
        p1 = Path("artifacts/ocr/DU-02/page_001.json")
        p5 = Path("artifacts/ocr/DU-02/page_005.json")
        if not (p1.exists() and p5.exists()):
            pytest.skip("DU-02 artifacts missing")

        pe1 = ocr_json_to_page_evidence(p1)
        pe5 = ocr_json_to_page_evidence(p5)

        res = group_document([pe1, pe5])
        assert len(res.groups) == 2
        assert res.groups[0].page_numbers == [1]
        assert res.groups[1].page_numbers == [5]
