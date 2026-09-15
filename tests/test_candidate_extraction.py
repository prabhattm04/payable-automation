"""tests/test_candidate_extraction.py — Test Suite for Phase 9B-1 Candidate Extraction.

Tests all required scenarios:
A. invoice number candidate
B. invoice date candidate
C. due date candidate
D. invoice type from evidence
E. filename ignored
F. currency candidate
G. supplier identity candidates
H. buyer identity candidates
I. PO number candidate
J. PO master ID not substituted
K. billed line candidate
L. component-detail candidate
M. missing quantity remains missing
N. missing unit price remains missing
O. missing amount remains missing
P. tax candidate
Q. header tax vs line tax
R. discount candidate
S. charge candidate
T. printed totals
U. multiple evidence IDs
V. conflicting OCR/Qwen candidates coexist
W. raw + normalized value preservation
X. Decimal handling
Y. multi-page group association
Z. supporting-document facts remain distinguishable
AA. no arithmetic
AB. no tax calculation
AC. no master-data matching
AD. deterministic candidate serialization
AE. no document-specific rules
AF. no Qwen call when routing says OCR sufficient
AG. Qwen evidence consumed when routing says route_to_vision

Refinements:
- EUR vs € normalization with/without contextual evidence
- invoice-type contextual references (body mention does not override header)
- identical OCR/Qwen values remaining separate candidates in 9B-1
- general properties/provenance integration tests on INV-01, HLD-01, INV-02, DU-02, HLD-03
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.extraction.candidates import (
    ChargeCandidate,
    DiscountCandidate,
    DocumentIdentityCandidate,
    ExtractionCandidates,
    LineCandidate,
    PartyIdentityCandidate,
    POCandidate,
    TaxCandidate,
    TotalCandidate,
    adapt_vision_evidence_to_candidates,
    extract_candidates_from_document,
    extract_candidates_from_page,
    normalize_currency_with_context,
)
from src.understanding.document_facts import (
    InvoiceType,
    Placement,
    SemanticRole,
)
from src.understanding.document_grouper import DocumentGroup
from src.understanding.evidence import (
    Evidence,
    EvidenceSource,
    PageEvidence,
    ocr_json_to_page_evidence,
    vision_response_to_evidence,
)
from src.understanding.page_classifier import PageRole, PayableRelevance, classify_page
from src.understanding.qwen_router import RouterDecisionType, RoutingDecision, route_page
from src.vision.provider import VisionProvider, VisionResponse



# ══════════════════════════════════════════════════════════════════════════
# Synthetic Test Helpers
# ══════════════════════════════════════════════════════════════════════════

def _make_page(
    lines: list[str],
    doc_id: str = "DOC_TEST",
    page_num: int = 1,
    boxes: list[list[float]] | None = None,
) -> PageEvidence:
    """Helper to build a synthetic PageEvidence with realistic Evidence items."""
    page_ev = PageEvidence(document_id=doc_id, page_number=page_num)
    for idx, text in enumerate(lines):
        bbox = (
            boxes[idx]
            if boxes and idx < len(boxes)
            else [10.0, float(idx * 30), 400.0, float(idx * 30 + 25)]
        )
        ev = Evidence.create(
            content=text,
            document_id=doc_id,
            page_number=page_num,
            source=EvidenceSource.OCR,
            extraction_method="RapidOCR/PP-OCRv5",
            confidence=0.98,
            bbox=bbox,
        )
        page_ev.add_item(ev)
    return page_ev


# ══════════════════════════════════════════════════════════════════════════
# 1. Document Identity Tests (A - F)
# ══════════════════════════════════════════════════════════════════════════

class TestDocumentIdentityExtraction:
    def test_a_invoice_number_candidate(self) -> None:
        """A. Invoice number candidate extracted with label, value, and evidence ID."""
        pe = _make_page(["COMMERCIAL INVOICE", "Invoice No: INV-2026-9001", "Total: 100 EUR"])
        res = extract_candidates_from_page(pe)

        inv_cands = [c for c in res.identity_candidates if c.field_name == "invoice_number"]
        assert len(inv_cands) >= 1
        cand = inv_cands[0]
        assert cand.raw_value == "INV-2026-9001"
        assert cand.normalized_value == "INV-2026-9001"
        assert len(cand.evidence_ids) >= 1
        assert cand.source == "ocr"

    def test_b_invoice_date_candidate(self) -> None:
        """B. Invoice date candidate extracted with normalization."""
        pe = _make_page(["TAX INVOICE", "Invoice Date: 15.03.2026", "Total: 50 EUR"])
        res = extract_candidates_from_page(pe)

        date_cands = [c for c in res.identity_candidates if c.field_name == "invoice_date"]
        assert len(date_cands) >= 1
        cand = date_cands[0]
        assert cand.raw_value == "15.03.2026"
        assert cand.normalized_value == "15.03.2026"
        assert len(cand.evidence_ids) >= 1

    def test_c_due_date_candidate(self) -> None:
        """C. Due date candidate extracted with label and evidence."""
        pe = _make_page(["INVOICE", "Due Date: 30-04-2026", "Total: 50 EUR"])
        res = extract_candidates_from_page(pe)

        due_cands = [c for c in res.identity_candidates if c.field_name == "due_date"]
        assert len(due_cands) >= 1
        cand = due_cands[0]
        assert cand.raw_value == "30-04-2026"
        assert cand.normalized_value == "30-04-2026"
        assert len(cand.evidence_ids) >= 1

    @pytest.mark.parametrize(
        "header_text,expected_type",
        [
            ("COMMERCIAL INVOICE", InvoiceType.INVOICE),
            ("CREDIT MEMO", InvoiceType.CREDIT_MEMO),
            ("GUTSCHRIFT", InvoiceType.CREDIT_MEMO),
            ("DEBIT NOTE", InvoiceType.DEBIT_MEMO),
            ("Random Document Title", InvoiceType.UNKNOWN),
        ],
    )
    def test_d_invoice_type_from_evidence(self, header_text: str, expected_type: InvoiceType) -> None:
        """D. Invoice type extracted strictly from document evidence."""
        pe = _make_page([header_text, "Number: 123", "Amount: 100 EUR"])
        res = extract_candidates_from_page(pe)

        type_cands = [c for c in res.identity_candidates if c.field_name == "invoice_type"]
        assert len(type_cands) >= 1
        assert type_cands[0].invoice_type_value == expected_type

    def test_e_filename_ignored(self) -> None:
        """E. Filename/document_id is strictly ignored when determining invoice type."""
        # Document named credit_memo_999.pdf but document content is a standard TAX INVOICE
        pe = _make_page(["TAX INVOICE", "Invoice No: 111"], doc_id="credit_memo_999.pdf")
        res = extract_candidates_from_page(pe)

        type_cands = [c for c in res.identity_candidates if c.field_name == "invoice_type"]
        assert len(type_cands) >= 1
        assert type_cands[0].invoice_type_value == InvoiceType.INVOICE
        assert type_cands[0].invoice_type_value != InvoiceType.CREDIT_MEMO

    def test_f_currency_candidate(self) -> None:
        """F. Currency candidate extracted with evidence."""
        pe = _make_page(["INVOICE", "Total Amount Due: 1,500.00 THB"])
        res = extract_candidates_from_page(pe)

        curr_cands = [c for c in res.identity_candidates if c.field_name == "currency"]
        assert len(curr_cands) >= 1
        assert curr_cands[0].raw_value == "THB"
        assert curr_cands[0].normalized_value == "THB"
        assert len(curr_cands[0].evidence_ids) >= 1


# ══════════════════════════════════════════════════════════════════════════
# 2. Contextual Refinements (Corrections 2 & 3)
# ══════════════════════════════════════════════════════════════════════════

class TestContextualRefinements:
    def test_eur_vs_euro_symbol_with_context(self) -> None:
        """Correction 2: € normalized to EUR when European context justifies it."""
        norm, justified = normalize_currency_with_context("€", ["DE123456789", "Berlin", "Germany"])
        assert norm == "EUR"
        assert justified is True

    def test_euro_symbol_without_context(self) -> None:
        """Correction 2: € preserved as symbol when no context is available."""
        norm, justified = normalize_currency_with_context("€", ["Unknown", "Entity"])
        assert norm == "€"
        assert justified is False

    def test_dollar_symbol_without_context(self) -> None:
        """Correction 2: $ preserved as symbol when ambiguous without US context."""
        norm, justified = normalize_currency_with_context("$", ["International", "Trading"])
        assert norm == "$"
        assert justified is False

    def test_dollar_symbol_with_us_context(self) -> None:
        """Correction 2: $ normalized to USD when explicit US context exists."""
        norm, justified = normalize_currency_with_context("$", ["USA", "New York"])
        assert norm == "USD"
        assert justified is True

    def test_invoice_type_contextual_reference(self) -> None:
        """Correction 3: Body mention of credit memo does not override header INVOICE."""
        pe = _make_page([
            "TAX INVOICE",
            "Invoice No: 8852",
            "Special Note: In reference to Credit Memo #CM-9988",
            "Total: 100 EUR",
        ])
        res = extract_candidates_from_page(pe)

        type_cands = [c for c in res.identity_candidates if c.field_name == "invoice_type"]
        assert len(type_cands) >= 1
        # Header "TAX INVOICE" takes precedence over body reference to Credit Memo
        assert type_cands[0].invoice_type_value == InvoiceType.INVOICE


# ══════════════════════════════════════════════════════════════════════════
# 3. Party & PO Candidates (G - J)
# ══════════════════════════════════════════════════════════════════════════

class TestPartyAndPOCandidates:
    def test_g_supplier_identity_candidates(self) -> None:
        """G. Supplier VAT ID, email, and IBAN candidates extracted."""
        pe = _make_page([
            "Acme Industrial Solutions",
            "VAT ID: DE812345678",
            "Email: contact@acme.de",
            "IBAN: DE89370400440532013000",
        ])
        res = extract_candidates_from_page(pe)

        vat_cands = [c for c in res.party_candidates if c.field_name == "vat_id"]
        assert len(vat_cands) >= 1
        assert vat_cands[0].raw_value == "DE812345678"
        assert vat_cands[0].normalized_value == "DE812345678"

        email_cands = [c for c in res.party_candidates if c.field_name == "email"]
        assert len(email_cands) >= 1
        assert email_cands[0].normalized_value == "contact@acme.de"

        iban_cands = [c for c in res.party_candidates if c.field_name == "bank_iban"]
        assert len(iban_cands) >= 1
        assert iban_cands[0].normalized_value == "DE89370400440532013000"

    def test_h_buyer_identity_candidates(self) -> None:
        """H. Buyer identity candidates extracted from bill-to context."""
        pe = _make_page([
            "Seller Corp",
            "Bill To:",
            "Global Logistics GmbH",
            "Invoice No: 12",
        ])
        res = extract_candidates_from_page(pe)

        buyer_cands = [c for c in res.party_candidates if c.party_role == "buyer"]
        assert len(buyer_cands) >= 1
        assert buyer_cands[0].raw_value == "Global Logistics GmbH"

    def test_i_po_number_candidate(self) -> None:
        """I. PO number candidate extracted with label and context."""
        pe = _make_page(["INVOICE", "Customer PO: 4500123456", "Total: 100 EUR"])
        res = extract_candidates_from_page(pe)

        assert len(res.po_candidates) >= 1
        cand = res.po_candidates[0]
        assert cand.raw_value == "4500123456"
        assert cand.normalized_value == "4500123456"
        assert "Customer PO" in cand.label

    def test_j_po_master_id_not_substituted(self) -> None:
        """J. Master PO ID is NOT substituted into candidate extraction."""
        pe = _make_page(["INVOICE", "Customer PO: 4500123456"])
        res = extract_candidates_from_page(pe)

        cand = res.po_candidates[0]
        # Observed PO number remains strictly what was on the document
        assert cand.raw_value == "4500123456"
        assert cand.normalized_value == "4500123456"


# ══════════════════════════════════════════════════════════════════════════
# 4. Table & Line Candidates (K - O, AA)
# ══════════════════════════════════════════════════════════════════════════

class TestLineCandidates:
    def test_k_billed_line_candidate(self) -> None:
        """K. Billed line candidate with raw_cells and source_row_number."""
        boxes = [
            [10.0, 10.0, 300.0, 30.0],
            [10.0, 50.0, 50.0, 70.0],
            [60.0, 50.0, 200.0, 70.0],
            [210.0, 50.0, 250.0, 70.0],
            [260.0, 50.0, 320.0, 70.0],
            [330.0, 50.0, 390.0, 70.0],
        ]
        lines = [
            "COMMERCIAL INVOICE",
            "1",
            "Heavy Machinery Part A",
            "2",
            "150.00",
            "300.00",
        ]
        pe = _make_page(lines, boxes=boxes)
        res = extract_candidates_from_page(pe)

        assert len(res.line_candidates) >= 1
        lc = res.line_candidates[0]
        assert lc.source_row_number == 1
        assert "Heavy Machinery Part A" in lc.description
        assert lc.semantic_role == SemanticRole.BILLED_LINE
        assert len(lc.raw_cells) > 0
        assert len(lc.evidence_ids) >= 1

    def test_l_component_detail_candidate(self) -> None:
        """L. Component detail / breakdown row receives COMPONENT_DETAIL role."""
        boxes = [
            [10.0, 50.0, 300.0, 70.0],
            [310.0, 50.0, 380.0, 70.0],
        ]
        lines = [
            "Sub-item breakdown (info only)",
            "50.00",
        ]
        pe = _make_page(lines, boxes=boxes)
        res = extract_candidates_from_page(pe)

        assert len(res.line_candidates) >= 1
        lc = res.line_candidates[0]
        assert lc.semantic_role == SemanticRole.COMPONENT_DETAIL

    def test_m_missing_quantity_remains_missing(self) -> None:
        """M. Missing quantity remains None; never guessed or defaulted."""
        boxes = [
            [10.0, 50.0, 250.0, 70.0],
            [260.0, 50.0, 350.0, 70.0],
        ]
        lines = ["Consulting Services Fixed Fee", "500.00"]
        pe = _make_page(lines, boxes=boxes)
        res = extract_candidates_from_page(pe)

        assert len(res.line_candidates) >= 1
        lc = res.line_candidates[0]
        assert lc.quantity is None
        assert lc.amount == Decimal("500.00")

    def test_n_missing_unit_price_remains_missing(self) -> None:
        """N. Missing unit price is NEVER computed from amount / quantity."""
        lc = LineCandidate(
            source_row_number=1,
            description="Item without unit price",
            quantity=Decimal("4"),
            amount=Decimal("100.00"),
            evidence_ids=("ev_1",),
        )
        # Unit price must be None, NOT 25.00!
        assert lc.unit_price is None

    def test_o_missing_amount_remains_missing(self) -> None:
        """O. Missing amount is NEVER computed from quantity * unit_price."""
        lc = LineCandidate(
            source_row_number=1,
            description="Item without total amount",
            quantity=Decimal("3"),
            unit_price=Decimal("40.00"),
            evidence_ids=("ev_1",),
        )
        # Amount must be None, NOT 120.00!
        assert lc.amount is None

    def test_aa_no_arithmetic(self) -> None:
        """AA. Candidate extraction performs zero arithmetic calculations."""
        # Row with qty=5 and amount=200.00
        boxes = [
            [10.0, 50.0, 200.0, 70.0],
            [210.0, 50.0, 250.0, 70.0],
            [260.0, 50.0, 350.0, 70.0],
        ]
        lines = ["Software Licenses", "5", "200.00"]
        pe = _make_page(lines, boxes=boxes)
        res = extract_candidates_from_page(pe)

        lc = res.line_candidates[0]
        assert lc.quantity == Decimal("5")
        assert lc.amount == Decimal("200.00")
        assert lc.unit_price is None  # Zero division / math performed!


# ══════════════════════════════════════════════════════════════════════════
# 5. Taxes, Discounts, Charges, and Totals (P - T, AB)
# ══════════════════════════════════════════════════════════════════════════

class TestFinancialComponents:
    def test_p_tax_candidate(self) -> None:
        """P. Tax candidate extracted with rate and amount."""
        pe = _make_page(["INVOICE", "MwSt 19%: 38.00", "Total: 238.00"])
        res = extract_candidates_from_page(pe)

        assert len(res.tax_candidates) >= 1
        tc = res.tax_candidates[0]
        assert tc.rate == Decimal("19")
        assert tc.amount == Decimal("38.00")
        assert len(tc.evidence_ids) >= 1

    def test_q_header_tax_vs_line_tax(self) -> None:
        """Q. Tax placement candidate preserved as HEADER vs LINE."""
        tc_hdr = TaxCandidate(tax_name="VAT Summary", rate=Decimal("19"), amount=Decimal("190.00"), placement=Placement.HEADER, evidence_ids=("ev_h",))
        tc_line = TaxCandidate(tax_name="VAT Line", rate=Decimal("19"), amount=Decimal("19.00"), placement=Placement.LINE, evidence_ids=("ev_l",))

        assert tc_hdr.placement == Placement.HEADER
        assert tc_line.placement == Placement.LINE

    def test_r_discount_candidate(self) -> None:
        """R. Discount candidate extracted separately from lines."""
        pe = _make_page(["INVOICE", "Special Discount: 25.00", "Total: 100.00"])
        res = extract_candidates_from_page(pe)

        assert len(res.discount_candidates) >= 1
        dc = res.discount_candidates[0]
        assert dc.amount == Decimal("25.00")
        assert dc.placement == Placement.HEADER

    def test_s_charge_candidate(self) -> None:
        """S. Charge candidate (shipping/freight) extracted separately from lines."""
        pe = _make_page(["INVOICE", "Shipping & Handling: 45.00", "Total: 200.00"])
        res = extract_candidates_from_page(pe)

        assert len(res.charge_candidates) >= 1
        cc = res.charge_candidates[0]
        assert cc.amount == Decimal("45.00")
        assert cc.placement == Placement.HEADER

    def test_t_printed_totals(self) -> None:
        """T. Printed totals preserved separately with labels and types."""
        pe = _make_page([
            "Subtotal: 1,000.00",
            "Nettobetrag: 950.00",
            "MwSt. Gesamt: 180.50",
            "Gesamtsumme: 1,130.50",
            "Endbetrag: 1,130.50",
        ])
        res = extract_candidates_from_page(pe)

        assert len(res.total_candidates) >= 3
        types = {tc.total_type for tc in res.total_candidates}
        assert "subtotal" in types
        assert "net" in types
        assert "gross_total" in types

    def test_ab_no_tax_calculation(self) -> None:
        """AB. Tax candidate rates are NOT derived from amount / base."""
        # Tax stated without rate
        pe = _make_page(["INVOICE", "MwSt: 50.00", "Net: 500.00"])
        res = extract_candidates_from_page(pe)

        tc = res.tax_candidates[0]
        assert tc.amount == Decimal("50.00")
        # Rate must remain None! Never computed as 50 / 500 = 10%
        assert tc.rate is None


# ══════════════════════════════════════════════════════════════════════════
# 6. Evidence Conflicts & OCR vs Qwen Coexistence (U, V, W, X, AD, AF, AG)
# ══════════════════════════════════════════════════════════════════════════

class TestEvidenceConflictsAndQwenCoexistence:
    def test_u_multiple_evidence_ids(self) -> None:
        """U. Candidates can possess multiple supporting evidence IDs."""
        pe = _make_page(["Invoice Number:", "INV-8889"])
        res = extract_candidates_from_page(pe)

        inv_cands = [c for c in res.identity_candidates if c.field_name == "invoice_number"]
        assert len(inv_cands) >= 1
        # Multi-evidence from label + value blocks
        assert len(inv_cands[0].evidence_ids) >= 2

    def test_v_conflicting_ocr_and_qwen_candidates_coexist(self) -> None:
        """V. Conflicting OCR and Qwen candidates coexist without forced resolution."""
        pe = _make_page(["Invoice No: 9972-907", "Total: 100 EUR"])

        # Create simulated Qwen Vision evidence that extracted slightly different value
        vision_ev = Evidence.create(
            content="Invoice Number: 9972-970\nTotal: 100 EUR",
            document_id="DOC_TEST",
            page_number=1,
            source=EvidenceSource.VISION,
            extraction_method="Puter/Qwen3-VL",
            confidence=None,
        )

        res = extract_candidates_from_page(pe, vision_evidence=vision_ev)

        inv_cands = [c for c in res.identity_candidates if c.field_name == "invoice_number"]
        # Both OCR and Vision candidates are preserved!
        values = {c.raw_value for c in inv_cands}
        assert "9972-907" in values
        assert "9972-970" in values
        assert len(inv_cands) == 2

    def test_identical_ocr_and_qwen_remain_separate_candidates(self) -> None:
        """Correction 7: Identical OCR and Qwen observations remain separate candidates in 9B-1."""
        pe = _make_page(["Invoice No: 9972-907", "Total: 100 EUR"])

        vision_ev = Evidence.create(
            content="Invoice Number: 9972-907\nTotal: 100 EUR",
            document_id="DOC_TEST",
            page_number=1,
            source=EvidenceSource.VISION,
            extraction_method="Puter/Qwen3-VL",
            confidence=None,
        )

        res = extract_candidates_from_page(pe, vision_evidence=vision_ev)

        inv_cands = [c for c in res.identity_candidates if c.field_name == "invoice_number"]
        # In 9B-1, both are kept as distinct candidates with their respective sources
        assert len(inv_cands) == 2
        sources = {c.source for c in inv_cands}
        assert "ocr" in sources
        assert "vision" in sources

    def test_w_raw_and_normalized_value_preservation(self) -> None:
        """W. Both raw observed string and normalized Decimal are preserved."""
        pe = _make_page(["Gesamtsumme: 1.234,50 EUR"])
        res = extract_candidates_from_page(pe)

        tc = res.total_candidates[0]
        assert tc.raw_value == "1.234,50"
        assert tc.normalized_value == Decimal("1234.50")

    def test_x_decimal_handling(self) -> None:
        """X. Exact Decimal precision is preserved without float rounding errors."""
        tc = TotalCandidate(
            total_type="gross_total",
            raw_label="Total",
            raw_value="0.10",
            normalized_value=Decimal("0.10"),
            evidence_ids=("ev_x",),
        )
        assert isinstance(tc.normalized_value, Decimal)
        d = tc.to_dict()
        assert d["normalized_value"] == "0.10"
        reconstructed = TotalCandidate.from_dict(d)
        assert reconstructed.normalized_value == Decimal("0.10")

    def test_ad_deterministic_candidate_serialization(self) -> None:
        """AD. Deterministic candidate serialization round-trip."""
        pe = _make_page(["TAX INVOICE", "Invoice No: 123", "Total: 100 EUR"])
        res = extract_candidates_from_page(pe)

        j1 = res.to_json()
        j2 = res.to_json()
        assert j1 == j2

        reconstructed = ExtractionCandidates.from_json(j1)
        assert reconstructed == res

    def test_af_no_qwen_call_when_routing_says_ocr_sufficient(self) -> None:
        """AF. Qwen provider is NEVER called when routing decides OCR_SUFFICIENT."""
        pe = _make_page(["INVOICE", "Invoice No: 111", "Total: 100 EUR"])
        mock_provider = MagicMock(spec=VisionProvider)

        # High confidence, clear layout -> ocr_sufficient
        res = extract_candidates_from_page(pe, vision_provider=mock_provider)

        assert mock_provider.analyze_image.called is False
        assert res.metadata["routing_decision"] == RouterDecisionType.OCR_SUFFICIENT.value

    def test_ag_qwen_evidence_consumed_when_routing_says_route_to_vision(self) -> None:
        """AG. Qwen evidence is consumed when routing decides ROUTE_TO_VISION."""
        # Ambiguous classification forces route_to_vision
        pe = _make_page(["Unknown Document Header", "Line Item 1"])
        mock_provider = MagicMock(spec=VisionProvider)
        mock_provider.analyze_image.return_value = VisionResponse(
            success=True,
            content="Invoice Number: INV-VISION-999\nGross Total: 500.00 EUR",
            model="qwen-vl",
            provider="Puter",
        )
        pe.image_path = "dummy_page_001.png"

        res = extract_candidates_from_page(pe, vision_provider=mock_provider)

        assert mock_provider.analyze_image.called is True
        assert res.metadata["routing_decision"] == RouterDecisionType.ROUTE_TO_VISION.value
        # Vision candidates are present!
        vision_cands = [c for c in res.identity_candidates if c.source == "vision"]
        assert len(vision_cands) >= 1
        assert vision_cands[0].raw_value == "INV-VISION-999"


# ══════════════════════════════════════════════════════════════════════════
# 7. Multi-Page & Supporting Documents (Y, Z, AC, AE)
# ══════════════════════════════════════════════════════════════════════════

class TestMultiPageAndSupportingDocuments:
    def test_y_multi_page_group_association(self) -> None:
        """Y. Candidates preserve multi-page group association."""
        p1 = _make_page(["INVOICE", "Invoice No: 101", "Page 1 of 2"], page_num=1)
        p2 = _make_page(["Page 2 of 2", "Total: 500 EUR"], page_num=2)

        results = extract_candidates_from_document([p1, p2])
        assert len(results) == 2
        # Both share the same group_id
        assert results[0].group_id is not None
        assert results[0].group_id == results[1].group_id

    def test_z_supporting_document_facts_remain_distinguishable(self) -> None:
        """Z. Supporting document facts are flagged with SUPPORTING relevance."""
        p1 = _make_page(["COMMERCIAL INVOICE", "Invoice No: 101", "Total Amount Due: 100.00 EUR"], page_num=1)
        p2 = _make_page(["PACKING LIST", "Gross Weight: 50 KG", "Package Total: 500"], page_num=2)

        results = extract_candidates_from_document([p1, p2])
        assert len(results) == 2
        assert results[0].payable_relevance == PayableRelevance.PAYABLE_CANDIDATE
        assert results[1].payable_relevance == PayableRelevance.SUPPORTING
        # Page 2 candidates are isolated in supporting group
        assert results[0].group_id != results[1].group_id

    def test_ac_no_master_data_matching(self) -> None:
        """AC. Phase 8 matchers are NOT invoked; candidate output has no master_id."""
        pe = _make_page(["Acme Corp", "Invoice No: 1", "Total: 10 EUR"])
        res = extract_candidates_from_page(pe)

        as_dict = res.to_dict()
        # No master_id, match_status, or master record injected
        assert "master_id" not in as_dict
        for pc in as_dict["party_candidates"]:
            assert "master_id" not in pc

    def test_ae_no_document_specific_rules(self) -> None:
        """AE. Extractor operates via generic patterns without document ID checks."""
        pe1 = _make_page(["INVOICE", "Invoice No: 555"], doc_id="ANY_ARBITRARY_ID_1")
        pe2 = _make_page(["INVOICE", "Invoice No: 555"], doc_id="DIFFERENT_ARBITRARY_ID_2")

        res1 = extract_candidates_from_page(pe1)
        res2 = extract_candidates_from_page(pe2)

        assert res1.identity_candidates[0].raw_value == res2.identity_candidates[0].raw_value


# ══════════════════════════════════════════════════════════════════════════
# 8. Representative Real Corpus Integration Tests (General Properties)
# ══════════════════════════════════════════════════════════════════════════

class TestRealCorpusGeneralProperties:
    def test_inv_01_general_properties(self) -> None:
        """INV-01 (German invoice): extracts identity, parties, totals with valid provenance."""
        p = Path("artifacts/ocr/INV-01/page_001.json")
        if not p.exists():
            pytest.skip("INV-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        res = extract_candidates_from_page(pe)

        assert "INV-01" in res.document_id
        assert res.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE
        # Verify valid evidence provenance for all extracted candidates
        for c in res.identity_candidates:
            assert len(c.evidence_ids) > 0
        for c in res.party_candidates:
            assert len(c.evidence_ids) > 0
        for c in res.total_candidates:
            assert len(c.evidence_ids) > 0

    def test_hld_01_general_properties(self) -> None:
        """HLD-01 (Thai invoice): extracts tax ID, email, dates with valid provenance."""
        p = Path("artifacts/ocr/HLD-01/page_001.json")
        if not p.exists():
            pytest.skip("HLD-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        res = extract_candidates_from_page(pe)

        assert "HLD-01" in res.document_id
        for c in res.party_candidates:
            assert len(c.evidence_ids) > 0

    def test_inv_02_general_properties(self) -> None:
        """INV-02 (Estonian invoice): extracts IBAN, dates with valid provenance."""
        p = Path("artifacts/ocr/INV-02/page_001.json")
        if not p.exists():
            pytest.skip("INV-02 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        res = extract_candidates_from_page(pe)

        assert "INV-02" in res.document_id
        # IBAN candidate should be present
        iban_cands = [c for c in res.party_candidates if c.field_name == "bank_iban"]
        assert len(iban_cands) >= 1
        assert "EE" in iban_cands[0].normalized_value

    def test_du_02_multi_page_general_properties(self) -> None:
        """DU-02 (multi-page customs dossier): page 1 and page 5 extracted with distinct roles."""
        p1 = Path("artifacts/ocr/DU-02/page_001.json")
        p5 = Path("artifacts/ocr/DU-02/page_005.json")
        if not p1.exists() or not p5.exists():
            pytest.skip("DU-02 artifacts missing")
        pe1 = ocr_json_to_page_evidence(p1)
        pe5 = ocr_json_to_page_evidence(p5)

        results = extract_candidates_from_document([pe1, pe5])
        assert len(results) == 2
        # Page 5 is supporting packing list; distinct from page 1 invoice
        assert results[1].payable_relevance == PayableRelevance.SUPPORTING

    def test_hld_03_general_properties(self) -> None:
        """HLD-03 (Portuguese invoice with complex structure): routes properly and preserves provenance."""
        p = Path("artifacts/ocr/HLD-03/page_001.json")
        if not p.exists():
            pytest.skip("HLD-03 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        res = extract_candidates_from_page(pe)

        assert "HLD-03" in res.document_id
        assert res.metadata["routing_decision"] == RouterDecisionType.ROUTE_TO_VISION.value
