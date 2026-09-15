"""tests/test_candidate_consolidation.py — Comprehensive Test Suite for Phase 9B-2 Consolidation.

Test Categories:
A. Identity Consolidation (invoice number, dates, type, currency, conflicts)
B. Supplier Consolidation (consensus, complementary fields, conflicts)
C. Buyer Consolidation (hierarchy preservation, no master inference)
D. PO Consolidation (consensus, conflicts, amount cannot substitute PO)
E. Line Consolidation (consensus, repeated identical lines, component details, no arithmetic)
F. Tax Consolidation (complementary partial observations, placement preservation, no rate calculation)
G. Discount & Charge Consolidation (complementary partial observations, separation from lines)
H. Printed Totals Consolidation (consensus, conflict preservation, no balancing)
I. Multi-Page & Supporting Document Isolation (group context, supporting group isolation)
J. Provenance (evidence retention, multiple evidence IDs, origin verification)
K. Determinism (ordering invariance)
L. Generalization & Property-Style Tests (invariance to filename, non-invention)
M. Real Corpus Integration Tests (INV-01, HLD-01, INV-02, DU-02, HLD-03)
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
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
    extract_candidates_from_page,
    extract_candidates_from_document,
)
from src.extraction.consolidation import (
    ConsolidatedField,
    FieldStatus,
    consolidate_candidates,
    consolidate_document_groups,
    normalize_date_safe,
)
from src.understanding.document_facts import (
    FactOrigin,
    InvoiceType,
    Placement,
    SemanticRole,
)
from src.understanding.evidence import (
    Evidence,
    EvidenceSource,
    PageEvidence,
    ocr_json_to_page_evidence,
)
from src.understanding.page_classifier import PageRole, PayableRelevance


# ══════════════════════════════════════════════════════════════════════════
# Synthetic Test Helpers
# ══════════════════════════════════════════════════════════════════════════

def _make_candidate_container(
    doc_id: str = "DOC-TEST",
    group_id: str = "grp_1",
    page_numbers: tuple[int, ...] = (1,),
    payable_relevance: PayableRelevance = PayableRelevance.PAYABLE_CANDIDATE,
    page_role: PageRole = PageRole.INVOICE,
    identities: tuple[DocumentIdentityCandidate, ...] = (),
    parties: tuple[PartyIdentityCandidate, ...] = (),
    pos: tuple[POCandidate, ...] = (),
    lines: tuple[LineCandidate, ...] = (),
    taxes: tuple[TaxCandidate, ...] = (),
    discounts: tuple[DiscountCandidate, ...] = (),
    charges: tuple[ChargeCandidate, ...] = (),
    totals: tuple[TotalCandidate, ...] = (),
) -> ExtractionCandidates:
    return ExtractionCandidates(
        document_id=doc_id,
        group_id=group_id,
        page_numbers=page_numbers,
        payable_relevance=payable_relevance,
        page_role=page_role,
        identity_candidates=identities,
        party_candidates=parties,
        po_candidates=pos,
        line_candidates=lines,
        tax_candidates=taxes,
        discount_candidates=discounts,
        charge_candidates=charges,
        total_candidates=totals,
    )


# ══════════════════════════════════════════════════════════════════════════
# A. Identity Consolidation
# ══════════════════════════════════════════════════════════════════════════

class TestIdentityConsolidation:
    def test_a1_equivalent_invoice_numbers_consolidate(self) -> None:
        """Equivalent invoice numbers from OCR and Vision consolidate with combined evidence."""
        c_ocr = DocumentIdentityCandidate(
            field_name="invoice_number",
            raw_value="INV-2026-0042",
            normalized_value="INV-2026-0042",
            evidence_ids=("E17",),
            source="ocr",
        )
        c_vis = DocumentIdentityCandidate(
            field_name="invoice_number",
            raw_value="INV-2026-0042",
            normalized_value="INV-2026-0042",
            evidence_ids=("E91",),
            source="vision",
        )

        cands = _make_candidate_container(identities=(c_ocr, c_vis))
        facts = consolidate_candidates(cands)

        assert facts.identity.invoice_number == "INV-2026-0042"
        assert set(facts.identity.field_evidence_ids["invoice_number"]) == {"E17", "E91"}
        assert len(facts.conflicting_facts) == 0

    def test_a2_conflicting_invoice_numbers_remain_unresolved(self) -> None:
        """Conflicting invoice numbers (e.g. OCR vs Vision) remain unresolved in facts and logged in conflicts."""
        c_ocr = DocumentIdentityCandidate(
            field_name="invoice_number",
            raw_value="INV-2026-0042",
            normalized_value="INV-2026-0042",
            evidence_ids=("E17",),
            source="ocr",
        )
        c_vis = DocumentIdentityCandidate(
            field_name="invoice_number",
            raw_value="INV-2026-0047",
            normalized_value="INV-2026-0047",
            evidence_ids=("E91",),
            source="vision",
        )

        cands = _make_candidate_container(identities=(c_ocr, c_vis))
        facts = consolidate_candidates(cands)

        # Neither source arbitrarily wins
        assert facts.identity.invoice_number is None
        assert len(facts.conflicting_facts) == 1
        assert facts.conflicting_facts[0]["field"] == "invoice_number"

    def test_a3_equivalent_dates_consolidate(self) -> None:
        """Equivalent date formats (YYYY-MM-DD vs DD/MM/YYYY) consolidate to standard ISO date."""
        c1 = DocumentIdentityCandidate(
            field_name="invoice_date",
            raw_value="2026-03-18",
            normalized_value="2026-03-18",
            evidence_ids=("E1",),
            source="ocr",
        )
        c2 = DocumentIdentityCandidate(
            field_name="invoice_date",
            raw_value="18/03/2026",
            normalized_value="18/03/2026",
            evidence_ids=("E2",),
            source="vision",
        )

        cands = _make_candidate_container(identities=(c1, c2))
        facts = consolidate_candidates(cands)

        assert facts.identity.invoice_date == "2026-03-18"
        assert set(facts.identity.field_evidence_ids["invoice_date"]) == {"E1", "E2"}

    def test_a4_ambiguous_date_not_guessed(self) -> None:
        """Ambiguous date formats (e.g. 05/06/2026) are not guessed; preserved raw when agreeing."""
        c1 = DocumentIdentityCandidate(
            field_name="invoice_date",
            raw_value="05/06/2026",
            normalized_value="05/06/2026",
            evidence_ids=("E1",),
            source="ocr",
        )
        c2 = DocumentIdentityCandidate(
            field_name="invoice_date",
            raw_value="05/06/2026",
            normalized_value="05/06/2026",
            evidence_ids=("E2",),
            source="vision",
        )

        cands = _make_candidate_container(identities=(c1, c2))
        facts = consolidate_candidates(cands)

        # Preserves exact raw agreement without inventing day/month orientation
        assert facts.identity.invoice_date == "05/06/2026"

    def test_a5_contextual_invoice_type_preserved(self) -> None:
        """Header invoice type is preserved; body mention does not override."""
        c_hdr = DocumentIdentityCandidate(
            field_name="invoice_type",
            raw_value="INVOICE",
            normalized_value="invoice",
            invoice_type_value=InvoiceType.INVOICE,
            placement=Placement.HEADER,
            evidence_ids=("E_HDR",),
        )
        c_body = DocumentIdentityCandidate(
            field_name="invoice_type",
            raw_value="Reference to Credit Memo CM-9988",
            normalized_value="credit_memo",
            invoice_type_value=InvoiceType.CREDIT_MEMO,
            placement=Placement.LINE,
            evidence_ids=("E_BODY",),
        )

        cands = _make_candidate_container(identities=(c_hdr, c_body))
        facts = consolidate_candidates(cands)

        assert facts.identity.invoice_type == InvoiceType.INVOICE

    def test_a6_conflicting_header_invoice_types_unresolved(self) -> None:
        """Conflicting header invoice types (e.g. INVOICE vs CREDIT MEMO) yield UNKNOWN."""
        c1 = DocumentIdentityCandidate(
            field_name="invoice_type",
            raw_value="COMMERCIAL INVOICE",
            normalized_value="invoice",
            invoice_type_value=InvoiceType.INVOICE,
            placement=Placement.HEADER,
            evidence_ids=("E1",),
        )
        c2 = DocumentIdentityCandidate(
            field_name="invoice_type",
            raw_value="CREDIT NOTE",
            normalized_value="credit_memo",
            invoice_type_value=InvoiceType.CREDIT_MEMO,
            placement=Placement.HEADER,
            evidence_ids=("E2",),
        )

        cands = _make_candidate_container(identities=(c1, c2))
        facts = consolidate_candidates(cands)

        assert facts.identity.invoice_type == InvoiceType.UNKNOWN
        assert len(facts.conflicting_facts) == 1
        assert facts.conflicting_facts[0]["field"] == "invoice_type"


# ══════════════════════════════════════════════════════════════════════════
# B. Supplier Consolidation
# ══════════════════════════════════════════════════════════════════════════

class TestSupplierConsolidation:
    def test_b1_supplier_consensus_and_complementary_fields(self) -> None:
        """Agreeing supplier name and complementary VAT ID survive together."""
        c_name_ocr = PartyIdentityCandidate(
            party_role="supplier",
            field_name="name",
            raw_value="Meridian Logistics OÜ",
            normalized_value="Meridian Logistics OÜ",
            evidence_ids=("E1",),
            source="ocr",
        )
        c_name_vis = PartyIdentityCandidate(
            party_role="supplier",
            field_name="name",
            raw_value="Meridian Logistics OÜ",
            normalized_value="Meridian Logistics OÜ",
            evidence_ids=("E2",),
            source="vision",
        )
        c_vat_vis = PartyIdentityCandidate(
            party_role="supplier",
            field_name="vat_id",
            raw_value="EE123456789",
            normalized_value="EE123456789",
            evidence_ids=("E3",),
            source="vision",
        )

        cands = _make_candidate_container(parties=(c_name_ocr, c_name_vis, c_vat_vis))
        facts = consolidate_candidates(cands)

        sup = facts.parties.supplier
        assert sup.observed_name == "Meridian Logistics OÜ"
        assert set(sup.field_evidence_ids["observed_name"]) == {"E1", "E2"}
        assert sup.vat_id == "EE123456789"
        assert set(sup.field_evidence_ids["vat_id"]) == {"E3"}
        assert len(facts.conflicting_facts) == 0

    def test_b2_conflicting_supplier_vat_remains_unresolved(self) -> None:
        """Conflicting supplier identifiers (e.g. DE111 vs DE222) remain unresolved."""
        c1 = PartyIdentityCandidate(
            party_role="supplier",
            field_name="vat_id",
            raw_value="DE111111111",
            normalized_value="DE111111111",
            evidence_ids=("E1",),
            source="ocr",
        )
        c2 = PartyIdentityCandidate(
            party_role="supplier",
            field_name="vat_id",
            raw_value="DE222222222",
            normalized_value="DE222222222",
            evidence_ids=("E2",),
            source="vision",
        )

        cands = _make_candidate_container(parties=(c1, c2))
        facts = consolidate_candidates(cands)

        assert facts.parties.supplier.vat_id is None
        assert len(facts.conflicting_facts) == 1
        assert facts.conflicting_facts[0]["field"] == "supplier.vat_id"


# ══════════════════════════════════════════════════════════════════════════
# C. Buyer Consolidation
# ══════════════════════════════════════════════════════════════════════════

class TestBuyerConsolidation:
    def test_c1_buyer_hierarchy_preserved(self) -> None:
        """Buyer company, BU, and location remain distinct without hierarchy collapse."""
        c_comp = PartyIdentityCandidate(
            party_role="buyer",
            field_name="company",
            raw_value="Acme Global Inc",
            normalized_value="Acme Global Inc",
            evidence_ids=("E1",),
        )
        c_bu = PartyIdentityCandidate(
            party_role="buyer",
            field_name="business_unit",
            raw_value="Logistics Division",
            normalized_value="Logistics Division",
            evidence_ids=("E2",),
        )
        c_loc = PartyIdentityCandidate(
            party_role="buyer",
            field_name="location",
            raw_value="Berlin Hub",
            normalized_value="Berlin Hub",
            evidence_ids=("E3",),
        )

        cands = _make_candidate_container(parties=(c_comp, c_bu, c_loc))
        facts = consolidate_candidates(cands)

        buy = facts.parties.buyer
        assert buy.observed_company == "Acme Global Inc"
        assert buy.business_unit == "Logistics Division"
        assert buy.location == "Berlin Hub"
        assert len(facts.conflicting_facts) == 0


# ══════════════════════════════════════════════════════════════════════════
# D. PO Consolidation
# ══════════════════════════════════════════════════════════════════════════

class TestPOConsolidation:
    def test_d1_identical_po_observations_consolidate(self) -> None:
        """Identical PO observations from OCR and Vision consolidate with combined evidence."""
        c1 = POCandidate(field_name="po_number", raw_value="PO-45009988", normalized_value="45009988", evidence_ids=("E1",), source="ocr")
        c2 = POCandidate(field_name="po_number", raw_value="PO: 45009988", normalized_value="45009988", evidence_ids=("E2",), source="vision")

        cands = _make_candidate_container(pos=(c1, c2))
        facts = consolidate_candidates(cands)

        assert facts.po.observed_po_number == "45009988"
        assert set(facts.po.field_evidence_ids["observed_po_number"]) == {"E1", "E2"}

    def test_d2_conflicting_po_observations_remain_unresolved(self) -> None:
        """Conflicting PO observations remain unresolved and enter conflicting_facts."""
        c1 = POCandidate(field_name="po_number", raw_value="PO-100", normalized_value="100", evidence_ids=("E1",), source="ocr")
        c2 = POCandidate(field_name="po_number", raw_value="PO-200", normalized_value="200", evidence_ids=("E2",), source="vision")

        cands = _make_candidate_container(pos=(c1, c2))
        facts = consolidate_candidates(cands)

        assert facts.po.observed_po_number is None
        assert len(facts.conflicting_facts) == 1
        assert facts.conflicting_facts[0]["field"] == "observed_po_number"

    def test_d3_amount_cannot_create_or_substitute_po(self) -> None:
        """PO candidate is never created or inferred from total amount."""
        tot = TotalCandidate(total_type="gross_total", raw_label="Total", raw_value="100.00", normalized_value=Decimal("100.00"), evidence_ids=("ET",))
        cands = _make_candidate_container(totals=(tot,))
        facts = consolidate_candidates(cands)

        assert facts.po.observed_po_number is None


# ══════════════════════════════════════════════════════════════════════════
# E. Line Item Consolidation (Amendment 2: Repeated Identical Lines)
# ══════════════════════════════════════════════════════════════════════════

class TestLineConsolidation:
    def test_e1_same_ocr_and_vision_line_consolidates(self) -> None:
        """OCR and Vision candidates for the same table row consolidate with combined evidence."""
        ln_ocr = LineCandidate(
            source_row_number=1,
            description="Consulting Services",
            quantity=Decimal("10"),
            unit_price=Decimal("100.00"),
            amount=Decimal("1000.00"),
            evidence_ids=("E_L1_OCR",),
            source="ocr",
            page_number=1,
        )
        ln_vis = LineCandidate(
            source_row_number=1,
            description="Consulting Services",
            quantity=Decimal("10"),
            unit_price=Decimal("100.00"),
            amount=Decimal("1000.00"),
            evidence_ids=("E_L1_VIS",),
            source="vision",
            page_number=1,
        )

        cands = _make_candidate_container(lines=(ln_ocr, ln_vis))
        facts = consolidate_candidates(cands)

        assert len(facts.financials.lines) == 1
        line = facts.financials.lines[0]
        assert line.description == "Consulting Services"
        assert line.quantity == Decimal("10")
        assert line.unit_price == Decimal("100.00")
        assert line.amount == Decimal("1000.00")
        assert set(line.evidence_ids) == {"E_L1_OCR", "E_L1_VIS"}

    def test_e2_repeated_identical_lines_remain_separate(self) -> None:
        """Two identical lines on the same document remain two distinct LineFacts (Amendment 2)."""
        # Document with two identical rows: e.g. 2 separate shipments of the same item
        ln1 = LineCandidate(
            source_row_number=1,
            description="Standard Widget",
            quantity=Decimal("1"),
            unit_price=Decimal("50.00"),
            amount=Decimal("50.00"),
            evidence_ids=("E_ROW_1",),
            source="ocr",
            page_number=1,
        )
        ln2 = LineCandidate(
            source_row_number=2,
            description="Standard Widget",
            quantity=Decimal("1"),
            unit_price=Decimal("50.00"),
            amount=Decimal("50.00"),
            evidence_ids=("E_ROW_2",),
            source="ocr",
            page_number=1,
        )

        cands = _make_candidate_container(lines=(ln1, ln2))
        facts = consolidate_candidates(cands)

        assert len(facts.financials.lines) == 2
        assert facts.financials.lines[0].line_number == 1
        assert facts.financials.lines[1].line_number == 2
        assert facts.financials.lines[0].evidence_ids == ("E_ROW_1",)
        assert facts.financials.lines[1].evidence_ids == ("E_ROW_2",)

    def test_e3_missing_line_values_remain_missing_no_arithmetic(self) -> None:
        """Missing unit price or amount is never computed via quantity * price."""
        # Row with qty=2, amount=100, unit_price=None
        ln = LineCandidate(
            source_row_number=1,
            description="Service Fee",
            quantity=Decimal("2"),
            unit_price=None,
            amount=Decimal("100.00"),
            evidence_ids=("E1",),
            source="ocr",
        )

        cands = _make_candidate_container(lines=(ln,))
        facts = consolidate_candidates(cands)

        assert facts.financials.lines[0].unit_price is None
        assert facts.financials.lines[0].quantity == Decimal("2")
        assert facts.financials.lines[0].amount == Decimal("100.00")

    def test_e4_component_detail_distinction_preserved(self) -> None:
        """Component details remain COMPONENT_DETAIL and are not added to billed lines."""
        ln_main = LineCandidate(
            source_row_number=1,
            description="Full Installation",
            amount=Decimal("5000.00"),
            semantic_role=SemanticRole.BILLED_LINE,
            evidence_ids=("E1",),
        )
        ln_sub1 = LineCandidate(
            source_row_number=2,
            description="Labor component",
            amount=Decimal("3000.00"),
            semantic_role=SemanticRole.COMPONENT_DETAIL,
            evidence_ids=("E2",),
        )
        ln_sub2 = LineCandidate(
            source_row_number=3,
            description="Parts component",
            amount=Decimal("2000.00"),
            semantic_role=SemanticRole.COMPONENT_DETAIL,
            evidence_ids=("E3",),
        )

        cands = _make_candidate_container(lines=(ln_main, ln_sub1, ln_sub2))
        facts = consolidate_candidates(cands)

        billed = [ln for ln in facts.financials.lines if ln.semantic_role == SemanticRole.BILLED_LINE]
        comp = [ln for ln in facts.financials.lines if ln.semantic_role == SemanticRole.COMPONENT_DETAIL]

        assert len(billed) == 1
        assert len(comp) == 2
        assert billed[0].description == "Full Installation"


# ══════════════════════════════════════════════════════════════════════════
# F. Tax Consolidation (Amendment 1: Complementary Partial Observations)
# ══════════════════════════════════════════════════════════════════════════

class TestTaxConsolidation:
    def test_f1_complementary_partial_tax_observations_consolidate(self) -> None:
        """OCR with rate+amount and Vision with amount-only merge without calculating missing rate (Amendment 1)."""
        # OCR: VAT / HEADER / 24% / 117.72
        # Vision: VAT / HEADER / None / 117.72
        tx_ocr = TaxCandidate(
            tax_name="VAT",
            rate=Decimal("24.0"),
            amount=Decimal("117.72"),
            placement=Placement.HEADER,
            evidence_ids=("E_TX_OCR",),
            source="ocr",
        )
        tx_vis = TaxCandidate(
            tax_name="VAT",
            rate=None,
            amount=Decimal("117.72"),
            placement=Placement.HEADER,
            evidence_ids=("E_TX_VIS",),
            source="vision",
        )

        cands = _make_candidate_container(taxes=(tx_ocr, tx_vis))
        facts = consolidate_candidates(cands)

        assert len(facts.financials.taxes) == 1
        tx = facts.financials.taxes[0]
        assert tx.tax_name == "VAT"
        assert tx.rate == Decimal("24.0")
        assert tx.amount == Decimal("117.72")
        assert tx.placement == Placement.HEADER
        assert set(tx.evidence_ids) == {"E_TX_OCR", "E_TX_VIS"}

    def test_f2_conflicting_tax_rates_remain_conflicted(self) -> None:
        """Genuine disagreement (e.g. 24% / 117.72 vs 18% / 88.20) is not merged."""
        tx1 = TaxCandidate(
            tax_name="VAT",
            rate=Decimal("24.0"),
            amount=Decimal("117.72"),
            placement=Placement.HEADER,
            evidence_ids=("E1",),
        )
        tx2 = TaxCandidate(
            tax_name="VAT",
            rate=Decimal("18.0"),
            amount=Decimal("88.20"),
            placement=Placement.HEADER,
            evidence_ids=("E2",),
        )

        cands = _make_candidate_container(taxes=(tx1, tx2))
        facts = consolidate_candidates(cands)

        # Remain distinct observations
        assert len(facts.financials.taxes) == 2

    def test_f3_line_vs_header_placement_preserved(self) -> None:
        """Line-level tax is never merged into header tax."""
        tx_line = TaxCandidate(
            tax_name="VAT",
            rate=Decimal("19.0"),
            amount=Decimal("19.00"),
            placement=Placement.LINE,
            evidence_ids=("E_LINE",),
        )
        tx_hdr = TaxCandidate(
            tax_name="VAT",
            rate=Decimal("19.0"),
            amount=Decimal("19.00"),
            placement=Placement.HEADER,
            evidence_ids=("E_HDR",),
        )

        cands = _make_candidate_container(taxes=(tx_line, tx_hdr))
        facts = consolidate_candidates(cands)

        assert len(facts.financials.taxes) == 2
        placements = {t.placement for t in facts.financials.taxes}
        assert placements == {Placement.LINE, Placement.HEADER}


# ══════════════════════════════════════════════════════════════════════════
# G. Discounts and Charges Consolidation
# ══════════════════════════════════════════════════════════════════════════

class TestDiscountsAndChargesConsolidation:
    def test_g1_complementary_discount_consolidation(self) -> None:
        """Discounts with complementary rate and amount merge without calculating."""
        d1 = DiscountCandidate(
            label="Cash Discount",
            rate=Decimal("2.0"),
            amount=Decimal("20.00"),
            placement=Placement.HEADER,
            evidence_ids=("E1",),
        )
        d2 = DiscountCandidate(
            label="Cash Discount",
            rate=None,
            amount=Decimal("20.00"),
            placement=Placement.HEADER,
            evidence_ids=("E2",),
        )

        cands = _make_candidate_container(discounts=(d1, d2))
        facts = consolidate_candidates(cands)

        assert len(facts.financials.discounts) == 1
        disc = facts.financials.discounts[0]
        assert disc.name == "Cash Discount"
        assert disc.rate == Decimal("2.0")
        assert disc.amount == Decimal("20.00")
        assert set(disc.evidence_ids) == {"E1", "E2"}

    def test_g2_charges_separate_from_lines(self) -> None:
        """Freight charge is stored in charges, not in lines."""
        chg = ChargeCandidate(
            label="Freight / Shipping",
            amount=Decimal("45.00"),
            placement=Placement.HEADER,
            evidence_ids=("E_CHG",),
        )
        ln = LineCandidate(
            source_row_number=1,
            description="Widget",
            amount=Decimal("100.00"),
            evidence_ids=("E_LN",),
        )

        cands = _make_candidate_container(charges=(chg,), lines=(ln,))
        facts = consolidate_candidates(cands)

        assert len(facts.financials.charges) == 1
        assert len(facts.financials.lines) == 1
        assert facts.financials.charges[0].amount == Decimal("45.00")


# ══════════════════════════════════════════════════════════════════════════
# H. Printed Totals Consolidation
# ══════════════════════════════════════════════════════════════════════════

class TestPrintedTotalsConsolidation:
    def test_h1_agreeing_printed_totals_consolidate(self) -> None:
        """Agreeing subtotal, tax_total, and gross_total consolidate with evidence."""
        t1 = TotalCandidate(total_type="gross_total", raw_label="Total", raw_value="608.23", normalized_value=Decimal("608.23"), evidence_ids=("E1",), source="ocr")
        t2 = TotalCandidate(total_type="gross_total", raw_label="Total", raw_value="608.23", normalized_value=Decimal("608.23"), evidence_ids=("E2",), source="vision")

        cands = _make_candidate_container(totals=(t1, t2))
        facts = consolidate_candidates(cands)

        assert facts.financials.printed_totals is not None
        assert facts.financials.printed_totals.gross_total == Decimal("608.23")
        assert set(facts.financials.printed_totals.field_evidence_ids["gross_total"]) == {"E1", "E2"}

    def test_h2_conflicting_totals_remain_unresolved_no_balancing(self) -> None:
        """Conflicting gross totals (e.g. 500.00 vs 505.00) remain unresolved and enter conflicting_facts."""
        t1 = TotalCandidate(total_type="gross_total", raw_label="Total", raw_value="500.00", normalized_value=Decimal("500.00"), evidence_ids=("E1",), source="ocr")
        t2 = TotalCandidate(total_type="gross_total", raw_label="Total", raw_value="505.00", normalized_value=Decimal("505.00"), evidence_ids=("E2",), source="vision")

        cands = _make_candidate_container(totals=(t1, t2))
        facts = consolidate_candidates(cands)

        # Neither wins
        assert facts.financials.printed_totals is not None
        assert facts.financials.printed_totals.gross_total is None
        assert len(facts.conflicting_facts) == 1
        assert facts.conflicting_facts[0]["field"] == "printed_totals.gross_total"

    def test_h3_no_arithmetic_balancing_of_unbalanced_totals(self) -> None:
        """Document with subtotal=100, tax=10, gross=120 (printed discrepancy) is recorded as printed without balancing."""
        t_sub = TotalCandidate(total_type="subtotal", raw_label="Subtotal", raw_value="100.00", normalized_value=Decimal("100.00"), evidence_ids=("E1",))
        t_tax = TotalCandidate(total_type="tax_total", raw_label="Tax", raw_value="10.00", normalized_value=Decimal("10.00"), evidence_ids=("E2",))
        t_gross = TotalCandidate(total_type="gross_total", raw_label="Total", raw_value="120.00", normalized_value=Decimal("120.00"), evidence_ids=("E3",))

        cands = _make_candidate_container(totals=(t_sub, t_tax, t_gross))
        facts = consolidate_candidates(cands)

        pt = facts.financials.printed_totals
        assert pt is not None
        assert pt.subtotal == Decimal("100.00")
        assert pt.tax_total == Decimal("10.00")
        assert pt.gross_total == Decimal("120.00")


# ══════════════════════════════════════════════════════════════════════════
# I. Multi-Page & Supporting Document Isolation
# ══════════════════════════════════════════════════════════════════════════

class TestMultiPageAndSupportingDocuments:
    def test_i1_multi_page_same_group_consolidates(self) -> None:
        """Multiple pages within the same group consolidate into one DocumentFacts."""
        p1 = _make_candidate_container(
            group_id="grp_1",
            page_numbers=(1,),
            identities=(
                DocumentIdentityCandidate(field_name="invoice_number", raw_value="INV-100", normalized_value="INV-100", evidence_ids=("E1",)),
            ),
        )
        p2 = _make_candidate_container(
            group_id="grp_1",
            page_numbers=(2,),
            totals=(
                TotalCandidate(total_type="gross_total", raw_label="Total", raw_value="200.00", normalized_value=Decimal("200.00"), evidence_ids=("E2",)),
            ),
        )

        facts = consolidate_candidates([p1, p2])

        assert facts.group_id == "grp_1"
        assert facts.page_numbers == (1, 2)
        assert facts.identity.invoice_number == "INV-100"
        assert facts.financials.printed_totals is not None
        assert facts.financials.printed_totals.gross_total == Decimal("200.00")

    def test_i2_supporting_document_facts_remain_distinct(self) -> None:
        """Supporting document group is referenced via supporting_group_ids; its financials are NOT added."""
        p_payable = _make_candidate_container(
            group_id="grp_payable",
            page_numbers=(1,),
            payable_relevance=PayableRelevance.PAYABLE_CANDIDATE,
            totals=(
                TotalCandidate(total_type="gross_total", raw_label="Total", raw_value="1000.00", normalized_value=Decimal("1000.00"), evidence_ids=("E_PAY",)),
            ),
        )
        p_supporting = _make_candidate_container(
            group_id="grp_packing_list",
            page_numbers=(2,),
            payable_relevance=PayableRelevance.SUPPORTING,
            page_role=PageRole.SUPPORTING_DOCUMENT,
            totals=(
                TotalCandidate(total_type="gross_total", raw_label="Gross Weight", raw_value="500.00", normalized_value=Decimal("500.00"), evidence_ids=("E_SUP",)),
            ),
        )

        facts = consolidate_candidates([p_payable, p_supporting])

        assert facts.group_id == "grp_payable"
        assert facts.supporting_group_ids == ("grp_packing_list",)
        # Payable total is NOT increased or mixed with supporting numbers
        assert facts.financials.printed_totals is not None
        assert facts.financials.printed_totals.gross_total == Decimal("1000.00")


# ══════════════════════════════════════════════════════════════════════════
# J. Provenance & Evidence Contracts
# ══════════════════════════════════════════════════════════════════════════

class TestProvenancePreservation:
    def test_j1_every_observed_fact_retains_evidence(self) -> None:
        """All populated accounting fields trace to non-empty evidence IDs."""
        c_inv = DocumentIdentityCandidate(field_name="invoice_number", raw_value="INV-1", normalized_value="INV-1", evidence_ids=("E_INV",))
        c_sup = PartyIdentityCandidate(party_role="supplier", field_name="name", raw_value="Supplier A", normalized_value="Supplier A", evidence_ids=("E_SUP",))
        c_tot = TotalCandidate(total_type="gross_total", raw_label="Total", raw_value="100.00", normalized_value=Decimal("100.00"), evidence_ids=("E_TOT",))

        cands = _make_candidate_container(identities=(c_inv,), parties=(c_sup,), totals=(c_tot,))
        facts = consolidate_candidates(cands)

        assert len(facts.identity.evidence_ids) > 0
        assert len(facts.parties.supplier.evidence_ids) > 0
        assert facts.financials.printed_totals is not None
        assert len(facts.financials.printed_totals.evidence_ids) > 0
        assert set(facts.evidence_ids) == {"E_INV", "E_SUP", "E_TOT"}


# ══════════════════════════════════════════════════════════════════════════
# K. Determinism (Amendment 3: Ordering Invariance, No Source Precedence)
# ══════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    def test_k1_candidate_input_ordering_invariance(self) -> None:
        """[OCR, Vision] and [Vision, OCR] produce byte-for-byte identical DocumentFacts."""
        c_ocr = DocumentIdentityCandidate(field_name="invoice_number", raw_value="INV-001", normalized_value="INV-001", evidence_ids=("E1",), source="ocr")
        c_vis = DocumentIdentityCandidate(field_name="invoice_number", raw_value="INV-001", normalized_value="INV-001", evidence_ids=("E2",), source="vision")

        cands_order1 = _make_candidate_container(identities=(c_ocr, c_vis))
        cands_order2 = _make_candidate_container(identities=(c_vis, c_ocr))

        facts1 = consolidate_candidates(cands_order1)
        facts2 = consolidate_candidates(cands_order2)

        assert facts1.to_json() == facts2.to_json()

    def test_k2_source_ordering_never_selects_winner_in_conflict(self) -> None:
        """Vision does NOT override OCR, and OCR does NOT override Vision in conflict (Amendment 3)."""
        c_ocr = DocumentIdentityCandidate(field_name="invoice_number", raw_value="INV-A", normalized_value="INV-A", evidence_ids=("E1",), source="ocr")
        c_vis = DocumentIdentityCandidate(field_name="invoice_number", raw_value="INV-B", normalized_value="INV-B", evidence_ids=("E2",), source="vision")

        facts_ocr_first = consolidate_candidates(_make_candidate_container(identities=(c_ocr, c_vis)))
        facts_vis_first = consolidate_candidates(_make_candidate_container(identities=(c_vis, c_ocr)))

        # Neither produces INV-A or INV-B as winning fact; both produce None
        assert facts_ocr_first.identity.invoice_number is None
        assert facts_vis_first.identity.invoice_number is None
        assert facts_ocr_first.to_json() == facts_vis_first.to_json()


# ══════════════════════════════════════════════════════════════════════════
# L. Generalization & Property-Style Tests
# ══════════════════════════════════════════════════════════════════════════

class TestGeneralizationProperties:
    def test_l1_no_arbitrary_third_value_invented(self) -> None:
        """Consolidation never produces an invented third value during consensus or conflict."""
        c1 = DocumentIdentityCandidate(field_name="invoice_number", raw_value="INV-999", normalized_value="INV-999", evidence_ids=("E1",))
        cands = _make_candidate_container(identities=(c1,))
        facts = consolidate_candidates(cands)

        assert facts.identity.invoice_number == "INV-999"

    def test_l2_filename_invariance(self) -> None:
        """Arbitrary document_id does not trigger document-specific rules."""
        c1 = DocumentIdentityCandidate(field_name="invoice_number", raw_value="123", normalized_value="123", evidence_ids=("E1",))
        f1 = consolidate_candidates(_make_candidate_container(doc_id="ARBITRARY_NAME_1", identities=(c1,)))
        f2 = consolidate_candidates(_make_candidate_container(doc_id="ARBITRARY_NAME_2", identities=(c1,)))

        assert f1.identity.invoice_number == f2.identity.invoice_number


# ══════════════════════════════════════════════════════════════════════════
# M. Representative Real Corpus Integration Tests
# ══════════════════════════════════════════════════════════════════════════

class TestRealCorpusConsolidation:
    def test_m1_inv_01_real_corpus_consolidation(self) -> None:
        """INV-01 (German invoice): extracts candidates, consolidates to DocumentFacts with valid provenance."""
        p = Path("artifacts/ocr/INV-01/page_001.json")
        if not p.exists():
            pytest.skip("INV-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)

        assert "INV-01" in facts.document_id
        assert facts.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE
        assert len(facts.evidence_ids) > 0
        # Check round-trip serialization
        serialized = facts.to_json()
        assert "document_id" in serialized

    def test_m2_hld_01_real_corpus_consolidation(self) -> None:
        """HLD-01 (Thai invoice): consolidates observed parties and dates without Thai calendar hacks."""
        p = Path("artifacts/ocr/HLD-01/page_001.json")
        if not p.exists():
            pytest.skip("HLD-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)

        assert "HLD-01" in facts.document_id
        assert len(facts.evidence_ids) > 0

    def test_m3_inv_02_real_corpus_consolidation(self) -> None:
        """INV-02 (Estonian invoice): consolidates IBAN and party attributes with valid evidence."""
        p = Path("artifacts/ocr/INV-02/page_001.json")
        if not p.exists():
            pytest.skip("INV-02 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)

        assert "INV-02" in facts.document_id
        if facts.parties.supplier.bank_iban:
            assert "EE" in facts.parties.supplier.bank_iban

    def test_m4_du_02_multi_page_dossier_consolidation(self) -> None:
        """DU-02: page 1 invoice and page 5 packing list isolate supporting group in DocumentFacts."""
        p1 = Path("artifacts/ocr/DU-02/page_001.json")
        p5 = Path("artifacts/ocr/DU-02/page_005.json")
        if not p1.exists() or not p5.exists():
            pytest.skip("DU-02 artifacts missing")
        pe1 = ocr_json_to_page_evidence(p1)
        pe5 = ocr_json_to_page_evidence(p5)

        cands = extract_candidates_from_document([pe1, pe5])
        doc_facts = consolidate_candidates(cands)

        # Primary facts represent payable candidate
        assert doc_facts.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE
        # Supporting group is referenced
        assert len(doc_facts.supporting_group_ids) >= 1

    def test_m5_hld_03_real_corpus_consolidation(self) -> None:
        """HLD-03: complex Portuguese invoice consolidates without errors."""
        p = Path("artifacts/ocr/HLD-03/page_001.json")
        if not p.exists():
            pytest.skip("HLD-03 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)

        assert "HLD-03" in facts.document_id
        assert len(facts.evidence_ids) > 0


# ══════════════════════════════════════════════════════════════════════════
# N. Additional Edge Cases & Boundaries
# ══════════════════════════════════════════════════════════════════════════

class TestAdditionalBoundaries:
    def test_n1_currency_consensus_and_conflict(self) -> None:
        """Agreeing currencies consolidate; conflicting currencies remain unresolved."""
        # Consensus
        c_eur1 = DocumentIdentityCandidate(field_name="currency", raw_value="EUR", normalized_value="EUR", evidence_ids=("E1",))
        c_eur2 = DocumentIdentityCandidate(field_name="currency", raw_value="EUR", normalized_value="EUR", evidence_ids=("E2",))
        f_eur = consolidate_candidates(_make_candidate_container(identities=(c_eur1, c_eur2)))
        assert f_eur.identity.currency == "EUR"
        assert set(f_eur.identity.field_evidence_ids["currency"]) == {"E1", "E2"}

        # Conflict
        c_usd = DocumentIdentityCandidate(field_name="currency", raw_value="USD", normalized_value="USD", evidence_ids=("E3",))
        f_conf = consolidate_candidates(_make_candidate_container(identities=(c_eur1, c_usd)))
        assert f_conf.identity.currency is None
        assert len(f_conf.conflicting_facts) == 1
        assert f_conf.conflicting_facts[0]["field"] == "currency"

    def test_n2_no_master_ids_injected_into_consolidated_facts(self) -> None:
        """Consolidated fact dictionaries must NOT contain Phase 8 master IDs."""
        c_sup = PartyIdentityCandidate(party_role="supplier", field_name="name", raw_value="Acme Supplier", normalized_value="Acme Supplier", evidence_ids=("E1",))
        c_buy = PartyIdentityCandidate(party_role="buyer", field_name="company", raw_value="Acme Buyer", normalized_value="Acme Buyer", evidence_ids=("E2",))
        c_po = POCandidate(field_name="po_number", raw_value="PO-123", normalized_value="123", evidence_ids=("E3",))

        facts = consolidate_candidates(_make_candidate_container(parties=(c_sup, c_buy), pos=(c_po,)))
        as_dict = facts.to_dict()

        assert "master_id" not in as_dict
        assert as_dict["parties"]["supplier"]["matched_result"] is None
        assert as_dict["parties"]["buyer"]["matched_result"] is None
        assert as_dict["po"]["matched_po_result"] is None

    def test_n3_consolidate_document_groups_multiple_results(self) -> None:
        """consolidate_document_groups returns facts for each distinct document group."""
        g1 = _make_candidate_container(group_id="GRP_A", page_numbers=(1,), payable_relevance=PayableRelevance.PAYABLE_CANDIDATE, identities=(DocumentIdentityCandidate(field_name="invoice_number", raw_value="INV-A", normalized_value="INV-A", evidence_ids=("EA",)),))
        g2 = _make_candidate_container(group_id="GRP_B", page_numbers=(2,), payable_relevance=PayableRelevance.SUPPORTING, identities=(DocumentIdentityCandidate(field_name="invoice_number", raw_value="DOC-B", normalized_value="DOC-B", evidence_ids=("EB",)),))

        results = consolidate_document_groups([g1, g2])
        assert len(results) == 2
        assert results[0].group_id == "GRP_A"
        assert results[0].supporting_group_ids == ("GRP_B",)
        assert results[1].group_id == "GRP_B"
        assert results[1].payable_relevance == PayableRelevance.SUPPORTING

    def test_n4_full_round_trip_serialization(self) -> None:
        """Consolidated DocumentFacts serializes and deserializes deterministically."""
        c_inv = DocumentIdentityCandidate(field_name="invoice_number", raw_value="INV-RT", normalized_value="INV-RT", evidence_ids=("E1",))
        c_tot = TotalCandidate(total_type="gross_total", raw_label="Total", raw_value="250.00", normalized_value=Decimal("250.00"), evidence_ids=("E2",))
        facts = consolidate_candidates(_make_candidate_container(identities=(c_inv,), totals=(c_tot,)))

        json_str = facts.to_json()
        round_tripped = facts.from_json(json_str)

        assert round_tripped.document_id == facts.document_id
        assert round_tripped.identity.invoice_number == "INV-RT"
        assert round_tripped.financials.printed_totals.gross_total == Decimal("250.00")
        assert round_tripped.to_json() == json_str

