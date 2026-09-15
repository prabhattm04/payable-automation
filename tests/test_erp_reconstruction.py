"""tests/test_erp_reconstruction.py — Comprehensive Test Suite for Phase 9C-2 ERP Reconstruction.

Test Categories:
A — Basic ERP reconstruction (valid normalized invoice -> successful ERP reconstruction)
B — Quantity × price calculation via erp.py
C — Missing quantity (preserves None, flags condition, zero invented qty)
D — Missing unit price (preserves None, flags condition, zero invented price)
E — Explicit line amount (observed amount and ERP calculated amount remain separate)
F — Line discount (ERP applies explicit line discount according to existing contract)
G — Header discount (ERP applies explicit header discount according to existing contract)
H — Line tax (line tax remains line-level)
I — Header tax (header tax remains header-level)
J — Other charges (explicit charges enter ERP reconstruction)
K — Missing tax amount (ERP derives tax from rate according to erp.py contract)
L — Printed total discrepancy (observed printed total and ERP reconstructed total coexist; no repair)
M — No balancing component (verify no synthetic tax/discount/charge is created to force balance)
N — Semantic role filtering (COMPONENT_DETAIL is preserved as non-posting and excluded from ERP line items)
O — Supporting isolation (DU-02 EUR primary vs TRY supporting documents remain isolated)
P — PO preservation (printed PO remains observational; no master matching)
Q — Party preservation (observed parties remain observed; no master IDs)
R — Credit memo (tested according to actual ERP semantics; positive magnitudes)
S — Debit memo (tested according to actual ERP semantics)
T — Currency (no currency conversion)
U — Provenance (ERP calculated values identify erp.py contract and calculation function)
V — No mutation (input FinancialStructure remains strictly unchanged)
W — Determinism (repeated reconstruction produces deterministic, identical output)
X — Serialization (round-trip to_dict/from_dict and to_json/from_json)
Y — Multiple payables (independent financial structures reconstructed independently)
Z — ERP contract compliance & delegation (verifies 9C-2 delegates calculations to erp.py via monkeypatch)
RC — Real corpus verification (INV-01, HLD-01, INV-02, DU-02, HLD-03)
"""
from __future__ import annotations

import copy
import json
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
    build_erp_payload,
    reconstruct_erp,
    reconstruct_erp_batch,
)
from src.extraction.financial_assembly import (
    FinancialDocumentAssembly,
    assemble_financial_documents,
)
from src.extraction.validation import validate_financial_document_assembly
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
from src.understanding.page_classifier import PageRole, PayableRelevance, classify_page


# ══════════════════════════════════════════════════════════════════════════
# Test Fact Builders
# ══════════════════════════════════════════════════════════════════════════

def _make_norm_line(
    line_id: str = "line-1",
    line_num: Optional[int] = 1,
    desc: str = "Cloud Infrastructure",
    qty: Optional[str] = "10",
    price: Optional[str] = "50.00",
    amount: Optional[str] = "500.00",
    role: SemanticRole = SemanticRole.BILLED_LINE,
    discounts: Tuple[NormalizedDiscount, ...] = (),
    taxes: Tuple[NormalizedTax, ...] = (),
) -> NormalizedLine:
    return NormalizedLine(
        source_line_id=line_id,
        line_number=line_num,
        description=desc,
        quantity=Decimal(qty) if qty is not None else None,
        unit_price=Decimal(price) if price is not None else None,
        amount=Decimal(amount) if amount is not None else None,
        currency="EUR",
        semantic_role=role,
        discounts=discounts,
        taxes=taxes,
        origin=FactOrigin.OBSERVED,
        evidence_ids=("EV_L1",),
    )


def _make_norm_structure(
    assembly_id: str = "ASM-001",
    doc_id: str = "DOC-001",
    currency: str = "EUR",
    doc_type: InvoiceType = InvoiceType.INVOICE,
    lines: Tuple[NormalizedLine, ...] = (),
    header_discounts: Tuple[NormalizedDiscount, ...] = (),
    header_charges: Tuple[NormalizedCharge, ...] = (),
    header_taxes: Tuple[NormalizedTax, ...] = (),
    printed_totals: Optional[NormalizedPrintedTotals] = None,
    supporting_group_ids: Tuple[str, ...] = (),
) -> FinancialStructure:
    if not lines:
        lines = (_make_norm_line(),)

    return FinancialStructure(
        assembly_id=assembly_id,
        document_id=doc_id,
        document_type=doc_type,
        invoice_number="INV-2026-99",
        invoice_date="2026-02-01",
        currency=currency,
        supplier=NormalizedParty(name="Test Supplier GmbH", origin=FactOrigin.OBSERVED),
        buyer=NormalizedParty(name="Bolt Operations OU", origin=FactOrigin.OBSERVED),
        purchase_order=NormalizedPO(po_number="PO-12345", origin=FactOrigin.OBSERVED),
        lines=lines,
        header_discounts=header_discounts,
        header_charges=header_charges,
        header_taxes=header_taxes,
        printed_totals=printed_totals,
        supporting_group_ids=supporting_group_ids,
        evidence_ids=("EV_DOC",),
    )


# ══════════════════════════════════════════════════════════════════════════
# A. Basic ERP Reconstruction
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryABasicReconstruction:
    def test_a1_valid_invoice_reconstructs_cleanly(self) -> None:
        """A valid normalized invoice produces an ERPReconstruction using erp.py."""
        struct = _make_norm_structure()
        recon = reconstruct_erp(struct)

        assert isinstance(recon, ERPReconstruction)
        assert recon.assembly_id == "ASM-001"
        assert recon.document_id == "DOC-001"
        assert recon.currency == "EUR"
        assert recon.reconstructed_gross_total == Decimal("500.00")
        assert recon.reconstructed_subtotal == Decimal("500.00")
        assert len(recon.lines) == 1
        assert recon.lines[0].reconstructed_base == Decimal("500.00")
        assert recon.lines[0].origin == FactOrigin.DERIVED


# ══════════════════════════════════════════════════════════════════════════
# B. Quantity × Price Calculation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryBQuantityPrice:
    def test_b1_quantity_times_unit_price_reconstructed_by_erp(self) -> None:
        """Line quantity * unit_price is calculated via erp.py."""
        line = _make_norm_line(qty="4", price="73.00", amount=None)
        struct = _make_norm_structure(lines=(line,))
        recon = reconstruct_erp(struct)

        # 4 * 73.00 = 292.00
        assert recon.lines[0].reconstructed_base == Decimal("292.00")
        assert recon.lines[0].observed_amount is None
        assert recon.reconstructed_gross_total == Decimal("292.00")


# ══════════════════════════════════════════════════════════════════════════
# C. Missing Quantity
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryCMissingQuantity:
    def test_c1_missing_quantity_handled_without_invention(self) -> None:
        """Missing quantity produces base 0.0 in ERP without inventing quantity=1."""
        line = _make_norm_line(qty=None, price="100.00", amount=None)
        struct = _make_norm_structure(lines=(line,))
        recon = reconstruct_erp(struct)

        assert recon.lines[0].observed_quantity is None
        assert recon.lines[0].reconstructed_base == Decimal("0.0")
        # Issue logged
        assert any(i.code == "MISSING_QUANTITY" for i in recon.issues)


# ══════════════════════════════════════════════════════════════════════════
# D. Missing Unit Price
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryDMissingUnitPrice:
    def test_d1_missing_unit_price_handled_without_invention(self) -> None:
        """Missing unit price produces base 0.0 without inventing price from amount."""
        line = _make_norm_line(qty="5", price=None, amount="500.00")
        struct = _make_norm_structure(lines=(line,))
        recon = reconstruct_erp(struct)

        assert recon.lines[0].observed_unit_price is None
        assert recon.lines[0].observed_amount == Decimal("500.00")
        assert recon.lines[0].reconstructed_base == Decimal("0.0")
        assert any(i.code == "MISSING_UNIT_PRICE" for i in recon.issues)


# ══════════════════════════════════════════════════════════════════════════
# E. Explicit Line Amount Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryEExplicitLineAmount:
    def test_e1_observed_amount_and_reconstructed_base_coexist(self) -> None:
        """Observed document amount and ERP reconstructed base coexist without overwrite."""
        # Document prints 180.00, but 10 * 20.00 = 200.00
        line = _make_norm_line(qty="10", price="20.00", amount="180.00")
        struct = _make_norm_structure(lines=(line,))
        recon = reconstruct_erp(struct)

        assert recon.lines[0].observed_amount == Decimal("180.00")
        assert recon.lines[0].reconstructed_base == Decimal("200.00")


# ══════════════════════════════════════════════════════════════════════════
# F. Line Discount Application
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryFLineDiscount:
    def test_f1_percentage_discount_applied_to_line(self) -> None:
        """ERP applies line percentage discount: subtotal - (subtotal * pct / 100)."""
        disc = NormalizedDiscount(
            name="10% promo",
            rate=Decimal("10"),
            scope=Placement.LINE,
            origin=FactOrigin.OBSERVED,
        )
        line = _make_norm_line(qty="10", price="100.00", amount=None, discounts=(disc,))
        struct = _make_norm_structure(lines=(line,))
        recon = reconstruct_erp(struct)

        # 10 * 100 = 1000, less 10% = 900.00
        assert recon.lines[0].reconstructed_base == Decimal("900.00")
        assert recon.reconstructed_gross_total == Decimal("900.00")

    def test_f2_amount_discount_applied_to_line(self) -> None:
        """ERP applies line amount discount: (price - disc / qty) * qty."""
        disc = NormalizedDiscount(
            name="Fixed $50 off",
            amount=Decimal("50.00"),
            scope=Placement.LINE,
            origin=FactOrigin.OBSERVED,
        )
        line = _make_norm_line(qty="5", price="100.00", amount=None, discounts=(disc,))
        struct = _make_norm_structure(lines=(line,))
        recon = reconstruct_erp(struct)

        # 500 - 50 = 450.00
        assert recon.lines[0].reconstructed_base == Decimal("450.00")
        assert recon.reconstructed_gross_total == Decimal("450.00")


# ══════════════════════════════════════════════════════════════════════════
# G. Header Discount Application
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryGHeaderDiscount:
    def test_g1_header_discount_subtracted_from_net_base(self) -> None:
        """ERP subtracts header discount from item discounted total."""
        h_disc = NormalizedDiscount(
            name="Special Contract Discount",
            amount=Decimal("100.00"),
            scope=Placement.HEADER,
            origin=FactOrigin.OBSERVED,
        )
        line = _make_norm_line(qty="1", price="1000.00", amount="1000.00")
        struct = _make_norm_structure(lines=(line,), header_discounts=(h_disc,))
        recon = reconstruct_erp(struct)

        # 1000 base - 100 discount = 900 gross
        assert recon.reconstructed_subtotal == Decimal("1000.00")
        assert recon.reconstructed_net == Decimal("900.00")
        assert recon.reconstructed_gross_total == Decimal("900.00")


# ══════════════════════════════════════════════════════════════════════════
# H. Line Tax Scope
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryHLineTax:
    def test_h1_line_tax_applied_to_line_base(self) -> None:
        """Line tax is computed on line base and added to gross."""
        tx = NormalizedTax(
            tax_name="VAT 20%",
            rate=Decimal("20"),
            amount=None,
            scope=Placement.LINE,
            origin=FactOrigin.OBSERVED,
        )
        line = _make_norm_line(qty="2", price="100.00", amount="200.00", taxes=(tx,))
        struct = _make_norm_structure(lines=(line,))
        recon = reconstruct_erp(struct)

        # Base = 200, Tax = 40.00, Gross = 240.00
        assert recon.lines[0].reconstructed_tax == Decimal("40.00")
        assert recon.reconstructed_gross_total == Decimal("240.00")


# ══════════════════════════════════════════════════════════════════════════
# I. Header Tax Scope
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryIHeaderTax:
    def test_i1_header_tax_applied_on_net_base(self) -> None:
        """Header tax is computed on net base (after header discount)."""
        h_disc = NormalizedDiscount(
            amount=Decimal("200.00"),
            scope=Placement.HEADER,
            origin=FactOrigin.OBSERVED,
        )
        h_tx = NormalizedTax(
            tax_name="VAT 19%",
            rate=Decimal("19"),
            amount=None,
            scope=Placement.HEADER,
            origin=FactOrigin.OBSERVED,
        )
        line = _make_norm_line(qty="1", price="1200.00", amount="1200.00")
        struct = _make_norm_structure(lines=(line,), header_discounts=(h_disc,), header_taxes=(h_tx,))
        recon = reconstruct_erp(struct)

        # Net base = 1200 - 200 = 1000.00
        # Tax = 1000 * 0.19 = 190.00
        # Gross = 1000 + 190 = 1190.00
        assert recon.reconstructed_net == Decimal("1000.00")
        assert recon.header_taxes[0].reconstructed_amount == Decimal("190.00")
        assert recon.reconstructed_gross_total == Decimal("1190.00")


# ══════════════════════════════════════════════════════════════════════════
# J. Other Charges
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryJOtherCharges:
    def test_j1_freight_and_insurance_charges_added(self) -> None:
        """Header charges are mapped to ERP other_charges and added to gross."""
        chg_fr = NormalizedCharge(
            name="Freight Delivery",
            amount=Decimal("50.00"),
            charge_category="freight",
            scope=Placement.HEADER,
            origin=FactOrigin.OBSERVED,
        )
        chg_ins = NormalizedCharge(
            name="Transit Insurance",
            amount=Decimal("25.00"),
            charge_category="insurance",
            scope=Placement.HEADER,
            origin=FactOrigin.OBSERVED,
        )
        line = _make_norm_line(qty="1", price="500.00", amount="500.00")
        struct = _make_norm_structure(lines=(line,), header_charges=(chg_fr, chg_ins))
        recon = reconstruct_erp(struct)

        # 500 base + 50 freight + 25 insurance = 575.00
        assert recon.reconstructed_gross_total == Decimal("575.00")


# ══════════════════════════════════════════════════════════════════════════
# K. Missing Tax Amount Reconstructed via ERP
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryKMissingTaxAmount:
    def test_k1_erp_calculates_tax_without_mutating_source(self) -> None:
        """ERP derives tax amount from rate, while source remains amount=None."""
        tx = NormalizedTax(
            tax_name="VAT 10%",
            rate=Decimal("10"),
            amount=None,
            scope=Placement.HEADER,
            origin=FactOrigin.OBSERVED,
        )
        line = _make_norm_line(qty="1", price="100.00", amount="100.00")
        struct = _make_norm_structure(lines=(line,), header_taxes=(tx,))
        recon = reconstruct_erp(struct)

        # Observed rate was 10, observed amount was None
        assert struct.header_taxes[0].amount is None
        # ERP reconstructed amount is 10.00
        assert recon.header_taxes[0].reconstructed_amount == Decimal("10.00")
        assert recon.header_taxes[0].observed_amount is None


# ══════════════════════════════════════════════════════════════════════════
# L. Printed Total Discrepancy Coexistence
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryLPrintedTotalDiscrepancy:
    def test_l1_printed_total_and_reconstructed_total_coexist(self) -> None:
        """Observed printed total (e.g. 1000) and ERP reconstructed total (e.g. 1050) coexist without repair."""
        line = _make_norm_line(qty="1", price="1050.00", amount="1050.00")
        pt = NormalizedPrintedTotals(gross_total=Decimal("1000.00"), origin=FactOrigin.OBSERVED)
        struct = _make_norm_structure(lines=(line,), printed_totals=pt)
        recon = reconstruct_erp(struct)

        assert recon.printed_totals is not None
        assert recon.printed_totals.gross_total == Decimal("1000.00")
        assert recon.reconstructed_gross_total == Decimal("1050.00")
        # Ensure neither overwrote the other
        assert recon.reconstructed_gross_total != recon.printed_totals.gross_total


# ══════════════════════════════════════════════════════════════════════════
# M. No Synthetic Balancing Components
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryMNoBalancingComponents:
    def test_m1_no_synthetic_balancing_charges_or_discounts(self) -> None:
        """When document total differs from line sum, no balancing discount is created."""
        line = _make_norm_line(qty="1", price="100.00", amount="100.00")
        pt = NormalizedPrintedTotals(gross_total=Decimal("80.00"), origin=FactOrigin.OBSERVED)
        struct = _make_norm_structure(lines=(line,), printed_totals=pt)
        recon = reconstruct_erp(struct)

        # No synthetic 20.00 discount created
        assert len(recon.header_discounts) == 0
        assert len(recon.line_discounts) == 0
        assert recon.reconstructed_gross_total == Decimal("100.00")


# ══════════════════════════════════════════════════════════════════════════
# N. Semantic Role Filtering
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryNSemanticRoleFiltering:
    def test_n1_component_detail_excluded_from_erp_booking(self) -> None:
        """COMPONENT_DETAIL rows are preserved with is_posting=False and excluded from ERP booking."""
        billed_line = _make_norm_line(
            line_id="line-post",
            qty="1",
            price="500.00",
            amount="500.00",
            role=SemanticRole.BILLED_LINE,
        )
        detail_line = _make_norm_line(
            line_id="line-detail",
            qty="10",
            price="50.00",
            amount="500.00",
            role=SemanticRole.COMPONENT_DETAIL,
        )
        struct = _make_norm_structure(lines=(billed_line, detail_line))
        recon = reconstruct_erp(struct)

        assert len(recon.lines) == 2
        # Posting line
        assert recon.lines[0].is_posting is True
        assert recon.lines[0].reconstructed_base == Decimal("500.00")
        # Detail line
        assert recon.lines[1].is_posting is False
        assert recon.lines[1].reconstructed_base is None
        # Gross only books the posting line (500.00, NOT 1000.00)
        assert recon.reconstructed_gross_total == Decimal("500.00")


# ══════════════════════════════════════════════════════════════════════════
# O. Supporting Document Isolation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryOSupportingIsolation:
    def test_o1_supporting_groups_remain_references_only(self) -> None:
        """Supporting group IDs remain isolated; no secondary facts enter ERP reconstruction."""
        struct = _make_norm_structure(
            currency="EUR",
            supporting_group_ids=("GRP-CUSTOMS-TRY-01", "GRP-PACKING-02"),
        )
        recon = reconstruct_erp(struct)

        assert recon.currency == "EUR"
        assert recon.supporting_group_ids == ("GRP-CUSTOMS-TRY-01", "GRP-PACKING-02")


# ══════════════════════════════════════════════════════════════════════════
# P. PO Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryPPOPreservation:
    def test_p1_po_preserved_without_master_matching(self) -> None:
        """PO number remains observational; no master matching called."""
        struct = _make_norm_structure()
        recon = reconstruct_erp(struct)
        assert struct.purchase_order is not None
        assert struct.purchase_order.po_number == "PO-12345"


# ══════════════════════════════════════════════════════════════════════════
# Q. Party Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryQPartyPreservation:
    def test_q1_party_details_remain_unmodified(self) -> None:
        """Supplier and Buyer data survive unchanged."""
        struct = _make_norm_structure()
        _ = reconstruct_erp(struct)
        assert struct.supplier is not None
        assert struct.supplier.name == "Test Supplier GmbH"


# ══════════════════════════════════════════════════════════════════════════
# R. Credit Memo Handling
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryRCreditMemo:
    def test_r1_credit_memo_positive_magnitude_per_erp(self) -> None:
        """Credit memo is booked as positive magnitude according to AUTODRAFT_SCHEMA and erp.py."""
        line = _make_norm_line(qty="2", price="50.00", amount="100.00")
        struct = _make_norm_structure(lines=(line,), doc_type=InvoiceType.CREDIT_MEMO)
        recon = reconstruct_erp(struct)

        assert recon.document_type == InvoiceType.CREDIT_MEMO
        assert recon.reconstructed_gross_total == Decimal("100.00")


# ══════════════════════════════════════════════════════════════════════════
# S. Debit Memo Handling
# ══════════════════════════════════════════════════════════════════════════

class TestCategorySDebitMemo:
    def test_s1_debit_memo_reconstruction(self) -> None:
        """Debit memo produces canonical reconstruction."""
        line = _make_norm_line(qty="1", price="150.00", amount="150.00")
        struct = _make_norm_structure(lines=(line,), doc_type=InvoiceType.DEBIT_MEMO)
        recon = reconstruct_erp(struct)

        assert recon.document_type == InvoiceType.DEBIT_MEMO
        assert recon.reconstructed_gross_total == Decimal("150.00")


# ══════════════════════════════════════════════════════════════════════════
# T. Currency Scope
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryTCurrency:
    def test_t1_currency_preserved_no_conversion(self) -> None:
        """Currency is preserved verbatim without exchange rate conversion."""
        struct = _make_norm_structure(currency="TRY")
        recon = reconstruct_erp(struct)
        assert recon.currency == "TRY"


# ══════════════════════════════════════════════════════════════════════════
# U. Provenance
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryUProvenance:
    def test_u1_reconstructed_values_have_provenance_to_erp(self) -> None:
        """Reconstructed line bases indicate provenance to erp.py functions."""
        struct = _make_norm_structure()
        recon = reconstruct_erp(struct)

        assert recon.provenance["erp_engine"] == "erp.py"
        assert "erp._line_base" in recon.lines[0].provenance["erp_function"]


# ══════════════════════════════════════════════════════════════════════════
# V. Non-Mutation Guarantee
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryVNonMutation:
    def test_v1_input_structure_is_never_mutated(self) -> None:
        """Input FinancialStructure remains completely unchanged after reconstruction."""
        struct = _make_norm_structure()
        struct_copy = copy.deepcopy(struct)

        _ = reconstruct_erp(struct)

        assert struct == struct_copy


# ══════════════════════════════════════════════════════════════════════════
# W. Determinism
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryWDeterminism:
    def test_w1_repeated_reconstruction_identical(self) -> None:
        """reconstruct_erp(x) produces identical output on repeated runs."""
        struct = _make_norm_structure()
        recon1 = reconstruct_erp(struct)
        recon2 = reconstruct_erp(struct)

        assert recon1.to_dict() == recon2.to_dict()


# ══════════════════════════════════════════════════════════════════════════
# X. Serialization Round-Trip
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryXSerialization:
    def test_x1_to_dict_and_from_dict(self) -> None:
        """to_dict and from_dict preserve all values and types."""
        struct = _make_norm_structure()
        recon = reconstruct_erp(struct)
        d = recon.to_dict()
        restored = ERPReconstruction.from_dict(d)

        assert recon.to_dict() == restored.to_dict()

    def test_x2_to_json_and_from_json(self) -> None:
        """to_json and from_json serialize and deserialize deterministically."""
        struct = _make_norm_structure()
        recon = reconstruct_erp(struct)
        j_str = recon.to_json()
        restored = ERPReconstruction.from_json(j_str)

        assert recon.to_dict() == restored.to_dict()


# ══════════════════════════════════════════════════════════════════════════
# Y. Multiple Payables Isolation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryYMultiplePayables:
    def test_y1_batch_reconstruction_isolates_payables(self) -> None:
        """Multiple financial structures are reconstructed independently without leakage."""
        s1 = _make_norm_structure(assembly_id="ASM-1", currency="EUR")
        s2 = _make_norm_structure(assembly_id="ASM-2", currency="USD")

        results = reconstruct_erp_batch([s1, s2])
        assert len(results) == 2
        assert results[0].assembly_id == "ASM-1"
        assert results[0].currency == "EUR"
        assert results[1].assembly_id == "ASM-2"
        assert results[1].currency == "USD"


# ══════════════════════════════════════════════════════════════════════════
# Z. ERP Contract Compliance & Delegation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryZERPAuditAndDelegation:
    def test_z1_erp_delegation_via_monkeypatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify 9C-2 delegates accounting calculations to erp.py rather than calculating locally."""
        struct = _make_norm_structure()

        # Monkeypatch erp.erp_book to return an artificial value
        monkeypatch.setattr(erp, "erp_book", lambda payload: {"will_book_gross": 9999.99, "currency": "EUR"})

        recon = reconstruct_erp(struct)

        # Confirm the output reflects the monkeypatched erp.py result
        assert recon.reconstructed_gross_total == Decimal("9999.99")

    def test_z2_no_forbidden_network_or_matching(self) -> None:
        """Inspect src/accounting/erp_reconstruction.py for forbidden imports."""
        src_path = Path("src/accounting/erp_reconstruction.py")
        content = src_path.read_text(encoding="utf-8")

        forbidden = [
            "rapidocr",
            "paddleocr",
            "qwen",
            "requests",
            "urllib.request",
            "httpx",
            "supplier_matcher",
            "buyer_matcher",
            "po_matcher",
            "tax_matcher",
        ]
        for f in forbidden:
            assert f not in content, f"Forbidden import '{f}' detected in erp_reconstruction.py"


# ══════════════════════════════════════════════════════════════════════════
# RC. Real Corpus Verification (INV-01, HLD-01, INV-02, DU-02, HLD-03)
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryRealCorpusReconstruction:
    def test_rc1_inv_01_erp_reconstruction(self) -> None:
        """INV-01: Reconstructs German invoice using erp.py."""
        p = Path("artifacts/ocr/INV-01/page_001.json")
        if not p.exists():
            pytest.skip("INV-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        struct = normalize_financial_structure(assemblies[0])

        recon = reconstruct_erp(struct)

        assert recon.document_id == "INV-01.pdf"
        assert recon.currency == "EUR"
        assert len(recon.lines) > 0
        assert recon.printed_totals is not None
        assert recon.printed_totals.gross_total == Decimal("438.00")
        assert recon.reconstructed_gross_total is not None

    def test_rc2_hld_01_erp_reconstruction(self) -> None:
        """HLD-01: Reconstructs Thai invoice; header tax and billed lines entered into erp.py."""
        p = Path("artifacts/ocr/HLD-01/page_001.json")
        if not p.exists():
            pytest.skip("HLD-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        struct = normalize_financial_structure(assemblies[0])

        recon = reconstruct_erp(struct)

        assert recon.document_id == "HLD-01.pdf"
        assert len(recon.lines) > 0
        assert len(recon.header_taxes) > 0
        assert recon.reconstructed_gross_total is not None

    def test_rc3_inv_02_erp_reconstruction(self) -> None:
        """INV-02: Reconstructs Estonian invoice without PO matching."""
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

        assert recon.document_id == "INV-02.pdf"
        assert recon.currency == "EUR"
        assert len(recon.lines) > 0

    def test_rc4_du_02_erp_reconstruction(self) -> None:
        """DU-02: Only EUR primary invoice enters ERP reconstruction; supporting TRY pages isolated."""
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

        assert recon.currency == "EUR"
        assert len(recon.supporting_group_ids) > 0
        # Reconstructed gross is in EUR
        assert recon.erp_book_result["currency"] == "EUR"

    def test_rc5_hld_03_erp_reconstruction(self) -> None:
        """HLD-03: Reconstructs Portuguese invoice."""
        p = Path("artifacts/ocr/HLD-03/page_001.json")
        if not p.exists():
            pytest.skip("HLD-03 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        struct = normalize_financial_structure(assemblies[0])

        recon = reconstruct_erp(struct)

        assert recon.document_id == "HLD-03.pdf"
        assert recon.currency == "EUR"
        assert len(recon.lines) > 0
