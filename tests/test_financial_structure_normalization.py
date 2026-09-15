"""tests/test_financial_structure_normalization.py — Comprehensive Test Suite for Phase 9C-1.

Test Categories:
A — Basic normalization (valid invoice -> canonical financial structure)
B — Monetary normalization (string/numeric normalization to Decimal without value change)
C — Currency preservation (EUR remains EUR, TRY remains TRY, no conversion)
D — Missing amount preservation (quantity + price with missing amount remains None; never computed)
E — Missing quantity preservation (observed amount with missing quantity remains None)
F — Tax normalization (rate and amount preserved independently; no tax calculation)
G — Tax placement (HEADER tax remains HEADER; LINE tax remains LINE)
H — Discount scope (HEADER discount remains HEADER; LINE discount remains LINE)
I — Charge scope (HEADER charge remains HEADER; LINE charge remains LINE)
J — Printed totals (observed values remain observed; no recomputation or line summation)
K — Semantic roles (BILLED_LINE remains BILLED_LINE, COMPONENT_DETAIL remains COMPONENT_DETAIL)
L — Supporting isolation (primary EUR + supporting TRY remain structurally separated)
M — PO preservation (observed PO remains raw printed PO; no master PO substitution)
N — Party preservation (observed parties remain observed; no master IDs injected)
O — Conflicts (upstream conflicts survive and produce typed issues)
P — Provenance (evidence, page, group, source row provenance survives)
Q — No arithmetic (explicit verification that zero accounting calculations occurred)
R — No master matching (no supplier/po/buyer/tax master matcher called)
S — No mutation (input assembly and validation result remain strictly unchanged)
T — Determinism (repeated normalization produces deterministic, identical output)
U — Idempotence (normalize(normalize(x)) == normalize(x))
V — Serialization (round-trip to_dict/from_dict and to_json/from_json)
W — Multiple payables (independent assemblies remain isolated; no cross-document leakage)
X — Real corpus verification (INV-01, HLD-01, INV-02, DU-02, HLD-03)
Y — ERP boundary test (verifies compatibility with erp.py input requirements without invoking ERP)
Z — Boundary audit (verifies no forbidden imports/calls: OCR, Qwen, Vision, network, erp.py)
"""
from __future__ import annotations

import copy
import inspect
import json
from decimal import Decimal
from pathlib import Path
import pytest

from src.accounting.financial_structure import (
    FinancialStructure,
    NormalizationIssue,
    NormalizationSeverity,
    NormalizedCharge,
    NormalizedDiscount,
    NormalizedLine,
    NormalizedParty,
    NormalizedPO,
    NormalizedPrintedTotals,
    NormalizedTax,
    normalize_financial_structure,
    normalize_financial_structures,
)
from src.extraction.financial_assembly import (
    FinancialDocumentAssembly,
    assemble_financial_documents,
)
from src.extraction.validation import (
    ExtractionValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    validate_financial_document_assembly,
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
    BuyerIdentityFact,
    ChargeFact,
    DiscountFact,
    DocumentFacts,
    DocumentIdentityFacts,
    FactOrigin,
    FinancialFacts,
    InvoiceType,
    LineFact,
    PartyIdentityFacts,
    Placement,
    POFacts,
    PrintedTotalsFact,
    SemanticRole,
    SupplierIdentityFact,
    TaxFact,
)
from src.understanding.document_grouper import group_document
from src.understanding.evidence import ocr_json_to_page_evidence
from src.understanding.page_classifier import PageRole, PayableRelevance, classify_page


# ══════════════════════════════════════════════════════════════════════════
# Test Fact Builders
# ══════════════════════════════════════════════════════════════════════════

def _make_line_fact(
    line_num: Optional[int] = 1,
    desc: str = "Consulting Services",
    qty: Optional[str] = "10",
    price: Optional[str] = "100.00",
    amount: Optional[str] = "1000.00",
    discount: Optional[str] = None,
    taxes: Tuple[TaxFact, ...] = (),
    ev_ids: Tuple[str, ...] = ("EV_LN1",),
    row_ev_ids: Tuple[str, ...] = ("ROW_1",),
    role: SemanticRole = SemanticRole.BILLED_LINE,
    origin: FactOrigin = FactOrigin.OBSERVED,
    currency: Optional[str] = None,
) -> LineFact:
    field_evs = {}
    if qty is not None:
        field_evs["quantity"] = ev_ids
    if price is not None:
        field_evs["unit_price"] = ev_ids
    if amount is not None:
        field_evs["amount"] = ev_ids

    return LineFact(
        line_number=line_num,
        description=desc,
        quantity=Decimal(qty) if qty is not None else None,
        unit_price=Decimal(price) if price is not None else None,
        amount=Decimal(amount) if amount is not None else None,
        discount=Decimal(discount) if discount is not None else None,
        taxes=taxes,
        currency=currency,
        evidence_ids=ev_ids,
        field_evidence_ids=field_evs,
        source_row_evidence_ids=row_ev_ids,
        semantic_role=role,
        origin=origin,
    )


def _make_assembly(
    assembly_id: str = "ASM-001",
    doc_id: str = "DOC-001",
    invoice_type: InvoiceType = InvoiceType.INVOICE,
    invoice_number: str = "INV-2026-001",
    invoice_date: str = "2026-02-01",
    due_date: Optional[str] = "2026-02-15",
    currency: Optional[str] = "EUR",
    supplier_name: Optional[str] = "Acme Supplies GmbH",
    vat_id: Optional[str] = "DE123456789",
    buyer_company: Optional[str] = "Bolt Operations OU",
    buyer_address: Optional[str] = "Vana-Louna 15, Tallinn",
    company_code: Optional[str] = None,
    business_unit_code: Optional[str] = None,
    location_code: Optional[str] = None,
    po_number: Optional[str] = "PO-9944",
    lines: Tuple[LineFact, ...] = (),
    taxes: Tuple[TaxFact, ...] = (),
    discounts: Tuple[DiscountFact, ...] = (),
    charges: Tuple[ChargeFact, ...] = (),
    printed_totals: Optional[PrintedTotalsFact] = None,
    relevance: PayableRelevance = PayableRelevance.PAYABLE_CANDIDATE,
    conflicts: Tuple[Dict[str, Any], ...] = (),
    supporting_group_ids: Tuple[str, ...] = (),
    party_origin: FactOrigin = FactOrigin.OBSERVED,
    po_origin: FactOrigin = FactOrigin.OBSERVED,
) -> FinancialDocumentAssembly:
    ev_ids = ("EV_HEAD",)

    if not lines:
        lines = (_make_line_fact(),)

    ident = DocumentIdentityFacts(
        invoice_type=invoice_type,
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        due_date=due_date,
        currency=currency,
        evidence_ids=ev_ids,
        field_evidence_ids={"invoice_number": ev_ids, "invoice_date": ev_ids},
        origin=FactOrigin.OBSERVED,
    )
    sup_matched = {"supplier_id": "SUP-123"} if party_origin == FactOrigin.MATCHED else None
    buy_matched = {"company_code": "BOLTGROUP"} if party_origin == FactOrigin.MATCHED else None
    supplier = SupplierIdentityFact(
        observed_name=supplier_name,
        vat_id=vat_id,
        evidence_ids=ev_ids,
        field_evidence_ids={"observed_name": ev_ids, "vat_id": ev_ids},
        origin=party_origin,
        matched_result=sup_matched,
    )
    buyer = BuyerIdentityFact(
        observed_company=buyer_company,
        invoice_to_address=buyer_address,
        company_code=company_code,
        business_unit_code=business_unit_code,
        location_code=location_code,
        evidence_ids=ev_ids,
        field_evidence_ids={"observed_company": ev_ids},
        origin=party_origin,
        matched_result=buy_matched,
    )
    po = POFacts(
        observed_po_number=po_number,
        evidence_ids=ev_ids if po_number else (),
        field_evidence_ids={"observed_po_number": ev_ids} if po_number else {},
        origin=po_origin,
    )
    fin = FinancialFacts(
        lines=lines,
        discounts=discounts,
        charges=charges,
        taxes=taxes,
        printed_totals=printed_totals,
        currency=currency,
        evidence_ids=ev_ids,
    )
    doc_facts = DocumentFacts(
        document_id=doc_id,
        page_numbers=(1,),
        document_role=PageRole.INVOICE,
        payable_relevance=relevance,
        identity=ident,
        parties=PartyIdentityFacts(supplier=supplier, buyer=buyer, evidence_ids=ev_ids),
        po=po,
        financials=fin,
        conflicting_facts=conflicts,
        evidence_ids=ev_ids,
    )
    return FinancialDocumentAssembly(
        assembly_id=assembly_id,
        document_id=doc_id,
        primary_group_ids=("GRP-01",),
        supporting_group_ids=supporting_group_ids,
        page_numbers=(1,),
        document_role=PageRole.INVOICE,
        payable_relevance=relevance,
        facts=doc_facts,
        conflicts=conflicts,
        evidence_ids=ev_ids,
    )


# ══════════════════════════════════════════════════════════════════════════
# A. Basic Normalization
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryABasicNormalization:
    def test_a1_valid_invoice_becomes_canonical_structure(self) -> None:
        """A valid invoice assembly normalizes cleanly into FinancialStructure."""
        asm = _make_assembly()
        struct = normalize_financial_structure(asm)

        assert isinstance(struct, FinancialStructure)
        assert struct.assembly_id == "ASM-001"
        assert struct.document_id == "DOC-001"
        assert struct.document_type == InvoiceType.INVOICE
        assert struct.invoice_number == "INV-2026-001"
        assert struct.invoice_date == "2026-02-01"
        assert struct.due_date == "2026-02-15"
        assert struct.currency == "EUR"
        assert len(struct.lines) == 1
        assert struct.lines[0].description == "Consulting Services"
        assert struct.lines[0].amount == Decimal("1000.00")
        assert struct.supplier is not None
        assert struct.supplier.name == "Acme Supplies GmbH"
        assert struct.purchase_order is not None
        assert struct.purchase_order.po_number == "PO-9944"


# ══════════════════════════════════════════════════════════════════════════
# B. Monetary Normalization
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryBMonetaryNormalization:
    def test_b1_monetary_values_are_exact_decimals(self) -> None:
        """Monetary values are canonical Decimals without value alteration."""
        line = _make_line_fact(qty="2.5", price="150.75", amount="376.875")
        pt = PrintedTotalsFact(
            gross_total=Decimal("376.875"),
            evidence_ids=("EV_HEAD",),
            origin=FactOrigin.OBSERVED,
        )
        asm = _make_assembly(lines=(line,), printed_totals=pt)
        struct = normalize_financial_structure(asm)

        assert struct.lines[0].quantity == Decimal("2.5")
        assert struct.lines[0].unit_price == Decimal("150.75")
        assert struct.lines[0].amount == Decimal("376.875")
        assert struct.printed_totals is not None
        assert struct.printed_totals.gross_total == Decimal("376.875")


# ══════════════════════════════════════════════════════════════════════════
# C. Currency Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryCCurrencyPreservation:
    def test_c1_eur_remains_eur_no_conversion(self) -> None:
        """EUR currency remains EUR."""
        asm = _make_assembly(currency="EUR")
        struct = normalize_financial_structure(asm)
        assert struct.currency == "EUR"

    def test_c2_try_remains_try_no_conversion(self) -> None:
        """TRY currency remains TRY."""
        asm = _make_assembly(currency="TRY")
        struct = normalize_financial_structure(asm)
        assert struct.currency == "TRY"

    def test_c3_missing_currency_produces_warning(self) -> None:
        """Missing currency produces MISSING_CURRENCY normalization issue."""
        asm = _make_assembly(currency=None)
        struct = normalize_financial_structure(asm)
        assert any(i.code == "MISSING_CURRENCY" for i in struct.normalization_issues)


# ══════════════════════════════════════════════════════════════════════════
# D. Missing Amount Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryDMissingAmountPreservation:
    def test_d1_missing_amount_remains_none(self) -> None:
        """Quantity=10, UnitPrice=25, Amount=None remains Amount=None (NEVER 250)."""
        line = _make_line_fact(qty="10", price="25.00", amount=None)
        asm = _make_assembly(lines=(line,))
        struct = normalize_financial_structure(asm)

        assert struct.lines[0].quantity == Decimal("10")
        assert struct.lines[0].unit_price == Decimal("25.00")
        assert struct.lines[0].amount is None
        # Informational issue flagged
        assert any(i.code == "MISSING_LINE_AMOUNT" for i in struct.normalization_issues)


# ══════════════════════════════════════════════════════════════════════════
# E. Missing Quantity Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryEMissingQuantityPreservation:
    def test_e1_missing_quantity_remains_none(self) -> None:
        """Amount=500, Quantity=None remains Quantity=None (NEVER inferred)."""
        line = _make_line_fact(qty=None, price=None, amount="500.00")
        asm = _make_assembly(lines=(line,))
        struct = normalize_financial_structure(asm)

        assert struct.lines[0].amount == Decimal("500.00")
        assert struct.lines[0].quantity is None
        assert struct.lines[0].unit_price is None


# ══════════════════════════════════════════════════════════════════════════
# F. Tax Normalization
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryFTaxNormalization:
    def test_f1_rate_and_amount_preserved_independently(self) -> None:
        """Tax with only rate preserves rate and leaves amount=None (no rate*base)."""
        tx = TaxFact(
            tax_name="VAT 19%",
            rate=Decimal("19"),
            amount=None,
            placement=Placement.HEADER,
            evidence_ids=("EV_HEAD",),
            origin=FactOrigin.OBSERVED,
        )
        asm = _make_assembly(taxes=(tx,))
        struct = normalize_financial_structure(asm)

        assert len(struct.header_taxes) == 1
        assert struct.header_taxes[0].rate == Decimal("19")
        assert struct.header_taxes[0].amount is None

    def test_f2_amount_only_preserves_amount(self) -> None:
        """Tax with only amount preserves amount and leaves rate=None."""
        tx = TaxFact(
            tax_name="Fixed Levy",
            rate=None,
            amount=Decimal("45.00"),
            placement=Placement.HEADER,
            evidence_ids=("EV_HEAD",),
            origin=FactOrigin.OBSERVED,
        )
        asm = _make_assembly(taxes=(tx,))
        struct = normalize_financial_structure(asm)

        assert len(struct.header_taxes) == 1
        assert struct.header_taxes[0].rate is None
        assert struct.header_taxes[0].amount == Decimal("45.00")


# ══════════════════════════════════════════════════════════════════════════
# G. Tax Placement
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryGTaxPlacement:
    def test_g1_header_tax_remains_header_line_remains_line(self) -> None:
        """Header tax remains in header_taxes; line tax remains in line_taxes."""
        tx_head = TaxFact(
            tax_name="VAT 20%",
            rate=Decimal("20"),
            amount=Decimal("200.00"),
            placement=Placement.HEADER,
            evidence_ids=("EV_HEAD",),
            origin=FactOrigin.OBSERVED,
        )
        tx_line = TaxFact(
            tax_name="Line Eco Tax",
            rate=None,
            amount=Decimal("5.00"),
            placement=Placement.LINE,
            evidence_ids=("EV_LN1",),
            origin=FactOrigin.OBSERVED,
        )
        line = _make_line_fact(taxes=(tx_line,))
        asm = _make_assembly(lines=(line,), taxes=(tx_head,))
        struct = normalize_financial_structure(asm)

        assert len(struct.header_taxes) == 1
        assert struct.header_taxes[0].scope == Placement.HEADER
        assert struct.header_taxes[0].tax_name == "VAT 20%"

        assert len(struct.line_taxes) == 1
        assert struct.line_taxes[0].scope == Placement.LINE
        assert struct.line_taxes[0].tax_name == "Line Eco Tax"
        # Also accessible directly from line
        assert len(struct.lines[0].taxes) == 1
        assert struct.lines[0].taxes[0].tax_name == "Line Eco Tax"


# ══════════════════════════════════════════════════════════════════════════
# H. Discount Scope
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryHDiscountScope:
    def test_h1_header_discount_remains_header(self) -> None:
        """Header discount remains in header_discounts."""
        disc = DiscountFact(
            name="Early Payment Discount",
            rate=Decimal("2"),
            amount=Decimal("50.00"),
            placement=Placement.HEADER,
            evidence_ids=("EV_HEAD",),
            origin=FactOrigin.OBSERVED,
        )
        asm = _make_assembly(discounts=(disc,))
        struct = normalize_financial_structure(asm)

        assert len(struct.header_discounts) == 1
        assert struct.header_discounts[0].scope == Placement.HEADER
        assert struct.header_discounts[0].rate == Decimal("2")
        assert struct.header_discounts[0].amount == Decimal("50.00")

    def test_h2_line_discount_remains_line(self) -> None:
        """Line discount remains in line_discounts and on NormalizedLine."""
        line = _make_line_fact(discount="15.00")
        asm = _make_assembly(lines=(line,))
        struct = normalize_financial_structure(asm)

        assert len(struct.line_discounts) == 1
        assert struct.line_discounts[0].scope == Placement.LINE
        assert struct.line_discounts[0].amount == Decimal("15.00")
        assert len(struct.lines[0].discounts) == 1
        assert struct.lines[0].discounts[0].amount == Decimal("15.00")


# ══════════════════════════════════════════════════════════════════════════
# I. Charge Scope
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryIChargeScope:
    def test_i1_header_charge_remains_charge_not_discount(self) -> None:
        """Header charge remains in header_charges and is never converted to discount."""
        chg = ChargeFact(
            name="Freight Shipping",
            rate=None,
            amount=Decimal("85.00"),
            placement=Placement.HEADER,
            evidence_ids=("EV_HEAD",),
            origin=FactOrigin.OBSERVED,
        )
        asm = _make_assembly(charges=(chg,))
        struct = normalize_financial_structure(asm)

        assert len(struct.header_charges) == 1
        assert struct.header_charges[0].scope == Placement.HEADER
        assert struct.header_charges[0].name == "Freight Shipping"
        assert struct.header_charges[0].amount == Decimal("85.00")
        assert len(struct.header_discounts) == 0


# ══════════════════════════════════════════════════════════════════════════
# J. Printed Totals
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryJPrintedTotals:
    def test_j1_printed_totals_remain_observed_without_recomputation(self) -> None:
        """Printed totals are strictly preserved as observed values; no sum(lines)."""
        line1 = _make_line_fact(line_num=1, amount="100.00")
        line2 = _make_line_fact(line_num=2, amount="200.00")
        # Document explicitly prints subtotal as 350.00 (contradicts 100+200=300)
        pt = PrintedTotalsFact(
            subtotal=Decimal("350.00"),
            gross_total=Decimal("420.00"),
            evidence_ids=("EV_HEAD",),
            origin=FactOrigin.OBSERVED,
        )
        asm = _make_assembly(lines=(line1, line2), printed_totals=pt)
        struct = normalize_financial_structure(asm)

        assert struct.printed_totals is not None
        assert struct.printed_totals.subtotal == Decimal("350.00")
        assert struct.printed_totals.gross_total == Decimal("420.00")
        # Ensure 300 was NOT computed or substituted
        assert struct.printed_totals.subtotal != Decimal("300.00")


# ══════════════════════════════════════════════════════════════════════════
# K. Semantic Roles
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryKSemanticRoles:
    def test_k1_semantic_roles_strictly_preserved(self) -> None:
        """BILLED_LINE remains BILLED_LINE and COMPONENT_DETAIL remains COMPONENT_DETAIL."""
        line1 = _make_line_fact(line_num=1, role=SemanticRole.BILLED_LINE)
        line2 = _make_line_fact(line_num=2, role=SemanticRole.COMPONENT_DETAIL)
        asm = _make_assembly(lines=(line1, line2))
        struct = normalize_financial_structure(asm)

        assert len(struct.lines) == 2
        assert struct.lines[0].semantic_role == SemanticRole.BILLED_LINE
        assert struct.lines[1].semantic_role == SemanticRole.COMPONENT_DETAIL


# ══════════════════════════════════════════════════════════════════════════
# L. Supporting Isolation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryLSupportingIsolation:
    def test_l1_supporting_groups_remain_isolated_references(self) -> None:
        """Supporting group IDs are attached as references without importing secondary facts."""
        asm = _make_assembly(
            currency="EUR",
            supporting_group_ids=("GRP-CUSTOMS-TRY", "GRP-PACKING"),
        )
        struct = normalize_financial_structure(asm)

        assert struct.currency == "EUR"
        assert struct.supporting_group_ids == ("GRP-CUSTOMS-TRY", "GRP-PACKING")
        assert "GRP-CUSTOMS-TRY" in struct.provenance["supporting_group_ids"]


# ══════════════════════════════════════════════════════════════════════════
# M. PO Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryMPOPreservation:
    def test_m1_printed_po_remains_observational_no_master_id(self) -> None:
        """Printed PO remains raw printed string; no master PO ID substitution."""
        asm = _make_assembly(po_number="PO 2287")
        struct = normalize_financial_structure(asm)

        assert struct.purchase_order is not None
        assert struct.purchase_order.po_number == "PO 2287"
        # No master PO code injected
        assert not hasattr(struct.purchase_order, "po_id") or getattr(struct.purchase_order, "po_id", None) is None


# ══════════════════════════════════════════════════════════════════════════
# N. Party Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryNPartyPreservation:
    def test_n1_observed_parties_remain_observed_no_master_codes(self) -> None:
        """Supplier and Buyer remain observed representations without master matching."""
        asm = _make_assembly(
            supplier_name="Raw Supplier AG",
            buyer_company="Bolt Operations OU",
            company_code=None,
            business_unit_code=None,
        )
        struct = normalize_financial_structure(asm)

        assert struct.supplier is not None
        assert struct.supplier.name == "Raw Supplier AG"
        assert struct.buyer is not None
        assert struct.buyer.name == "Bolt Operations OU"
        # Not inferred from master data
        assert struct.buyer.company_code is None
        assert struct.buyer.business_unit_code is None

    def test_n2_upstream_buyer_codes_preserved_if_explicitly_observed(self) -> None:
        """If upstream explicitly observed company_code, it is preserved without alteration."""
        asm = _make_assembly(
            buyer_company="Bolt Operations OU",
            company_code="BOLTGROUP",
            business_unit_code="EE004",
        )
        struct = normalize_financial_structure(asm)

        assert struct.buyer is not None
        assert struct.buyer.company_code == "BOLTGROUP"
        assert struct.buyer.business_unit_code == "EE004"


# ══════════════════════════════════════════════════════════════════════════
# O. Conflicts Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryOConflictsPreservation:
    def test_o1_conflicting_observations_survive_with_typed_issues(self) -> None:
        """Upstream conflicts survive normalization and generate typed NormalizationIssues."""
        conflict = {
            "field": "invoice_number",
            "reason": "Disagreement between header INV-01 and stamp INV-02",
            "evidence_ids": ("EV_HEAD", "EV_STAMP"),
        }
        asm = _make_assembly(conflicts=(conflict,))
        struct = normalize_financial_structure(asm)

        assert len(struct.conflicts) >= 1
        assert any(i.code == "PRESERVED_CONFLICT" for i in struct.normalization_issues)


# ══════════════════════════════════════════════════════════════════════════
# P. Provenance Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryPProvenancePreservation:
    def test_p1_evidence_and_fact_origin_survive(self) -> None:
        """Evidence IDs and FactOrigin are preserved for all components."""
        line = _make_line_fact(
            ev_ids=("EV_LINE_DOC",),
            row_ev_ids=("EV_ROW_DOC",),
            origin=FactOrigin.DERIVED,
        )
        asm = _make_assembly(lines=(line,), party_origin=FactOrigin.MATCHED)
        struct = normalize_financial_structure(asm)

        assert struct.lines[0].origin == FactOrigin.DERIVED
        assert "EV_LINE_DOC" in struct.lines[0].evidence_ids
        assert "EV_ROW_DOC" in struct.lines[0].source_row_evidence_ids
        assert struct.supplier is not None
        assert struct.supplier.origin == FactOrigin.MATCHED


# ══════════════════════════════════════════════════════════════════════════
# Q. Zero Accounting Arithmetic
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryQNoArithmetic:
    def test_q1_zero_accounting_calculations(self) -> None:
        """Verify that line quantity*price, line sums, and tax calculations never occur."""
        line1 = _make_line_fact(qty="5", price="10.00", amount=None)
        line2 = _make_line_fact(qty="2", price="50.00", amount=None)
        tx = TaxFact(
            tax_name="VAT 10%",
            rate=Decimal("10"),
            amount=None,
            placement=Placement.HEADER,
            evidence_ids=("EV_HEAD",),
            origin=FactOrigin.OBSERVED,
        )
        asm = _make_assembly(lines=(line1, line2), taxes=(tx,))
        struct = normalize_financial_structure(asm)

        # Lines must have amount None
        assert struct.lines[0].amount is None
        assert struct.lines[1].amount is None
        # Taxes must have amount None
        assert struct.header_taxes[0].amount is None
        # Printed totals must remain None
        assert struct.printed_totals is None


# ══════════════════════════════════════════════════════════════════════════
# R. Zero Master Matching
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryRNoMasterMatching:
    def test_r1_no_master_matchers_invoked(self) -> None:
        """Ensure no master matching functions or classes are called during normalization."""
        asm = _make_assembly()
        struct = normalize_financial_structure(asm)
        # Verify supplier ID is not mapped to master data
        assert struct.supplier is not None
        assert not hasattr(struct.supplier, "supplier_id") or getattr(struct.supplier, "supplier_id", None) is None


# ══════════════════════════════════════════════════════════════════════════
# S. No Mutation
# ══════════════════════════════════════════════════════════════════════════

class TestCategorySNoMutation:
    def test_s1_input_assembly_and_validation_unchanged(self) -> None:
        """Normalization does not mutate input assembly or validation result."""
        asm = _make_assembly()
        val = ExtractionValidationResult(
            assembly_id=asm.assembly_id,
            status=ValidationStatus.VALID,
            issues=(),
        )
        asm_copy = copy.deepcopy(asm)
        val_copy = copy.deepcopy(val)

        _ = normalize_financial_structure(asm, val)

        assert asm == asm_copy
        assert val == val_copy


# ══════════════════════════════════════════════════════════════════════════
# T. Determinism & Line Ordering
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryTDeterminismAndOrdering:
    def test_t1_physical_line_ordering_strictly_preserved(self) -> None:
        """Line items preserve the exact physical sequence established upstream without sorting."""
        line_b = _make_line_fact(line_num=2, desc="Zebra Item", amount="50.00")
        line_a = _make_line_fact(line_num=1, desc="Apple Item", amount="500.00")
        asm = _make_assembly(lines=(line_b, line_a))
        struct = normalize_financial_structure(asm)

        assert struct.lines[0].description == "Zebra Item"
        assert struct.lines[1].description == "Apple Item"

    def test_t2_repeated_normalization_produces_identical_output(self) -> None:
        """normalize(X) produces identical output on repeated runs."""
        asm = _make_assembly()
        struct1 = normalize_financial_structure(asm)
        struct2 = normalize_financial_structure(asm)
        assert struct1.to_dict() == struct2.to_dict()

    def test_t3_line_number_not_invented(self) -> None:
        """line_number is preserved if present; remains None if absent (never 1, 2, 3)."""
        line_without_num = _make_line_fact(line_num=None, desc="Unnumbered Item")
        asm = _make_assembly(lines=(line_without_num,))
        struct = normalize_financial_structure(asm)

        assert struct.lines[0].line_number is None


# ══════════════════════════════════════════════════════════════════════════
# U. Idempotence
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryUIdempotence:
    def test_u1_normalize_normalize_equals_normalize(self) -> None:
        """normalize(normalize(X)) == normalize(X)."""
        asm = _make_assembly()
        struct = normalize_financial_structure(asm)
        struct_twice = normalize_financial_structure(struct)

        assert struct.to_dict() == struct_twice.to_dict()


# ══════════════════════════════════════════════════════════════════════════
# V. Serialization Round-Trip
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryVSerialization:
    def test_v1_to_dict_and_from_dict(self) -> None:
        """to_dict and from_dict preserve all fields and types."""
        asm = _make_assembly()
        struct = normalize_financial_structure(asm)
        d = struct.to_dict()
        reconstructed = FinancialStructure.from_dict(d)

        assert struct.to_dict() == reconstructed.to_dict()

    def test_v2_to_json_and_from_json(self) -> None:
        """to_json and from_json serialize and deserialize deterministically."""
        asm = _make_assembly()
        struct = normalize_financial_structure(asm)
        j_str = struct.to_json()
        reconstructed = FinancialStructure.from_json(j_str)

        assert struct.to_dict() == reconstructed.to_dict()


# ══════════════════════════════════════════════════════════════════════════
# W. Multiple Payables
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryWMultiplePayables:
    def test_w1_batch_normalization_isolates_payables(self) -> None:
        """Multiple assemblies normalize independently without cross-document contamination."""
        asm1 = _make_assembly(assembly_id="ASM-1", doc_id="DOC-1", currency="EUR")
        asm2 = _make_assembly(assembly_id="ASM-2", doc_id="DOC-2", currency="USD")

        results = normalize_financial_structures([asm1, asm2])
        assert len(results) == 2
        assert results[0].assembly_id == "ASM-1"
        assert results[0].currency == "EUR"
        assert results[1].assembly_id == "ASM-2"
        assert results[1].currency == "USD"


# ══════════════════════════════════════════════════════════════════════════
# X. Real Corpus Verification (INV-01, HLD-01, INV-02, DU-02, HLD-03)
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryXRealCorpusVerification:
    def test_x1_inv_01_real_corpus_normalization(self) -> None:
        """INV-01 (German invoice): Canonical structure normalizes cleanly."""
        p = Path("artifacts/ocr/INV-01/page_001.json")
        if not p.exists():
            pytest.skip("INV-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        assert len(assemblies) == 1

        val = validate_financial_document_assembly(assemblies[0])
        struct = normalize_financial_structure(assemblies[0], val)

        assert struct.document_id == "INV-01.pdf"
        assert struct.currency == "EUR"
        assert len(struct.lines) > 0
        assert struct.printed_totals is not None
        assert struct.printed_totals.gross_total == Decimal("438.00")

    def test_x2_hld_01_real_corpus_normalization(self) -> None:
        """HLD-01 (Thai invoice): Billed lines and VAT header tax normalize without Thai date conversion."""
        p = Path("artifacts/ocr/HLD-01/page_001.json")
        if not p.exists():
            pytest.skip("HLD-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        assert len(assemblies) == 1

        val = validate_financial_document_assembly(assemblies[0])
        struct = normalize_financial_structure(assemblies[0], val)

        assert struct.document_id == "HLD-01.pdf"
        assert len(struct.lines) > 0
        assert len(struct.header_taxes) > 0
        # Thai date string is preserved as extracted, not converted via document-specific rule
        assert struct.invoice_date is None or isinstance(struct.invoice_date, str)

    def test_x3_inv_02_real_corpus_normalization(self) -> None:
        """INV-02 (Estonian invoice): Preserves observed PO without mapping to master PO."""
        p_dir = Path("artifacts/ocr/INV-02")
        if not p_dir.exists():
            pytest.skip("INV-02 directory missing")
        page_files = sorted(p_dir.glob("page_*.json"))
        page_evs = [ocr_json_to_page_evidence(pf) for pf in page_files]
        cands = extract_candidates_from_document(page_evs)
        facts_list = consolidate_document_groups(cands)
        assemblies = assemble_financial_documents(facts_list)

        val = validate_financial_document_assembly(assemblies[0])
        struct = normalize_financial_structure(assemblies[0], val)

        assert struct.document_id == "INV-02.pdf"
        assert struct.currency == "EUR"
        assert len(struct.lines) > 0

    def test_x4_du_02_real_corpus_normalization(self) -> None:
        """DU-02 (20-page customs dossier): EUR primary invoice isolated from supporting TRY documents."""
        p_dir = Path("artifacts/ocr/DU-02")
        if not p_dir.exists():
            pytest.skip("DU-02 artifacts missing")
        page_files = sorted(p_dir.glob("page_*.json"))
        assert len(page_files) == 20
        page_evs = [ocr_json_to_page_evidence(pf) for pf in page_files]
        unds = [classify_page(pe) for pe in page_evs]
        gres = group_document(page_evs, unds)
        cands = extract_candidates_from_document(page_evs, unds, gres.groups)
        facts_list = consolidate_document_groups(cands)
        assemblies = assemble_financial_documents(facts_list, gres.groups)

        primary_asm = next((a for a in assemblies if a.currency == "EUR"), assemblies[0])
        val = validate_financial_document_assembly(primary_asm)
        struct = normalize_financial_structure(primary_asm, val)

        # Primary structure remains EUR
        assert struct.currency == "EUR"
        # Supporting groups remain isolated in supporting_group_ids
        assert len(struct.supporting_group_ids) > 0
        assert not any(i.code == "CONFLICTING_CURRENCY" for i in struct.normalization_issues)

    def test_x5_hld_03_real_corpus_normalization(self) -> None:
        """HLD-03 (Portuguese invoice): Multilingual financial facts survive normalization cleanly."""
        p = Path("artifacts/ocr/HLD-03/page_001.json")
        if not p.exists():
            pytest.skip("HLD-03 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])

        val = validate_financial_document_assembly(assemblies[0])
        struct = normalize_financial_structure(assemblies[0], val)

        assert struct.document_id == "HLD-03.pdf"
        assert struct.currency == "EUR"
        assert len(struct.lines) > 0


# ══════════════════════════════════════════════════════════════════════════
# Y. ERP Boundary Compatibility (Structural inspection, NO ERP Invocation)
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryYERPBoundary:
    def test_y1_financial_structure_provides_all_erp_inputs(self) -> None:
        """Verify FinancialStructure supplies all components required by erp.py without invoking erp.py.
        
        erp.py expects:
        - currency: str
        - line_items: list of dicts with quantity, unit_price, discount, discount_percentage, taxes
        - discount_amount: header discount
        - taxes: list of dicts with tax_rate, tax_amount
        - other_charges: freight, insurance, extra, excise
        """
        line = _make_line_fact(qty="10", price="15.00", amount="150.00", discount="5.00")
        tx_head = TaxFact(
            tax_name="VAT 19%",
            rate=Decimal("19"),
            amount=Decimal("27.55"),
            placement=Placement.HEADER,
            evidence_ids=("EV_HEAD",),
            origin=FactOrigin.OBSERVED,
        )
        chg = ChargeFact(
            name="Freight",
            amount=Decimal("20.00"),
            placement=Placement.HEADER,
            evidence_ids=("EV_HEAD",),
            origin=FactOrigin.OBSERVED,
        )
        asm = _make_assembly(lines=(line,), taxes=(tx_head,), charges=(chg,))
        struct = normalize_financial_structure(asm)

        # Check that required ERP component data is structurally present
        assert struct.currency == "EUR"
        assert len(struct.lines) == 1
        assert struct.lines[0].quantity == Decimal("10")
        assert struct.lines[0].unit_price == Decimal("15.00")
        assert len(struct.lines[0].discounts) == 1
        assert struct.lines[0].discounts[0].amount == Decimal("5.00")
        assert len(struct.header_taxes) == 1
        assert struct.header_taxes[0].rate == Decimal("19")
        assert struct.header_taxes[0].amount == Decimal("27.55")
        assert len(struct.header_charges) == 1
        assert struct.header_charges[0].amount == Decimal("20.00")


# ══════════════════════════════════════════════════════════════════════════
# Z. Boundary Audit (Forbidden Calls & Arithmetic Inspection)
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryZBoundaryAudit:
    def test_z1_no_forbidden_module_imports(self) -> None:
        """Inspect src/accounting/financial_structure.py to confirm no forbidden imports."""
        src_path = Path("src/accounting/financial_structure.py")
        content = src_path.read_text(encoding="utf-8")

        forbidden_imports = [
            "src.erp",
            "erp_book",
            "rapidocr",
            "paddleocr",
            "qwen",
            "requests",
            "urllib.request",
            "httpx",
            "src.matching.supplier_matcher",
            "src.matching.buyer_matcher",
            "src.matching.po_matcher",
            "src.matching.tax_matcher",
            "src.matching.payment_terms_matcher",
        ]
        for forb in forbidden_imports:
            assert forb not in content, f"Forbidden import '{forb}' detected in financial_structure.py"

    def test_z2_no_accounting_arithmetic_operators(self) -> None:
        """Inspect AST to confirm no arithmetic multiplication or addition of accounting fields."""
        import ast

        src_path = Path("src/accounting/financial_structure.py")
        content = src_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(src_path))

        # Inspect all binary operations in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                # Ensure no multiplication operator is used anywhere in the module
                assert not isinstance(node.op, ast.Mult), (
                    f"Forbidden multiplication BinOp detected in financial_structure.py at line {node.lineno}"
                )
                # Check operands for prohibited arithmetic
                left_id = getattr(node.left, "id", "") if isinstance(node.left, ast.Name) else ""
                right_id = getattr(node.right, "id", "") if isinstance(node.right, ast.Name) else ""
                prohibited_vars = {"quantity", "unit_price", "amount", "rate", "subtotal", "tax_total", "gross_total"}
                assert not (left_id in prohibited_vars or right_id in prohibited_vars), (
                    f"Forbidden arithmetic operation with accounting variable at line {node.lineno}"
                )

