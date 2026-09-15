"""tests/test_document_facts.py — Test Suite for Phase 9A Document Fact Model.

Covers all 18 specified scenarios:
A. document identity serialization
B. invoice/credit/debit/unknown types
C. supplier observed values + evidence
D. buyer observed values + evidence
E. PO observed value preserved
F. line vs component_detail distinction
G. header vs line tax placement
H. discounts separate from lines
I. charges separate from lines
J. printed totals preserved separately
K. Decimal numeric preservation
L. missing optional fields
M. conflicting evidence can coexist
N. no naked accounting field without evidence
O. multi-page supporting facts remain distinguishable
P. deterministic serialization round-trip
Q. no arithmetic/calculation in fact construction
R. no filename-based invoice classification
"""
import json
import pytest
from decimal import Decimal

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
    to_decimal,
)
from src.understanding.page_classifier import PageRole, PayableRelevance


# ══════════════════════════════════════════════════════════════════════════
# A. Document Identity Serialization
# ══════════════════════════════════════════════════════════════════════════

class TestDocumentIdentitySerialization:
    """Test A: Document identity serialization to/from dict and JSON."""

    def test_document_identity_serialization_round_trip(self):
        ident = DocumentIdentityFacts(
            invoice_number="INV-2026-001",
            invoice_date="2026-03-01",
            due_date="2026-03-31",
            invoice_type=InvoiceType.INVOICE,
            currency="EUR",
            evidence_ids=("ev_id_1", "ev_id_2"),
            field_evidence_ids={
                "invoice_number": ("ev_id_1",),
                "invoice_date": ("ev_id_2",),
            },
            origin=FactOrigin.OBSERVED,
            raw_values={"invoice_number": "INV-2026-001"},
        )
        as_dict = ident.to_dict()
        assert as_dict["invoice_number"] == "INV-2026-001"
        assert as_dict["invoice_type"] == "invoice"
        assert as_dict["currency"] == "EUR"
        assert as_dict["origin"] == "observed"
        assert as_dict["evidence_ids"] == ["ev_id_1", "ev_id_2"]
        assert as_dict["field_evidence_ids"]["invoice_number"] == ["ev_id_1"]

        reconstructed = DocumentIdentityFacts.from_dict(as_dict)
        assert reconstructed == ident
        assert reconstructed.invoice_type == InvoiceType.INVOICE
        assert reconstructed.origin == FactOrigin.OBSERVED


# ══════════════════════════════════════════════════════════════════════════
# B. Invoice/Credit/Debit/Unknown Types
# ══════════════════════════════════════════════════════════════════════════

class TestInvoiceTypes:
    """Test B: All invoice types supported and preserved."""

    @pytest.mark.parametrize(
        "inv_type,expected_str",
        [
            (InvoiceType.INVOICE, "invoice"),
            (InvoiceType.CREDIT_MEMO, "credit_memo"),
            (InvoiceType.DEBIT_MEMO, "debit_memo"),
            (InvoiceType.UNKNOWN, "unknown"),
        ],
    )
    def test_invoice_type_values_and_round_trip(self, inv_type, expected_str):
        ident = DocumentIdentityFacts(
            invoice_type=inv_type,
            evidence_ids=("ev_type",),
        )
        assert ident.invoice_type == inv_type
        assert ident.invoice_type.value == expected_str

        serialized = ident.to_dict()
        assert serialized["invoice_type"] == expected_str

        deserialized = DocumentIdentityFacts.from_dict(serialized)
        assert deserialized.invoice_type == inv_type


# ══════════════════════════════════════════════════════════════════════════
# C. Supplier Observed Values + Evidence
# ══════════════════════════════════════════════════════════════════════════

class TestSupplierObservedValuesAndEvidence:
    """Test C: Supplier observed values + evidence IDs preserved separately from match."""

    def test_supplier_observed_values_and_match_separation(self):
        matched_payload = {
            "supplier_id": "SUP-001",
            "status": "matched",
            "method": "exact_vat",
        }
        sup = SupplierIdentityFact(
            observed_name="Acme Industrial GmbH",
            vat_id="DE123456789",
            country="DE",
            email="billing@acme.de",
            bank_iban="DE89370400440532013000",
            address="Musterstraße 12, 10115 Berlin",
            evidence_ids=("ev_sup_name", "ev_sup_vat"),
            field_evidence_ids={
                "observed_name": ("ev_sup_name",),
                "vat_id": ("ev_sup_vat",),
            },
            origin=FactOrigin.OBSERVED,
            matched_result=matched_payload,
        )

        assert sup.observed_name == "Acme Industrial GmbH"
        assert sup.vat_id == "DE123456789"
        assert sup.matched_result == matched_payload
        assert sup.matched_result["supplier_id"] == "SUP-001"
        # Observed name is NOT replaced by master data name
        assert sup.observed_name != sup.matched_result["supplier_id"]

        d = sup.to_dict()
        assert d["observed_name"] == "Acme Industrial GmbH"
        assert d["matched_result"]["supplier_id"] == "SUP-001"

        reconstructed = SupplierIdentityFact.from_dict(d)
        assert reconstructed == sup


# ══════════════════════════════════════════════════════════════════════════
# D. Buyer Observed Values + Evidence
# ══════════════════════════════════════════════════════════════════════════

class TestBuyerObservedValuesAndEvidence:
    """Test D: Buyer observed values, codes, addresses, and evidence preserved."""

    def test_buyer_hierarchy_and_evidence(self):
        buyer = BuyerIdentityFact(
            observed_company="Global Corp Holding",
            business_unit="Automotive Systems",
            location="Plant Leipzig",
            company_code="CORP01",
            business_unit_code="BU-AUTO",
            location_code="LOC-LEI",
            invoice_to_address="Werkallee 5, 04109 Leipzig",
            evidence_ids=("ev_buyer_hdr", "ev_buyer_addr"),
            field_evidence_ids={
                "observed_company": ("ev_buyer_hdr",),
                "company_code": ("ev_buyer_hdr",),
                "invoice_to_address": ("ev_buyer_addr",),
            },
            origin=FactOrigin.OBSERVED,
        )

        assert buyer.observed_company == "Global Corp Holding"
        assert buyer.company_code == "CORP01"
        assert buyer.business_unit_code == "BU-AUTO"
        assert buyer.location_code == "LOC-LEI"
        assert "ev_buyer_hdr" in buyer.evidence_ids
        assert buyer.field_evidence_ids["observed_company"] == ("ev_buyer_hdr",)

        reconstructed = BuyerIdentityFact.from_dict(buyer.to_dict())
        assert reconstructed == buyer


# ══════════════════════════════════════════════════════════════════════════
# E. PO Observed Value Preserved
# ══════════════════════════════════════════════════════════════════════════

class TestPOObservedValuePreserved:
    """Test E: PO observed number preserved without substituting matched po_id."""

    def test_po_observed_number_not_substituted(self):
        po = POFacts(
            observed_po_number="PO-PRINTED-4500012345",
            evidence_ids=("ev_po_box",),
            field_evidence_ids={"observed_po_number": ("ev_po_box",)},
            origin=FactOrigin.OBSERVED,
            matched_po_result={
                "po_id": "PO-ERP-9999",
                "status": "matched",
                "method": "exact_po_number",
            },
        )

        # The observed field remains strictly the printed value
        assert po.observed_po_number == "PO-PRINTED-4500012345"
        assert po.matched_po_result["po_id"] == "PO-ERP-9999"
        assert po.observed_po_number != po.matched_po_result["po_id"]

        reconstructed = POFacts.from_dict(po.to_dict())
        assert reconstructed.observed_po_number == "PO-PRINTED-4500012345"
        assert reconstructed.matched_po_result["po_id"] == "PO-ERP-9999"


# ══════════════════════════════════════════════════════════════════════════
# F. Line vs Component Detail Distinction
# ══════════════════════════════════════════════════════════════════════════

class TestLineVsComponentDetailDistinction:
    """Test F: Distinguish billed lines from breakdown/component detail rows."""

    def test_billed_line_and_component_detail_roles(self):
        billed_line = LineFact(
            line_number=1,
            description="Complete Turbine Assembly",
            quantity=Decimal("1"),
            unit_price=Decimal("15000.00"),
            amount=Decimal("15000.00"),
            semantic_role=SemanticRole.BILLED_LINE,
            evidence_ids=("ev_ln_1",),
        )

        component_detail = LineFact(
            line_number=2,
            description="Turbine Rotor Breakdown (Info only)",
            quantity=Decimal("1"),
            unit_price=Decimal("9000.00"),
            amount=Decimal("9000.00"),
            semantic_role=SemanticRole.COMPONENT_DETAIL,
            evidence_ids=("ev_ln_2",),
        )

        assert billed_line.semantic_role == SemanticRole.BILLED_LINE
        assert component_detail.semantic_role == SemanticRole.COMPONENT_DETAIL
        assert billed_line.semantic_role != component_detail.semantic_role

        # Roles round-trip cleanly
        assert LineFact.from_dict(billed_line.to_dict()).semantic_role == SemanticRole.BILLED_LINE
        assert LineFact.from_dict(component_detail.to_dict()).semantic_role == SemanticRole.COMPONENT_DETAIL


# ══════════════════════════════════════════════════════════════════════════
# G. Header vs Line Tax Placement
# ══════════════════════════════════════════════════════════════════════════

class TestHeaderVsLineTaxPlacement:
    """Test G: Tax placement preserved as header vs line without moving levels."""

    def test_tax_placement_preservation(self):
        line_tax = TaxFact(
            tax_name="VAT Standard",
            tax_type="VAT",
            rate=Decimal("19"),
            amount=Decimal("38.00"),
            placement=Placement.LINE,
            evidence_ids=("ev_tx_line",),
        )

        header_tax = TaxFact(
            tax_name="VAT Summary Total",
            tax_type="VAT",
            rate=Decimal("19"),
            amount=Decimal("190.00"),
            placement=Placement.HEADER,
            evidence_ids=("ev_tx_hdr",),
        )

        line = LineFact(
            line_number=1,
            description="Item 1",
            amount=Decimal("200.00"),
            taxes=(line_tax,),
            evidence_ids=("ev_l1",),
        )

        fin = FinancialFacts(
            lines=(line,),
            taxes=(header_tax,),
            evidence_ids=("ev_fin",),
        )

        assert fin.lines[0].taxes[0].placement == Placement.LINE
        assert fin.taxes[0].placement == Placement.HEADER

        reconstructed = FinancialFacts.from_dict(fin.to_dict())
        assert reconstructed.lines[0].taxes[0].placement == Placement.LINE
        assert reconstructed.taxes[0].placement == Placement.HEADER


# ══════════════════════════════════════════════════════════════════════════
# H. Discounts Separate from Lines
# ══════════════════════════════════════════════════════════════════════════

class TestDiscountsSeparateFromLines:
    """Test H: Discounts held separately in discounts tuple, never converted to lines."""

    def test_discounts_separate_from_lines(self):
        discount = DiscountFact(
            name="Special Volume Discount",
            rate=Decimal("5"),
            amount=Decimal("50.00"),
            placement=Placement.HEADER,
            evidence_ids=("ev_disc_1",),
        )

        fin = FinancialFacts(
            lines=(),
            discounts=(discount,),
            evidence_ids=("ev_fin",),
        )

        assert len(fin.lines) == 0
        assert len(fin.discounts) == 1
        assert fin.discounts[0].name == "Special Volume Discount"
        assert fin.discounts[0].amount == Decimal("50.00")

        reconstructed = FinancialFacts.from_dict(fin.to_dict())
        assert len(reconstructed.lines) == 0
        assert len(reconstructed.discounts) == 1
        assert isinstance(reconstructed.discounts[0], DiscountFact)


# ══════════════════════════════════════════════════════════════════════════
# I. Charges Separate from Lines
# ══════════════════════════════════════════════════════════════════════════

class TestChargesSeparateFromLines:
    """Test I: Charges held separately in charges tuple, never converted to lines."""

    def test_charges_separate_from_lines(self):
        charge = ChargeFact(
            name="Freight & Packaging",
            amount=Decimal("75.00"),
            placement=Placement.HEADER,
            evidence_ids=("ev_chg_1",),
        )

        fin = FinancialFacts(
            lines=(),
            charges=(charge,),
            evidence_ids=("ev_fin",),
        )

        assert len(fin.lines) == 0
        assert len(fin.charges) == 1
        assert fin.charges[0].name == "Freight & Packaging"
        assert fin.charges[0].amount == Decimal("75.00")

        reconstructed = FinancialFacts.from_dict(fin.to_dict())
        assert len(reconstructed.lines) == 0
        assert len(reconstructed.charges) == 1
        assert isinstance(reconstructed.charges[0], ChargeFact)


# ══════════════════════════════════════════════════════════════════════════
# J. Printed Totals Preserved Separately
# ══════════════════════════════════════════════════════════════════════════

class TestPrintedTotalsPreservedSeparately:
    """Test J: Document-printed totals preserved separately without reconciliation."""

    def test_printed_totals_preserved_without_reconciliation(self):
        # Deliberately divergent figures to prove no reconciliation occurs
        totals = PrintedTotalsFact(
            subtotal=Decimal("1000.00"),
            net=Decimal("950.00"),
            taxable_base=Decimal("950.00"),
            tax_total=Decimal("180.50"),
            gross_total=Decimal("1130.50"),
            amount_due=Decimal("1130.50"),
            payment_total=Decimal("0.00"),
            additional_totals={"Zwischensumme": Decimal("950.00")},
            evidence_ids=("ev_tot_1",),
            field_evidence_ids={
                "gross_total": ("ev_tot_gross",),
                "net": ("ev_tot_net",),
            },
            origin=FactOrigin.OBSERVED,
        )

        assert totals.subtotal == Decimal("1000.00")
        assert totals.net == Decimal("950.00")
        assert totals.tax_total == Decimal("180.50")
        assert totals.gross_total == Decimal("1130.50")
        assert totals.additional_totals["Zwischensumme"] == Decimal("950.00")

        reconstructed = PrintedTotalsFact.from_dict(totals.to_dict())
        assert reconstructed == totals
        assert reconstructed.gross_total == Decimal("1130.50")
        assert reconstructed.additional_totals["Zwischensumme"] == Decimal("950.00")


# ══════════════════════════════════════════════════════════════════════════
# K. Decimal Numeric Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestDecimalNumericPreservation:
    """Test K: Exact Decimal numeric representation without floating-point errors."""

    def test_decimal_numeric_preservation(self):
        # Floating point 0.1 + 0.2 != 0.3, but Decimal('0.1') + Decimal('0.2') == Decimal('0.3')
        d1 = to_decimal("0.1")
        d2 = to_decimal("0.2")
        assert d1 + d2 == Decimal("0.3")

        line = LineFact(
            quantity=Decimal("3.000"),
            unit_price=Decimal("19.99"),
            amount=Decimal("59.97"),
            evidence_ids=("ev_k",),
        )
        assert isinstance(line.quantity, Decimal)
        assert isinstance(line.unit_price, Decimal)
        assert isinstance(line.amount, Decimal)

        serialized = line.to_dict()
        assert serialized["quantity"] == "3.000"
        assert serialized["unit_price"] == "19.99"
        assert serialized["amount"] == "59.97"

        deserialized = LineFact.from_dict(serialized)
        assert deserialized.quantity == Decimal("3.000")
        assert deserialized.unit_price == Decimal("19.99")
        assert deserialized.amount == Decimal("59.97")

    def test_to_decimal_string_formats(self):
        assert to_decimal("1,234.56") == Decimal("1234.56")
        assert to_decimal("1.234,56") == Decimal("1234.56")
        assert to_decimal("  $ 500.00 ") == Decimal("500.00")
        assert to_decimal("19%") == Decimal("19")
        assert to_decimal(None) is None
        assert to_decimal("") is None


# ══════════════════════════════════════════════════════════════════════════
# L. Missing Optional Fields
# ══════════════════════════════════════════════════════════════════════════

class TestMissingOptionalFields:
    """Test L: Optional fields can be empty/None without error."""

    def test_minimal_instantiations(self):
        assert DocumentIdentityFacts().invoice_number is None
        assert SupplierIdentityFact().observed_name is None
        assert BuyerIdentityFact().observed_company is None
        assert PartyIdentityFacts().evidence_ids == ()
        assert POFacts().observed_po_number is None
        assert LineFact().amount is None
        assert TaxFact().rate is None
        assert DiscountFact().amount is None
        assert ChargeFact().amount is None
        assert PrintedTotalsFact().gross_total is None
        assert FinancialFacts().printed_totals is None

        doc = DocumentFacts(document_id="DOC-MINIMAL")
        assert doc.document_id == "DOC-MINIMAL"
        assert doc.page_numbers == ()
        assert doc.supporting_group_ids == ()
        assert doc.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE

        # Serializes cleanly
        d = doc.to_dict()
        reconstructed = DocumentFacts.from_dict(d)
        assert reconstructed.document_id == "DOC-MINIMAL"


# ══════════════════════════════════════════════════════════════════════════
# M. Conflicting Evidence Can Coexist
# ══════════════════════════════════════════════════════════════════════════

class TestConflictingEvidenceCanCoexist:
    """Test M: Conflicting OCR vs Vision or multi-source evidence can coexist without forced resolution."""

    def test_conflicting_evidence_in_field_evidence_and_conflicts_list(self):
        # 1. Field has multiple evidence IDs representing OCR vs Vision observations
        line = LineFact(
            description="Item description",
            amount=Decimal("100.00"),
            evidence_ids=("ev_ocr_amount_100", "ev_vision_amount_105"),
            field_evidence_ids={
                "amount": ("ev_ocr_amount_100", "ev_vision_amount_105"),
            },
            raw_values={
                "amount:ocr": "100.00",
                "amount:vision": "105.00",
            },
        )

        assert len(line.field_evidence_ids["amount"]) == 2
        assert "ev_ocr_amount_100" in line.field_evidence_ids["amount"]
        assert "ev_vision_amount_105" in line.field_evidence_ids["amount"]
        assert line.raw_values["amount:ocr"] == "100.00"
        assert line.raw_values["amount:vision"] == "105.00"

        # 2. Conflicting document-level candidate facts in DocumentFacts
        doc = DocumentFacts(
            document_id="DOC-CONFLICT",
            conflicting_facts=(
                {
                    "field": "invoice_number",
                    "candidate_a": "INV-100",
                    "evidence_a": "ev_ocr_1",
                    "candidate_b": "INV-101",
                    "evidence_b": "ev_vision_1",
                },
            ),
            evidence_ids=("ev_ocr_1", "ev_vision_1"),
        )
        assert len(doc.conflicting_facts) == 1
        assert doc.conflicting_facts[0]["candidate_a"] == "INV-100"
        assert doc.conflicting_facts[0]["candidate_b"] == "INV-101"

        reconstructed = DocumentFacts.from_dict(doc.to_dict())
        assert reconstructed.conflicting_facts[0]["candidate_a"] == "INV-100"


# ══════════════════════════════════════════════════════════════════════════
# N. No Naked Accounting Field Without Evidence
# ══════════════════════════════════════════════════════════════════════════

class TestNoNakedAccountingFieldWithoutEvidence:
    """Test N: Accounting fields must retain evidence / provenance based on origin."""

    def test_observed_line_without_evidence_raises_error(self):
        with pytest.raises(ValueError, match="Naked accounting field"):
            LineFact(amount=Decimal("100.00"), origin=FactOrigin.OBSERVED)

    def test_observed_line_with_evidence_succeeds(self):
        line = LineFact(amount=Decimal("100.00"), evidence_ids=("ev_ok",))
        assert line.amount == Decimal("100.00")

    def test_observed_printed_totals_without_evidence_raises_error(self):
        with pytest.raises(ValueError, match="Naked accounting field"):
            PrintedTotalsFact(gross_total=Decimal("500.00"), origin=FactOrigin.OBSERVED)

    def test_observed_document_identity_without_evidence_raises_error(self):
        with pytest.raises(ValueError, match="Naked accounting field"):
            DocumentIdentityFacts(invoice_number="INV-123", origin=FactOrigin.OBSERVED)

    def test_observed_tax_fact_without_evidence_raises_error(self):
        with pytest.raises(ValueError, match="Naked accounting field"):
            TaxFact(rate=Decimal("19"), origin=FactOrigin.OBSERVED)

    def test_derived_value_with_derivation_rule_succeeds(self):
        # Derived facts do not require direct document OCR evidence if derivation rule is given
        line = LineFact(
            amount=Decimal("100.00"),
            origin=FactOrigin.DERIVED,
            derivation_rule="sum_of_components",
            derivation_source_fields=("line_1", "line_2"),
        )
        assert line.amount == Decimal("100.00")
        assert line.origin == FactOrigin.DERIVED

    def test_derived_value_without_provenance_raises_error(self):
        with pytest.raises(ValueError, match="Derived accounting field"):
            LineFact(amount=Decimal("100.00"), origin=FactOrigin.DERIVED)

    def test_matched_value_with_match_provenance_succeeds(self):
        sup = SupplierIdentityFact(
            observed_name="Acme",
            origin=FactOrigin.MATCHED,
            matched_result={"supplier_id": "SUP-1", "method": "vat"},
            evidence_ids=("ev_matched_sup",),
        )
        assert sup.observed_name == "Acme"
        assert sup.origin == FactOrigin.MATCHED

    def test_matched_value_without_match_result_raises_error(self):
        with pytest.raises(ValueError, match="Matched accounting field"):
            SupplierIdentityFact(
                observed_name="Acme",
                origin=FactOrigin.MATCHED,
                evidence_ids=("ev_1",),
            )


# ══════════════════════════════════════════════════════════════════════════
# O. Multi-Page Supporting Facts Remain Distinguishable
# ══════════════════════════════════════════════════════════════════════════

class TestMultiPageSupportingFactsDistinguishable:
    """Test O: Supporting document facts remain distinguishable from payable facts."""

    def test_payable_and_supporting_facts_separation(self):
        payable_doc = DocumentFacts(
            document_id="DOC-INV-01",
            group_id="GRP-PAYABLE-1",
            page_numbers=(1, 2),
            document_role=PageRole.INVOICE,
            payable_relevance=PayableRelevance.PAYABLE_CANDIDATE,
            supporting_group_ids=("GRP-SUPP-1",),
            identity=DocumentIdentityFacts(invoice_number="INV-001", evidence_ids=("ev_p1",)),
            financials=FinancialFacts(
                lines=(LineFact(amount=Decimal("500.00"), evidence_ids=("ev_l1",)),),
                evidence_ids=("ev_fin_1",),
            ),
            evidence_ids=("ev_p1", "ev_p2"),
        )

        supporting_doc = DocumentFacts(
            document_id="DOC-INV-01",
            group_id="GRP-SUPP-1",
            page_numbers=(3,),
            document_role=PageRole.SUPPORTING_DOCUMENT,
            payable_relevance=PayableRelevance.SUPPORTING,
            financials=FinancialFacts(
                lines=(LineFact(amount=Decimal("9999.00"), evidence_ids=("ev_supp_l1",)),),
                evidence_ids=("ev_supp_fin",),
            ),
            evidence_ids=("ev_p3",),
        )

        assert payable_doc.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE
        assert supporting_doc.payable_relevance == PayableRelevance.SUPPORTING
        assert payable_doc.supporting_group_ids == ("GRP-SUPP-1",)

        # Financials are distinct; supporting figures are not merged into payable figures
        assert payable_doc.financials.lines[0].amount == Decimal("500.00")
        assert supporting_doc.financials.lines[0].amount == Decimal("9999.00")


# ══════════════════════════════════════════════════════════════════════════
# P. Deterministic Serialization Round-Trip
# ══════════════════════════════════════════════════════════════════════════

class TestDeterministicSerializationRoundTrip:
    """Test P: Deterministic serialization to/from JSON with byte-identical output."""

    def test_full_document_facts_round_trip(self):
        doc = DocumentFacts(
            document_id="INV-2026-TEST",
            group_id="GRP-01",
            page_numbers=(1, 2),
            document_role=PageRole.INVOICE,
            payable_relevance=PayableRelevance.PAYABLE_CANDIDATE,
            supporting_group_ids=("GRP-02",),
            identity=DocumentIdentityFacts(
                invoice_number="INV-2026-TEST",
                invoice_date="2026-03-01",
                due_date="2026-03-31",
                invoice_type=InvoiceType.INVOICE,
                currency="EUR",
                evidence_ids=("ev_inv",),
            ),
            parties=PartyIdentityFacts(
                supplier=SupplierIdentityFact(
                    observed_name="Alpha Tech",
                    vat_id="DE999",
                    evidence_ids=("ev_sup",),
                ),
                buyer=BuyerIdentityFact(
                    observed_company="Beta Corp",
                    evidence_ids=("ev_buy",),
                ),
                evidence_ids=("ev_parties",),
            ),
            po=POFacts(
                observed_po_number="PO-4455",
                evidence_ids=("ev_po",),
            ),
            financials=FinancialFacts(
                lines=(
                    LineFact(
                        line_number=1,
                        description="Consulting",
                        quantity=Decimal("10"),
                        unit_price=Decimal("100.00"),
                        amount=Decimal("1000.00"),
                        semantic_role=SemanticRole.BILLED_LINE,
                        evidence_ids=("ev_ln1",),
                    ),
                ),
                discounts=(
                    DiscountFact(
                        name="Loyalty",
                        amount=Decimal("50.00"),
                        evidence_ids=("ev_d1",),
                    ),
                ),
                charges=(
                    ChargeFact(
                        name="Service fee",
                        amount=Decimal("25.00"),
                        evidence_ids=("ev_c1",),
                    ),
                ),
                taxes=(
                    TaxFact(
                        tax_name="VAT 19%",
                        rate=Decimal("19"),
                        amount=Decimal("185.25"),
                        placement=Placement.HEADER,
                        evidence_ids=("ev_t1",),
                    ),
                ),
                printed_totals=PrintedTotalsFact(
                    net=Decimal("975.00"),
                    tax_total=Decimal("185.25"),
                    gross_total=Decimal("1160.25"),
                    evidence_ids=("ev_tot",),
                ),
                currency="EUR",
                evidence_ids=("ev_fin",),
            ),
            evidence_ids=("ev_doc_1",),
            metadata={"source": "test"},
            provenance={"created_by": "Phase9A"},
        )

        json_str_1 = doc.to_json()
        json_str_2 = doc.to_json()
        # Byte-identical deterministic output
        assert json_str_1 == json_str_2

        reconstructed = DocumentFacts.from_json(json_str_1)
        assert reconstructed == doc
        assert reconstructed.to_json() == json_str_1


# ══════════════════════════════════════════════════════════════════════════
# Q. No Arithmetic/Calculation in Fact Construction
# ══════════════════════════════════════════════════════════════════════════

class TestNoArithmeticInFactConstruction:
    """Test Q: Observed values must remain observed; never calculate missing amounts/prices."""

    def test_no_calculation_of_missing_unit_price(self):
        # Line has quantity and amount, but unit price was NOT printed on document
        line = LineFact(
            quantity=Decimal("5"),
            amount=Decimal("100.00"),
            evidence_ids=("ev_q",),
        )
        # unit_price must remain None, NOT 20.00!
        assert line.unit_price is None

    def test_no_calculation_of_missing_amount(self):
        # Line has quantity and unit price, but amount was NOT printed on document
        line = LineFact(
            quantity=Decimal("4"),
            unit_price=Decimal("25.00"),
            evidence_ids=("ev_q",),
        )
        # amount must remain None, NOT 100.00!
        assert line.amount is None

    def test_no_calculation_of_missing_totals(self):
        # Printed totals has net and tax, but gross total was omitted on document
        totals = PrintedTotalsFact(
            net=Decimal("100.00"),
            tax_total=Decimal("19.00"),
            evidence_ids=("ev_q",),
        )
        # gross_total must remain None, NOT 119.00!
        assert totals.gross_total is None


# ══════════════════════════════════════════════════════════════════════════
# R. No Filename-Based Invoice Classification
# ══════════════════════════════════════════════════════════════════════════

class TestNoFilenameBasedInvoiceClassification:
    """Test R: invoice_type must NOT be inferred from filenames or document IDs."""

    @pytest.mark.parametrize(
        "filename_document_id",
        [
            "credit_memo_12345.pdf",
            "CREDIT_NOTE_888",
            "debit_memo_999.pdf",
            "invoice_001.pdf",
        ],
    )
    def test_document_id_does_not_infer_invoice_type(self, filename_document_id):
        # Creating facts with filename-like document_id
        doc = DocumentFacts(document_id=filename_document_id)
        # invoice_type must remain UNKNOWN when no explicit document evidence is provided
        assert doc.identity.invoice_type == InvoiceType.UNKNOWN
        assert doc.identity.invoice_type != InvoiceType.CREDIT_MEMO
        assert doc.identity.invoice_type != InvoiceType.DEBIT_MEMO
