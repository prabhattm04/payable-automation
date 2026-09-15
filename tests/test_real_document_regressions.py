import pytest
from decimal import Decimal
from typing import Optional, Sequence

from src.understanding.evidence import Evidence, PageEvidence, EvidenceProvenance, EvidenceSource
from src.understanding.document_facts import SemanticRole, Placement, InvoiceType
from src.extraction.candidates import (
    extract_lines_from_evidence,
    extract_totals_and_taxes_from_evidence,
)
from src.accounting.payable_builder import build_payable
from src.accounting.decision import PayableDecision, PayableDecisionStatus
from src.accounting.erp_reconstruction import reconstruct_erp
from src.accounting.financial_structure import (
    FinancialStructure,
    NormalizedLine,
    NormalizedParty,
    NormalizedPrintedTotals,
    NormalizedTax,
)

_DEFAULT_PROV = EvidenceProvenance(
    source=EvidenceSource.OCR,
    document_id="test.pdf",
    page_number=1,
    extraction_method="test_ocr",
)


def _ev(evidence_id: str, content: str, bbox: list[float]) -> Evidence:
    return Evidence(
        provenance=_DEFAULT_PROV,
        evidence_id=evidence_id,
        content=content,
        bbox=bbox,
    )


def _page(items: list[Evidence], page_number: int = 1) -> PageEvidence:
    return PageEvidence(document_id="test.pdf", page_number=page_number, items=items)


def _make_safe_decision(status: PayableDecisionStatus = PayableDecisionStatus.SAFE_TO_AUTODRAFT) -> PayableDecision:
    return PayableDecision(
        assembly_id="asm-test",
        document_id="test.pdf",
        status=status,
        primary_reason="Test reason",
    )


def _make_financial_structure(
    header_taxes: Sequence[NormalizedTax] = (),
) -> FinancialStructure:
    return FinancialStructure(
        assembly_id="asm-test",
        document_id="test.pdf",
        document_type=InvoiceType.INVOICE,
        invoice_number="INV-TEST-001",
        invoice_date="2026-01-01",
        currency="EUR",
        printed_totals=NormalizedPrintedTotals(gross_total=Decimal("119.00"), subtotal=Decimal("100.00"), tax_total=Decimal("19.00")),
        lines=[
            NormalizedLine(
                source_line_id="line-1",
                line_number=1,
                description="Consulting Service",
                quantity=Decimal("1"),
                unit_price=Decimal("100.00"),
                amount=Decimal("100.00"),
                semantic_role=SemanticRole.BILLED_LINE,
                metadata={"item_type": "SERVICE"},
            )
        ],
        header_taxes=tuple(header_taxes),
        supplier=NormalizedParty(name="Test Vendor"),
    )


def test_metadata_rows_not_extracted_as_lines():
    """Non-table metadata rows (vendor info, bank details, addresses, VAT IDs) must not be extracted as lines."""
    items = [
        # Metadata rows before any table
        _ev("e1", "Vendor Name GmbH", [100.0, 50.0, 300.0, 70.0]),
        _ev("e2", "Musterstrasse 12, 10115 Berlin", [100.0, 75.0, 400.0, 95.0]),
        _ev("e3", "USt-IdNr.: DE123456789", [100.0, 100.0, 350.0, 120.0]),
        _ev("e4", "IBAN: DE89370400440532013000", [100.0, 125.0, 450.0, 145.0]),
        _ev("e5", "Rechnungsdatum: 15.03.2024", [100.0, 150.0, 350.0, 170.0]),
        # Table Header
        _ev("e6", "Pos", [50.0, 200.0, 90.0, 220.0]),
        _ev("e7", "Beschreibung", [100.0, 200.0, 300.0, 220.0]),
        _ev("e8", "Menge", [320.0, 200.0, 380.0, 220.0]),
        _ev("e9", "Einzelpreis", [400.0, 200.0, 480.0, 220.0]),
        _ev("e10", "Gesamtpreis", [500.0, 200.0, 580.0, 220.0]),
        # Table Row
        _ev("e11", "1", [50.0, 240.0, 90.0, 260.0]),
        _ev("e12", "Beratungsleistung", [100.0, 240.0, 300.0, 260.0]),
        _ev("e13", "5", [320.0, 240.0, 380.0, 260.0]),
        _ev("e14", "100,00", [400.0, 240.0, 480.0, 260.0]),
        _ev("e15", "500,00", [500.0, 240.0, 580.0, 260.0]),
    ]
    pe = _page(items)
    lines = extract_lines_from_evidence(pe)

    assert len(lines) == 1
    assert lines[0].description == "Beratungsleistung"
    assert lines[0].quantity == Decimal("5")
    assert lines[0].unit_price == Decimal("100.00")
    assert lines[0].amount == Decimal("500.00")


def test_index_column_not_mapped_to_quantity():
    """Pos / Item index column must not be misassigned as quantity or unit price."""
    items = [
        # Table Header
        _ev("h1", "Pos", [50.0, 100.0, 90.0, 120.0]),
        _ev("h2", "Description", [100.0, 100.0, 300.0, 120.0]),
        _ev("h3", "Unit Price", [320.0, 100.0, 400.0, 120.0]),
        _ev("h4", "Qty", [420.0, 100.0, 480.0, 120.0]),
        _ev("h5", "Total", [500.0, 100.0, 580.0, 120.0]),
        # Row 1 (Pos 1, Qty 4, Price 73.00, Total 292.00)
        _ev("r1_1", "1", [50.0, 140.0, 90.0, 160.0]),
        _ev("r1_2", "Standard Service", [100.0, 140.0, 300.0, 160.0]),
        _ev("r1_3", "73,00", [320.0, 140.0, 400.0, 160.0]),
        _ev("r1_4", "4", [420.0, 140.0, 480.0, 160.0]),
        _ev("r1_5", "292,00", [500.0, 140.0, 580.0, 160.0]),
    ]
    pe = _page(items)
    lines = extract_lines_from_evidence(pe)

    assert len(lines) == 1
    assert lines[0].quantity == Decimal("4")
    assert lines[0].unit_price == Decimal("73.00")
    assert lines[0].amount == Decimal("292.00")


def test_hts_tariff_code_not_mapped_to_unit_price():
    """Tariff / HTS codes (e.g. 3822190080) must not be interpreted as financial unit prices."""
    items = [
        # Table Header
        _ev("h1", "Item", [50.0, 100.0, 90.0, 120.0]),
        _ev("h2", "Description", [100.0, 100.0, 300.0, 120.0]),
        _ev("h3", "HTS Code", [320.0, 100.0, 400.0, 120.0]),
        _ev("h4", "Qty", [420.0, 100.0, 480.0, 120.0]),
        _ev("h5", "Price", [500.0, 100.0, 580.0, 120.0]),
        _ev("h6", "Amount", [600.0, 100.0, 680.0, 120.0]),
        # Row with HTS code
        _ev("r1_1", "1", [50.0, 140.0, 90.0, 160.0]),
        _ev("r1_2", "Diagnostic Kit", [100.0, 140.0, 300.0, 160.0]),
        _ev("r1_3", "3822190080", [320.0, 140.0, 400.0, 160.0]),
        _ev("r1_4", "10", [420.0, 140.0, 480.0, 160.0]),
        _ev("r1_5", "45.00", [500.0, 140.0, 580.0, 160.0]),
        _ev("r1_6", "450.00", [600.0, 140.0, 680.0, 160.0]),
    ]
    pe = _page(items)
    lines = extract_lines_from_evidence(pe)

    assert len(lines) == 1
    assert lines[0].unit_price == Decimal("45.00")
    assert lines[0].amount == Decimal("450.00")
    assert lines[0].quantity == Decimal("10")
    assert lines[0].unit_price != Decimal("3822190080")


def test_model_number_not_mapped_to_unit_price():
    """Model numbers in description (e.g. CDJ-3000) must stay in description, not become unit price."""
    items = [
        # Table Header
        _ev("h1", "Description", [100.0, 100.0, 350.0, 120.0]),
        _ev("h2", "Qty", [370.0, 100.0, 420.0, 120.0]),
        _ev("h3", "Price", [440.0, 100.0, 500.0, 120.0]),
        _ev("h4", "Total", [520.0, 100.0, 600.0, 120.0]),
        # Row with model number in description
        _ev("r1_1", "Pioneer CDJ-3000 Multi Player", [100.0, 140.0, 350.0, 160.0]),
        _ev("r1_2", "2", [370.0, 140.0, 420.0, 160.0]),
        _ev("r1_3", "75.00", [440.0, 140.0, 500.0, 160.0]),
        _ev("r1_4", "150.00", [520.0, 140.0, 600.0, 160.0]),
    ]
    pe = _page(items)
    lines = extract_lines_from_evidence(pe)

    assert len(lines) == 1
    assert "CDJ-3000" in lines[0].description
    assert lines[0].quantity == Decimal("2")
    assert lines[0].unit_price == Decimal("75.00")
    assert lines[0].amount == Decimal("150.00")
    assert lines[0].unit_price != Decimal("3000")


def test_tax_and_discount_rows_not_extracted_as_lines():
    """Tax and discount rows below table must not be extracted as LineCandidates."""
    items = [
        # Header
        _ev("h1", "Description", [100.0, 100.0, 350.0, 120.0]),
        _ev("h2", "Total", [500.0, 100.0, 600.0, 120.0]),
        # Commercial line
        _ev("l1_1", "Equipment Rental", [100.0, 140.0, 350.0, 160.0]),
        _ev("l1_2", "500.00", [500.0, 140.0, 600.0, 160.0]),
        # Subtotal, Discount, Tax, Total
        _ev("s1", "Vahesumma", [100.0, 200.0, 250.0, 220.0]),
        _ev("s2", "500.00", [500.0, 200.0, 600.0, 220.0]),
        _ev("d1", "Allahindlus 10 %", [100.0, 230.0, 250.0, 250.0]),
        _ev("d2", "-50.00", [500.0, 230.0, 600.0, 250.0]),
        _ev("t1", "Käibemaks 20%", [100.0, 260.0, 250.0, 280.0]),
        _ev("t2", "90.00", [500.0, 260.0, 600.0, 280.0]),
        _ev("g1", "Summa koos käibemaksuga", [100.0, 290.0, 300.0, 310.0]),
        _ev("g2", "540.00", [500.0, 290.0, 600.0, 310.0]),
    ]
    pe = _page(items)
    lines = extract_lines_from_evidence(pe)

    assert len(lines) == 1
    assert lines[0].description == "Equipment Rental"
    assert lines[0].amount == Decimal("500.00")

    totals, taxes, discounts, _ = extract_totals_and_taxes_from_evidence(pe)
    assert any(t.total_type == "subtotal" and t.normalized_value == Decimal("500.00") for t in totals)
    assert any(t.total_type == "gross_total" and t.normalized_value == Decimal("540.00") for t in totals)
    assert any(tx.rate == Decimal("20") and tx.amount == Decimal("90.00") for tx in taxes)
    assert any(d.rate == Decimal("10") and d.amount == Decimal("50.00") for d in discounts)


def test_missing_quantity_remains_none_not_one():
    """When a row only provides a lump-sum amount without qty/price, quantity must remain None."""
    items = [
        # Table Header
        _ev("h1", "Service Description", [100.0, 100.0, 400.0, 120.0]),
        _ev("h2", "Amount", [450.0, 100.0, 550.0, 120.0]),
        # Lump sum row
        _ev("r1_1", "Lump sum consultation fee", [100.0, 140.0, 400.0, 160.0]),
        _ev("r1_2", "1500.00", [450.0, 140.0, 550.0, 160.0]),
    ]
    pe = _page(items)
    lines = extract_lines_from_evidence(pe)

    assert len(lines) == 1
    assert lines[0].description == "Lump sum consultation fee"
    assert lines[0].amount == Decimal("1500.00")
    assert lines[0].quantity is None
    assert lines[0].unit_price is None


def test_unmatched_tax_master_data_emits_empty_string():
    """Unmatched tax master data must emit tax_type_code="", never fallback to raw tax_type."""
    fs = _make_financial_structure(
        header_taxes=[
            NormalizedTax(
                tax_type="VAT",
                tax_name="VAT 19%",
                rate=Decimal("19.00"),
                amount=Decimal("19.00"),
                scope=Placement.HEADER,
                metadata={},  # No matched tax_type_code
            )
        ]
    )
    recon = reconstruct_erp(fs)
    decision = _make_safe_decision()

    payable = build_payable(decision, fs, recon)
    assert len(payable["taxes"]) == 1
    assert payable["taxes"][0]["tax_type_code"] == ""
    assert payable["taxes"][0]["tax_type"] == "VAT"


def test_matched_tax_master_data_emits_correct_code():
    """When tax is genuinely matched to master data, matched tax_type_code must be emitted."""
    fs = _make_financial_structure(
        header_taxes=[
            NormalizedTax(
                tax_type="VAT",
                tax_name="VAT 19%",
                rate=Decimal("19.00"),
                amount=Decimal("19.00"),
                scope=Placement.HEADER,
                metadata={"tax_type_code": "TX-STD-19"},
            )
        ]
    )
    recon = reconstruct_erp(fs)
    decision = _make_safe_decision()

    payable = build_payable(decision, fs, recon)
    assert len(payable["taxes"]) == 1
    assert payable["taxes"][0]["tax_type_code"] == "TX-STD-19"


def test_inv19_preservation():
    """Preserve INV-19: row 2 equal unit_price/amount preserves qty=1, and header taxes preserve placement."""
    items = [
        # Table Header
        _ev("h1", "Item Description", [50.0, 100.0, 250.0, 120.0]),
        _ev("h2", "Qty", [280.0, 100.0, 330.0, 120.0]),
        _ev("h3", "Price", [350.0, 100.0, 420.0, 120.0]),
        _ev("h4", "Total", [450.0, 100.0, 520.0, 120.0]),
        # Row 1 (Meal per head: 60 @ 90.00 = 5400.00)
        _ev("r1_1", "Meal per head", [50.0, 140.0, 250.0, 160.0]),
        _ev("r1_2", "60", [280.0, 140.0, 330.0, 160.0]),
        _ev("r1_3", "90.00", [350.0, 140.0, 420.0, 160.0]),
        _ev("r1_4", "5400.00", [450.0, 140.0, 520.0, 160.0]),
        # Row 2 (Transportation and Set up: Price 1200.00, Total 1200.00)
        _ev("r2_1", "Transportation and Set up", [50.0, 180.0, 250.0, 200.0]),
        _ev("r2_3", "1200.00", [350.0, 180.0, 420.0, 200.0]),
        _ev("r2_4", "1200.00", [450.0, 180.0, 520.0, 200.0]),
        # Taxes in summary
        _ev("s1", "Subtotal", [50.0, 240.0, 200.0, 260.0]),
        _ev("s2", "6600.00", [450.0, 240.0, 520.0, 260.0]),
        _ev("t1", "NHIL (2.5%)", [50.0, 270.0, 200.0, 290.0]),
        _ev("t1_a", "165.00", [450.0, 270.0, 520.0, 290.0]),
        _ev("t2", "GETFund (2.5%)", [50.0, 300.0, 200.0, 320.0]),
        _ev("t2_a", "165.00", [450.0, 300.0, 520.0, 320.0]),
        _ev("t3", "COVID-19 Health Levy (1%)", [50.0, 330.0, 250.0, 350.0]),
        _ev("t3_a", "66.00", [450.0, 330.0, 520.0, 350.0]),
        _ev("t4", "VAT (15%)", [50.0, 360.0, 200.0, 380.0]),
        _ev("t4_a", "1049.40", [450.0, 360.0, 520.0, 380.0]),
        _ev("g1", "Total Due", [50.0, 400.0, 200.0, 420.0]),
        _ev("g2", "8045.40", [450.0, 400.0, 520.0, 420.0]),
    ]
    pe = _page(items)
    lines = extract_lines_from_evidence(pe)

    assert len(lines) == 2
    assert lines[0].description == "Meal per head"
    assert lines[0].quantity == Decimal("60")
    assert lines[0].unit_price == Decimal("90.00")
    assert lines[0].amount == Decimal("5400.00")

    assert lines[1].description == "Transportation and Set up"
    assert lines[1].unit_price == Decimal("1200.00")
    assert lines[1].amount == Decimal("1200.00")
    assert lines[1].quantity == Decimal("1")  # Equal price & amount preserves qty=1

    totals, taxes, _, _ = extract_totals_and_taxes_from_evidence(pe)
    assert len(taxes) == 4
    tax_names = [tx.tax_name for tx in taxes]
    assert any("NHIL" in n for n in tax_names)
    assert any("GETFund" in n for n in tax_names)
    assert any("COVID" in n for n in tax_names)
    assert any("VAT" in n for n in tax_names)
