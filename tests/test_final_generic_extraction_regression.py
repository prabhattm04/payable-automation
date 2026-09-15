"""tests/test_final_generic_extraction_regression.py — Targeted Generic Extraction Regression Test Suite.

Tests generic semantic extraction behaviors across 8 mandatory classes:
1. Leading quantity ("2 × Product", "2 x Product", "2 X Product", decimal, model number exclusion)
2. Table column interpretation (index + desc + qty + price + amount, product code exclusion, HTS exclusion)
3. Summary-row exclusion (subtotal, tax, discount, charge, total, amount due, tax rate vs amount)
4. Currency extraction (ISO code, symbol, currency name)
5. Conflicting totals (conflict preservation without arithmetic resolution)
6. Missing values (None preserved for qty, price, total without default invention)
7. Vision candidate handoff (Vision-derived candidates survive into consolidation)
8. Determinism (candidate ordering does not alter consolidated facts)
"""
from __future__ import annotations

from decimal import Decimal
import pytest

from src.understanding.evidence import Evidence, EvidenceSource, PageEvidence
from src.understanding.document_facts import SemanticRole
from src.extraction.candidates import (
    extract_candidates_from_page,
    extract_lines_from_evidence,
    extract_totals_and_taxes_from_evidence,
    extract_identity_from_evidence,
    normalize_currency_with_context,
    adapt_vision_evidence_to_candidates,
    LineCandidate,
    TotalCandidate,
    TaxCandidate,
    ExtractionCandidates,
)
from src.extraction.consolidation import consolidate_candidates


def _make_page(
    lines: list[str],
    boxes: list[list[float]] | None = None,
    doc_id: str = "SYNTH_DOC",
    page_num: int = 1,
) -> PageEvidence:
    """Build synthetic PageEvidence with explicit bounding boxes."""
    pe = PageEvidence(document_id=doc_id, page_number=page_num)
    for idx, text in enumerate(lines):
        bbox = (
            boxes[idx]
            if boxes and idx < len(boxes)
            else [10.0, float(idx * 30), 400.0, float(idx * 30 + 20)]
        )
        ev = Evidence.create(
            content=text,
            document_id=doc_id,
            page_number=page_num,
            source=EvidenceSource.OCR,
            extraction_method="test_engine",
            confidence=0.99,
            bbox=bbox,
        )
        pe.add_item(ev)
    return pe


# ==============================================================================
# 1. Leading Quantity Tests
# ==============================================================================

class TestLeadingQuantity:
    def test_leading_qty_multiplication_cross(self):
        """'2 × Product A' -> qty=2, description='Product A'."""
        pe = _make_page(["2 × Product A", "100.00"], boxes=[[10.0, 50.0, 200.0, 70.0], [250.0, 50.0, 320.0, 70.0]])
        lines = extract_lines_from_evidence(pe)
        assert len(lines) == 1
        assert lines[0].quantity == Decimal("2")
        assert lines[0].description == "Product A"
        assert lines[0].amount == Decimal("100.00")

    def test_leading_qty_lowercase_x(self):
        """'2 x Product A' -> qty=2, description='Product A'."""
        pe = _make_page(["2 x Product A", "100.00"], boxes=[[10.0, 50.0, 200.0, 70.0], [250.0, 50.0, 320.0, 70.0]])
        lines = extract_lines_from_evidence(pe)
        assert len(lines) == 1
        assert lines[0].quantity == Decimal("2")
        assert lines[0].description == "Product A"

    def test_leading_qty_uppercase_x(self):
        """'2 X Product A' -> qty=2, description='Product A'."""
        pe = _make_page(["2 X Product A", "100.00"], boxes=[[10.0, 50.0, 200.0, 70.0], [250.0, 50.0, 320.0, 70.0]])
        lines = extract_lines_from_evidence(pe)
        assert len(lines) == 1
        assert lines[0].quantity == Decimal("2")
        assert lines[0].description == "Product A"

    def test_leading_qty_decimal(self):
        """'2.5 x Bulk Chemical' -> qty=2.5, description='Bulk Chemical'."""
        pe = _make_page(["2.5 x Bulk Chemical", "250.00"], boxes=[[10.0, 50.0, 200.0, 70.0], [250.0, 50.0, 320.0, 70.0]])
        lines = extract_lines_from_evidence(pe)
        assert len(lines) == 1
        assert lines[0].quantity == Decimal("2.5")
        assert lines[0].description == "Bulk Chemical"

    def test_model_and_part_number_not_interpreted_as_quantity(self):
        """'Product 8000' and 'Model 2X9000' must NOT yield quantity."""
        pe1 = _make_page(["Product 8000", "500.00"], boxes=[[10.0, 50.0, 200.0, 70.0], [250.0, 50.0, 320.0, 70.0]])
        lines1 = extract_lines_from_evidence(pe1)
        assert len(lines1) == 1
        assert lines1[0].quantity is None
        assert "8000" in lines1[0].description

        pe2 = _make_page(["Model 2X9000", "750.00"], boxes=[[10.0, 50.0, 200.0, 70.0], [250.0, 50.0, 320.0, 70.0]])
        lines2 = extract_lines_from_evidence(pe2)
        assert len(lines2) == 1
        assert lines2[0].quantity is None
        assert "2X9000" in lines2[0].description


# ==============================================================================
# 2. Table Column Interpretation Tests
# ==============================================================================

class TestTableColumnInterpretation:
    def test_standard_table_columns(self):
        """Index + Description + Qty + Price + Amount mapped by geometry."""
        headers = ["Item", "Description", "Qty", "Unit Price", "Total"]
        h_boxes = [
            [50.0, 100.0, 90.0, 120.0],
            [120.0, 100.0, 300.0, 120.0],
            [350.0, 100.0, 400.0, 120.0],
            [450.0, 100.0, 550.0, 120.0],
            [600.0, 100.0, 700.0, 120.0],
        ]
        row = ["1", "Server Hosting Support", "5", "50.00", "250.00"]
        r_boxes = [
            [50.0, 130.0, 90.0, 150.0],
            [120.0, 130.0, 300.0, 150.0],
            [350.0, 130.0, 400.0, 150.0],
            [450.0, 130.0, 550.0, 150.0],
            [600.0, 130.0, 700.0, 150.0],
        ]
        pe = _make_page(headers + row, boxes=h_boxes + r_boxes)
        lines = extract_lines_from_evidence(pe)
        assert len(lines) == 1
        assert lines[0].description == "Server Hosting Support"
        assert lines[0].quantity == Decimal("5")
        assert lines[0].unit_price == Decimal("50.00")
        assert lines[0].amount == Decimal("250.00")

    def test_product_code_and_hts_exclusion(self):
        """Numeric product code and tariff code must NOT become quantity or price."""
        headers = ["Part No", "Description", "HTS Code", "Qty", "Price", "Amount"]
        h_boxes = [
            [50.0, 100.0, 120.0, 120.0],
            [140.0, 100.0, 300.0, 120.0],
            [320.0, 100.0, 420.0, 120.0],
            [450.0, 100.0, 500.0, 120.0],
            [530.0, 100.0, 600.0, 120.0],
            [630.0, 100.0, 720.0, 120.0],
        ]
        row = ["984210", "Industrial Sensor", "8542.31.00", "10", "15.00", "150.00"]
        r_boxes = [
            [50.0, 130.0, 120.0, 150.0],
            [140.0, 130.0, 300.0, 150.0],
            [320.0, 130.0, 420.0, 150.0],
            [450.0, 130.0, 500.0, 150.0],
            [530.0, 130.0, 600.0, 150.0],
            [630.0, 130.0, 720.0, 150.0],
        ]
        pe = _make_page(headers + row, boxes=h_boxes + r_boxes)
        lines = extract_lines_from_evidence(pe)
        assert len(lines) == 1
        assert lines[0].quantity == Decimal("10")
        assert lines[0].unit_price == Decimal("15.00")
        assert lines[0].amount == Decimal("150.00")
        assert lines[0].quantity != Decimal("984210")


# ==============================================================================
# 3. Summary-Row Exclusion Tests
# ==============================================================================

class TestSummaryRowExclusion:
    def test_summary_rows_not_commercial_lines(self):
        """Subtotal, Tax, Discount, Charge, and Total rows must NOT become billed lines."""
        pe = _make_page([
            "Widget Alpha", "100.00",
            "Subtotal: 100.00",
            "Discount: 10.00",
            "Freight: 15.00",
            "VAT 19%: 19.95",
            "Total: 124.95",
            "Amount Due: 124.95",
        ], boxes=[
            [10.0, 50.0, 150.0, 70.0], [200.0, 50.0, 260.0, 70.0],
            [10.0, 100.0, 150.0, 120.0],
            [10.0, 130.0, 150.0, 150.0],
            [10.0, 160.0, 150.0, 180.0],
            [10.0, 190.0, 150.0, 210.0],
            [10.0, 220.0, 150.0, 240.0],
            [10.0, 250.0, 150.0, 270.0],
        ])
        lines = extract_lines_from_evidence(pe)
        # Only the commercial item should be emitted as a billed line
        billed_lines = [l for l in lines if l.semantic_role == SemanticRole.BILLED_LINE]
        assert len(billed_lines) == 1
        assert billed_lines[0].description == "Widget Alpha"

    def test_tax_rate_versus_tax_amount(self):
        """A tax rate (e.g. 'VAT 19%') must NOT become tax amount 19."""
        pe = _make_page(["VAT 19%", "Total: 119.00"])
        _, tax_cands, _, _ = extract_totals_and_taxes_from_evidence(pe)
        assert len(tax_cands) >= 1
        assert tax_cands[0].rate == Decimal("19")
        # Amount must remain None when document only provides percentage rate
        assert tax_cands[0].amount is None

    def test_subtotal_does_not_become_tax(self):
        """A subtotal amount must NOT be stolen by a nearby tax label across intervening labels."""
        pe = _make_page(
            ["IVA", "Total ilíquido", "500.00"],
            boxes=[
                [100.0, 200.0, 150.0, 220.0],
                [200.0, 200.0, 320.0, 220.0],
                [400.0, 200.0, 480.0, 220.0],
            ]
        )
        totals, taxes, _, _ = extract_totals_and_taxes_from_evidence(pe)
        subtotals = [t for t in totals if t.total_type == "subtotal"]
        assert len(subtotals) >= 1
        assert subtotals[0].normalized_value == Decimal("500.00")
        # IVA must not claim 500.00 across the subtotal label
        iva_taxes = [t for t in taxes if "IVA" in (t.tax_name or "")]
        for t in iva_taxes:
            assert t.amount != Decimal("500.00")


# ==============================================================================
# 4. Currency Extraction Tests
# ==============================================================================

class TestCurrencyExtraction:
    def test_iso_code_currency(self):
        """Recognize explicit 3-letter ISO code."""
        norm, justified = normalize_currency_with_context("EUR")
        assert norm == "EUR"
        assert justified is True

        norm_thb, justified_thb = normalize_currency_with_context("THB")
        assert norm_thb == "THB"
        assert justified_thb is True

    def test_currency_symbol_with_context(self):
        """Recognize € symbol with Eurozone context."""
        norm, justified = normalize_currency_with_context("€", context_tokens=["GERMANY", "DEUTSCHLAND"])
        assert norm == "EUR"
        assert justified is True

    def test_currency_names(self):
        """Currency names BAHT, EURO, DOLLAR map to ISO codes."""
        norm_baht, _ = normalize_currency_with_context("BAHT")
        assert norm_baht == "THB"

        norm_thai, _ = normalize_currency_with_context("บาท /BAHT")
        assert norm_thai == "THB"

        norm_dollar, _ = normalize_currency_with_context("DOLLARS")
        assert norm_dollar == "USD"


# ==============================================================================
# 5. Conflicting Totals Tests
# ==============================================================================

class TestConflictingTotals:
    def test_conflicting_totals_preserved_without_arithmetic(self):
        """Conflicting gross totals from document must be preserved rather than resolved by math."""
        pe = _make_page(["Total: 100.00", "Grand Total: 105.00"])
        cands = extract_candidates_from_page(pe)
        gross_cands = [t for t in cands.total_candidates if t.total_type == "gross_total"]
        assert len(gross_cands) >= 2
        values = {t.normalized_value for t in gross_cands}
        assert Decimal("100.00") in values
        assert Decimal("105.00") in values


# ==============================================================================
# 6. Missing Values Preservation Tests
# ==============================================================================

class TestMissingValues:
    def test_quantity_remains_none(self):
        """Missing quantity must remain None without defaulting to 1."""
        pe = _make_page(["Consulting Services", "500.00"], boxes=[[10.0, 50.0, 200.0, 70.0], [250.0, 50.0, 320.0, 70.0]])
        lines = extract_lines_from_evidence(pe)
        assert len(lines) == 1
        assert lines[0].quantity is None
        assert lines[0].amount == Decimal("500.00")

    def test_unit_price_remains_none(self):
        """Missing unit price must remain None without dividing amount by quantity."""
        pe = _make_page(["5", "Support Hours", "250.00"], boxes=[[10.0, 50.0, 50.0, 70.0], [60.0, 50.0, 200.0, 70.0], [250.0, 50.0, 320.0, 70.0]])
        lines = extract_lines_from_evidence(pe)
        assert len(lines) == 1
        assert lines[0].quantity == Decimal("5")
        assert lines[0].amount == Decimal("250.00")
        assert lines[0].unit_price is None


# ==============================================================================
# 7. Vision Candidate Handoff Tests
# ==============================================================================

class TestVisionCandidateHandoff:
    def test_vision_candidates_handoff_and_consolidation(self):
        """Generic Vision-derived candidates survive into consolidation."""
        vis_content = (
            "Invoice Number: INV-99001\n"
            "Date: 2026-06-15\n"
            "Due Date: 2026-07-15\n"
            "Currency: USD\n"
            "PO: PO-7788\n"
            "Subtotal: 1000.00\n"
            "VAT 10%: 100.00\n"
            "Discount: 50.00\n"
            "Freight: 25.00\n"
            "Total: 1075.00\n"
            "Amount Due: 1075.00\n"
        )
        vis_ev = Evidence.create(
            content=vis_content,
            document_id="DOC_VIS",
            page_number=1,
            source=EvidenceSource.VISION,
            extraction_method="Puter/Qwen3-VL",
            confidence=None,
        )
        pe = _make_page(["Page 1 Placeholder"])
        cands = extract_candidates_from_page(pe, vision_evidence=vis_ev)

        # Vision candidates must be populated across all types
        assert any(c.field_name == "invoice_number" and c.source == "vision" for c in cands.identity_candidates)
        assert any(c.field_name == "invoice_date" and c.source == "vision" for c in cands.identity_candidates)
        assert any(c.field_name == "currency" and c.source == "vision" for c in cands.identity_candidates)
        assert any(c.field_name == "due_date" and c.source == "vision" for c in cands.identity_candidates)
        assert any(c.source == "vision" for c in cands.po_candidates)
        assert any(t.total_type == "subtotal" and t.source == "vision" for t in cands.total_candidates)
        assert any(t.total_type == "gross_total" and t.source == "vision" for t in cands.total_candidates)
        assert any(t.total_type == "amount_due" and t.source == "vision" for t in cands.total_candidates)
        assert any(t.source == "vision" for t in cands.tax_candidates)
        assert any(d.source == "vision" for d in cands.discount_candidates)
        assert any(c.source == "vision" for c in cands.charge_candidates)

        # Consolidate candidates
        facts = consolidate_candidates(cands)
        assert facts.identity.invoice_number == "INV-99001"
        assert facts.identity.currency == "USD"
        assert facts.financials.printed_totals is not None
        assert facts.financials.printed_totals.gross_total == Decimal("1075.00")
        assert facts.financials.printed_totals.subtotal == Decimal("1000.00")
        assert len(facts.financials.taxes) >= 1
        assert facts.financials.taxes[0].amount == Decimal("100.00")


# ==============================================================================
# 8. Determinism Tests
# ==============================================================================

class TestCandidateDeterminism:
    def test_candidate_order_does_not_change_consolidation(self):
        """Reversing candidate order yields identical consolidated fact interpretation."""
        pe = _make_page(["Invoice No: 12345", "Total: 500.00 EUR", "VAT 20%: 100.00"])
        cands = extract_candidates_from_page(pe)

        # Shuffle / reverse candidates
        rev_cands = ExtractionCandidates(
            document_id=cands.document_id,
            group_id=cands.group_id,
            page_numbers=cands.page_numbers,
            payable_relevance=cands.payable_relevance,
            page_role=cands.page_role,
            identity_candidates=tuple(reversed(cands.identity_candidates)),
            party_candidates=tuple(reversed(cands.party_candidates)),
            po_candidates=tuple(reversed(cands.po_candidates)),
            line_candidates=tuple(reversed(cands.line_candidates)),
            tax_candidates=tuple(reversed(cands.tax_candidates)),
            discount_candidates=tuple(reversed(cands.discount_candidates)),
            charge_candidates=tuple(reversed(cands.charge_candidates)),
            total_candidates=tuple(reversed(cands.total_candidates)),
        )

        facts1 = consolidate_candidates(cands)
        facts2 = consolidate_candidates(rev_cands)

        assert facts1.identity.invoice_number == facts2.identity.invoice_number
        assert facts1.identity.currency == facts2.identity.currency
        assert facts1.financials.printed_totals == facts2.financials.printed_totals
        assert len(facts1.financials.taxes) == len(facts2.financials.taxes)
