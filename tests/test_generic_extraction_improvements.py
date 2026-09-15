"""tests/test_generic_extraction_improvements.py — Test Suite for Generic Extraction Improvements.

Tests the approved generic improvements across Amendments 1–7:
1. Generic labelled field extraction (Name:, TIN:, Invoice No:, Date:, Account No:)
2. Generic buyer-labelled extraction (PREPARED FOR:, BILL TO:, SOLD TO:)
3. Section-heading handling (VENDOR DETAILS, BANK DETAILS must not become names)
4. Financial row classification (billed lines vs taxes vs totals vs banking/address)
5. Currency aliases (GH¢, GH₵, GHC, GH$, GH€, GHS, EUR, GBP context)
6. Compound tax with explicit printed tax amounts in ERP reconciliation
7. INV-19 end-to-end integration regression using actual OCR evidence
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import pytest

from src.understanding.evidence import Evidence, EvidenceSource, PageEvidence
from src.understanding.document_facts import InvoiceType, Placement, SemanticRole
from src.understanding.page_classifier import PageRole, PayableRelevance, classify_page
from src.extraction.candidates import (
    extract_identity_from_evidence,
    extract_party_and_po_from_evidence,
    extract_totals_and_taxes_from_evidence,
    extract_lines_from_evidence,
    normalize_currency_with_context,
    extract_candidates_from_document,
)
from src.extraction.consolidation import consolidate_document_groups
from src.extraction.financial_assembly import assemble_financial_documents
from src.extraction.validation import validate_financial_document_assembly
from src.accounting.financial_structure import normalize_financial_structure
from src.accounting.erp_reconstruction import reconstruct_erp
from src.accounting.reconciliation import reconcile
from src.matching.match_models import (
    ObservedSupplierIdentity,
    ObservedBuyerIdentity,
    ObservedPOIdentity,
    ObservedPOLineEvidence,
    ObservedPaymentTermIdentity,
    MatchStatus,
)
from run_pipeline import acquire_document_evidence, MasterMatchers
from src.pdf.loader import discover_pdfs
from src.understanding.document_grouper import group_document
from erp import erp_book


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


class TestGenericLabelledFieldExtraction:
    """1. Test generic labelled field extraction (<label>: <value> and split layouts)."""

    def test_labelled_fields_inline(self):
        pe = _make_page([
            "Invoice No: INV-2024-001",
            "Date: 15/09/2026",
            "VENDOR DETAILS",
            "Name: Alpha Solutions Ltd",
            "TIN: 9876543210",
            "Account No: 1234567890",
        ])

        id_cands = extract_identity_from_evidence(pe)
        party_cands, _ = extract_party_and_po_from_evidence(pe)

        inv_cands = [c for c in id_cands if c.field_name == "invoice_number"]
        assert len(inv_cands) >= 1
        assert "INV-2024-001" in inv_cands[0].raw_value

        date_cands = [c for c in id_cands if c.field_name == "invoice_date"]
        assert len(date_cands) >= 1
        assert "15/09/2026" in date_cands[0].raw_value

        sup_names = [c for c in party_cands if c.party_role == "supplier" and c.field_name == "name"]
        assert any(c.raw_value == "Alpha Solutions Ltd" for c in sup_names)

        sup_vats = [c for c in party_cands if c.party_role == "supplier" and c.field_name == "vat_id"]
        assert any("9876543210" in c.raw_value for c in sup_vats)

    def test_labelled_fields_split_layout(self):
        pe = _make_page(
            lines=[
                "Invoice Number",
                "INV-9999",
                "PREPARED FOR:",
                "Beta Corp International",
            ],
            boxes=[
                [10.0, 10.0, 100.0, 25.0],
                [110.0, 10.0, 200.0, 25.0],
                [100.0, 100.0, 200.0, 120.0],
                [100.0, 130.0, 300.0, 150.0],
            ],
        )

        id_cands = extract_identity_from_evidence(pe)
        party_cands, _ = extract_party_and_po_from_evidence(pe)

        inv_cands = [c for c in id_cands if c.field_name == "invoice_number"]
        assert any("INV-9999" in c.raw_value for c in inv_cands)

        buyer_names = [c for c in party_cands if c.party_role == "buyer" and c.field_name == "name"]
        assert any("Beta Corp International" in c.raw_value for c in buyer_names)


class TestGenericBuyerLabelledExtraction:
    """2. Test buyer labelled extraction across PREPARED FOR, BILL TO, SOLD TO."""

    @pytest.mark.parametrize("label", ["PREPARED FOR:", "BILL TO:", "SOLD TO:", "INVOICE TO:"])
    def test_buyer_labels(self, label):
        pe = _make_page(
            lines=[label, "Acme Global Logistics LLC"],
            boxes=[
                [50.0, 200.0, 150.0, 220.0],
                [50.0, 230.0, 300.0, 250.0],
            ],
        )
        party_cands, _ = extract_party_and_po_from_evidence(pe)
        buyer_names = [c for c in party_cands if c.party_role == "buyer" and c.field_name == "name"]
        assert len(buyer_names) >= 1
        assert "Acme Global Logistics LLC" in buyer_names[0].raw_value


class TestSectionHeadingHandling:
    """3. Test section headings are never emitted as identity values."""

    def test_section_headings_not_identity_values(self):
        pe = _make_page([
            "VENDOR DETAILS",
            "Name: Valid Vendor Inc",
            "BANK DETAILS",
            "Bank Name: City Bank",
            "CUSTOMER DETAILS",
            "Acme Buyer Ltd",
        ])
        party_cands, _ = extract_party_and_po_from_evidence(pe)
        all_raw_names = {c.raw_value.upper() for c in party_cands}

        assert "VENDOR DETAILS" not in all_raw_names
        assert "BANK DETAILS" not in all_raw_names
        assert "CUSTOMER DETAILS" not in all_raw_names
        assert "Valid Vendor Inc" in {c.raw_value for c in party_cands if c.party_role == "supplier"}


class TestFinancialRowClassification:
    """4. Test contextual financial row classification."""

    def test_financial_row_distinction(self):
        pe = _make_page(
            lines=[
                "ITEM QTY PRICE TOTAL",
                "Consulting Services",
                "10",
                "150.00",
                "1500.00",
                "Equipment Setup Fee",
                "500.00",
                "500.00",
                "VAT (15%)",
                "300.00",
                "Grand Total",
                "2300.00",
                "Account No: 9876543210 SORT code: 112233",
            ],
            boxes=[
                [10.0, 50.0, 400.0, 70.0],
                [10.0, 100.0, 150.0, 120.0],
                [160.0, 100.0, 180.0, 120.0],
                [200.0, 100.0, 250.0, 120.0],
                [300.0, 100.0, 360.0, 120.0],
                [10.0, 140.0, 150.0, 160.0],
                [200.0, 140.0, 250.0, 160.0],
                [300.0, 140.0, 360.0, 160.0],
                [10.0, 200.0, 150.0, 220.0],
                [300.0, 200.0, 360.0, 220.0],
                [10.0, 250.0, 150.0, 270.0],
                [300.0, 250.0, 360.0, 270.0],
                [10.0, 300.0, 350.0, 320.0],
            ],
        )

        lines = extract_lines_from_evidence(pe)
        totals, taxes, _, _ = extract_totals_and_taxes_from_evidence(pe)

        # Lines must contain commercial rows only
        assert len(lines) == 2
        descriptions = [l.description for l in lines]
        assert any("Consulting Services" in d for d in descriptions)
        assert any("Equipment Setup Fee" in d for d in descriptions)

        # Banking information must not become a line
        assert not any("Account No" in l.description for l in lines)
        assert not any("SORT" in l.description for l in lines)

        # Taxes
        assert len(taxes) >= 1
        assert any(t.tax_name == "VAT" and t.amount == Decimal("300.00") for t in taxes)

        # Totals
        assert len(totals) >= 1
        assert any(tot.total_type == "gross_total" and tot.normalized_value == Decimal("2300.00") for tot in totals)


class TestCurrencyAliases:
    """5. Test Ghanaian Cedi and other currency aliases with contextual justification."""

    @pytest.mark.parametrize("alias", ["GH¢", "GH₵", "GHC", "GH$", "GH€", "GHS"])
    def test_ghana_cedi_aliases(self, alias):
        norm, justified = normalize_currency_with_context(alias, context_tokens=["Invoice", "Accra", "GH"])
        assert norm == "GHS"
        assert justified is True

    def test_bare_dollar_with_ghana_context(self):
        norm, justified = normalize_currency_with_context("$", context_tokens=["VENDOR", "GHC", "ACCRA"])
        assert norm == "GHS"
        assert justified is True

    def test_bare_dollar_without_context(self):
        norm, justified = normalize_currency_with_context("$", context_tokens=["INVOICE"])
        assert norm == "$"
        assert justified is False

    def test_standard_currencies(self):
        assert normalize_currency_with_context("EUR")[0] == "EUR"
        assert normalize_currency_with_context("USD")[0] == "USD"
        assert normalize_currency_with_context("GBP")[0] == "GBP"


class TestCompoundTaxReconciliation:
    """6. Test compound tax with explicit printed amounts in ERP reconciliation."""

    def test_explicit_tax_amounts_reach_exact_erp_reconciliation(self):
        # Simulating INV-19 document-grounded input to erp_book:
        # Net = 6600.00, Levies = 165.00 + 165.00 + 66.00 = 396.00, VAT = 1049.40, Gross = 8045.40
        payable = {
            "currency": "GHS",
            "line_items": [
                {"quantity": "60", "unit_price": "90.00", "total": "5400.00"},
                {"quantity": "1", "unit_price": "1200.00", "total": "1200.00"},
            ],
            "taxes": [
                {"tax_name": "NHIL", "tax_rate": "2.5", "tax_amount": "165.00"},
                {"tax_name": "GETFund Levy", "tax_rate": "2.5", "tax_amount": "165.00"},
                {"tax_name": "COVID-19 Levy", "tax_rate": "1.0", "tax_amount": "66.00"},
                {"tax_name": "VAT", "tax_rate": "15.0", "tax_amount": "1049.40"},
            ],
        }

        result = erp_book(payable)

        assert round(result.get("will_book_gross", 0.0), 2) == 8045.40
        diff = Decimal("8045.40") - Decimal(str(round(result.get("will_book_gross", 0.0), 2)))
        assert diff == Decimal("0.00")


class TestInv19IntegrationRegression:
    """7. INV-19 end-to-end integration test using actual document evidence."""

    def test_inv19_pipeline_end_to_end(self):
        doc_path = next(p for p in discover_pdfs(Path("documents")) if p.stem == "INV-19")
        pes = acquire_document_evidence(doc_path)
        unds = [classify_page(pe) for pe in pes]
        grps = group_document(pes, unds).groups

        # Step 1: Candidate Extraction
        cands = extract_candidates_from_document(pes, unds, grps, vision_provider=None)
        assert len(cands) >= 1
        p1_cands = next(c for c in cands if c.page_numbers == (1,))

        # Verify Supplier candidate
        sup_names = [p.raw_value for p in p1_cands.party_candidates if p.party_role == "supplier" and p.field_name == "name"]
        assert "Redwater Technologies" in sup_names

        sup_vats = [p.raw_value for p in p1_cands.party_candidates if p.party_role == "supplier" and p.field_name == "vat_id"]
        assert "C9765403675" in sup_vats

        # Verify Buyer candidate
        buy_names = [p.raw_value for p in p1_cands.party_candidates if p.party_role == "buyer" and p.field_name == "name"]
        assert any("Northwind HOLDINGS" in n for n in buy_names)

        # Verify Currency
        currs = [c.normalized_value for c in p1_cands.identity_candidates if c.field_name == "currency"]
        assert all(c == "GHS" for c in currs if c)

        # Verify 2 Billed Lines
        assert len(p1_cands.line_candidates) == 2
        l1, l2 = p1_cands.line_candidates[0], p1_cands.line_candidates[1]
        assert l1.quantity == Decimal(60) and l1.unit_price == Decimal("90.00") and l1.amount == Decimal("5400.00")
        assert l2.quantity == Decimal(1) and l2.unit_price == Decimal("1200.00") and l2.amount == Decimal("1200.00")

        # Verify 4 Taxes
        assert len(p1_cands.tax_candidates) == 4
        tax_map = {t.tax_name: (t.rate, t.amount) for t in p1_cands.tax_candidates}
        assert tax_map["NHIL"] == (Decimal("2.5"), Decimal("165.00"))
        assert tax_map["GETFund Levy"] == (Decimal("2.5"), Decimal("165.00"))
        assert tax_map["COVID-19 Levy"] == (Decimal("1"), Decimal("66.00"))
        assert tax_map["VAT"] == (Decimal("15"), Decimal("1049.40"))

        # Step 2: Consolidation & Assembly
        facts_list = consolidate_document_groups(cands)
        asms = assemble_financial_documents(facts_list, grps)
        asm = asms[0]

        assert asm.currency == "GHS"
        assert asm.printed_totals is not None
        assert asm.printed_totals.gross_total == Decimal("8045.40")
        assert asm.printed_totals.subtotal == Decimal("6600.00")
        assert asm.printed_totals.taxable_base == Decimal("6996.00")

        # Step 3: Financial Structure & ERP Reconstruction
        val_result = validate_financial_document_assembly(asm)
        struct = normalize_financial_structure(asm, val_result)
        recon = reconstruct_erp(struct)

        assert recon.reconstructed_gross_total == Decimal("8045.40")

        # Step 4: Reconciliation
        recon_res = reconcile(recon, tolerance=Decimal("0.00"))
        assert recon_res.gross_comparison is not None
        assert recon_res.gross_comparison.difference == Decimal("0.00")
        assert recon_res.gross_comparison.status.value == "MATCH"
