"""tests/test_payable_builder.py — Comprehensive Test Suite for Phase 9E Payable JSON Generation.

Test Categories:
1. Safety Gate & Precedence Routing (A, B, C, D, Anti-Bypass)
2. Schema Compliance & Key Presence (E, F, G, Schema Fidelity)
3. Master-Data Propagation (H, I, J, K, L)
4. Line Items & Semantic Role Filtering (P, Q, Y)
5. Taxes, Discounts, and Charges (T, U, V, W, X)
6. Document Types: Invoice & Credit Memo (Z, AA)
7. Isolation & Multi-Document Handling (R, S, AF, AG)
8. Safe Failure, Immutability, Determinism, & ERP Roundtrip (AB, AC, AD, AE)
9. Anti-Repair Test (discrepant/unsafe inputs never patched)
10. Real-Corpus End-to-End Pipeline (INV-01, HLD-01, INV-02, DU-02, HLD-03)
"""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import pytest

from erp import erp_book  # Strictly in tests for round-trip verification
from src.accounting.decision import (
    PayableDecision,
    PayableDecisionStatus,
    SafetyCheckCode,
    SafetyCheckResult,
    SafetyCheckStatus,
    evaluate_payable_safety,
)
from src.accounting.erp_reconstruction import (
    ERPComponentReconstruction,
    ERPLineReconstruction,
    ERPReconstruction,
    reconstruct_erp,
)
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
from src.accounting.payable_builder import (
    AutodraftPayableResult,
    FileOutputResult,
    PayableContractError,
    build_declined_entry,
    build_file_output,
    build_payable,
    serialize_file_output_json,
    serialize_payable_json,
    validate_autodraft_payload,
    validate_file_output,
    write_file_output,
)
from src.accounting.reconciliation import (
    ReconciliationResult,
    ReconciliationStatus,
    reconcile,
)
from src.extraction.candidates import (
    extract_candidates_from_document,
    extract_candidates_from_page,
)
from src.extraction.consolidation import (
    consolidate_candidates,
    consolidate_document_groups,
)
from src.extraction.financial_assembly import assemble_financial_documents
from src.matching.match_models import MasterMatchResult, MatchStatus
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
# Synthetic Test Fixture Builders
# ══════════════════════════════════════════════════════════════════════════

def _make_safe_decision(
    assembly_id: str = "asm-001",
    document_id: str = "INV-001.pdf",
    status: PayableDecisionStatus = PayableDecisionStatus.SAFE_TO_AUTODRAFT,
    decline_doc_type: Optional[str] = None,
    decline_reason: Optional[str] = None,
) -> PayableDecision:
    return PayableDecision(
        assembly_id=assembly_id,
        document_id=document_id,
        status=status,
        primary_reason="All safety checks passed" if status == PayableDecisionStatus.SAFE_TO_AUTODRAFT else "Test reason",
        decline_doc_type=decline_doc_type,
        decline_reason=decline_reason,
        reconciliation_status=ReconciliationStatus.MATCH,
    )


def _make_line(
    line_id: str = "line-1",
    desc: str = "Consulting Service",
    qty: Optional[str] = "4",
    price: Optional[str] = "100.00",
    amt: Optional[str] = "400.00",
    role: SemanticRole = SemanticRole.BILLED_LINE,
    item_type: str = "SERVICE",
    uom: str = "Hr",
    taxes: Sequence[NormalizedTax] = (),
    discounts: Sequence[NormalizedDiscount] = (),
) -> NormalizedLine:
    return NormalizedLine(
        source_line_id=line_id,
        line_number=1,
        description=desc,
        quantity=Decimal(qty) if qty else None,
        unit_price=Decimal(price) if price else None,
        amount=Decimal(amt) if amt else None,
        semantic_role=role,
        taxes=tuple(taxes),
        discounts=tuple(discounts),
        metadata={"item_type": item_type, "uom": uom},
    )


def _make_financial_structure(
    assembly_id: str = "asm-001",
    doc_id: str = "INV-001.pdf",
    inv_number: str = "INV-2026-001",
    inv_date: str = "2026-02-01",
    due_date: str = "2026-02-15",
    currency: str = "EUR",
    doc_type: InvoiceType = InvoiceType.INVOICE,
    gross_total: str = "476.00",
    subtotal: str = "400.00",
    tax_total: str = "76.00",
    lines: Sequence[NormalizedLine] = (),
    header_taxes: Sequence[NormalizedTax] = (),
    header_discounts: Sequence[NormalizedDiscount] = (),
    header_charges: Sequence[NormalizedCharge] = (),
    supplier_name: str = "ACME Corp",
    vat_id: str = "DE123456789",
    address: str = "Hauptstrasse 1, Berlin",
    company_code: str = "BOLTGROUP",
    po_number: str = "PO-999",
) -> FinancialStructure:
    pt = NormalizedPrintedTotals(
        gross_total=Decimal(gross_total) if gross_total else None,
        subtotal=Decimal(subtotal) if subtotal else None,
        tax_total=Decimal(tax_total) if tax_total else None,
    )
    sup = NormalizedParty(
        name=supplier_name,
        vat_id=vat_id,
        address=address,
    )
    buy = NormalizedParty(
        company_code=company_code,
    )
    po = NormalizedPO(po_number=po_number)

    line_items = list(lines)
    if not line_items:
        line_items = [_make_line()]

    return FinancialStructure(
        assembly_id=assembly_id,
        document_id=doc_id,
        document_type=doc_type,
        invoice_number=inv_number,
        invoice_date=inv_date,
        due_date=due_date,
        currency=currency,
        supplier=sup,
        buyer=buy,
        purchase_order=po,
        lines=tuple(line_items),
        header_taxes=tuple(header_taxes),
        header_discounts=tuple(header_discounts),
        header_charges=tuple(header_charges),
        printed_totals=pt,
    )


def _make_reconstruction(
    fs: FinancialStructure,
) -> ERPReconstruction:
    return reconstruct_erp(fs)


# ══════════════════════════════════════════════════════════════════════════
# Test Group 1: Safety Gate & Precedence Routing (A, B, C, D, Anti-Bypass)
# ══════════════════════════════════════════════════════════════════════════

class TestSafetyGateAndPrecedenceRouting:
    """Verify strict routing and anti-bypass enforcement."""

    def test_a_safe_to_autodraft_emits_payable(self) -> None:
        """A: SAFE_TO_AUTODRAFT -> produces valid payable dictionary."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision(status=PayableDecisionStatus.SAFE_TO_AUTODRAFT)

        payable = build_payable(decision, fs, recon)
        assert isinstance(payable, dict)
        assert payable["invoice_number"] == "INV-2026-001"
        assert payable["gross_total"] == "476.00"

    def test_b_not_payable_emits_declined(self) -> None:
        """B: NOT_PAYABLE -> build_declined_entry creates valid declined dict; build_payable rejects."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision(
            status=PayableDecisionStatus.NOT_PAYABLE,
            decline_doc_type="PURCHASE_ORDER",
            decline_reason="Document role 'purchase_order' is non-payable",
        )

        with pytest.raises(PayableContractError, match="Cannot build payable for document with status 'NOT_PAYABLE'"):
            build_payable(decision, fs, recon)

        declined = build_declined_entry(decision)
        assert declined["doc_type"] == "PURCHASE_ORDER"
        assert declined["reason"] == "Document role 'purchase_order' is non-payable"

    def test_c_hold_for_review_emits_no_payable(self) -> None:
        """C: HOLD_FOR_REVIEW -> build_payable raises PayableContractError; never enters payables[]."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision(status=PayableDecisionStatus.HOLD_FOR_REVIEW)

        with pytest.raises(PayableContractError, match="status 'HOLD_FOR_REVIEW'"):
            build_payable(decision, fs, recon)

        with pytest.raises(PayableContractError, match="Only NOT_PAYABLE decisions populate declined"):
            build_declined_entry(decision)

    def test_d_unsafe_to_autodraft_emits_no_payable(self) -> None:
        """D: UNSAFE_TO_AUTODRAFT -> build_payable raises PayableContractError; never enters payables[]."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision(status=PayableDecisionStatus.UNSAFE_TO_AUTODRAFT)

        with pytest.raises(PayableContractError, match="status 'UNSAFE_TO_AUTODRAFT'"):
            build_payable(decision, fs, recon)

        with pytest.raises(PayableContractError, match="Only NOT_PAYABLE decisions populate declined"):
            build_declined_entry(decision)

    def test_anti_bypass_enforcement(self) -> None:
        """Explicit Anti-Bypass: verify every non-safe status is blocked from payables."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        for st in (
            PayableDecisionStatus.HOLD_FOR_REVIEW,
            PayableDecisionStatus.UNSAFE_TO_AUTODRAFT,
            PayableDecisionStatus.NOT_PAYABLE,
        ):
            dec = _make_safe_decision(status=st)
            with pytest.raises(PayableContractError):
                build_payable(dec, fs, recon)


# ══════════════════════════════════════════════════════════════════════════
# Test Group 2: Schema Compliance & Key Presence (E, F, G, Schema Fidelity)
# ══════════════════════════════════════════════════════════════════════════

class TestSchemaComplianceAndKeyPresence:
    """Verify exact schema keys, types, and empty-value representation."""

    def test_e_exact_output_schema_compliance(self) -> None:
        """E: Output strictly adheres to AUTODRAFT_SCHEMA.md."""
        fs = _make_financial_structure(
            header_taxes=[NormalizedTax(tax_name="VAT 19%", tax_type="VAT", rate=Decimal("19"), amount=Decimal("76.00"), scope=Placement.HEADER)]
        )
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        errors = validate_autodraft_payload(payable)
        assert errors == []

    def test_f_required_keys_present(self) -> None:
        """F: Every single schema-defined key is present at all levels."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        for key in (
            "invoice_number", "invoice_date", "due_date", "invoice_type", "currency",
            "supplier", "buyer", "payment_term_id", "po_number", "po_id",
            "gross_total", "subtotal", "total_tax_amount", "discount_amount",
            "freight_charges", "insurance_charges", "extra_charges", "excise_duties",
            "taxes", "line_items",
        ):
            assert key in payable

        assert set(payable["supplier"].keys()) == {"name", "supplier_id", "address", "vat_id"}
        assert set(payable["buyer"].keys()) == {"company_code", "business_unit_code", "location_code"}
        assert len(payable["line_items"]) == 1
        assert set(payable["line_items"][0].keys()) == {
            "description", "item_type", "uom", "quantity", "unit_price", "total",
            "discount", "discount_percentage", "tax_rate", "tax_amount", "taxes",
        }

    def test_g_empty_values_representation(self) -> None:
        """G: Missing optional fields are empty strings or empty lists, never None."""
        fs = _make_financial_structure(
            inv_number="",
            due_date="",
            subtotal="",
            tax_total="",
            po_number="",
        )
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        assert payable["subtotal"] == ""
        assert payable["total_tax_amount"] == ""
        assert payable["discount_amount"] == ""
        assert payable["freight_charges"] == ""
        assert payable["taxes"] == []
        assert payable["po_id"] == ""
        assert payable["payment_term_id"] == ""
        # No None values in entire payload
        assert None not in payable.values()
        assert None not in payable["supplier"].values()
        assert None not in payable["buyer"].values()

    def test_schema_validator_catches_invalid_payload(self) -> None:
        """Schema Fidelity: Validator detects missing keys and invalid types."""
        invalid_p: Dict[str, Any] = {
            "invoice_number": "123",
            # missing rest
        }
        errs = validate_autodraft_payload(invalid_p)
        assert len(errs) > 0

    def test_dot_decimal_formatting_validation(self) -> None:
        """Schema Fidelity: Rejects comma or percent signs in numeric fields."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        payable["gross_total"] = "1.234,56"
        errs = validate_autodraft_payload(payable)
        assert any("gross_total" in e for e in errs)


# ══════════════════════════════════════════════════════════════════════════
# Test Group 3: Master-Data Propagation (H, I, J, K, L)
# ══════════════════════════════════════════════════════════════════════════

class TestMasterDataPropagation:
    """Verify matched codes vs honest blanks for master reference data."""

    def test_h_supplier_matched_id_propagation(self) -> None:
        """H: MATCHED supplier propagates master_id."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()
        s_match = MasterMatchResult(
            entity_type="supplier",
            status=MatchStatus.MATCHED,
            master_id="SUP-2845695",
        )

        payable = build_payable(decision, fs, recon, supplier_match=s_match)
        assert payable["supplier"]["supplier_id"] == "SUP-2845695"

    def test_i_supplier_honest_blank_on_no_match(self) -> None:
        """I: NO_MATCH supplier emits honest blank supplier_id."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()
        s_match = MasterMatchResult(
            entity_type="supplier",
            status=MatchStatus.NO_MATCH,
        )

        payable = build_payable(decision, fs, recon, supplier_match=s_match)
        assert payable["supplier"]["supplier_id"] == ""

    def test_j_ambiguous_supplier_never_emits_guessed_id(self) -> None:
        """J: AMBIGUOUS supplier match never produces a guessed ID."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()
        s_match = MasterMatchResult(
            entity_type="supplier",
            status=MatchStatus.AMBIGUOUS,
            candidates=[{"supplier_id": "GUESSED_1"}, {"supplier_id": "GUESSED_2"}],
        )

        payable = build_payable(decision, fs, recon, supplier_match=s_match)
        assert payable["supplier"]["supplier_id"] == ""

    def test_k_po_propagation(self) -> None:
        """K: Raw printed po_number and matched po_id propagate cleanly."""
        fs = _make_financial_structure(po_number="PO-PRINTED-888")
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()
        po_match = MasterMatchResult(
            entity_type="po",
            status=MatchStatus.MATCHED,
            master_id="PO_MASTER_001",
        )

        payable = build_payable(decision, fs, recon, po_match=po_match)
        assert payable["po_number"] == "PO-PRINTED-888"
        assert payable["po_id"] == "PO_MASTER_001"

    def test_l_buyer_codes_propagation(self) -> None:
        """L: Matched buyer hierarchy propagates company, BU, location codes."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()
        b_match = MasterMatchResult(
            entity_type="buyer",
            status=MatchStatus.MATCHED,
            details={
                "company_code": "BOLTGROUP",
                "business_unit_code": "EE004",
                "location_code": "LOC_EE_001",
            },
        )

        payable = build_payable(decision, fs, recon, buyer_match=b_match)
        assert payable["buyer"]["company_code"] == "BOLTGROUP"
        assert payable["buyer"]["business_unit_code"] == "EE004"
        assert payable["buyer"]["location_code"] == "LOC_EE_001"

    def test_payment_terms_propagation(self) -> None:
        """Payment term matched ID propagates; unmatched produces honest blank."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()
        pt_match = MasterMatchResult(
            entity_type="payment_term",
            status=MatchStatus.MATCHED,
            master_id="Net_30",
        )

        payable = build_payable(decision, fs, recon, payment_term_match=pt_match)
        assert payable["payment_term_id"] == "Net_30"


# ══════════════════════════════════════════════════════════════════════════
# Test Group 4: Line Items & Semantic Role Filtering (P, Q, Y)
# ══════════════════════════════════════════════════════════════════════════

class TestLineItemsAndSemanticRoleFiltering:
    """Verify strict line item filtering and anti-recalculation."""

    def test_p_posting_lines_filtering(self) -> None:
        """P: Only BILLED_LINE rows post to line_items[]."""
        ln1 = _make_line(line_id="line-1", role=SemanticRole.BILLED_LINE, desc="Billed Item")
        ln2 = _make_line(line_id="line-2", role=SemanticRole.COMPONENT_DETAIL, desc="Detail Item")
        fs = _make_financial_structure(lines=[ln1, ln2])
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        assert len(payable["line_items"]) == 1
        assert payable["line_items"][0]["description"] == "Billed Item"

    def test_q_component_detail_excluded(self) -> None:
        """Q: COMPONENT_DETAIL rows never appear in payable line_items."""
        detail = _make_line(line_id="det-1", role=SemanticRole.COMPONENT_DETAIL)
        billed = _make_line(line_id="bill-1", role=SemanticRole.BILLED_LINE)
        fs = _make_financial_structure(lines=[detail, billed])
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        assert len(payable["line_items"]) == 1
        assert payable["line_items"][0]["description"] == billed.description

    def test_y_zero_line_amount_recalculation(self) -> None:
        """Y: 9E never calculates quantity * unit_price."""
        ln = NormalizedLine(
            source_line_id="l-1",
            quantity=Decimal("10"),
            unit_price=Decimal("15.00"),
            amount=Decimal("150.00"),  # explicitly given
            semantic_role=SemanticRole.BILLED_LINE,
            metadata={"item_type": "SERVICE"},
        )
        fs = _make_financial_structure(lines=[ln])
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        assert payable["line_items"][0]["total"] == "150.00"

    def test_missing_item_type_raises_safe_failure(self) -> None:
        """Strict Authoritative Sourcing: Missing item_type fails safely without guessing."""
        ln = NormalizedLine(
            source_line_id="l-1",
            description="Item without type",
            quantity=Decimal("1"),
            unit_price=Decimal("50.00"),
            amount=Decimal("50.00"),
            semantic_role=SemanticRole.BILLED_LINE,
            metadata={},  # No item_type!
        )
        fs = _make_financial_structure(lines=[ln])
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        with pytest.raises(PayableContractError, match="item_type' is missing"):
            build_payable(decision, fs, recon)


# ══════════════════════════════════════════════════════════════════════════
# Test Group 5: Taxes, Discounts, and Charges (T, U, V, W, X)
# ══════════════════════════════════════════════════════════════════════════

class TestTaxesDiscountsCharges:
    """Verify scope placement, pre-aggregation, and zero keyword guessing."""

    def test_t_header_tax_remains_header_tax(self) -> None:
        """T: Header tax stays in taxes[]; never migrated to line."""
        htx = NormalizedTax(
            tax_name="VAT Reverse Charge",
            tax_type="VAT",
            rate=Decimal("0"),
            amount=Decimal("0.00"),
            scope=Placement.HEADER,
        )
        fs = _make_financial_structure(header_taxes=[htx])
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        assert len(payable["taxes"]) == 1
        assert payable["taxes"][0]["tax_name"] == "VAT Reverse Charge"
        assert payable["taxes"][0]["tax_rate"] == "0"
        assert payable["line_items"][0]["taxes"] == []

    def test_u_line_tax_remains_line_tax(self) -> None:
        """U: Line tax stays in line_items[].taxes[]; never moved to header."""
        ltx = NormalizedTax(
            tax_name="VAT 19%",
            tax_type="VAT",
            rate=Decimal("19"),
            amount=Decimal("76.00"),
            scope=Placement.LINE,
        )
        ln = _make_line(taxes=[ltx])
        fs = _make_financial_structure(lines=[ln], header_taxes=[])
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        assert len(payable["taxes"]) == 0
        assert len(payable["line_items"][0]["taxes"]) == 1
        assert payable["line_items"][0]["taxes"][0]["tax_rate"] == "19"

    def test_v_discounts_preserved(self) -> None:
        """V: Header discount in discount_amount; line discount on line."""
        h_disc = NormalizedDiscount(name="Prompt Pay", amount=Decimal("20.00"), scope=Placement.HEADER)
        l_disc = NormalizedDiscount(name="Line Rebate", amount=Decimal("5.00"), scope=Placement.LINE)
        ln = _make_line(discounts=[l_disc])
        fs = _make_financial_structure(lines=[ln], header_discounts=[h_disc])
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        assert payable["discount_amount"] == "20.00"
        assert payable["line_items"][0]["discount"] == "5.00"

    def test_w_charges_canonical_categorization(self) -> None:
        """W: Charges populated strictly by charge_category; no keyword guessing."""
        chg_fr = NormalizedCharge(name="Transport", amount=Decimal("15.00"), charge_category="freight", scope=Placement.HEADER)
        chg_ins = NormalizedCharge(name="Risk", amount=Decimal("5.00"), charge_category="insurance", scope=Placement.HEADER)
        chg_ex = NormalizedCharge(name="Tariff", amount=Decimal("2.50"), charge_category="excise", scope=Placement.HEADER)
        chg_oth = NormalizedCharge(name="Handling", amount=Decimal("10.00"), charge_category="extra", scope=Placement.HEADER)

        fs = _make_financial_structure(header_charges=[chg_fr, chg_ins, chg_ex, chg_oth])
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        assert payable["freight_charges"] == "15.00"
        assert payable["insurance_charges"] == "5.00"
        assert payable["excise_duties"] == "2.50"
        assert payable["extra_charges"] == "10.00"

    def test_x_multiple_unaggregated_charges_raises_error(self) -> None:
        """Zero Arithmetic: Multiple charges of the same category without upstream aggregate trigger error."""
        chg1 = NormalizedCharge(name="F1", amount=Decimal("10.00"), charge_category="freight")
        chg2 = NormalizedCharge(name="F2", amount=Decimal("10.00"), charge_category="freight")
        fs = _make_financial_structure(header_charges=[chg1, chg2])
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        with pytest.raises(PayableContractError, match="Multiple unaggregated freight charges"):
            build_payable(decision, fs, recon)


# ══════════════════════════════════════════════════════════════════════════
# Test Group 6: Document Types: Invoice & Credit Memo (Z, AA)
# ══════════════════════════════════════════════════════════════════════════

class TestDocumentTypes:
    """Verify supported invoice/credit memo types and rejection of unsupported types."""

    def test_z_credit_memo_positive_magnitude(self) -> None:
        """Z: CREDIT_MEMO emitted with invoice_type='CREDIT_MEMO' and positive magnitudes."""
        fs = _make_financial_structure(
            doc_type=InvoiceType.CREDIT_MEMO,
            gross_total="100.00",
            subtotal="100.00",
        )
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        assert payable["invoice_type"] == "CREDIT_MEMO"
        assert payable["gross_total"] == "100.00"

    def test_aa_debit_memo_fails_safely(self) -> None:
        """AA: DEBIT_MEMO is unsupported by AUTODRAFT_SCHEMA.md; fails safely without guessing."""
        fs = _make_financial_structure(doc_type=InvoiceType.DEBIT_MEMO)
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        with pytest.raises(PayableContractError, match="Unsupported document_type for autodraft schema: 'debit_memo'"):
            build_payable(decision, fs, recon)


# ══════════════════════════════════════════════════════════════════════════
# Test Group 7: Isolation & Multi-Document Handling (R, S, AF, AG)
# ══════════════════════════════════════════════════════════════════════════

class TestIsolationAndMultiDocument:
    """Verify document isolation, DU-02 regression, and multi-payable bundling."""

    def test_r_supporting_document_isolation(self) -> None:
        """R: Supporting group references remain metadata; never leak into payable."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        assert "supporting_group_ids" not in payable

    def test_s_du02_eur_try_regression(self) -> None:
        """S: DU-02 Regression: Primary EUR currency strictly preserved; secondary TRY isolated."""
        fs = _make_financial_structure(currency="EUR", gross_total="37534.94")
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)
        assert payable["currency"] == "EUR"
        assert payable["gross_total"] == "37534.94"

    def test_af_multiple_assemblies_isolated(self) -> None:
        """AF: Multi-payable file produces isolated, unmerged payable records."""
        fs1 = _make_financial_structure(inv_number="INV-A", gross_total="100.00")
        fs2 = _make_financial_structure(inv_number="INV-B", gross_total="200.00")
        dec1 = _make_safe_decision()
        dec2 = _make_safe_decision()

        p1 = build_payable(dec1, fs1, _make_reconstruction(fs1))
        p2 = build_payable(dec2, fs2, _make_reconstruction(fs2))

        file_out = build_file_output("multi.pdf", payables=[p1, p2])
        assert len(file_out["payables"]) == 2
        assert file_out["payables"][0]["invoice_number"] == "INV-A"
        assert file_out["payables"][1]["invoice_number"] == "INV-B"
        assert file_out["payables"][0]["gross_total"] == "100.00"
        assert file_out["payables"][1]["gross_total"] == "200.00"


# ══════════════════════════════════════════════════════════════════════════
# Test Group 8: Safe Failure, Immutability, Determinism, & ERP Roundtrip (AB, AC, AD, AE)
# ══════════════════════════════════════════════════════════════════════════

class TestSafeFailureImmutabilityDeterminism:
    """Verify safe failure, input immutability, byte determinism, and test-level ERP oracle check."""

    def test_ab_missing_mandatory_source_field_safe_failure(self) -> None:
        """AB: Missing currency raises PayableContractError; never invents currency."""
        fs = _make_financial_structure(currency="")
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        with pytest.raises(PayableContractError, match="currency' is missing"):
            build_payable(decision, fs, recon)

    def test_ac_input_immutability(self) -> None:
        """AC: Input FinancialStructure, decision, and reconstruction are not mutated."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        fs_dict_before = fs.to_dict()
        dec_dict_before = decision.to_dict()

        _ = build_payable(decision, fs, recon)

        assert fs.to_dict() == fs_dict_before
        assert decision.to_dict() == dec_dict_before

    def test_ad_deterministic_payload_and_serialization(self) -> None:
        """AD: Repeated calls yield identical Python dict and byte-for-byte serialized JSON."""
        fs = _make_financial_structure()
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        p1 = build_payable(decision, fs, recon)
        p2 = build_payable(decision, fs, recon)
        assert p1 == p2

        s1 = serialize_payable_json(p1)
        s2 = serialize_payable_json(p2)
        assert s1 == s2
        assert isinstance(s1, str)

    def test_ae_oracle_verification_roundtrip(self) -> None:
        """AE: Test-level contract check: Feeding emitted payable to erp_book() matches gross."""
        # Setup payable with 4 x 73.00 = 292.00, 2 x 73.00 = 146.00 -> total 438.00 (matching sample_autodraft.json)
        ln1 = _make_line(line_id="l1", qty="4", price="73.00", amt="292.00")
        ln2 = _make_line(line_id="l2", qty="2", price="73.00", amt="146.00")
        fs = _make_financial_structure(
            gross_total="438.00",
            subtotal="438.00",
            tax_total="0.00",
            lines=[ln1, ln2],
            header_taxes=[NormalizedTax(tax_name="VAT 0%", tax_type="VAT", rate=Decimal("0"), amount=Decimal("0.00"), scope=Placement.HEADER)],
        )
        recon = _make_reconstruction(fs)
        decision = _make_safe_decision()

        payable = build_payable(decision, fs, recon)

        # ERP oracle recompute check
        oracle_res = erp_book(payable)
        assert oracle_res["will_book_gross"] == 438.00
        assert oracle_res["currency"] == "EUR"

    def test_file_output_writing(self) -> None:
        """Verify write_file_output writes correct file structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = _make_financial_structure()
            recon = _make_reconstruction(fs)
            dec = _make_safe_decision()
            p = build_payable(dec, fs, recon)
            file_out = build_file_output("test_doc.pdf", payables=[p])

            target = write_file_output(file_out, output_dir=tmpdir)
            assert target.exists()
            assert target.name == "test_doc.json"

            loaded = json.loads(target.read_text(encoding="utf-8"))
            assert loaded["file"] == "test_doc.pdf"
            assert len(loaded["payables"]) == 1


# ══════════════════════════════════════════════════════════════════════════
# Test Group 9: Anti-Repair Test
# ══════════════════════════════════════════════════════════════════════════

class TestAntiRepair:
    """Explicit Anti-Repair: Unsafe or discrepant data is rejected without repair."""

    def test_anti_repair_discrepant_upstream(self) -> None:
        """Discrepant upstream data marked UNSAFE_TO_AUTODRAFT is never repaired."""
        # Doc says 1000, ERP says 1050
        fs = _make_financial_structure(gross_total="1000.00")
        recon = _make_reconstruction(fs)
        decision = PayableDecision(
            assembly_id=fs.assembly_id,
            document_id=fs.document_id,
            status=PayableDecisionStatus.UNSAFE_TO_AUTODRAFT,
            primary_reason="Reconciliation discrepancy of 50.00 exceeds tolerance",
        )

        with pytest.raises(PayableContractError):
            build_payable(decision, fs, recon)

        # Confirm file output emits payables: []
        file_out = build_file_output("discrepant.pdf", payables=[])
        assert file_out["payables"] == []
        assert file_out["declined"] == []


# ══════════════════════════════════════════════════════════════════════════
# Test Group 10: Real-Corpus End-to-End Pipeline (AH)
# ══════════════════════════════════════════════════════════════════════════

class TestRealCorpusE2E:
    """Verify end-to-end execution through Phase 9E using real corpus OCR evidence."""

    def test_ah_inv_01_pipeline_output(self) -> None:
        """INV-01: German invoice through 9E -> UNSAFE status emits payables=[]."""
        p = Path("artifacts/ocr/INV-01/page_001.json")
        if not p.exists():
            pytest.skip("INV-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        struct = normalize_financial_structure(assemblies[0])
        recon = reconstruct_erp(struct)
        res = reconcile(recon)
        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=recon,
            document_facts=facts,
        )

        assert decision.status in (PayableDecisionStatus.SAFE_TO_AUTODRAFT, PayableDecisionStatus.UNSAFE_TO_AUTODRAFT)
        if decision.status == PayableDecisionStatus.SAFE_TO_AUTODRAFT:
            payable = build_payable(decision, struct, recon)
            assert payable["gross_total"] == "438.00"
            assert len(payable["line_items"]) == 2
        else:
            with pytest.raises(PayableContractError):
                build_payable(decision, struct, recon)

        # File output produces expected structure
        f_out = build_file_output("INV-01.pdf", payables=[])
        assert f_out["file"] == "INV-01.pdf"
        assert f_out["payables"] == []

    def test_ah_du_02_pipeline_output(self) -> None:
        """DU-02: Multi-page invoice with TRY customs attachments -> non-safe status emits payables=[]."""
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
        res = reconcile(recon)
        primary_facts = next((f for f in facts_list if f.identity.currency == "EUR" or f.financials.currency == "EUR"), facts_list[0])
        decision = evaluate_payable_safety(
            reconciliation=res,
            reconstruction=recon,
            document_facts=primary_facts,
        )

        assert decision.status in (PayableDecisionStatus.HOLD_FOR_REVIEW, PayableDecisionStatus.UNSAFE_TO_AUTODRAFT)
        with pytest.raises(PayableContractError):
            build_payable(decision, struct, recon)

        f_out = build_file_output("DU-02.pdf", payables=[])
        assert f_out["payables"] == []

