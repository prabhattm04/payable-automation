"""tests/test_extraction_validation.py — Comprehensive Test Suite for Phase 9B-4 Extraction Validation.

Test Categories:
A. Valid single-page invoice (structurally coherent -> VALID)
B. Missing optional fields (discount, charges, PO missing -> does not fail)
C. Missing critical identity (completely empty payable candidate -> BLOCKED)
D. Contradictory invoice number (unresolved conflict detected -> ERROR / INVALID)
E. Contradictory currency (unresolved currency conflict -> ERROR / INVALID)
F. Supporting currency isolation (primary EUR + supporting TRY -> no false conflict)
G. Semantic role validation (structural role mismatches flagged without text heuristics)
H. Tax placement (header remains header, line remains line, mismatches flagged)
I. Missing tax (no tax observed remains None, never becomes 0)
J. Missing discount (no discount observed remains None, never becomes 0)
K. Missing charge (no charge observed remains None, never becomes 0)
L. Repeated identical lines (legitimate identical lines pass without false duplicate error)
M. Provenance validation (missing evidence on observed facts produces ERROR)
N. Conflict preservation (upstream conflicts survive and produce typed issues)
O. No arithmetic (no quantity * price, no tax calculation, no gross derivation)
P. No master matching (no master matchers called, no master IDs injected)
Q. No mutation (deep equality of assembly before and after validation)
R. Determinism (same input produces identical validation results and ordering)
S. Serialization & Round-tripping (to_dict, from_dict, to_json, from_json)
T. Payable relevance (supporting/non-payable assemblies validated in role, not promoted)
U. Invoice type validation (contradictory invoice types flagged as ERROR)
V. Evidence completeness (accurate validated fact and evidence counts)
W. Multiple logical payables (independent validation without cross-contamination)
X. Real Corpus Verification (INV-01, HLD-01, INV-02, DU-02, HLD-03)
"""
from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path
import pytest

from src.extraction.financial_assembly import (
    FinancialDocumentAssembly,
    assemble_financial_documents,
)
from src.extraction.validation import (
    ExtractionValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    validate_financial_document_assemblies,
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

def _make_valid_line(
    line_num: int = 1,
    desc: str = "Consulting Services",
    qty: str = "10",
    price: str = "100.00",
    amount: str = "1000.00",
    ev_ids: tuple[str, ...] = ("EV_LN1",),
    row_ev_ids: tuple[str, ...] = ("ROW_1",),
    role: SemanticRole = SemanticRole.BILLED_LINE,
    taxes: tuple[TaxFact, ...] = (),
) -> LineFact:
    field_evs = {
        "quantity": ev_ids,
        "unit_price": ev_ids,
        "amount": ev_ids,
    }
    return LineFact(
        line_number=line_num,
        description=desc,
        quantity=Decimal(qty),
        unit_price=Decimal(price),
        amount=Decimal(amount),
        taxes=taxes,
        evidence_ids=ev_ids,
        field_evidence_ids=field_evs,
        source_row_evidence_ids=row_ev_ids,
        semantic_role=role,
        origin=FactOrigin.OBSERVED,
    )


def _make_valid_assembly(
    assembly_id: str = "ASM-001",
    doc_id: str = "DOC-001",
    invoice_number: str = "INV-2026-001",
    invoice_date: str = "2026-02-01",
    currency: str = "EUR",
    supplier_name: str = "Acme Supplies GmbH",
    vat_id: str = "DE123456789",
    lines: tuple[LineFact, ...] = (),
    taxes: tuple[TaxFact, ...] = (),
    gross_total: str = "1000.00",
    relevance: PayableRelevance = PayableRelevance.PAYABLE_CANDIDATE,
    conflicts: tuple[dict, ...] = (),
    supporting_group_ids: tuple[str, ...] = (),
) -> FinancialDocumentAssembly:
    if not lines:
        lines = (_make_valid_line(),)

    ev_ids = ("EV_HEAD",)
    ident = DocumentIdentityFacts(
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        currency=currency,
        evidence_ids=ev_ids,
        field_evidence_ids={
            "invoice_number": ev_ids,
            "invoice_date": ev_ids,
            "currency": ev_ids,
        },
        origin=FactOrigin.OBSERVED,
    )
    supplier = SupplierIdentityFact(
        observed_name=supplier_name,
        vat_id=vat_id,
        evidence_ids=ev_ids,
        field_evidence_ids={"observed_name": ev_ids, "vat_id": ev_ids},
        origin=FactOrigin.OBSERVED,
    )
    buyer = BuyerIdentityFact(
        observed_company="Client Corp",
        evidence_ids=ev_ids,
        field_evidence_ids={"observed_company": ev_ids},
        origin=FactOrigin.OBSERVED,
    )
    parties = PartyIdentityFacts(supplier=supplier, buyer=buyer, evidence_ids=ev_ids)
    pt = PrintedTotalsFact(
        gross_total=Decimal(gross_total),
        evidence_ids=ev_ids,
        field_evidence_ids={"gross_total": ev_ids},
        origin=FactOrigin.OBSERVED,
    )
    fin = FinancialFacts(
        lines=lines,
        taxes=taxes,
        printed_totals=pt,
        currency=currency,
        evidence_ids=ev_ids,
    )
    doc_facts = DocumentFacts(
        document_id=doc_id,
        page_numbers=(1,),
        payable_relevance=relevance,
        identity=ident,
        parties=parties,
        financials=fin,
        evidence_ids=ev_ids,
    )
    return FinancialDocumentAssembly(
        assembly_id=assembly_id,
        document_id=doc_id,
        primary_group_ids=("GRP-1",),
        supporting_group_ids=supporting_group_ids,
        page_numbers=(1,),
        payable_relevance=relevance,
        facts=doc_facts,
        conflicts=conflicts,
        evidence_ids=ev_ids,
    )


# ══════════════════════════════════════════════════════════════════════════
# A. Valid Single-Page Invoice
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryAValidInvoice:
    def test_a1_structurally_coherent_invoice_is_valid(self) -> None:
        """A complete, coherent invoice produces status VALID without errors."""
        asm = _make_valid_assembly()
        result = validate_financial_document_assembly(asm)

        assert result.status == ValidationStatus.VALID
        assert result.is_valid is True
        assert result.is_usable is True
        assert result.has_blocking is False
        assert result.has_errors is False
        assert result.validated_fact_count > 0
        assert result.validated_evidence_count > 0


# ══════════════════════════════════════════════════════════════════════════
# B. Missing Optional Fields
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryBMissingOptionalFields:
    def test_b1_missing_discounts_charges_and_po_does_not_fail(self) -> None:
        """Absence of discounts, charges, or PO does not fail validation."""
        asm = _make_valid_assembly()
        assert len(asm.discounts) == 0
        assert len(asm.charges) == 0
        assert asm.po.observed_po_number is None

        result = validate_financial_document_assembly(asm)
        assert result.status == ValidationStatus.VALID
        assert not result.has_errors
        assert not result.has_blocking

    def test_b2_missing_buyer_produces_info_not_error(self) -> None:
        """Missing buyer is recorded as INFO and does not invalidate the invoice."""
        asm = _make_valid_assembly()
        # Create facts with no buyer
        parties_no_buyer = PartyIdentityFacts(
            supplier=asm.supplier,
            buyer=BuyerIdentityFact(),
            evidence_ids=asm.evidence_ids,
        )
        facts_no_buyer = DocumentFacts(
            document_id=asm.document_id,
            identity=asm.facts.identity,
            parties=parties_no_buyer,
            financials=asm.facts.financials,
            evidence_ids=asm.evidence_ids,
        )
        asm_no_buyer = FinancialDocumentAssembly(
            assembly_id="ASM-NO-BUYER",
            document_id=asm.document_id,
            primary_group_ids=("GRP-1",),
            facts=facts_no_buyer,
            evidence_ids=asm.evidence_ids,
        )

        result = validate_financial_document_assembly(asm_no_buyer)
        assert result.status == ValidationStatus.VALID
        buyer_issues = result.get_issues_by_field("buyer")
        assert len(buyer_issues) == 1
        assert buyer_issues[0].code == "PARTY_BUYER_MISSING"
        assert buyer_issues[0].severity == ValidationSeverity.INFO


# ══════════════════════════════════════════════════════════════════════════
# C. Missing Critical Identity
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryCMissingCriticalIdentity:
    def test_c1_completely_empty_payable_is_blocked(self) -> None:
        """A payable candidate with no evidence, no lines, no totals, and no identity is BLOCKED."""
        empty_facts = DocumentFacts(
            document_id="EMPTY-DOC",
            payable_relevance=PayableRelevance.PAYABLE_CANDIDATE,
        )
        empty_asm = FinancialDocumentAssembly(
            assembly_id="ASM-EMPTY",
            document_id="EMPTY-DOC",
            primary_group_ids=("GRP-EMPTY",),
            payable_relevance=PayableRelevance.PAYABLE_CANDIDATE,
            facts=empty_facts,
        )

        result = validate_financial_document_assembly(empty_asm)
        assert result.status == ValidationStatus.BLOCKED
        assert result.has_blocking is True
        assert any(i.code == "IDENTITY_NO_EVIDENCE" for i in result.issues)

    def test_c2_missing_invoice_number_produces_warning(self) -> None:
        """A payable candidate with financials but no invoice number produces WARNING (usable)."""
        ev_ids = ("EV_1",)
        ident = DocumentIdentityFacts(currency="EUR", evidence_ids=ev_ids, field_evidence_ids={"currency": ev_ids})
        supplier = SupplierIdentityFact(observed_name="Supplier X", evidence_ids=ev_ids, field_evidence_ids={"observed_name": ev_ids})
        parties = PartyIdentityFacts(supplier=supplier, evidence_ids=ev_ids)
        line = _make_valid_line()
        fin = FinancialFacts(lines=(line,), currency="EUR", evidence_ids=ev_ids)
        facts = DocumentFacts(
            document_id="DOC-NO-INV-NUM",
            identity=ident,
            parties=parties,
            financials=fin,
            evidence_ids=ev_ids,
        )
        asm = FinancialDocumentAssembly(
            assembly_id="ASM-NO-INV",
            document_id="DOC-NO-INV-NUM",
            primary_group_ids=("GRP-1",),
            facts=facts,
            evidence_ids=ev_ids,
        )

        result = validate_financial_document_assembly(asm)
        assert result.status == ValidationStatus.WARNING
        assert result.is_usable is True
        assert any(i.code == "IDENTITY_MISSING_INVOICE_NUMBER" for i in result.issues)


# ══════════════════════════════════════════════════════════════════════════
# D. Contradictory Invoice Number
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryDContradictoryInvoiceNumber:
    def test_d1_conflicting_invoice_numbers_produce_error_invalid(self) -> None:
        """Conflicting explicit invoice numbers in assembly.conflicts yield ERROR and INVALID status."""
        conflict = {
            "field": "invoice_number",
            "conflict_type": "disagreement",
            "observations": [
                {"value": "INV-100", "sources": ["ocr"], "evidence_ids": ["EV_A"]},
                {"value": "INV-200", "sources": ["vision"], "evidence_ids": ["EV_B"]},
            ],
        }
        asm = _make_valid_assembly(conflicts=(conflict,))
        result = validate_financial_document_assembly(asm)

        assert result.status == ValidationStatus.INVALID
        assert result.has_errors is True
        inv_issues = result.get_issues_by_field("invoice_number")
        assert len(inv_issues) == 1
        assert inv_issues[0].code == "IDENTITY_CONTRADICTORY_INVOICE_NUMBER"
        assert inv_issues[0].severity == ValidationSeverity.ERROR
        # Neither value is chosen
        assert asm.invoice_number == "INV-2026-001"


# ══════════════════════════════════════════════════════════════════════════
# E. Contradictory Currency
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryEContradictoryCurrency:
    def test_e1_conflicting_currencies_in_primary_document_produce_error(self) -> None:
        """Conflicting currencies within the same logical document yield ERROR and INVALID status."""
        conflict = {
            "field": "currency",
            "conflict_type": "disagreement",
            "observations": [
                {"value": "EUR", "sources": ["ocr"], "evidence_ids": ["EV_A"]},
                {"value": "USD", "sources": ["vision"], "evidence_ids": ["EV_B"]},
            ],
        }
        asm = _make_valid_assembly(conflicts=(conflict,))
        result = validate_financial_document_assembly(asm)

        assert result.status == ValidationStatus.INVALID
        assert result.has_errors is True
        curr_issues = result.get_issues_by_field("currency")
        assert any(i.code == "CURRENCY_CONFLICT" and i.severity == ValidationSeverity.ERROR for i in curr_issues)


# ══════════════════════════════════════════════════════════════════════════
# F. Supporting Currency Isolation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryFSupportingCurrencyIsolation:
    def test_f1_supporting_document_currency_does_not_affect_primary(self) -> None:
        """Primary EUR payable referencing a supporting group does NOT trigger a currency conflict."""
        asm = _make_valid_assembly(
            currency="EUR",
            supporting_group_ids=("GRP-CUSTOMS-TRY",),
        )
        result = validate_financial_document_assembly(asm)

        assert result.status == ValidationStatus.VALID
        assert not any(i.code == "CURRENCY_CONFLICT" for i in result.issues)


# ══════════════════════════════════════════════════════════════════════════
# G. Semantic Role Validation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryGSemanticRoleValidation:
    def test_g1_structural_role_mismatch_flagged_without_text_inspection(self) -> None:
        """Line item with explicit semantic role TOTAL placed in lines container produces warning."""
        line_total_role = _make_valid_line(role=SemanticRole.TOTAL)
        asm = _make_valid_assembly(lines=(line_total_role,))

        result = validate_financial_document_assembly(asm)
        assert result.status == ValidationStatus.WARNING
        role_issues = [i for i in result.issues if i.code == "SEMANTIC_ROLE_STRUCTURAL_MISMATCH"]
        assert len(role_issues) == 1
        assert role_issues[0].severity == ValidationSeverity.WARNING
        # Verify original fact was untouched
        assert asm.lines[0].semantic_role == SemanticRole.TOTAL

    def test_g2_line_with_total_description_not_flagged_if_role_is_billed_line(self) -> None:
        """Description wording like 'Total Price' is NOT flagged if upstream role is BILLED_LINE."""
        line_desc_total = _make_valid_line(desc="Total Security Package", role=SemanticRole.BILLED_LINE)
        asm = _make_valid_assembly(lines=(line_desc_total,))

        result = validate_financial_document_assembly(asm)
        assert result.status == ValidationStatus.VALID
        assert not any(i.code == "SEMANTIC_ROLE_STRUCTURAL_MISMATCH" for i in result.issues)


# ══════════════════════════════════════════════════════════════════════════
# H. Tax Placement Invariants
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryHTaxPlacement:
    def test_h1_header_tax_with_line_placement_produces_error(self) -> None:
        """Header tax fact with Placement.LINE produces TAX_PLACEMENT_MISMATCH ERROR."""
        tax = TaxFact(
            tax_name="VAT 19%",
            rate=Decimal("19"),
            amount=Decimal("190.00"),
            placement=Placement.LINE,  # Mismatch for header container
            evidence_ids=("EV_TAX",),
            field_evidence_ids={"rate": ("EV_TAX",), "amount": ("EV_TAX",)},
        )
        asm = _make_valid_assembly(taxes=(tax,))

        result = validate_financial_document_assembly(asm)
        assert result.status == ValidationStatus.INVALID
        tax_issues = [i for i in result.issues if i.code == "TAX_PLACEMENT_MISMATCH"]
        assert len(tax_issues) == 1
        assert tax_issues[0].severity == ValidationSeverity.ERROR
        # Tax placement is NOT repaired or moved
        assert asm.taxes[0].placement == Placement.LINE

    def test_h2_line_tax_with_header_placement_produces_error(self) -> None:
        """Line-level tax fact with Placement.HEADER produces TAX_PLACEMENT_MISMATCH ERROR."""
        line_tax = TaxFact(
            tax_name="VAT 19%",
            rate=Decimal("19"),
            amount=Decimal("190.00"),
            placement=Placement.HEADER,  # Mismatch for line container
            evidence_ids=("EV_TAX",),
            field_evidence_ids={"rate": ("EV_TAX",), "amount": ("EV_TAX",)},
        )
        line = _make_valid_line(taxes=(line_tax,))
        asm = _make_valid_assembly(lines=(line,))

        result = validate_financial_document_assembly(asm)
        assert result.status == ValidationStatus.INVALID
        tax_issues = [i for i in result.issues if i.code == "TAX_PLACEMENT_MISMATCH"]
        assert len(tax_issues) == 1
        assert asm.lines[0].taxes[0].placement == Placement.HEADER

    def test_h3_tax_with_unknown_placement_produces_warning(self) -> None:
        """Tax fact with Placement.UNKNOWN produces TAX_PLACEMENT_UNKNOWN WARNING."""
        tax = TaxFact(
            tax_name="VAT 19%",
            rate=Decimal("19"),
            amount=Decimal("190.00"),
            placement=Placement.UNKNOWN,
            evidence_ids=("EV_TAX",),
            field_evidence_ids={"rate": ("EV_TAX",), "amount": ("EV_TAX",)},
        )
        asm = _make_valid_assembly(taxes=(tax,))

        result = validate_financial_document_assembly(asm)
        assert result.status == ValidationStatus.WARNING
        unknown_issues = [i for i in result.issues if i.code == "TAX_PLACEMENT_UNKNOWN"]
        assert len(unknown_issues) == 1


# ══════════════════════════════════════════════════════════════════════════
# I, J, K. Missing Tax, Discount, Charge
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryIJKMissingFinancials:
    def test_i1_missing_tax_remains_empty_not_zero(self) -> None:
        """No tax observed means taxes is empty, never tax = 0."""
        asm = _make_valid_assembly(taxes=())
        result = validate_financial_document_assembly(asm)

        assert result.status == ValidationStatus.VALID
        assert asm.taxes == ()

    def test_j1_missing_discount_remains_empty_not_zero(self) -> None:
        """No discount observed means discounts is empty, never discount = 0."""
        asm = _make_valid_assembly()
        result = validate_financial_document_assembly(asm)

        assert result.status == ValidationStatus.VALID
        assert asm.discounts == ()

    def test_k1_missing_charge_remains_empty_not_zero(self) -> None:
        """No charge observed means charges is empty, never charge = 0."""
        asm = _make_valid_assembly()
        result = validate_financial_document_assembly(asm)

        assert result.status == ValidationStatus.VALID
        assert asm.charges == ()


# ══════════════════════════════════════════════════════════════════════════
# L. Repeated Identical Lines
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryLRepeatedLines:
    def test_l1_legitimate_repeated_lines_remain_valid(self) -> None:
        """Identical line items with distinct row evidence remain valid and are not deduplicated."""
        ln1 = _make_valid_line(line_num=1, desc="Widget A", qty="2", price="50.00", amount="100.00", row_ev_ids=("ROW_1",))
        ln2 = _make_valid_line(line_num=2, desc="Widget A", qty="2", price="50.00", amount="100.00", row_ev_ids=("ROW_2",))
        asm = _make_valid_assembly(lines=(ln1, ln2))

        result = validate_financial_document_assembly(asm)
        assert result.status == ValidationStatus.VALID
        assert len(asm.lines) == 2


# ══════════════════════════════════════════════════════════════════════════
# M. Provenance Validation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryMProvenance:
    def test_m1_missing_evidence_on_observed_line_produces_error(self) -> None:
        """Observed LineFact with empty evidence IDs produces PROVENANCE_MISSING_LINE_EVIDENCE ERROR."""
        # Create line without evidence contract validation error by bypassing post_init or using derived/empty values
        # Since LineFact.__post_init__ enforces evidence when values exist with origin=OBSERVED,
        # we can test with a line where values are empty, or test Tax/Charge/Totals evidence
        # Or construct via object.__new__ to test validator's detection of missing evidence
        ln = object.__new__(LineFact)
        object.__setattr__(ln, "line_number", 1)
        object.__setattr__(ln, "description", "Item")
        object.__setattr__(ln, "quantity", Decimal("1"))
        object.__setattr__(ln, "unit_price", Decimal("10"))
        object.__setattr__(ln, "amount", Decimal("10"))
        object.__setattr__(ln, "taxes", ())
        object.__setattr__(ln, "evidence_ids", ())
        object.__setattr__(ln, "field_evidence_ids", {})
        object.__setattr__(ln, "source_row_evidence_ids", ())
        object.__setattr__(ln, "semantic_role", SemanticRole.BILLED_LINE)
        object.__setattr__(ln, "origin", FactOrigin.OBSERVED)

        asm = _make_valid_assembly(lines=(ln,))
        result = validate_financial_document_assembly(asm)

        assert result.status == ValidationStatus.INVALID
        prov_issues = [i for i in result.issues if i.code == "PROVENANCE_MISSING_LINE_EVIDENCE"]
        assert len(prov_issues) == 1
        assert prov_issues[0].severity == ValidationSeverity.ERROR


# ══════════════════════════════════════════════════════════════════════════
# N. Conflict Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryNConflictPreservation:
    def test_n1_conflicts_survive_and_are_reported(self) -> None:
        """Pre-existing conflicts in assembly.conflicts are preserved and reported."""
        conflict = {
            "field": "gross_total",
            "conflict_type": "disagreement",
            "observations": [
                {"value": "1000.00", "sources": ["ocr"], "evidence_ids": ["EV_1"]},
                {"value": "1100.00", "sources": ["vision"], "evidence_ids": ["EV_2"]},
            ],
        }
        asm = _make_valid_assembly(conflicts=(conflict,))
        result = validate_financial_document_assembly(asm)

        assert result.status == ValidationStatus.INVALID
        assert len(asm.conflicts) == 1
        tot_issues = [i for i in result.issues if i.code == "FINANCIALS_CONTRADICTORY_TOTAL"]
        assert len(tot_issues) == 1
        assert tot_issues[0].metadata == conflict


# ══════════════════════════════════════════════════════════════════════════
# O. Zero Arithmetic
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryOZeroArithmetic:
    def test_o1_missing_amount_is_never_computed_from_qty_and_price(self) -> None:
        """Missing line amount is NOT computed from quantity * unit_price."""
        ln = LineFact(
            line_number=1,
            description="Item",
            quantity=Decimal("5"),
            unit_price=Decimal("20.00"),
            amount=None,  # Not observed
            evidence_ids=("EV_1",),
            field_evidence_ids={"quantity": ("EV_1",), "unit_price": ("EV_1",)},
            origin=FactOrigin.OBSERVED,
        )
        asm = _make_valid_assembly(lines=(ln,))
        result = validate_financial_document_assembly(asm)

        # Still None! No 100.00 computed
        assert asm.lines[0].amount is None


# ══════════════════════════════════════════════════════════════════════════
# P. Zero Master Matching
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryPZeroMasterMatching:
    def test_p1_no_master_ids_injected_during_validation(self) -> None:
        """Validator does not inject master supplier, buyer, or PO IDs."""
        asm = _make_valid_assembly()
        assert asm.supplier.matched_result is None
        assert asm.buyer.company_code is None
        assert asm.po.matched_po_result is None

        validate_financial_document_assembly(asm)

        assert asm.supplier.matched_result is None
        assert asm.buyer.company_code is None
        assert asm.po.matched_po_result is None


# ══════════════════════════════════════════════════════════════════════════
# Q. Non-Destructive Mutation Audit
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryQNoMutation:
    def test_q1_assembly_is_deep_equal_before_and_after_validation(self) -> None:
        """Validating an assembly causes zero mutation to the input assembly."""
        asm = _make_valid_assembly()
        before_dict = copy.deepcopy(asm.to_dict())

        validate_financial_document_assembly(asm)

        after_dict = asm.to_dict()
        assert before_dict == after_dict


# ══════════════════════════════════════════════════════════════════════════
# R. Determinism
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryRDeterminism:
    def test_r1_repeated_validation_produces_identical_results(self) -> None:
        """Validating the same assembly multiple times yields identical results and issue order."""
        asm = _make_valid_assembly()
        res1 = validate_financial_document_assembly(asm)
        res2 = validate_financial_document_assembly(asm)

        assert res1.status == res2.status
        assert res1.validated_fact_count == res2.validated_fact_count
        assert res1.validated_evidence_count == res2.validated_evidence_count
        assert [i.code for i in res1.issues] == [i.code for i in res2.issues]
        assert res1.to_json() == res2.to_json()


# ══════════════════════════════════════════════════════════════════════════
# S. Serialization & Round-Tripping
# ══════════════════════════════════════════════════════════════════════════

class TestCategorySSerialization:
    def test_s1_full_json_round_trip(self) -> None:
        """ExtractionValidationResult serializes to JSON and reconstructs identically."""
        asm = _make_valid_assembly()
        result = validate_financial_document_assembly(asm)

        json_str = result.to_json()
        round_tripped = ExtractionValidationResult.from_json(json_str)

        assert round_tripped.assembly_id == result.assembly_id
        assert round_tripped.status == result.status
        assert round_tripped.validated_fact_count == result.validated_fact_count
        assert round_tripped.validated_evidence_count == result.validated_evidence_count
        assert round_tripped.to_json() == json_str


# ══════════════════════════════════════════════════════════════════════════
# T. Payable Relevance
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryTPayableRelevance:
    def test_t1_supporting_document_not_promoted_to_payable(self) -> None:
        """Supporting document assembly validates as supporting document and is not promoted."""
        asm = _make_valid_assembly(relevance=PayableRelevance.SUPPORTING)
        result = validate_financial_document_assembly(asm)

        assert asm.payable_relevance == PayableRelevance.SUPPORTING
        assert any(i.code == "RELEVANCE_SUPPORTING_DOCUMENT" for i in result.issues)

    def test_t2_non_payable_document_not_promoted(self) -> None:
        """Non-payable assembly validates in role."""
        asm = _make_valid_assembly(relevance=PayableRelevance.NON_PAYABLE)
        result = validate_financial_document_assembly(asm)

        assert asm.payable_relevance == PayableRelevance.NON_PAYABLE
        assert any(i.code == "RELEVANCE_NON_PAYABLE" for i in result.issues)


# ══════════════════════════════════════════════════════════════════════════
# U. Invoice Type Validation
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryUInvoiceType:
    def test_u1_conflicting_invoice_types_produce_error(self) -> None:
        """Conflicting invoice types in conflicts produce IDENTITY_CONTRADICTORY_INVOICE_TYPE ERROR."""
        conflict = {
            "field": "invoice_type",
            "conflict_type": "disagreement",
            "observations": [
                {"value": "invoice", "sources": ["ocr"]},
                {"value": "credit_memo", "sources": ["vision"]},
            ],
        }
        asm = _make_valid_assembly(conflicts=(conflict,))
        result = validate_financial_document_assembly(asm)

        assert result.status == ValidationStatus.INVALID
        type_issues = [i for i in result.issues if i.code == "IDENTITY_CONTRADICTORY_INVOICE_TYPE"]
        assert len(type_issues) == 1
        assert type_issues[0].severity == ValidationSeverity.ERROR


# ══════════════════════════════════════════════════════════════════════════
# V. Evidence Completeness
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryVEvidenceCompleteness:
    def test_v1_fact_and_evidence_counts(self) -> None:
        """Validated fact count and evidence count accurately reflect the assembly."""
        asm = _make_valid_assembly()
        result = validate_financial_document_assembly(asm)

        assert result.validated_fact_count >= 5  # Identity, parties, line, total, etc.
        assert result.validated_evidence_count >= 1


# ══════════════════════════════════════════════════════════════════════════
# W. Multiple Logical Payables
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryWMultipleLogicalPayables:
    def test_w1_batch_validation_preserves_independence(self) -> None:
        """Validating multiple assemblies batch-wise processes each independently."""
        asm_valid = _make_valid_assembly(assembly_id="ASM-VALID")
        asm_conflict = _make_valid_assembly(
            assembly_id="ASM-CONFLICT",
            conflicts=({
                "field": "invoice_number",
                "conflict_type": "disagreement",
            },),
        )

        results = validate_financial_document_assemblies([asm_valid, asm_conflict])
        assert len(results) == 2
        assert results[0].assembly_id == "ASM-VALID"
        assert results[0].status == ValidationStatus.VALID
        assert results[1].assembly_id == "ASM-CONFLICT"
        assert results[1].status == ValidationStatus.INVALID


# ══════════════════════════════════════════════════════════════════════════
# X. Real Corpus Verification (INV-01, HLD-01, INV-02, DU-02, HLD-03)
# ══════════════════════════════════════════════════════════════════════════

class TestCategoryXRealCorpusVerification:
    def test_x1_inv_01_real_corpus_validation(self) -> None:
        """INV-01 (German invoice): Assembles and validates cleanly with traceable provenance."""
        p = Path("artifacts/ocr/INV-01/page_001.json")
        if not p.exists():
            pytest.skip("INV-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        assert len(assemblies) == 1

        result = validate_financial_document_assembly(assemblies[0])
        assert result.is_usable is True
        assert not result.has_blocking
        assert result.validated_fact_count > 0
        assert result.validated_evidence_count > 0

    def test_x2_hld_01_real_corpus_validation(self) -> None:
        """HLD-01 (Thai invoice): Validates observed structure without Thai-specific normalization."""
        p = Path("artifacts/ocr/HLD-01/page_001.json")
        if not p.exists():
            pytest.skip("HLD-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        assert len(assemblies) == 1

        result = validate_financial_document_assembly(assemblies[0])
        assert result.is_usable is True
        assert not result.has_blocking
        # Financial facts remain observed
        assert assemblies[0].charges != () or assemblies[0].taxes != () or assemblies[0].lines != ()

    def test_x3_inv_02_real_corpus_validation(self) -> None:
        """INV-02 (Estonian invoice): Assembles and validates without hardcoded page count assumption."""
        p_dir = Path("artifacts/ocr/INV-02")
        if not p_dir.exists():
            pytest.skip("INV-02 directory missing")
        page_files = sorted(p_dir.glob("page_*.json"))
        assert len(page_files) >= 1
        page_evs = [ocr_json_to_page_evidence(pf) for pf in page_files]
        cands = extract_candidates_from_document(page_evs)
        facts_list = consolidate_document_groups(cands)
        assemblies = assemble_financial_documents(facts_list)
        assert len(assemblies) >= 1

        result = validate_financial_document_assembly(assemblies[0])
        assert result.is_usable is True
        assert not result.has_blocking

    def test_x4_du_02_real_corpus_validation(self) -> None:
        """DU-02 (20-page customs dossier): EUR primary invoice and TRY supporting documents validated with isolation."""
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

        assert len(assemblies) >= 1
        primary_asm = next((a for a in assemblies if a.currency == "EUR"), assemblies[0])
        result = validate_financial_document_assembly(primary_asm)

        # Primary invoice EUR is not in conflict with supporting TRY pages
        assert not any(i.code == "CURRENCY_CONFLICT" for i in result.issues)
        assert not result.has_blocking
        # Real DU-02 extraction has a preserved invoice number disagreement (554701215 vs TES) -> INVALID
        assert result.status == ValidationStatus.INVALID
        assert any(i.code == "IDENTITY_CONTRADICTORY_INVOICE_NUMBER" for i in result.issues)

    def test_x5_hld_03_real_corpus_validation(self) -> None:
        """HLD-03 (Portuguese invoice): Assembles and validates cleanly."""
        p = Path("artifacts/ocr/HLD-03/page_001.json")
        if not p.exists():
            pytest.skip("HLD-03 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])
        assert len(assemblies) == 1

        result = validate_financial_document_assembly(assemblies[0])
        assert result.is_usable is True
        assert not result.has_blocking
