"""tests/test_financial_assembly.py — Comprehensive Test Suite for Phase 9B-3 Financial Assembly.

Test Categories:
A. Single-page payable (1 logical payable -> 1 assembly)
B. Two-page continuation (page 1 + continuation page 2 -> 1 assembly with lines 1-10 + 11-20)
C. Three+ page continuation (page 1 header/lines, page 2 lines, page 3 totals -> 1 assembly)
D. Supporting group isolation (supporting group retained separately in supporting_group_ids)
E. DU-02-style case (EUR primary payable + TRY supporting pages -> EUR payable with TRY isolated)
F. Multiple logical payables (Invoice A, Invoice B produce separate assemblies)
G. Supporting material attached to correct payable (supporting groups attach to correct invoice)
H. Duplicate evidence (multiple evidence IDs for same fact do not duplicate financial facts)
I. Repeated identical invoice lines (genuine repeated rows remain separate)
J. Provenance preservation (evidence, page, group, source-row provenance survive)
K. Conflict preservation (unresolved conflicts remain unresolved)
L. Missing values (missing fields remain None)
M. No arithmetic (no missing amount/tax/total derived)
N. No master matching (no master-data IDs injected)
O. Determinism (permutation invariance)
P. No cross-group aggregation (unrelated groups remain separate)
Q. Semantic-role preservation (BILLED_LINE, COMPONENT_DETAIL, etc. correctly typed)
R. Tax-placement preservation (header tax remains header, line tax remains line)
S. Serialization & Round-tripping (to_dict, from_dict, to_json, from_json)
T. Real Corpus Integration Tests (INV-01, HLD-01, INV-02, DU-02, HLD-03)
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
import pytest

from src.extraction.financial_assembly import (
    FinancialDocumentAssembly,
    assemble_financial_documents,
)
from src.extraction.candidates import (
    DocumentIdentityCandidate,
    ExtractionCandidates,
    LineCandidate,
    TotalCandidate,
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
from src.understanding.document_grouper import (
    DocumentGroup,
    GroupingResult,
    group_document,
)
from src.understanding.evidence import (
    Evidence,
    EvidenceSource,
    PageEvidence,
    ocr_json_to_page_evidence,
)
from src.understanding.page_classifier import PageRole, PayableRelevance, classify_page


# ══════════════════════════════════════════════════════════════════════════
# Synthetic Test Fact Builders
# ══════════════════════════════════════════════════════════════════════════

def _make_line(
    line_num: int,
    desc: str,
    qty: Optional[str] = None,
    price: Optional[str] = None,
    amount: Optional[str] = None,
    ev_ids: tuple[str, ...] = ("EV_LN",),
    row_ev_ids: tuple[str, ...] = ("ROW_1",),
    role: SemanticRole = SemanticRole.BILLED_LINE,
    taxes: tuple[TaxFact, ...] = (),
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
        semantic_role=role,
        taxes=taxes,
        evidence_ids=ev_ids,
        field_evidence_ids=field_evs,
        source_row_evidence_ids=row_ev_ids,
    )


def _make_document_facts(
    doc_id: str = "DOC-TEST",
    group_id: str = "DOC-TEST:group:p1",
    page_numbers: tuple[int, ...] = (1,),
    role: PageRole = PageRole.INVOICE,
    relevance: PayableRelevance = PayableRelevance.PAYABLE_CANDIDATE,
    inv_number: Optional[str] = "INV-1001",
    currency: Optional[str] = "EUR",
    lines: tuple[LineFact, ...] = (),
    taxes: tuple[TaxFact, ...] = (),
    discounts: tuple[DiscountFact, ...] = (),
    charges: tuple[ChargeFact, ...] = (),
    gross_total: Optional[str] = None,
    supporting_group_ids: tuple[str, ...] = (),
    conflicts: tuple[dict, ...] = (),
    evidence_ids: tuple[str, ...] = ("EV_ROOT",),
) -> DocumentFacts:
    ident_evs = {"invoice_number": ("EV_NUM",), "currency": ("EV_CURR",)} if inv_number and currency else {}
    ident = DocumentIdentityFacts(
        invoice_number=inv_number,
        currency=currency,
        evidence_ids=("EV_IDENT",) if (inv_number or currency) else (),
        field_evidence_ids=ident_evs,
    )
    totals = None
    if gross_total is not None:
        totals = PrintedTotalsFact(
            gross_total=Decimal(gross_total),
            evidence_ids=("EV_TOT",),
            field_evidence_ids={"gross_total": ("EV_TOT",)},
        )
    fin_evs = list(evidence_ids)
    if totals:
        fin_evs.extend(totals.evidence_ids)
    for ln in lines:
        fin_evs.extend(ln.evidence_ids)

    fin = FinancialFacts(
        lines=lines,
        discounts=discounts,
        charges=charges,
        taxes=taxes,
        printed_totals=totals,
        currency=currency,
        evidence_ids=tuple(dict.fromkeys(fin_evs)),
    )
    return DocumentFacts(
        document_id=doc_id,
        group_id=group_id,
        page_numbers=page_numbers,
        document_role=role,
        payable_relevance=relevance,
        supporting_group_ids=supporting_group_ids,
        identity=ident,
        financials=fin,
        conflicting_facts=conflicts,
        evidence_ids=tuple(dict.fromkeys(evidence_ids + fin.evidence_ids)),
    )


# ══════════════════════════════════════════════════════════════════════════
# A. Single-Page Payable
# ══════════════════════════════════════════════════════════════════════════

class TestSinglePagePayable:
    def test_a1_single_page_payable_produces_one_assembly(self) -> None:
        """One logical payable on one page produces exactly one FinancialDocumentAssembly."""
        ln = _make_line(1, "Cloud Hosting", "1", "100.00", "100.00")
        f = _make_document_facts(
            doc_id="INV-001",
            group_id="INV-001:group:p1",
            page_numbers=(1,),
            inv_number="INV-2024-001",
            currency="EUR",
            lines=(ln,),
            gross_total="100.00",
        )
        assemblies = assemble_financial_documents([f])

        assert len(assemblies) == 1
        asm = assemblies[0]
        assert asm.document_id == "INV-001"
        assert asm.assembly_id == "INV-001:assembly:p1"
        assert asm.primary_group_ids == ("INV-001:group:p1",)
        assert asm.page_numbers == (1,)
        assert asm.invoice_number == "INV-2024-001"
        assert asm.currency == "EUR"
        assert len(asm.lines) == 1
        assert asm.printed_totals.gross_total == Decimal("100.00")


# ══════════════════════════════════════════════════════════════════════════
# B. Two-Page Continuation
# ══════════════════════════════════════════════════════════════════════════

class TestTwoPageContinuation:
    def test_b1_page1_and_page2_continuation_form_one_assembly(self) -> None:
        """Page 1 lines 1-10 + continuation page 2 lines 11-20 form ONE logical assembly."""
        p1_lines = tuple(_make_line(i, f"Widget {i}", "1", "10.00", "10.00", ev_ids=(f"E_P1_{i}",), row_ev_ids=(f"R_P1_{i}",)) for i in range(1, 11))
        p2_lines = tuple(_make_line(i, f"Widget {i}", "1", "10.00", "10.00", ev_ids=(f"E_P2_{i}",), row_ev_ids=(f"R_P2_{i}",)) for i in range(11, 21))

        f_p1 = _make_document_facts(
            doc_id="DOC-TWO-PAGE",
            group_id="DOC-TWO-PAGE:group:p1",
            page_numbers=(1,),
            role=PageRole.INVOICE,
            inv_number="INV-8888",
            currency="USD",
            lines=p1_lines,
        )
        f_p2 = _make_document_facts(
            doc_id="DOC-TWO-PAGE",
            group_id="DOC-TWO-PAGE:group:p2",
            page_numbers=(2,),
            role=PageRole.CONTINUATION,
            inv_number=None,  # Continuation does not repeat invoice number
            currency="USD",
            lines=p2_lines,
            gross_total="200.00",
        )

        assemblies = assemble_financial_documents([f_p1, f_p2])

        assert len(assemblies) == 1
        asm = assemblies[0]
        assert asm.page_numbers == (1, 2)
        assert asm.primary_group_ids == ("DOC-TWO-PAGE:group:p1", "DOC-TWO-PAGE:group:p2")
        assert asm.invoice_number == "INV-8888"  # Inherited from parent
        assert len(asm.lines) == 20
        assert asm.lines[0].description == "Widget 1"
        assert asm.lines[19].description == "Widget 20"
        assert asm.printed_totals.gross_total == Decimal("200.00")


# ══════════════════════════════════════════════════════════════════════════
# C. Three+ Page Continuation
# ══════════════════════════════════════════════════════════════════════════

class TestThreePageContinuation:
    def test_c1_three_page_continuation_assembles_cleanly(self) -> None:
        """Page 1 header/lines, Page 2 lines, Page 3 totals form ONE assembly."""
        ln_p1 = (_make_line(1, "Item A", "2", "50.00", "100.00", ev_ids=("E1",), row_ev_ids=("R1",)),)
        ln_p2 = (_make_line(2, "Item B", "1", "150.00", "150.00", ev_ids=("E2",), row_ev_ids=("R2",)),)
        
        f1 = _make_document_facts(doc_id="DOC-3P", group_id="g1", page_numbers=(1,), role=PageRole.INVOICE, inv_number="INV-3P", lines=ln_p1)
        f2 = _make_document_facts(doc_id="DOC-3P", group_id="g2", page_numbers=(2,), role=PageRole.CONTINUATION, inv_number=None, lines=ln_p2)
        f3 = _make_document_facts(doc_id="DOC-3P", group_id="g3", page_numbers=(3,), role=PageRole.CONTINUATION, inv_number=None, lines=(), gross_total="250.00")

        assemblies = assemble_financial_documents([f1, f2, f3])

        assert len(assemblies) == 1
        asm = assemblies[0]
        assert asm.page_numbers == (1, 2, 3)
        assert asm.primary_group_ids == ("g1", "g2", "g3")
        assert len(asm.lines) == 2
        assert asm.printed_totals.gross_total == Decimal("250.00")


# ══════════════════════════════════════════════════════════════════════════
# D. Supporting Group Isolation
# ══════════════════════════════════════════════════════════════════════════

class TestSupportingGroupIsolation:
    def test_d1_supporting_group_retained_separately(self) -> None:
        """Supporting groups are isolated in supporting_group_ids and not merged into financial facts."""
        f_inv = _make_document_facts(
            doc_id="DOC-SUPP",
            group_id="DOC-SUPP:group:p1",
            page_numbers=(1,),
            role=PageRole.INVOICE,
            inv_number="INV-MAIN",
            currency="EUR",
            lines=(_make_line(1, "Main Service", "1", "1000.00", "1000.00", ev_ids=("E_MAIN",)),),
            gross_total="1000.00",
        )
        f_pack = _make_document_facts(
            doc_id="DOC-SUPP",
            group_id="DOC-SUPP:group:p2",
            page_numbers=(2,),
            role=PageRole.SUPPORTING_DOCUMENT,
            relevance=PayableRelevance.SUPPORTING,
            inv_number=None,
            currency=None,
            lines=(_make_line(1, "Packing Box", "5", "0.00", "0.00", ev_ids=("E_PACK",)),),
            gross_total=None,
        )

        assemblies = assemble_financial_documents([f_inv, f_pack])

        assert len(assemblies) == 1
        asm = assemblies[0]
        assert asm.primary_group_ids == ("DOC-SUPP:group:p1",)
        assert asm.supporting_group_ids == ("DOC-SUPP:group:p2",)
        assert asm.page_numbers == (1,)
        # Packing line is NOT merged into primary lines
        assert len(asm.lines) == 1
        assert asm.lines[0].description == "Main Service"
        assert asm.printed_totals.gross_total == Decimal("1000.00")


# ══════════════════════════════════════════════════════════════════════════
# E. DU-02-Style Multi-Currency Dossier
# ══════════════════════════════════════════════════════════════════════════

class TestDU02DossierIsolation:
    def test_e1_eur_payable_with_try_supporting_material_no_cross_aggregation(self) -> None:
        """DU-02 scenario: EUR primary payable + TRY supporting sheets does NOT aggregate TRY amounts."""
        # Primary Customs Consolidated Invoice (EUR 37,534.94)
        f_prim = _make_document_facts(
            doc_id="DU-02.pdf",
            group_id="DU-02.pdf:group:p1",
            page_numbers=(1, 2),
            role=PageRole.INVOICE,
            relevance=PayableRelevance.PAYABLE_CANDIDATE,
            inv_number="554701215",
            currency="EUR",
            lines=(_make_line(1, "Consolidated Customs Entry", "1", "37534.94", "37534.94", ev_ids=("E_EUR",)),),
            gross_total="37534.94",
        )
        # Supporting Detailed Invoice in TRY
        f_supp_try = _make_document_facts(
            doc_id="DU-02.pdf",
            group_id="DU-02.pdf:group:p3",
            page_numbers=(3,),
            role=PageRole.INVOICE,
            relevance=PayableRelevance.SUPPORTING,
            inv_number="554701215",
            currency="TRY",
            lines=(_make_line(1, "Detailed Sheet 1", "10", "187.21", "1872.11", ev_ids=("E_TRY",)),),
            gross_total="1872.11",
        )
        # Supporting Packing Sheet
        f_supp_pack = _make_document_facts(
            doc_id="DU-02.pdf",
            group_id="DU-02.pdf:group:p5",
            page_numbers=(5,),
            role=PageRole.SUPPORTING_DOCUMENT,
            relevance=PayableRelevance.SUPPORTING,
            inv_number=None,
            currency=None,
        )

        assemblies = assemble_financial_documents([f_prim, f_supp_try, f_supp_pack])

        assert len(assemblies) == 1
        asm = assemblies[0]
        assert asm.primary_group_ids == ("DU-02.pdf:group:p1",)
        assert "DU-02.pdf:group:p3" in asm.supporting_group_ids
        assert "DU-02.pdf:group:p5" in asm.supporting_group_ids

        # Financial facts remain purely EUR
        assert asm.currency == "EUR"
        assert asm.printed_totals.gross_total == Decimal("37534.94")
        assert len(asm.lines) == 1
        assert asm.lines[0].amount == Decimal("37534.94")


# ══════════════════════════════════════════════════════════════════════════
# F. Multiple Logical Payables in One PDF
# ══════════════════════════════════════════════════════════════════════════

class TestMultiplePayables:
    def test_f1_two_independent_invoices_form_separate_assemblies(self) -> None:
        """Invoice A and Invoice B in the same document produce separate assemblies."""
        f_a = _make_document_facts(
            doc_id="MULTI-DOC",
            group_id="MULTI-DOC:group:p1",
            page_numbers=(1,),
            inv_number="INV-A",
            currency="EUR",
            lines=(_make_line(1, "Product A", "1", "100.00", "100.00", ev_ids=("EA",)),),
            gross_total="100.00",
        )
        f_b = _make_document_facts(
            doc_id="MULTI-DOC",
            group_id="MULTI-DOC:group:p2",
            page_numbers=(2,),
            inv_number="INV-B",
            currency="EUR",
            lines=(_make_line(1, "Product B", "1", "200.00", "200.00", ev_ids=("EB",)),),
            gross_total="200.00",
        )

        assemblies = assemble_financial_documents([f_a, f_b])

        assert len(assemblies) == 2
        assert assemblies[0].invoice_number == "INV-A"
        assert assemblies[0].printed_totals.gross_total == Decimal("100.00")
        assert assemblies[1].invoice_number == "INV-B"
        assert assemblies[1].printed_totals.gross_total == Decimal("200.00")


# ══════════════════════════════════════════════════════════════════════════
# G. Supporting Material Attached to Correct Payable
# ══════════════════════════════════════════════════════════════════════════

class TestSupportingAttachment:
    def test_g1_supporting_attaches_to_preceding_payable(self) -> None:
        """Invoice A + Supp A followed by Invoice B + Supp B attaches correctly."""
        f_a = _make_document_facts(doc_id="DOC-AB", group_id="g_a", page_numbers=(1,), inv_number="INV-A")
        f_supp_a = _make_document_facts(doc_id="DOC-AB", group_id="g_sa", page_numbers=(2,), role=PageRole.SUPPORTING_DOCUMENT, relevance=PayableRelevance.SUPPORTING, inv_number=None)
        f_b = _make_document_facts(doc_id="DOC-AB", group_id="g_b", page_numbers=(3,), inv_number="INV-B")
        f_supp_b = _make_document_facts(doc_id="DOC-AB", group_id="g_sb", page_numbers=(4,), role=PageRole.SUPPORTING_DOCUMENT, relevance=PayableRelevance.SUPPORTING, inv_number=None)

        assemblies = assemble_financial_documents([f_a, f_supp_a, f_b, f_supp_b])

        assert len(assemblies) == 2
        assert assemblies[0].invoice_number == "INV-A"
        assert assemblies[0].supporting_group_ids == ("g_sa",)
        assert assemblies[1].invoice_number == "INV-B"
        assert assemblies[1].supporting_group_ids == ("g_sb",)


# ══════════════════════════════════════════════════════════════════════════
# H. Duplicate Evidence
# ══════════════════════════════════════════════════════════════════════════

class TestDuplicateEvidence:
    def test_h1_duplicate_evidence_does_not_duplicate_financial_facts(self) -> None:
        """The exact same line observation across groups preserves the line once while merging evidence."""
        ln1 = _make_line(1, "Consulting", "1", "500.00", "500.00", ev_ids=("EV_L1",), row_ev_ids=("ROW_1",))
        ln2 = _make_line(1, "Consulting", "1", "500.00", "500.00", ev_ids=("EV_L1",), row_ev_ids=("ROW_1",))

        f1 = _make_document_facts(doc_id="DOC-DUP", group_id="g1", page_numbers=(1,), lines=(ln1,))
        f2 = _make_document_facts(doc_id="DOC-DUP", group_id="g2", page_numbers=(2,), role=PageRole.CONTINUATION, inv_number=None, lines=(ln2,))

        assemblies = assemble_financial_documents([f1, f2])

        assert len(assemblies) == 1
        assert len(assemblies[0].lines) == 1


# ══════════════════════════════════════════════════════════════════════════
# I. Repeated Identical Invoice Lines
# ══════════════════════════════════════════════════════════════════════════

class TestRepeatedLines:
    def test_i1_genuine_repeated_invoice_lines_remain_separate(self) -> None:
        """Distinct invoice line items with identical text/qty/amount remain separate lines."""
        ln1 = _make_line(1, "Standard License", "1", "100.00", "100.00", ev_ids=("EV_ROW1",), row_ev_ids=("R1",))
        ln2 = _make_line(2, "Standard License", "1", "100.00", "100.00", ev_ids=("EV_ROW2",), row_ev_ids=("R2",))

        f1 = _make_document_facts(doc_id="DOC-REP", group_id="g1", page_numbers=(1,), lines=(ln1,))
        f2 = _make_document_facts(doc_id="DOC-REP", group_id="g2", page_numbers=(2,), role=PageRole.CONTINUATION, inv_number=None, lines=(ln2,))

        assemblies = assemble_financial_documents([f1, f2])

        assert len(assemblies) == 1
        assert len(assemblies[0].lines) == 2
        assert assemblies[0].lines[0].line_number == 1
        assert assemblies[0].lines[1].line_number == 2


# ══════════════════════════════════════════════════════════════════════════
# J. Provenance Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestProvenancePreservation:
    def test_j1_evidence_ids_and_source_row_context_survive_assembly(self) -> None:
        """All evidence IDs and source row context survive assembly."""
        ln = _make_line(1, "Hardware", "1", "800.00", "800.00", ev_ids=("EV_LINE_800",), row_ev_ids=("ROW_EVID_12",))
        f = _make_document_facts(doc_id="DOC-PROV", lines=(ln,), evidence_ids=("EV_HEADER",))

        asm = assemble_financial_documents([f])[0]

        assert "EV_HEADER" in asm.evidence_ids
        assert "EV_LINE_800" in asm.facts.financials.lines[0].evidence_ids
        assert ("ROW_EVID_12",) == asm.facts.financials.lines[0].source_row_evidence_ids


# ══════════════════════════════════════════════════════════════════════════
# K. Conflict Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestConflictPreservation:
    def test_k1_conflicting_printed_totals_preserved_in_assembly(self) -> None:
        """Contradictory printed totals between continuation pages are recorded as conflicts without guessing."""
        f1 = _make_document_facts(doc_id="DOC-CONF", group_id="g1", page_numbers=(1,), role=PageRole.INVOICE, gross_total="100.00")
        f2 = _make_document_facts(doc_id="DOC-CONF", group_id="g2", page_numbers=(2,), role=PageRole.CONTINUATION, inv_number=None, gross_total="200.00")

        assemblies = assemble_financial_documents([f1, f2])

        assert len(assemblies) == 1
        asm = assemblies[0]
        assert len(asm.conflicts) >= 1
        assert any(c.get("field") == "gross_total" for c in asm.conflicts)


# ══════════════════════════════════════════════════════════════════════════
# L. Missing Values
# ══════════════════════════════════════════════════════════════════════════

class TestMissingValues:
    def test_l1_missing_fields_remain_none(self) -> None:
        """Missing optional fields (e.g. PO, due date) remain None."""
        f = _make_document_facts(doc_id="DOC-MISSING", inv_number="INV-NO-PO")
        asm = assemble_financial_documents([f])[0]

        assert asm.po.observed_po_number is None
        assert asm.facts.identity.due_date is None


# ══════════════════════════════════════════════════════════════════════════
# M. No Accounting Arithmetic
# ══════════════════════════════════════════════════════════════════════════

class TestNoArithmetic:
    def test_m1_no_missing_amount_or_total_derived(self) -> None:
        """Line with quantity and price but no amount must NOT have amount computed."""
        ln = LineFact(
            line_number=1,
            description="Item without amount",
            quantity=Decimal("2"),
            unit_price=Decimal("50.00"),
            amount=None,  # Missing
            evidence_ids=("E1",),
            field_evidence_ids={"quantity": ("E1",), "unit_price": ("E1",)},
        )
        f = _make_document_facts(doc_id="DOC-NO-ARITH", lines=(ln,), gross_total=None)
        asm = assemble_financial_documents([f])[0]

        # Must strictly remain None (do NOT compute 2 * 50 = 100)
        assert asm.lines[0].amount is None
        assert asm.printed_totals is None


# ══════════════════════════════════════════════════════════════════════════
# N. No Master Matching
# ══════════════════════════════════════════════════════════════════════════

class TestNoMasterMatching:
    def test_n1_no_master_ids_injected_into_assembly(self) -> None:
        """Assembled facts do not inject master match IDs."""
        f = _make_document_facts(doc_id="DOC-NO-MATCH")
        asm = assemble_financial_documents([f])[0]

        assert "master_id" not in asm.to_dict()
        assert asm.supplier.matched_result is None
        assert asm.buyer.matched_result is None
        assert asm.po.matched_po_result is None


# ══════════════════════════════════════════════════════════════════════════
# O. Determinism
# ══════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    def test_o1_permutation_invariance(self) -> None:
        """Input order permutation produces equivalent deterministic assemblies."""
        f1 = _make_document_facts(doc_id="DOC-DET", group_id="g1", page_numbers=(1,), inv_number="INV-DET", gross_total="100.00")
        f2 = _make_document_facts(doc_id="DOC-DET", group_id="g2", page_numbers=(2,), role=PageRole.CONTINUATION, inv_number=None)

        res1 = assemble_financial_documents([f1, f2])
        res2 = assemble_financial_documents([f2, f1])

        assert res1[0].assembly_id == res2[0].assembly_id
        assert res1[0].page_numbers == res2[0].page_numbers
        assert res1[0].to_dict() == res2[0].to_dict()


# ══════════════════════════════════════════════════════════════════════════
# P. No Cross-Group Aggregation
# ══════════════════════════════════════════════════════════════════════════

class TestNoCrossGroupAggregation:
    def test_p1_unrelated_groups_remain_separate(self) -> None:
        """Unrelated groups without continuation signals remain separate assemblies."""
        f1 = _make_document_facts(doc_id="DOC-SEP", group_id="g1", page_numbers=(1,), inv_number="INV-100", currency="USD")
        f2 = _make_document_facts(doc_id="DOC-SEP", group_id="g2", page_numbers=(2,), inv_number="INV-200", currency="GBP")

        assemblies = assemble_financial_documents([f1, f2])
        assert len(assemblies) == 2


# ══════════════════════════════════════════════════════════════════════════
# Q. Semantic-Role Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestSemanticRolePreservation:
    def test_q1_semantic_roles_strictly_preserved(self) -> None:
        """BILLED_LINE, COMPONENT_DETAIL, etc. are preserved without promotion."""
        ln_billed = _make_line(1, "Billed Header", "1", "100.00", "100.00", role=SemanticRole.BILLED_LINE)
        ln_detail = _make_line(2, "Component Detail", "1", "50.00", "50.00", role=SemanticRole.COMPONENT_DETAIL)

        f = _make_document_facts(doc_id="DOC-ROLES", lines=(ln_billed, ln_detail))
        asm = assemble_financial_documents([f])[0]

        assert asm.lines[0].semantic_role == SemanticRole.BILLED_LINE
        assert asm.lines[1].semantic_role == SemanticRole.COMPONENT_DETAIL


# ══════════════════════════════════════════════════════════════════════════
# R. Tax-Placement Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestTaxPlacementPreservation:
    def test_r1_header_tax_remains_header_line_tax_remains_line(self) -> None:
        """Header tax remains at header level; line tax remains attached to LineFact."""
        line_tax = TaxFact(tax_name="VAT", rate=Decimal("20.00"), amount=Decimal("20.00"), placement=Placement.LINE, evidence_ids=("E_TX_LN",))
        header_tax = TaxFact(tax_name="VAT Total", rate=Decimal("20.00"), amount=Decimal("20.00"), placement=Placement.HEADER, evidence_ids=("E_TX_HDR",))

        ln = _make_line(1, "Item with Tax", "1", "100.00", "100.00", taxes=(line_tax,))
        f = _make_document_facts(doc_id="DOC-TAX", lines=(ln,), taxes=(header_tax,))

        asm = assemble_financial_documents([f])[0]

        assert len(asm.taxes) == 1
        assert asm.taxes[0].placement == Placement.HEADER
        assert len(asm.lines[0].taxes) == 1
        assert asm.lines[0].taxes[0].placement == Placement.LINE


# ══════════════════════════════════════════════════════════════════════════
# S. Serialization & Round-Tripping
# ══════════════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_s1_full_round_trip_serialization(self) -> None:
        """FinancialDocumentAssembly serializes to JSON and reconstructs identically."""
        ln = _make_line(1, "Item", "1", "100.00", "100.00")
        f = _make_document_facts(doc_id="DOC-SER", lines=(ln,), gross_total="100.00")
        asm = assemble_financial_documents([f])[0]

        json_str = asm.to_json()
        round_tripped = FinancialDocumentAssembly.from_json(json_str)

        assert round_tripped.assembly_id == asm.assembly_id
        assert round_tripped.document_id == asm.document_id
        assert round_tripped.page_numbers == asm.page_numbers
        assert round_tripped.lines[0].description == "Item"
        assert round_tripped.printed_totals.gross_total == Decimal("100.00")
        assert round_tripped.to_json() == json_str


# ══════════════════════════════════════════════════════════════════════════
# T. Real Corpus Integration Tests (INV-01, HLD-01, INV-02, DU-02, HLD-03)
# ══════════════════════════════════════════════════════════════════════════

class TestRealCorpusIntegration:
    def test_t1_inv_01_real_corpus_assembly(self) -> None:
        """INV-01: Single-page invoice assembles into exactly 1 FinancialDocumentAssembly."""
        p = Path("artifacts/ocr/INV-01/page_001.json")
        if not p.exists():
            pytest.skip("INV-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])

        assert len(assemblies) == 1
        asm = assemblies[0]
        assert "INV-01" in asm.document_id
        assert asm.page_numbers == (1,)
        assert asm.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE

    def test_t2_hld_01_real_corpus_assembly(self) -> None:
        """HLD-01: German invoice assembles into exactly 1 FinancialDocumentAssembly."""
        p = Path("artifacts/ocr/HLD-01/page_001.json")
        if not p.exists():
            pytest.skip("HLD-01 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])

        assert len(assemblies) == 1
        asm = assemblies[0]
        assert "HLD-01" in asm.document_id
        assert asm.page_numbers == (1,)

    def test_t3_inv_02_real_corpus_assembly(self) -> None:
        """INV-02: 2-page invoice with terms assembles into 1 FinancialDocumentAssembly."""
        p1 = Path("artifacts/ocr/INV-02/page_001.json")
        p2 = Path("artifacts/ocr/INV-02/page_002.json")
        if not p1.exists() or not p2.exists():
            pytest.skip("INV-02 artifacts missing")
        pe1 = ocr_json_to_page_evidence(p1)
        pe2 = ocr_json_to_page_evidence(p2)
        cands = extract_candidates_from_document([pe1, pe2])
        facts_list = consolidate_document_groups(cands)
        assemblies = assemble_financial_documents(facts_list)

        assert len(assemblies) == 1
        asm = assemblies[0]
        assert "INV-02" in asm.document_id
        assert asm.page_numbers == (1, 2)
        assert asm.invoice_number == "9972-907"
        assert asm.currency == "EUR"

    def test_t4_du_02_real_corpus_assembly(self) -> None:
        """DU-02: 20-page customs dossier assembles EUR payable and isolates supporting pages."""
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

        # Primary payable assembly exists and represents EUR invoice
        assert len(assemblies) >= 1
        primary_asm = next((a for a in assemblies if a.currency == "EUR"), assemblies[0])
        assert "DU-02" in primary_asm.document_id
        assert primary_asm.currency == "EUR"
        assert 1 in primary_asm.page_numbers
        assert 2 in primary_asm.page_numbers
        # Supporting groups (customs detailed sheets & packing lists) are referenced
        assert len(primary_asm.supporting_group_ids) >= 1

    def test_t5_hld_03_real_corpus_assembly(self) -> None:
        """HLD-03: Portuguese invoice assembles cleanly into 1 FinancialDocumentAssembly."""
        p = Path("artifacts/ocr/HLD-03/page_001.json")
        if not p.exists():
            pytest.skip("HLD-03 artifact missing")
        pe = ocr_json_to_page_evidence(p)
        cands = extract_candidates_from_page(pe)
        facts = consolidate_candidates(cands)
        assemblies = assemble_financial_documents([facts])

        assert len(assemblies) == 1
        asm = assemblies[0]
        assert "HLD-03" in asm.document_id
        assert asm.page_numbers == (1,)
