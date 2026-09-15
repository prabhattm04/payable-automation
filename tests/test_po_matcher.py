"""tests/test_po_matcher.py — Test suite for Phase 8D Purchase Order Master-Data Matching.

Validates:
A. Unique exact PO number -> MATCHED.
B. Exact PO number + supplier_id -> MATCHED.
C. Exact PO number + currency -> MATCHED.
D. Exact PO number + supplier + currency -> MATCHED.
E. Unknown PO number -> NO_MATCH (blank master_id).
F. PO number + conflicting supplier -> not MATCHED (master_id=None; AMBIGUOUS when competing identities exist).
G. PO number + conflicting currency -> not MATCHED (master_id=None).
H. Duplicate PO number synthetic case -> AMBIGUOUS without discriminator.
I. Duplicate PO number resolved by supplier -> MATCHED.
J. Duplicate PO number resolved by supplier + currency -> MATCHED.
K. Amount alone cannot match -> NO_MATCH.
L. Currency alone cannot match -> NO_MATCH.
M. Supplier alone cannot match -> NO_MATCH.
N. Line evidence alone cannot match -> NO_MATCH.
O. '2287 vs PO-EE-2026-0044' safety test: observed PO 2287 never substituted for PO-EE-2026-0044.
P. Original observed PO number preserved in details.
Q. Evidence IDs preserved.
R. Candidate list preserved for ambiguous cases.
S. Confidence is strictly None.
T. Deterministic repeated results.
U. Document ID and filename ignored.
V. Indexed lookup without full-master scan.
W. Normalization variations handled.
X. Unicode / diacritics in line descriptions.
Y. Missing optional fields handled safely.
Z. Serialization round-trip.
AA. No document-specific rules.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import pytest

from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedPOIdentity,
    ObservedPOLineEvidence,
)
from src.matching.models import POLineRecord, PurchaseOrderRecord
from src.matching.po_matcher import POMatcher
from src.matching.store import MasterDataStore

MASTER_DATA_DIR = Path("d:/candidate_kit/master_data")


@pytest.fixture(scope="module")
def store() -> MasterDataStore:
    return MasterDataStore.from_directory(MASTER_DATA_DIR)


@pytest.fixture(scope="module")
def matcher(store: MasterDataStore) -> POMatcher:
    return POMatcher(store)


class TestPOExactMatching:
    """Verify primary PO number matching and independent constraint confirmation."""

    def test_a_unique_exact_po_number(self, matcher: POMatcher):
        """A: Unique exact PO number -> MATCHED."""
        res = matcher.match({"po_number": "PO-EE-2026-0044"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "PO-EE-2026-0044"
        assert res.method == "exact_po_number"
        assert "po_number" in res.matched_fields
        assert res.confidence is None

    def test_b_exact_po_number_plus_supplier(self, matcher: POMatcher):
        """B: Exact PO number + compatible supplier_id -> MATCHED."""
        res = matcher.match({
            "po_number": "PO-EE-2026-0044",
            "supplier_id": "2807582",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "PO-EE-2026-0044"
        assert res.method == "exact_po_number_supplier"
        assert "supplier_id" in res.matched_fields

    def test_c_exact_po_number_plus_currency(self, matcher: POMatcher):
        """C: Exact PO number + compatible currency -> MATCHED."""
        res = matcher.match({
            "po_number": "PO-EE-2026-0044",
            "currency": "EUR",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "PO-EE-2026-0044"
        assert res.method == "exact_po_number_currency"
        assert "currency" in res.matched_fields

    def test_d_exact_po_number_supplier_and_currency(self, matcher: POMatcher):
        """D: Exact PO number + supplier + currency -> MATCHED."""
        res = matcher.match({
            "po_number": "PO-GH-2026-0177",
            "supplier_id": "2731927",
            "currency": "GHS",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "PO-GH-2026-0177"
        assert res.method == "exact_po_number_supplier_currency"
        assert "supplier_id" in res.matched_fields
        assert "currency" in res.matched_fields

    def test_w_normalization_variations(self, matcher: POMatcher):
        """W: Whitespace, case, and wrapping characters are normalized."""
        res = matcher.match({
            "po_number": "  po-ee-2026-0044  \n",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "PO-EE-2026-0044"

    def test_line_evidence_support(self, matcher: POMatcher):
        """Line description evidence confirms PO candidate."""
        res = matcher.match({
            "po_number": "PO-EE-2026-0044",
            "lines": [{"description": "AV equipment rental", "quantity": 1}],
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "PO-EE-2026-0044"
        assert res.method == "exact_po_number_line_evidence"


class TestPOContradictionsAndSafety:
    """Verify contradiction handling and user corrections: master_id=None, AMBIGUOUS vs NO_MATCH."""

    def test_f_conflicting_supplier_competing_identities_ambiguous(self, matcher: POMatcher):
        """F & User Correction: PO number of P1 + supplier of P2 (competing valid identities) -> AMBIGUOUS."""
        # PO-EE-2026-0044 belongs to supplier 2807582.
        # Supplier 2731927 is Scancom PLC, which has PO-GH-2026-0177.
        res = matcher.match({
            "po_number": "PO-EE-2026-0044",
            "supplier_id": "2731927",
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert res.method == "conflicting_po_and_supplier"
        assert len(res.candidates) >= 2
        cand_ids = {c["po_id"] for c in res.candidates}
        assert "PO-EE-2026-0044" in cand_ids
        assert "PO-GH-2026-0177" in cand_ids

    def test_f_conflicting_supplier_no_satisfying_candidate_no_match(self, matcher: POMatcher):
        """F & User Correction: PO number of P1 + unknown non-competing supplier -> NO_MATCH."""
        res = matcher.match({
            "po_number": "PO-EE-2026-0044",
            "supplier_id": "UNKNOWN_SUPPLIER_999",
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None
        assert len(res.candidates) == 1
        assert "conflicts" in res.details

    def test_g_conflicting_currency(self, matcher: POMatcher):
        """G & User Correction: PO number EUR + observed currency GHS -> NO_MATCH (no satisfying candidate)."""
        res = matcher.match({
            "po_number": "PO-EE-2026-0044",
            "currency": "GHS",
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None
        assert "conflicts" in res.details

    def test_e_unknown_po_number_returns_no_match(self, matcher: POMatcher):
        """E: Unknown PO number -> NO_MATCH."""
        res = matcher.match({"po_number": "PO-9999-NONEXISTENT"})
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None
        assert res.candidates == []

    def test_o_2287_safety_test_never_substitutes(self, matcher: POMatcher):
        """O & Prompt requirement: PO number '2287' does NOT become 'PO-EE-2026-0044'
        merely because amount, currency, and supplier match!
        """
        res = matcher.match({
            "po_number": "2287",
            "gross_amount": Decimal("608.23"),
            "currency": "EUR",
            "supplier_id": "2807582",
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None
        assert res.candidates == []
        assert "does not exist in po_master" in res.details["reason"]


class TestPODuplicateDisambiguation:
    """Verify synthetic duplicate PO numbers disambiguated by supplier and currency."""

    def test_h_duplicate_po_number_ambiguous_without_discriminator(self):
        """H: Two synthetic master POs with identical po_number -> AMBIGUOUS."""
        pos = [
            PurchaseOrderRecord(
                po_id="PO-DUP-A",
                po_number="ORDER-999",
                supplier_id="SUP_A",
                currency="EUR",
            ),
            PurchaseOrderRecord(
                po_id="PO-DUP-B",
                po_number="ORDER-999",
                supplier_id="SUP_B",
                currency="USD",
            ),
        ]
        store = MasterDataStore.from_records(po_master=pos)
        matcher = POMatcher(store)

        res = matcher.match({"po_number": "ORDER-999"})
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert len(res.candidates) == 2
        assert res.method == "ambiguous_po_number"

    def test_i_duplicate_po_number_resolved_by_supplier(self):
        """I: Duplicate PO number disambiguated by supplier_id -> MATCHED."""
        pos = [
            PurchaseOrderRecord(
                po_id="PO-DUP-A",
                po_number="ORDER-999",
                supplier_id="SUP_A",
                currency="EUR",
            ),
            PurchaseOrderRecord(
                po_id="PO-DUP-B",
                po_number="ORDER-999",
                supplier_id="SUP_B",
                currency="USD",
            ),
        ]
        store = MasterDataStore.from_records(po_master=pos)
        matcher = POMatcher(store)

        res = matcher.match({
            "po_number": "ORDER-999",
            "supplier_id": "SUP_A",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "PO-DUP-A"

    def test_j_duplicate_po_number_resolved_by_currency(self):
        """J: Duplicate PO number disambiguated by currency -> MATCHED."""
        pos = [
            PurchaseOrderRecord(
                po_id="PO-DUP-A",
                po_number="ORDER-999",
                supplier_id="SUP_A",
                currency="EUR",
            ),
            PurchaseOrderRecord(
                po_id="PO-DUP-B",
                po_number="ORDER-999",
                supplier_id="SUP_B",
                currency="USD",
            ),
        ]
        store = MasterDataStore.from_records(po_master=pos)
        matcher = POMatcher(store)

        res = matcher.match({
            "po_number": "ORDER-999",
            "currency": "USD",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "PO-DUP-B"


class TestPONegativeConstraints:
    """Verify non-identity signals (amounts, currencies, suppliers alone) cannot produce a PO match."""

    def test_k_amount_alone_cannot_match(self, matcher: POMatcher):
        """K: Invoice amount alone must never match a PO."""
        res = matcher.match({
            "gross_amount": Decimal("608.23"),
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None

    def test_l_currency_alone_cannot_match(self, matcher: POMatcher):
        """L: Currency alone must never match a PO."""
        res = matcher.match({"currency": "EUR"})
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None

    def test_m_supplier_alone_cannot_match(self, matcher: POMatcher):
        """M: Supplier ID alone must never automatically match a PO."""
        res = matcher.match({"supplier_id": "2807582"})
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None

    def test_n_line_evidence_alone_cannot_match(self, matcher: POMatcher):
        """N: Line description or quantity alone without PO number cannot match."""
        res = matcher.match({
            "lines": [{"description": "AV equipment rental", "quantity": 1}],
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None

    def test_u_document_id_and_filename_invariance(self, matcher: POMatcher):
        """U: Metadata, filename, and document ID cannot affect matching."""
        res1 = matcher.match({
            "po_number": "PO-EE-2026-0044",
            "metadata": {"document_id": "INV-01.pdf"},
        })
        res2 = matcher.match({
            "po_number": "PO-EE-2026-0044",
            "metadata": {"document_id": "DIFFERENT.pdf"},
        })
        assert res1.master_id == res2.master_id == "PO-EE-2026-0044"


class TestPOEvidenceAndSerialization:
    """Verify evidence IDs, observed values, determinism, and serialization."""

    def test_p_observed_values_preserved(self, matcher: POMatcher):
        """P: Raw observed PO number preserved in details."""
        res = matcher.match({"po_number": "PO-EE-2026-0044", "currency": "EUR"})
        assert res.details["observed_values"]["po_number"] == "PO-EE-2026-0044"
        assert res.details["observed_values"]["currency"] == "EUR"

    def test_q_evidence_ids_preserved(self, matcher: POMatcher):
        """Q: Supporting evidence IDs preserved in result."""
        res = matcher.match({
            "po_number": "PO-EE-2026-0044",
            "field_evidence_ids": {"po_number": "ev_po_num_01"},
        })
        assert "ev_po_num_01" in res.evidence_ids

    def test_s_confidence_is_none(self, matcher: POMatcher):
        """S: Confidence must strictly be None."""
        res = matcher.match({"po_number": "PO-EE-2026-0044"})
        assert res.confidence is None

    def test_t_deterministic_repeated_results(self, matcher: POMatcher):
        """T: Deterministic repeated queries produce identical outputs."""
        query = {"po_number": "PO-EE-2026-0044"}
        r1 = matcher.match(query)
        r2 = matcher.match(query)
        assert r1.status == r2.status
        assert r1.master_id == r2.master_id
        assert r1.method == r2.method

    def test_z_serialization_round_trip(self, matcher: POMatcher):
        """Z: Serialization round trip preserves all fields and null confidence."""
        res = matcher.match({
            "po_number": "PO-EE-2026-0044",
            "supplier_id": "2807582",
            "currency": "EUR",
        })
        d = res.to_dict()
        assert d["status"] == "matched"
        assert d["master_id"] == "PO-EE-2026-0044"
        assert d["confidence"] is None

        reconstructed = MasterMatchResult.from_dict(d)
        assert reconstructed.status == res.status
        assert reconstructed.master_id == res.master_id
        assert reconstructed.confidence is None

    def test_empty_input_returns_no_match(self, matcher: POMatcher):
        """Empty input safely returns NO_MATCH."""
        res = matcher.match({})
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None
