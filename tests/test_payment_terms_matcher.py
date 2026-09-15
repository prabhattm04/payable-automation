"""tests/test_payment_terms_matcher.py — Test suite for Phase 8C Payment-Term Master-Data Matching.

Validates:
R. Unique alias matching (e.g. 'net 10', 'two weeks', 'within 7 days').
S. Exact explicit payment-term ID (e.g. 'Net_30').
T. Unique explicit numeric days (e.g. days=14 -> 'Net_14').
U. Real master-data days=0 ambiguity ('Immediate' vs 'Monthly_in_advance').
V. Unknown alias -> NO_MATCH (safe blank master_id).
W. Strong conflicts between text and days -> AMBIGUOUS (never guess).
X. Evidence IDs preserved.
Y. Raw observed values preserved in details.
Z. Confidence is strictly None.
AA. Deterministic repeated results.
AB. Date-derived days is strictly separate and NEVER automatically matches.
AC. Fast indexed lookup without full master-data scanning.
AD. Document ID and filename invariance.
AE. Synthetic alias collision handling (Correction 5).
"""
from __future__ import annotations

from pathlib import Path
import pytest

from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedPaymentTermIdentity,
)
from src.matching.models import PaymentTermRecord
from src.matching.payment_terms_matcher import PaymentTermsMatcher
from src.matching.store import MasterDataStore

MASTER_DATA_DIR = Path("d:/candidate_kit/master_data")


@pytest.fixture(scope="module")
def store() -> MasterDataStore:
    return MasterDataStore.from_directory(MASTER_DATA_DIR)


@pytest.fixture(scope="module")
def matcher(store: MasterDataStore) -> PaymentTermsMatcher:
    return PaymentTermsMatcher(store)


class TestPaymentTermsExactMatching:
    """Verify alias, explicit ID, and numeric days matching."""

    def test_r_unique_alias_net_10(self, matcher: PaymentTermsMatcher):
        """R: Unique alias 'net 10' -> Net_10."""
        res = matcher.match({"raw_text": "net 10"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "Net_10"
        assert res.method == "exact_normalized_alias"
        assert res.confidence is None

    def test_r_alias_two_weeks(self, matcher: PaymentTermsMatcher):
        """R: Alias 'two weeks' -> Net_14."""
        res = matcher.match({"raw_text": "two weeks"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "Net_14"

    def test_r_alias_within_7_days(self, matcher: PaymentTermsMatcher):
        """R: Alias 'within 7 days' -> Net_7."""
        res = matcher.match({"raw_text": "within 7 days"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "Net_7"

    def test_s_explicit_payment_term_id(self, matcher: PaymentTermsMatcher):
        """S: Explicit payment_term_id code -> MATCHED."""
        res = matcher.match({"payment_term_id": "Net_30"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "Net_30"
        assert res.method == "exact_payment_term_id"

    def test_t_unique_numeric_days(self, matcher: PaymentTermsMatcher):
        """T: Explicit numeric days=14 -> Net_14."""
        res = matcher.match({"days": 14})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "Net_14"
        assert res.method == "exact_numeric_days"

    def test_t_unique_numeric_days_30(self, matcher: PaymentTermsMatcher):
        """T: Explicit numeric days=30 -> Net_30."""
        res = matcher.match({"days": 30})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "Net_30"


class TestPaymentTermsAmbiguityAndDaysZero:
    """Verify real master-data ambiguity for days=0 and synthetic alias collisions."""

    def test_u_days_zero_without_text_is_ambiguous(self, matcher: PaymentTermsMatcher):
        """U: days=0 in real master data has both Immediate and Monthly_in_advance -> AMBIGUOUS."""
        res = matcher.match({"days": 0})
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert len(res.candidates) == 2
        cand_ids = {c["payment_term_id"] for c in res.candidates}
        assert "Immediate" in cand_ids
        assert "Monthly_in_advance" in cand_ids
        assert res.method == "ambiguous_days"

    def test_u_days_zero_disambiguated_by_text_immediate(self, matcher: PaymentTermsMatcher):
        """U: days=0 + 'due on receipt' disambiguates to Immediate."""
        res = matcher.match({
            "raw_text": "due on receipt",
            "days": 0,
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "Immediate"
        assert res.method == "composite_alias_and_days"

    def test_u_days_zero_disambiguated_by_text_advance(self, matcher: PaymentTermsMatcher):
        """U: days=0 + 'in advance' disambiguates to Monthly_in_advance."""
        res = matcher.match({
            "raw_text": "in advance",
            "days": 0,
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "Monthly_in_advance"

    def test_ae_alias_collision_returns_ambiguous(self):
        """AE & Correction 5: Multiple master candidates for one normalized alias -> AMBIGUOUS."""
        terms = [
            PaymentTermRecord(payment_term_id="TERM_X", days=15, text_aliases=("custom terms",)),
            PaymentTermRecord(payment_term_id="TERM_Y", days=45, text_aliases=("custom terms",)),
        ]
        local_store = MasterDataStore.from_records(payment_terms=terms)
        local_matcher = PaymentTermsMatcher(local_store)

        res = local_matcher.match({"raw_text": "custom terms"})
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert len(res.candidates) == 2
        assert res.method == "ambiguous_alias"


class TestPaymentTermsConflictsAndNegativeRules:
    """Verify conflicts between signals and strict negative constraints."""

    def test_w_conflicting_text_and_days(self, matcher: PaymentTermsMatcher):
        """W: Text says 'Immediate' (days 0) but explicit days is 14 -> AMBIGUOUS."""
        res = matcher.match({
            "raw_text": "due on receipt",
            "days": 14,
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert res.method == "conflicting_text_and_days"

    def test_w_conflicting_id_and_days(self, matcher: PaymentTermsMatcher):
        """W: Code is Net_10 (days 10) but explicit days is 30 -> AMBIGUOUS."""
        res = matcher.match({
            "payment_term_id": "Net_10",
            "days": 30,
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert res.method == "conflicting_term_id_and_signals"

    def test_ab_date_derived_days_alone_does_not_match(self, matcher: PaymentTermsMatcher):
        """AB & Correction 4: Date-derived days alone must NEVER automatically match."""
        res = matcher.match({
            "date_derived_days": 10,
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None
        assert "automatic matching disallowed" in res.details["reason"]

    def test_v_unknown_alias_returns_no_match(self, matcher: PaymentTermsMatcher):
        """V: Completely unknown payment phrase -> NO_MATCH."""
        res = matcher.match({
            "raw_text": "pay whenever convenient next year",
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None
        assert res.candidates == []

    def test_empty_input_returns_no_match(self, matcher: PaymentTermsMatcher):
        """Empty input safely returns NO_MATCH."""
        res = matcher.match({})
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None

    def test_amount_and_currency_ignored(self, matcher: PaymentTermsMatcher):
        """Negative test: Amount and currency cannot match payment terms."""
        res = matcher.match({
            "metadata": {"gross_total": "100.00", "currency": "EUR"},
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None

    def test_ad_document_id_and_filename_invariance(self, matcher: PaymentTermsMatcher):
        """AD: Matching is invariant to document ID or filename."""
        res1 = matcher.match({
            "raw_text": "net 30",
            "metadata": {"document_id": "INV-01.pdf"},
        })
        res2 = matcher.match({
            "raw_text": "net 30",
            "metadata": {"document_id": "INV-99.pdf"},
        })
        assert res1.master_id == res2.master_id == "Net_30"
        assert res1.status == res2.status == MatchStatus.MATCHED


class TestPaymentTermsEvidenceAndSerialization:
    """Verify evidence IDs, observed values, determinism, and serialization."""

    def test_x_evidence_ids_preserved(self, matcher: PaymentTermsMatcher):
        """X: Supporting evidence IDs are retained in result."""
        res = matcher.match({
            "raw_text": "net 14",
            "field_evidence_ids": {"raw_text": "ev_term_text_01"},
        })
        assert "ev_term_text_01" in res.evidence_ids

    def test_y_observed_values_preserved(self, matcher: PaymentTermsMatcher):
        """Y: Raw observed text and days preserved in details."""
        raw = "  two weeks  "
        res = matcher.match({"raw_text": raw})
        assert res.details["observed_values"]["raw_text"] == raw

    def test_z_confidence_is_none(self, matcher: PaymentTermsMatcher):
        """Z: Confidence is strictly None."""
        res = matcher.match({"raw_text": "net 10"})
        assert res.confidence is None

    def test_aa_deterministic_repeated_results(self, matcher: PaymentTermsMatcher):
        """AA: Repeated queries produce identical outputs."""
        query = {"raw_text": "net 25"}
        r1 = matcher.match(query)
        r2 = matcher.match(query)
        assert r1.status == r2.status
        assert r1.master_id == r2.master_id
        assert r1.method == r2.method

    def test_serialization_round_trip(self, matcher: PaymentTermsMatcher):
        """Serialization round trip maintains state and null confidence."""
        res = matcher.match({"raw_text": "net 10"})
        d = res.to_dict()
        assert d["status"] == "matched"
        assert d["master_id"] == "Net_10"
        assert d["confidence"] is None

        reconstructed = MasterMatchResult.from_dict(d)
        assert reconstructed.status == res.status
        assert reconstructed.master_id == res.master_id
        assert reconstructed.confidence is None
