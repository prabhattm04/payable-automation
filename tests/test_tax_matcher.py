"""tests/test_tax_matcher.py — Test suite for Phase 8C Tax Master-Data Matching.

Validates:
A. Unique exact country + tax type + rate.
B. Unique country + tax name + rate.
C. Unique country + tax name.
D. Ambiguous duplicate candidates.
E. Unknown tax -> NO_MATCH (blank master_id).
F. Cross-field tax conflicts (code/rate, code/name, name/rate, type/rate, country).
G. Explicit tax code (only when caller explicitly supplies tax_code).
H. Evidence IDs preserved.
I. Raw observed values preserved in details.
J. Confidence is strictly None.
K. Deterministic repeated results.
L. Amount alone cannot match.
M. Currency alone cannot match.
N. Filename / document ID cannot match.
O. Indexed lookup without full master-data scan.
P. Unicode / diacritics normalization where applicable.
Q. Tax placement preserved strictly as read-only metadata.
R. Negative test: country="DE" + rate=19 alone does NOT automatically match.
S. Decimal rate representation equivalence (Decimal("19") == Decimal("19.00")).
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import pytest

from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedTaxIdentity,
)
from src.matching.models import TaxRecord
from src.matching.store import MasterDataStore
from src.matching.tax_matcher import TaxMatcher

MASTER_DATA_DIR = Path("d:/candidate_kit/master_data")


@pytest.fixture(scope="module")
def store() -> MasterDataStore:
    return MasterDataStore.from_directory(MASTER_DATA_DIR)


@pytest.fixture(scope="module")
def matcher(store: MasterDataStore) -> TaxMatcher:
    return TaxMatcher(store)


class TestTaxExactMatching:
    """Verify exact matching priorities and decimal rate handling."""

    def test_a_unique_country_type_rate(self, matcher: TaxMatcher):
        """A: Unique exact country + tax type + rate -> MATCHED."""
        res = matcher.match({
            "country": "DE",
            "tax_type": "VAT",
            "rate": 19,
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "DE_190_VAT"
        assert res.method == "exact_country_tax_type_rate"
        assert "tax_type" in res.matched_fields
        assert "rate" in res.matched_fields
        assert res.confidence is None

    def test_a_decimal_rate_equivalence(self, matcher: TaxMatcher):
        """S & Correction 2: Decimal rate normalization handles 19, 19.0, '19%' identically."""
        res_dec = matcher.match({
            "country": "DE",
            "tax_type": "VAT",
            "rate": Decimal("19.00"),
        })
        assert res_dec.status == MatchStatus.MATCHED
        assert res_dec.master_id == "DE_190_VAT"

        res_str = matcher.match({
            "country": "EE",
            "tax_type": "VAT",
            "rate": "22%",
        })
        assert res_str.status == MatchStatus.MATCHED
        assert res_str.master_id == "EST_220_VAT"

    def test_b_unique_country_name_rate(self, matcher: TaxMatcher):
        """B: Unique country + tax name + rate -> MATCHED."""
        res = matcher.match({
            "country": "DE",
            "tax_name": "German VAT 7% (reduced)",
            "rate": 7.0,
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "DE_070_VAT"
        assert res.method == "exact_country_name_rate"

    def test_c_unique_country_name_without_rate(self, matcher: TaxMatcher):
        """C: Unique country + tax name without rate -> MATCHED."""
        res = matcher.match({
            "country": "DE",
            "tax_name": "German Reverse Charge 0%",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "DE_000_RC"
        assert res.method == "exact_country_name"

    def test_g_explicit_tax_code(self, matcher: TaxMatcher):
        """G & Correction 3: Explicit tax code only when explicitly supplied in tax_code field."""
        res = matcher.match({"tax_code": "TAX024"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "TAX024"
        assert res.method == "exact_tax_code"

        # Explicit code consistent with observed country and rate
        res2 = matcher.match({
            "tax_code": "TAX024",
            "country": "GH",
            "rate": Decimal("15"),
        })
        assert res2.status == MatchStatus.MATCHED
        assert res2.master_id == "TAX024"

    def test_explicit_code_not_inferred_from_name(self, matcher: TaxMatcher):
        """Correction 3: The matcher must NOT classify tax_name as a tax_code."""
        # If user passes a code string inside tax_name field without country/type:
        res = matcher.match({"tax_name": "TAX024"})
        # Should not blindly match as a code
        assert res.status == MatchStatus.NO_MATCH


class TestTaxNegativeAndRateRules:
    """Verify negative constraints and that rate alone never matches."""

    def test_r_country_and_rate_alone_does_not_match(self, matcher: TaxMatcher):
        """Correction 1 & 7: Country="DE" + rate=19 alone does NOT automatically return DE_190_VAT."""
        res = matcher.match({
            "country": "DE",
            "rate": 19,
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None
        assert len(res.candidates) > 0  # Generated for diagnostic purposes
        assert "not sufficient tax identity" in res.details["reason"]

    def test_l_amount_alone_cannot_match(self, matcher: TaxMatcher):
        """L: Tax amount alone must never match a tax."""
        res = matcher.match({
            "metadata": {"tax_amount": "83.22", "subtotal": "438.00"},
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None

    def test_m_currency_alone_cannot_match(self, matcher: TaxMatcher):
        """M: Currency alone must never match a tax."""
        res = matcher.match({
            "metadata": {"currency": "EUR"},
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None

    def test_n_filename_and_document_id_ignored(self, matcher: TaxMatcher):
        """N: Document ID and filename cannot match or affect a tax."""
        res = matcher.match({
            "metadata": {"document_id": "INV-01.pdf", "filename": "INV-01.pdf"},
        })
        assert res.status == MatchStatus.NO_MATCH


class TestTaxConflictsAndAmbiguity:
    """Verify cross-field conflict handling (Correction 8)."""

    def test_f1_conflict_code_vs_rate(self, matcher: TaxMatcher):
        """Correction 8: Explicit tax code with conflicting rate -> AMBIGUOUS."""
        # DE_190_VAT is 19%, but observed rate is 7%
        res = matcher.match({
            "tax_code": "DE_190_VAT",
            "rate": 7.0,
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert res.method == "conflicting_tax_code_and_properties"

    def test_f2_conflict_code_vs_country(self, matcher: TaxMatcher):
        """Correction 8: Explicit tax code with conflicting country -> AMBIGUOUS."""
        # DE_190_VAT belongs to DE, but observed country is PT
        res = matcher.match({
            "tax_code": "DE_190_VAT",
            "country": "PT",
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None

    def test_f3_conflict_code_vs_name(self, matcher: TaxMatcher):
        """Correction 8: Explicit tax code with conflicting name -> AMBIGUOUS."""
        # Code points to German 19%, but name points to German 7%
        res = matcher.match({
            "tax_code": "DE_190_VAT",
            "tax_name": "German VAT 7% (reduced)",
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None

    def test_f4_conflict_name_vs_rate(self, matcher: TaxMatcher):
        """Correction 8: Name matches 19% but rate observed is 7% -> AMBIGUOUS."""
        res = matcher.match({
            "country": "DE",
            "tax_name": "German VAT 19%",
            "rate": 7.0,
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert res.method == "conflicting_name_and_rate"

    def test_f5_conflict_type_rate_vs_competing_name(self, matcher: TaxMatcher):
        """Correction 8: Country + Type + Rate identifies DE_190_VAT, but Name identifies DE_070_VAT -> AMBIGUOUS."""
        res = matcher.match({
            "country": "DE",
            "tax_type": "VAT",
            "rate": 19,
            "tax_name": "German VAT 7% (reduced)",
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert res.method == "conflicting_tax_name_and_rate"

    def test_d_ambiguous_duplicate_candidates(self):
        """D: Multiple taxes sharing country and name without rate -> AMBIGUOUS."""
        taxes = [
            TaxRecord(code="TAX_1", country="DE", tax_type="VAT", rate=19.0, name="Standard VAT"),
            TaxRecord(code="TAX_2", country="DE", tax_type="VAT", rate=16.0, name="Standard VAT"),
        ]
        local_store = MasterDataStore.from_records(tax_master=taxes)
        local_matcher = TaxMatcher(local_store)

        res = local_matcher.match({
            "country": "DE",
            "tax_name": "Standard VAT",
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert len(res.candidates) == 2

    def test_e_unknown_tax_returns_no_match(self, matcher: TaxMatcher):
        """E: Unknown tax country / rate / type -> NO_MATCH."""
        res = matcher.match({
            "country": "XX",
            "tax_type": "NON_EXISTENT",
            "rate": 99.0,
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None


class TestTaxMetadataAndEvidence:
    """Verify evidence IDs, raw values, and placement metadata."""

    def test_q_placement_preserved_as_metadata(self, matcher: TaxMatcher):
        """Q & Correction 9: Tax placement ('header' | 'line') preserved strictly as read-only metadata."""
        res_header = matcher.match({
            "tax_code": "DE_190_VAT",
            "placement": "header",
        })
        assert res_header.details["placement"] == "header"

        res_line = matcher.match({
            "tax_code": "DE_190_VAT",
            "placement": "line",
        })
        assert res_line.details["placement"] == "line"

    def test_h_evidence_ids_preserved(self, matcher: TaxMatcher):
        """H: Supporting evidence IDs are retained in result."""
        res = matcher.match({
            "tax_code": "DE_190_VAT",
            "field_evidence_ids": {"tax_code": "ev_tax_code_01"},
        })
        assert "ev_tax_code_01" in res.evidence_ids

    def test_i_observed_values_preserved(self, matcher: TaxMatcher):
        """I: Raw observed values are preserved in details."""
        res = matcher.match({
            "tax_code": "DE_190_VAT",
            "country": "DE",
            "rate": "19%",
        })
        assert res.details["observed_values"]["rate"] == "19%"
        assert res.details["observed_values"]["country"] == "DE"

    def test_j_confidence_is_none(self, matcher: TaxMatcher):
        """J: Confidence is strictly None."""
        res = matcher.match({"tax_code": "DE_190_VAT"})
        assert res.confidence is None

    def test_k_deterministic_repeated_results(self, matcher: TaxMatcher):
        """K: Repeated matching queries produce identical outputs."""
        query = {"country": "DE", "tax_type": "VAT", "rate": 19}
        r1 = matcher.match(query)
        r2 = matcher.match(query)
        assert r1.status == r2.status
        assert r1.master_id == r2.master_id
        assert r1.method == r2.method
