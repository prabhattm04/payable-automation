"""tests/test_buyer_matcher.py — Test suite for Phase 8B Buyer Master-Data Matching.

Validates:
M. Company match (retains company code without fabricating leaf).
N. Business unit match.
O. Location match.
P. Duplicate location code (`LOC_EE_001`) without parent BU -> AMBIGUOUS.
Q. Hierarchical identity (BU + Location) resolves unique leaf buyer.
R. Unknown buyer -> NO_MATCH (safe empty master_id).
S. Evidence IDs preserved.
T. Original observed identity preserved.
U. Cross-hierarchy conflicts (e.g. BU in GH with Location in EE) -> AMBIGUOUS.
V. Document-ID invariance & determinism.
W. Serialization round-trip & confidence is strictly None.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from src.matching.buyer_matcher import BuyerMatcher
from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedBuyerIdentity,
)
from src.matching.store import MasterDataStore

MASTER_DATA_DIR = Path("d:/candidate_kit/master_data")


@pytest.fixture(scope="module")
def store() -> MasterDataStore:
    return MasterDataStore.from_directory(MASTER_DATA_DIR)


@pytest.fixture(scope="module")
def matcher(store: MasterDataStore) -> BuyerMatcher:
    return BuyerMatcher(store)


class TestBuyerHierarchicalMatching:
    """Verify hierarchical levels and explicit master_id semantics."""

    def test_m_company_level_match(self, matcher: BuyerMatcher):
        """M & Correction 3 & 4: Company-only match does NOT fabricate leaf codes."""
        res = matcher.match({"company_code": "BOLTGROUP"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "BOLTGROUP"
        assert res.details["hierarchy_level"] == "company"
        assert res.details["company_code"] == "BOLTGROUP"
        assert res.details["business_unit_code"] is None
        assert res.details["location_code"] is None

    def test_company_name_match(self, matcher: BuyerMatcher):
        """M: Company name match resolves company."""
        res = matcher.match({"company_name": "Bolt Group"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "BOLTGROUP"
        assert res.details["company_code"] == "BOLTGROUP"

    def test_n_business_unit_match(self, matcher: BuyerMatcher):
        """N: Business unit match resolves BU and its associated company."""
        res = matcher.match({"business_unit_code": "GH001"})
        assert res.status == MatchStatus.MATCHED
        assert res.details["hierarchy_level"] == "business_unit"
        assert res.details["company_code"] == "BOLTGROUP"
        assert res.details["business_unit_code"] == "GH001"

    def test_o_unique_location_match(self, matcher: BuyerMatcher):
        """O: Unique location code resolves leaf buyer."""
        res = matcher.match({"location_code": "LOC_GH_001"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "LOC_GH_001"
        assert res.details["hierarchy_level"] == "location"
        assert res.details["company_code"] == "BOLTGROUP"
        assert res.details["business_unit_code"] == "GH001"
        assert res.details["location_code"] == "LOC_GH_001"

    def test_p_duplicate_location_code_returns_ambiguous(self, matcher: BuyerMatcher):
        """P & Correction 3: Duplicate location code LOC_EE_001 under multiple BUs is AMBIGUOUS."""
        # LOC_EE_001 exists under EE001 (Bolt Technology OU) and EE004 (Bolt Holdings OU)
        res = matcher.match({"location_code": "LOC_EE_001"})
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert len(res.candidates) == 2
        assert res.method == "ambiguous_location_code"
        assert "EE001" in res.details["candidate_bu_codes"]
        assert "EE004" in res.details["candidate_bu_codes"]

    def test_q1_hierarchical_identity_resolves_unique_buyer_ee001(self, matcher: BuyerMatcher):
        """Q: BU EE001 + Location LOC_EE_001 resolves unambiguous leaf buyer."""
        res = matcher.match({
            "business_unit_code": "EE001",
            "location_code": "LOC_EE_001",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "LOC_EE_001"
        assert res.details["company_code"] == "BOLTGROUP"
        assert res.details["business_unit_code"] == "EE001"
        assert res.details["location_code"] == "LOC_EE_001"

    def test_q2_hierarchical_identity_resolves_unique_buyer_ee004(self, matcher: BuyerMatcher):
        """Q: BU EE004 (Bolt Holdings OU) + Location LOC_EE_001 resolves unambiguous leaf buyer."""
        res = matcher.match({
            "business_unit_name": "Bolt Holdings OU",
            "location_code": "LOC_EE_001",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "LOC_EE_001"
        assert res.details["company_code"] == "BOLTGROUP"
        assert res.details["business_unit_code"] == "EE004"
        assert res.details["location_code"] == "LOC_EE_001"


class TestBuyerConflictsAndNoMatch:
    """Verify cross-hierarchy conflict rejection and missing buyer handling."""

    def test_cross_hierarchy_conflict_bu_vs_location(self, matcher: BuyerMatcher):
        """U: Ghana BU code (GH001) + Estonia Location code (LOC_EE_001) -> AMBIGUOUS."""
        res = matcher.match({
            "business_unit_code": "GH001",
            "location_code": "LOC_EE_001",
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert res.method == "conflicting_bu_and_location"

    def test_r_unknown_buyer_returns_no_match(self, matcher: BuyerMatcher):
        """R: Completely unknown company or location -> NO_MATCH."""
        res = matcher.match({
            "company_code": "UNKNOWN_GROUP",
            "location_code": "LOC_UNKNOWN",
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None
        assert res.candidates == []

    def test_empty_buyer_identity_returns_no_match(self, matcher: BuyerMatcher):
        """R: Empty input returns NO_MATCH safely."""
        res = matcher.match({})
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None


class TestBuyerEvidenceAndTraceability:
    """Verify evidence IDs and raw observed values are preserved."""

    def test_s_evidence_ids_preserved(self, matcher: BuyerMatcher):
        """S: Supporting evidence IDs are retained in match result."""
        identity = ObservedBuyerIdentity(
            business_unit_code="EE001",
            location_code="LOC_EE_001",
            field_evidence_ids={
                "business_unit_code": "ev_bu_101",
                "location_code": "ev_loc_102",
            },
        )
        res = matcher.match(identity)
        assert res.status == MatchStatus.MATCHED
        assert "ev_bu_101" in res.evidence_ids
        assert "ev_loc_102" in res.evidence_ids

    def test_t_original_observed_identity_preserved(self, matcher: BuyerMatcher):
        """T: Original observed buyer values are preserved in details."""
        raw_loc = "LOC_GH_001"
        res = matcher.match({"location_code": raw_loc})
        assert res.details["observed_values"]["location_code"] == raw_loc


class TestBuyerSerializationAndInvariance:
    """Verify serialization, document ID invariance, and confidence is None."""

    def test_serialization_round_trip(self, matcher: BuyerMatcher):
        """W: MasterMatchResult converts to dict and reconstructs identically."""
        res = matcher.match({"location_code": "LOC_GH_001"})
        d = res.to_dict()
        assert d["status"] == "matched"
        assert d["master_id"] == "LOC_GH_001"
        assert d["confidence"] is None

        reconstructed = MasterMatchResult.from_dict(d)
        assert reconstructed.status == res.status
        assert reconstructed.master_id == res.master_id
        assert reconstructed.details["company_code"] == "BOLTGROUP"

    def test_v_document_id_and_filename_invariance(self, matcher: BuyerMatcher):
        """V: Matching is invariant to document ID or filename."""
        res1 = matcher.match({
            "location_code": "LOC_GH_001",
            "metadata": {"document_id": "INV-01.pdf"},
        })
        res2 = matcher.match({
            "location_code": "LOC_GH_001",
            "metadata": {"document_id": "HLD-03.pdf"},
        })
        assert res1.master_id == res2.master_id == "LOC_GH_001"
        assert res1.status == res2.status == MatchStatus.MATCHED

    def test_x_confidence_is_none(self, matcher: BuyerMatcher):
        """X: Confidence must strictly be None."""
        res = matcher.match({"location_code": "LOC_GH_001"})
        assert res.confidence is None
