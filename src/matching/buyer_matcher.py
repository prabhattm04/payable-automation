"""src/matching/buyer_matcher.py — Hierarchical buyer master-data matcher.

Phase 8B: Supplier + Buyer Master-Data Matching.

Design Principles:
1. Strict Hierarchy Traversal:
   - Company -> Business Unit -> Location.
   - Company, BU, and Location represent distinct hierarchical evidence levels.
2. Safe Leaf Resolution:
   - Does NOT automatically convert a partial hierarchy match (e.g. company only)
     into a leaf buyer code unless the resulting identity is unambiguous.
   - If a location code appears under multiple business units (e.g. `LOC_EE_001`),
     it remains AMBIGUOUS unless disambiguated by parent BU/Company evidence.
3. Explicit Master ID Semantics & Separate Codes:
   - Preserves `company_code`, `business_unit_code`, and `location_code` separately
     in `details`.
   - `master_id` holds the code of the most specific unambiguous level resolved
     (leaf `location_code`, `business_unit_code`, or `company_code`), or None if ambiguous/unmatched.
4. Conflict Rejection:
   - Conflicting hierarchical signals (e.g. BU 'GH001' with Location 'LOC_EE_001')
     result in AMBIGUOUS or NO_MATCH, never an unsafe guess.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedBuyerIdentity,
)
from src.matching.models import BusinessUnitRecord, CompanyRecord, LocationRecord
from src.matching.normalization import (
    normalize_code,
    normalize_name,
    normalize_text,
)
from src.matching.store import MasterDataStore


class BuyerMatcher:
    """Matches observed document buyer identity against chart-of-books master data."""

    def __init__(self, store: MasterDataStore) -> None:
        self.store = store

    def match(
        self,
        observed: Union[ObservedBuyerIdentity, Dict[str, Any]],
    ) -> MasterMatchResult:
        """Match observed buyer identity signals against chart of books."""
        if isinstance(observed, dict):
            obs = ObservedBuyerIdentity.from_dict(observed)
        elif isinstance(observed, ObservedBuyerIdentity):
            obs = observed
        else:
            raise TypeError(f"observed must be ObservedBuyerIdentity or dict, got {type(observed)}")

        observed_values: Dict[str, Any] = {
            "company_name": obs.company_name,
            "company_code": obs.company_code,
            "business_unit_name": obs.business_unit_name,
            "business_unit_code": obs.business_unit_code,
            "location_name": obs.location_name,
            "location_code": obs.location_code,
            "invoice_to_address": obs.invoice_to_address,
        }

        # Helper for evidence IDs
        def get_evidence_ids(fields: List[str]) -> List[str]:
            ev_set: Set[str] = set()
            for f in fields:
                if f in obs.field_evidence_ids and obs.field_evidence_ids[f]:
                    ev_set.add(obs.field_evidence_ids[f])
            if not ev_set and obs.evidence_ids:
                return list(obs.evidence_ids)
            return sorted(ev_set)

        # Normalize observed codes and names
        c_code = normalize_code(obs.company_code)
        c_name = normalize_name(obs.company_name)
        bu_code = normalize_code(obs.business_unit_code)
        bu_name = normalize_name(obs.business_unit_name)
        loc_code = normalize_code(obs.location_code)
        loc_name = normalize_name(obs.location_name)
        inv_addr = normalize_text(obs.invoice_to_address)

        # ──────────────────────────────────────────────────────────────────────
        # Level 1: Location Evidence Observed (Leaf Level)
        # ──────────────────────────────────────────────────────────────────────
        if loc_code or loc_name or inv_addr:
            loc_candidates: List[LocationRecord] = []
            matched_f: List[str] = []

            if loc_code:
                loc_candidates = self.store.find_locations_by_code(loc_code)
                if loc_candidates:
                    matched_f.append("location_code")
            elif loc_name:
                loc_candidates = self.store.chart_of_books_indexes.by_normalized_location_name.lookup(loc_name)
                if loc_candidates:
                    matched_f.append("location_name")

            if not loc_candidates:
                # No location matched in chart of books
                return MasterMatchResult(
                    entity_type="buyer",
                    status=MatchStatus.NO_MATCH,
                    master_id=None,
                    candidates=[],
                    method="none",
                    evidence_ids=get_evidence_ids(matched_f or ["location_code"]),
                    matched_fields=[],
                    details={
                        "reason": "Observed location code/name does not exist in chart of books",
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "BuyerMatcher", "phase": "8B"},
                )

            # Check hierarchical consistency with observed parent BU and Company
            if bu_code:
                loc_candidates = [l for l in loc_candidates if l.business_unit_code == bu_code]
                if not loc_candidates:
                    # Contradiction: location exists, but not under the specified BU!
                    return MasterMatchResult(
                        entity_type="buyer",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[],
                        method="conflicting_bu_and_location",
                        evidence_ids=get_evidence_ids(["business_unit_code", "location_code"]),
                        matched_fields=["location_code"],
                        details={
                            "conflict": f"Location '{loc_code}' does not belong to business unit '{bu_code}'",
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "BuyerMatcher", "phase": "8B"},
                    )
                matched_f.append("business_unit_code")

            if bu_name:
                # Disambiguate by BU name if BU code was not provided
                loc_candidates_bu = []
                for l in loc_candidates:
                    bu_record_list = self.store.chart_of_books_indexes.by_bu_code.lookup(l.business_unit_code)
                    if bu_record_list and normalize_name(bu_record_list[0].business_unit_name) == bu_name:
                        loc_candidates_bu.append(l)
                if loc_candidates_bu:
                    loc_candidates = loc_candidates_bu
                    matched_f.append("business_unit_name")

            if c_code:
                loc_candidates = [l for l in loc_candidates if l.company_code == c_code]
                if not loc_candidates:
                    return MasterMatchResult(
                        entity_type="buyer",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[],
                        method="conflicting_company_and_location",
                        evidence_ids=get_evidence_ids(["company_code", "location_code"]),
                        matched_fields=["location_code"],
                        details={
                            "conflict": f"Location '{loc_code}' does not belong to company '{c_code}'",
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "BuyerMatcher", "phase": "8B"},
                    )
                matched_f.append("company_code")

            # Final check of candidate count at location level
            if len(loc_candidates) == 1:
                loc = loc_candidates[0]
                return MasterMatchResult(
                    entity_type="buyer",
                    status=MatchStatus.MATCHED,
                    master_id=loc.location_code,
                    candidates=[loc.to_dict()],
                    method="hierarchical_location" if len(matched_f) > 1 else "exact_location_code",
                    evidence_ids=get_evidence_ids(matched_f),
                    matched_fields=matched_f,
                    details={
                        "hierarchy_level": "location",
                        "company_code": loc.company_code,
                        "business_unit_code": loc.business_unit_code,
                        "location_code": loc.location_code,
                        "location_name": loc.location_name,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "BuyerMatcher", "phase": "8B"},
                )
            elif len(loc_candidates) > 1:
                # Ambiguous location code (e.g. LOC_EE_001 under multiple BUs without BU context)
                return MasterMatchResult(
                    entity_type="buyer",
                    status=MatchStatus.AMBIGUOUS,
                    master_id=None,
                    candidates=[l.to_dict() for l in loc_candidates],
                    method="ambiguous_location_code",
                    evidence_ids=get_evidence_ids(matched_f),
                    matched_fields=matched_f,
                    details={
                        "reason": f"Location '{loc_code}' exists under multiple business units; business unit evidence required to disambiguate",
                        "candidate_bu_codes": [l.business_unit_code for l in loc_candidates],
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "BuyerMatcher", "phase": "8B"},
                )

        # ──────────────────────────────────────────────────────────────────────
        # Level 2: Business Unit Evidence Observed (No Location Observed)
        # ──────────────────────────────────────────────────────────────────────
        if bu_code or bu_name:
            bu_candidates: List[BusinessUnitRecord] = []
            matched_f = []

            if bu_code:
                bu_candidates = self.store.chart_of_books_indexes.by_bu_code.lookup(bu_code)
                if bu_candidates:
                    matched_f.append("business_unit_code")
            elif bu_name:
                bu_candidates = self.store.chart_of_books_indexes.by_normalized_bu_name.lookup(bu_name)
                if bu_candidates:
                    matched_f.append("business_unit_name")

            if not bu_candidates:
                return MasterMatchResult(
                    entity_type="buyer",
                    status=MatchStatus.NO_MATCH,
                    master_id=None,
                    candidates=[],
                    method="none",
                    evidence_ids=get_evidence_ids(matched_f or ["business_unit_code"]),
                    matched_fields=[],
                    details={
                        "reason": "Observed business unit code/name does not exist in chart of books",
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "BuyerMatcher", "phase": "8B"},
                )

            # Consistency with company code if observed
            if c_code:
                bu_candidates = [b for b in bu_candidates if b.company_code == c_code]
                if not bu_candidates:
                    return MasterMatchResult(
                        entity_type="buyer",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[],
                        method="conflicting_company_and_bu",
                        evidence_ids=get_evidence_ids(["company_code", "business_unit_code"]),
                        matched_fields=["business_unit_code"],
                        details={
                            "conflict": f"Business unit does not belong to company '{c_code}'",
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "BuyerMatcher", "phase": "8B"},
                    )
                matched_f.append("company_code")

            if len(bu_candidates) == 1:
                bu = bu_candidates[0]
                # Check if this BU has an unambiguous single leaf location
                unambiguous_leaf_loc = (
                    bu.locations[0].location_code if len(bu.locations) == 1 else None
                )

                return MasterMatchResult(
                    entity_type="buyer",
                    status=MatchStatus.MATCHED,
                    # When BU is the highest level resolved, master_id is the BU code (or leaf if unambiguous)
                    master_id=unambiguous_leaf_loc or bu.business_unit_code,
                    candidates=[bu.to_dict()],
                    method="exact_business_unit_code" if "business_unit_code" in matched_f else "exact_normalized_bu_name",
                    evidence_ids=get_evidence_ids(matched_f),
                    matched_fields=matched_f,
                    details={
                        "hierarchy_level": "business_unit",
                        "company_code": bu.company_code,
                        "business_unit_code": bu.business_unit_code,
                        "business_unit_name": bu.business_unit_name,
                        "location_code": unambiguous_leaf_loc,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "BuyerMatcher", "phase": "8B"},
                )
            elif len(bu_candidates) > 1:
                return MasterMatchResult(
                    entity_type="buyer",
                    status=MatchStatus.AMBIGUOUS,
                    master_id=None,
                    candidates=[b.to_dict() for b in bu_candidates],
                    method="ambiguous_business_unit",
                    evidence_ids=get_evidence_ids(matched_f),
                    matched_fields=matched_f,
                    details={
                        "reason": "Multiple business units match observed name/code",
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "BuyerMatcher", "phase": "8B"},
                )

        # ──────────────────────────────────────────────────────────────────────
        # Level 3: Company Evidence Observed (No BU or Location Observed)
        # ──────────────────────────────────────────────────────────────────────
        if c_code or c_name:
            comp_rec: Optional[CompanyRecord] = None
            matched_f = []

            if c_code:
                comp_rec = self.store.get_company_by_code(c_code)
                if comp_rec:
                    matched_f.append("company_code")
            elif c_name:
                for c in self.store.chart_of_books:
                    if normalize_name(c.company_name) == c_name:
                        comp_rec = c
                        matched_f.append("company_name")
                        break

            if comp_rec:
                # Do NOT convert partial company match to leaf code if multiple BUs exist!
                # Keep location_code and business_unit_code as None
                return MasterMatchResult(
                    entity_type="buyer",
                    status=MatchStatus.MATCHED,
                    master_id=comp_rec.company_code,
                    candidates=[comp_rec.to_dict()],
                    method="exact_company_code" if "company_code" in matched_f else "exact_company_name",
                    evidence_ids=get_evidence_ids(matched_f),
                    matched_fields=matched_f,
                    details={
                        "hierarchy_level": "company",
                        "company_code": comp_rec.company_code,
                        "company_name": comp_rec.company_name,
                        "business_unit_code": None,
                        "location_code": None,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "BuyerMatcher", "phase": "8B"},
                )
            else:
                return MasterMatchResult(
                    entity_type="buyer",
                    status=MatchStatus.NO_MATCH,
                    master_id=None,
                    candidates=[],
                    method="none",
                    evidence_ids=get_evidence_ids(["company_code"]),
                    matched_fields=[],
                    details={
                        "reason": "Observed company code/name does not exist in chart of books",
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "BuyerMatcher", "phase": "8B"},
                )

        # ──────────────────────────────────────────────────────────────────────
        # No Buyer Evidence Provided
        # ──────────────────────────────────────────────────────────────────────
        return MasterMatchResult(
            entity_type="buyer",
            status=MatchStatus.NO_MATCH,
            master_id=None,
            candidates=[],
            method="none",
            evidence_ids=list(obs.evidence_ids),
            matched_fields=[],
            details={
                "reason": "No buyer identity evidence provided",
                "observed_values": observed_values,
            },
            confidence=None,
            provenance={"matcher": "BuyerMatcher", "phase": "8B"},
        )
