"""src/matching/payment_terms_matcher.py — Safe, evidence-aware payment-term matcher.

Phase 8C: Tax + Payment-Term Master-Data Matching.

Design Principles:
1. Matching Hierarchy:
   - 1. Exact explicit payment-term ID/code (only if explicitly present on document)
   - 2. Exact normalized alias/text
   - 3. Explicit numeric days (e.g. printed "14 days")
   - 4. Composite evidence (text + explicit numeric days)
2. Real Master Ambiguity for days=0:
   - `days=0` in real master data corresponds to both `Immediate` and `Monthly_in_advance`.
   - `days=0` without alias/text to disambiguate strictly returns AMBIGUOUS.
3. Separation of Date-Derived Days:
   - Date-derived days (from invoice/due date difference) are strictly separated and
     NEVER automatically match a payment term without explicit document evidence.
4. Conflict Outcomes:
   - Competing candidates (e.g. text "Immediate" + days 14) -> AMBIGUOUS.
   - Unknown values -> NO_MATCH.
5. Invariance:
   - Amounts, currencies, PO numbers, filenames, and document IDs cannot match a payment term.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union

from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedPaymentTermIdentity,
)
from src.matching.models import PaymentTermRecord
from src.matching.normalization import (
    normalize_code,
    normalize_text,
)
from src.matching.store import MasterDataStore


class PaymentTermsMatcher:
    """Matches observed document payment-term identity against MasterDataStore.payment_term_indexes."""

    def __init__(self, store: MasterDataStore) -> None:
        self.store = store

    def match(
        self,
        observed: Union[ObservedPaymentTermIdentity, Dict[str, Any]],
    ) -> MasterMatchResult:
        """Match observed payment-term identity against master data."""
        if isinstance(observed, dict):
            obs = ObservedPaymentTermIdentity.from_dict(observed)
        elif isinstance(observed, ObservedPaymentTermIdentity):
            obs = observed
        else:
            raise TypeError(f"observed must be ObservedPaymentTermIdentity or dict, got {type(observed)}")

        observed_values: Dict[str, Any] = {
            "payment_term_id": obs.payment_term_id,
            "raw_text": obs.raw_text,
            "days": obs.days,
            "date_derived_days": obs.date_derived_days,
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

        norm_id = normalize_code(obs.payment_term_id)
        norm_text = normalize_text(obs.raw_text)
        obs_days = obs.days

        # ──────────────────────────────────────────────────────────────────────
        # Priority 1: Exact Explicit Payment-Term ID
        # ──────────────────────────────────────────────────────────────────────
        if norm_id:
            cand = self.store.get_payment_term_by_id(norm_id)
            if cand:
                matched_f = ["payment_term_id"]
                conflicts: List[str] = []

                # Conflict with explicit days if present
                if obs_days is not None and obs_days != cand.days:
                    conflicts.append(f"Observed days '{obs_days}' conflicts with master days '{cand.days}'")

                # Conflict with explicit text if present
                if norm_text:
                    alias_cands = self.store.find_payment_terms_by_alias(norm_text)
                    if alias_cands and cand not in alias_cands:
                        conflicts.append(f"Observed text '{obs.raw_text}' identifies competing candidate '{alias_cands[0].payment_term_id}'")

                if conflicts:
                    return MasterMatchResult(
                        entity_type="payment_term",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[cand.to_dict()],
                        method="conflicting_term_id_and_signals",
                        evidence_ids=get_evidence_ids(["payment_term_id", "days", "raw_text"]),
                        matched_fields=matched_f,
                        details={
                            "conflicts": conflicts,
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "PaymentTermsMatcher", "phase": "8C"},
                    )

                if obs_days is not None and obs_days == cand.days:
                    matched_f.append("days")

                return MasterMatchResult(
                    entity_type="payment_term",
                    status=MatchStatus.MATCHED,
                    master_id=cand.payment_term_id,
                    candidates=[cand.to_dict()],
                    method="exact_payment_term_id",
                    evidence_ids=get_evidence_ids(matched_f),
                    matched_fields=matched_f,
                    details={
                        "payment_term_id": cand.payment_term_id,
                        "days": cand.days,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "PaymentTermsMatcher", "phase": "8C"},
                )
            else:
                # Explicit ID provided but does not exist in master data
                return MasterMatchResult(
                    entity_type="payment_term",
                    status=MatchStatus.NO_MATCH,
                    master_id=None,
                    candidates=[],
                    method="none",
                    evidence_ids=get_evidence_ids(["payment_term_id"]),
                    matched_fields=[],
                    details={
                        "reason": f"Explicit payment_term_id '{norm_id}' not found in master data",
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "PaymentTermsMatcher", "phase": "8C"},
                )

        # ──────────────────────────────────────────────────────────────────────
        # Priority 2: Exact Normalized Alias / Text (+ Composite Days)
        # ──────────────────────────────────────────────────────────────────────
        if norm_text:
            alias_cands = self.store.find_payment_terms_by_alias(norm_text)

            if alias_cands:
                matched_f = ["raw_text"]

                # Cross-check with explicit days if observed
                if obs_days is not None:
                    # Check for contradiction with explicit days
                    matching_days = [c for c in alias_cands if c.days == obs_days]
                    if not matching_days:
                        # Strong conflict: e.g. text "Immediate" (days 0) with explicit days 14 (Net_14)
                        return MasterMatchResult(
                            entity_type="payment_term",
                            status=MatchStatus.AMBIGUOUS,
                            master_id=None,
                            candidates=[c.to_dict() for c in alias_cands],
                            method="conflicting_text_and_days",
                            evidence_ids=get_evidence_ids(["raw_text", "days"]),
                            matched_fields=["raw_text"],
                            details={
                                "conflict": f"Observed text points to '{[c.payment_term_id for c in alias_cands]}' (days {[c.days for c in alias_cands]}) but observed explicit days is '{obs_days}'",
                                "observed_values": observed_values,
                            },
                            confidence=None,
                            provenance={"matcher": "PaymentTermsMatcher", "phase": "8C"},
                        )
                    alias_cands = matching_days
                    matched_f.append("days")

                if len(alias_cands) == 1:
                    cand = alias_cands[0]
                    return MasterMatchResult(
                        entity_type="payment_term",
                        status=MatchStatus.MATCHED,
                        master_id=cand.payment_term_id,
                        candidates=[cand.to_dict()],
                        method="composite_alias_and_days" if "days" in matched_f else "exact_normalized_alias",
                        evidence_ids=get_evidence_ids(matched_f),
                        matched_fields=matched_f,
                        details={
                            "payment_term_id": cand.payment_term_id,
                            "days": cand.days,
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "PaymentTermsMatcher", "phase": "8C"},
                    )
                elif len(alias_cands) > 1:
                    # Alias collision / multiple candidates
                    return MasterMatchResult(
                        entity_type="payment_term",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[c.to_dict() for c in alias_cands],
                        method="ambiguous_alias",
                        evidence_ids=get_evidence_ids(matched_f),
                        matched_fields=matched_f,
                        details={
                            "reason": f"Normalized text '{norm_text}' matches multiple payment terms",
                            "candidate_ids": [c.payment_term_id for c in alias_cands],
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "PaymentTermsMatcher", "phase": "8C"},
                    )

        # ──────────────────────────────────────────────────────────────────────
        # Priority 3: Explicit Numeric Days (Without Text)
        # ──────────────────────────────────────────────────────────────────────
        if obs_days is not None:
            days_cands = self.store.find_payment_terms_by_days(obs_days)

            if len(days_cands) == 1:
                cand = days_cands[0]
                return MasterMatchResult(
                    entity_type="payment_term",
                    status=MatchStatus.MATCHED,
                    master_id=cand.payment_term_id,
                    candidates=[cand.to_dict()],
                    method="exact_numeric_days",
                    evidence_ids=get_evidence_ids(["days"]),
                    matched_fields=["days"],
                    details={
                        "payment_term_id": cand.payment_term_id,
                        "days": cand.days,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "PaymentTermsMatcher", "phase": "8C"},
                )
            elif len(days_cands) > 1:
                # Real Master Ambiguity: days=0 corresponds to both Immediate and Monthly_in_advance
                return MasterMatchResult(
                    entity_type="payment_term",
                    status=MatchStatus.AMBIGUOUS,
                    master_id=None,
                    candidates=[c.to_dict() for c in days_cands],
                    method="ambiguous_days",
                    evidence_ids=get_evidence_ids(["days"]),
                    matched_fields=["days"],
                    details={
                        "reason": f"Observed days '{obs_days}' matches multiple payment terms in master data; text alias required to disambiguate",
                        "candidate_ids": [c.payment_term_id for c in days_cands],
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "PaymentTermsMatcher", "phase": "8C"},
                )
            else:
                return MasterMatchResult(
                    entity_type="payment_term",
                    status=MatchStatus.NO_MATCH,
                    master_id=None,
                    candidates=[],
                    method="none",
                    evidence_ids=get_evidence_ids(["days"]),
                    matched_fields=[],
                    details={
                        "reason": f"Explicit days '{obs_days}' does not correspond to any master payment term",
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "PaymentTermsMatcher", "phase": "8C"},
                )

        # ──────────────────────────────────────────────────────────────────────
        # Correction 4: Date-Derived Days Alone Disallowed from Automatic Match
        # ──────────────────────────────────────────────────────────────────────
        if obs.date_derived_days is not None:
            return MasterMatchResult(
                entity_type="payment_term",
                status=MatchStatus.NO_MATCH,
                master_id=None,
                candidates=[],
                method="none",
                evidence_ids=get_evidence_ids(["date_derived_days"]),
                matched_fields=[],
                details={
                    "reason": "Date-derived days is not explicit payment-term evidence; automatic matching disallowed",
                    "observed_values": observed_values,
                },
                confidence=None,
                provenance={"matcher": "PaymentTermsMatcher", "phase": "8C"},
            )

        # ──────────────────────────────────────────────────────────────────────
        # No Match Found / Empty Input
        # ──────────────────────────────────────────────────────────────────────
        return MasterMatchResult(
            entity_type="payment_term",
            status=MatchStatus.NO_MATCH,
            master_id=None,
            candidates=[],
            method="none",
            evidence_ids=list(obs.evidence_ids),
            matched_fields=[],
            details={
                "reason": "No payment-term identity evidence provided",
                "observed_values": observed_values,
            },
            confidence=None,
            provenance={"matcher": "PaymentTermsMatcher", "phase": "8C"},
        )
