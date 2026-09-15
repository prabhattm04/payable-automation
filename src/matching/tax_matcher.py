"""src/matching/tax_matcher.py — Safe, evidence-aware tax master-data matcher.

Phase 8C: Tax + Payment-Term Master-Data Matching.

Design Principles:
1. Matching Hierarchy:
   - 1. Exact explicit tax code (only when caller explicitly identifies a tax_code field)
   - 2. Exact country + tax type + rate (deterministic Decimal rate comparison)
   - 3. Exact country + normalized tax name + rate
   - 4. Exact country + normalized tax name (only MATCHED if uniquely identifiable)
   - Country + rate alone is NOT sufficient tax identity and NEVER automatically matches.
2. Deterministic Decimal Rate Comparison:
   - All tax rates are compared using `Decimal` values normalized to eliminate floating-point jitter.
3. Conflict Outcomes:
   - Competing valid candidates supported by contradictory observed fields -> AMBIGUOUS.
   - No candidate satisfying the constraints -> NO_MATCH.
4. Preserved Metadata & No Amount Derivations:
   - `placement` ("header" | "line" | "unknown") is preserved strictly as read-only metadata.
   - Tax amount, gross amount, currency, PO number, filename, and document ID cannot match a tax.
   - Rates are never derived or calculated from amounts.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Union

from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedTaxIdentity,
    to_decimal_rate,
)
from src.matching.models import TaxRecord
from src.matching.normalization import (
    normalize_code,
    normalize_name,
)
from src.matching.store import MasterDataStore


class TaxMatcher:
    """Matches observed document tax identity against MasterDataStore.tax_indexes."""

    def __init__(self, store: MasterDataStore) -> None:
        self.store = store

    def match(
        self,
        observed: Union[ObservedTaxIdentity, Dict[str, Any]],
    ) -> MasterMatchResult:
        """Match an observed tax identity against tax master data."""
        if isinstance(observed, dict):
            obs = ObservedTaxIdentity.from_dict(observed)
        elif isinstance(observed, ObservedTaxIdentity):
            obs = observed
        else:
            raise TypeError(f"observed must be ObservedTaxIdentity or dict, got {type(observed)}")

        observed_values: Dict[str, Any] = {
            "tax_code": obs.tax_code,
            "tax_name": obs.tax_name,
            "tax_type": obs.tax_type,
            "rate": str(obs.rate) if isinstance(obs.rate, Decimal) else obs.rate,
            "country": obs.country,
            "placement": obs.placement,
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

        # Normalize observed identity signals
        norm_code = normalize_code(obs.tax_code)
        norm_country = normalize_code(obs.country).upper()
        norm_type = normalize_code(obs.tax_type).upper()
        norm_name = normalize_name(obs.tax_name)
        obs_rate_dec = to_decimal_rate(obs.rate)

        # ──────────────────────────────────────────────────────────────────────
        # Priority 1: Exact Explicit Tax Code (Only if explicitly supplied)
        # ──────────────────────────────────────────────────────────────────────
        if norm_code:
            code_cand = self.store.get_tax_by_code(norm_code)
            if code_cand:
                matched_f = ["tax_code"]
                cand_rate_dec = to_decimal_rate(code_cand.rate)

                # Check for cross-field conflicts against the explicit code
                conflicts: List[str] = []
                if norm_country and code_cand.country.upper() != norm_country:
                    conflicts.append(f"Observed country '{norm_country}' != master country '{code_cand.country}'")
                if obs_rate_dec is not None and cand_rate_dec is not None and obs_rate_dec != cand_rate_dec:
                    conflicts.append(f"Observed rate '{obs_rate_dec}' != master rate '{cand_rate_dec}'")
                if norm_type and code_cand.tax_type.upper() != norm_type:
                    conflicts.append(f"Observed tax type '{norm_type}' != master tax type '{code_cand.tax_type}'")
                if norm_name and normalize_name(code_cand.name) != norm_name:
                    # Check if the observed name matches a different valid tax candidate in the same country
                    alt_candidates = [
                        t for t in self.store.tax_indexes.by_normalized_name.lookup(norm_name)
                        if t.country.upper() == code_cand.country.upper()
                    ]
                    if alt_candidates and alt_candidates[0].code != code_cand.code:
                        conflicts.append(f"Observed tax name matches competing candidate '{alt_candidates[0].code}'")

                if conflicts:
                    return MasterMatchResult(
                        entity_type="tax",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[code_cand.to_dict()],
                        method="conflicting_tax_code_and_properties",
                        evidence_ids=get_evidence_ids(["tax_code", "rate", "country", "tax_name", "tax_type"]),
                        matched_fields=matched_f,
                        details={
                            "conflicts": conflicts,
                            "placement": obs.placement,
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "TaxMatcher", "phase": "8C"},
                    )

                if norm_country and code_cand.country.upper() == norm_country:
                    matched_f.append("country")
                if obs_rate_dec is not None and cand_rate_dec == obs_rate_dec:
                    matched_f.append("rate")
                if norm_type and code_cand.tax_type.upper() == norm_type:
                    matched_f.append("tax_type")

                return MasterMatchResult(
                    entity_type="tax",
                    status=MatchStatus.MATCHED,
                    master_id=code_cand.code,
                    candidates=[code_cand.to_dict()],
                    method="exact_tax_code",
                    evidence_ids=get_evidence_ids(matched_f),
                    matched_fields=matched_f,
                    details={
                        "code": code_cand.code,
                        "rate": str(cand_rate_dec),
                        "country": code_cand.country,
                        "tax_type": code_cand.tax_type,
                        "placement": obs.placement,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "TaxMatcher", "phase": "8C"},
                )
            else:
                # Explicit code supplied but does not exist in master data
                return MasterMatchResult(
                    entity_type="tax",
                    status=MatchStatus.NO_MATCH,
                    master_id=None,
                    candidates=[],
                    method="none",
                    evidence_ids=get_evidence_ids(["tax_code"]),
                    matched_fields=[],
                    details={
                        "reason": f"Explicit tax code '{norm_code}' not found in tax master",
                        "placement": obs.placement,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "TaxMatcher", "phase": "8C"},
                )

        # ──────────────────────────────────────────────────────────────────────
        # Priority 2: Exact Country + Tax Type + Rate
        # ──────────────────────────────────────────────────────────────────────
        if norm_country and norm_type and obs_rate_dec is not None:
            # Look up candidates matching country and tax type
            cands_by_country = self.store.tax_indexes.by_country.lookup(norm_country)
            type_rate_cands = [
                t for t in cands_by_country
                if t.tax_type.upper() == norm_type and to_decimal_rate(t.rate) == obs_rate_dec
            ]

            # Check if name is also observed and contradicts
            if norm_name:
                name_cands = [
                    t for t in cands_by_country
                    if normalize_name(t.name) == norm_name
                ]
                if name_cands:
                    competing = [t for t in name_cands if t not in type_rate_cands]
                    if competing and not (set(name_cands) & set(type_rate_cands)):
                        # Name matches candidate A, but type+rate matches candidate B
                        all_cands = list({t.code: t for t in type_rate_cands + name_cands}.values())
                        return MasterMatchResult(
                            entity_type="tax",
                            status=MatchStatus.AMBIGUOUS,
                            master_id=None,
                            candidates=[t.to_dict() for t in all_cands],
                            method="conflicting_tax_name_and_rate",
                            evidence_ids=get_evidence_ids(["country", "tax_type", "rate", "tax_name"]),
                            matched_fields=["country", "tax_type", "rate", "tax_name"],
                            details={
                                "conflict": "Observed tax name contradicts observed rate/type candidates",
                                "placement": obs.placement,
                                "observed_values": observed_values,
                            },
                            confidence=None,
                            provenance={"matcher": "TaxMatcher", "phase": "8C"},
                        )

            if len(type_rate_cands) == 1:
                cand = type_rate_cands[0]
                return MasterMatchResult(
                    entity_type="tax",
                    status=MatchStatus.MATCHED,
                    master_id=cand.code,
                    candidates=[cand.to_dict()],
                    method="exact_country_tax_type_rate",
                    evidence_ids=get_evidence_ids(["country", "tax_type", "rate"]),
                    matched_fields=["country", "tax_type", "rate"],
                    details={
                        "code": cand.code,
                        "rate": str(to_decimal_rate(cand.rate)),
                        "country": cand.country,
                        "tax_type": cand.tax_type,
                        "placement": obs.placement,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "TaxMatcher", "phase": "8C"},
                )
            elif len(type_rate_cands) > 1:
                return MasterMatchResult(
                    entity_type="tax",
                    status=MatchStatus.AMBIGUOUS,
                    master_id=None,
                    candidates=[t.to_dict() for t in type_rate_cands],
                    method="ambiguous_country_tax_type_rate",
                    evidence_ids=get_evidence_ids(["country", "tax_type", "rate"]),
                    matched_fields=["country", "tax_type", "rate"],
                    details={
                        "reason": "Multiple master taxes share identical country, tax type, and rate",
                        "placement": obs.placement,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "TaxMatcher", "phase": "8C"},
                )

        # ──────────────────────────────────────────────────────────────────────
        # Priority 3: Exact Country + Normalized Tax Name + Rate
        # ──────────────────────────────────────────────────────────────────────
        if norm_country and norm_name and obs_rate_dec is not None:
            cands_by_country = self.store.tax_indexes.by_country.lookup(norm_country)
            name_rate_cands = [
                t for t in cands_by_country
                if normalize_name(t.name) == norm_name and to_decimal_rate(t.rate) == obs_rate_dec
            ]

            if len(name_rate_cands) == 1:
                cand = name_rate_cands[0]
                return MasterMatchResult(
                    entity_type="tax",
                    status=MatchStatus.MATCHED,
                    master_id=cand.code,
                    candidates=[cand.to_dict()],
                    method="exact_country_name_rate",
                    evidence_ids=get_evidence_ids(["country", "tax_name", "rate"]),
                    matched_fields=["country", "tax_name", "rate"],
                    details={
                        "code": cand.code,
                        "rate": str(to_decimal_rate(cand.rate)),
                        "country": cand.country,
                        "placement": obs.placement,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "TaxMatcher", "phase": "8C"},
                )
            elif len(name_rate_cands) > 1:
                return MasterMatchResult(
                    entity_type="tax",
                    status=MatchStatus.AMBIGUOUS,
                    master_id=None,
                    candidates=[t.to_dict() for t in name_rate_cands],
                    method="ambiguous_country_name_rate",
                    evidence_ids=get_evidence_ids(["country", "tax_name", "rate"]),
                    matched_fields=["country", "tax_name", "rate"],
                    details={
                        "reason": "Multiple master taxes share identical country, name, and rate",
                        "placement": obs.placement,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "TaxMatcher", "phase": "8C"},
                )
            else:
                # Name matched, but rate contradicted in master data
                cands_name_only = [t for t in cands_by_country if normalize_name(t.name) == norm_name]
                if cands_name_only:
                    return MasterMatchResult(
                        entity_type="tax",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[t.to_dict() for t in cands_name_only],
                        method="conflicting_name_and_rate",
                        evidence_ids=get_evidence_ids(["country", "tax_name", "rate"]),
                        matched_fields=["country", "tax_name"],
                        details={
                            "conflict": f"Observed rate '{obs_rate_dec}' conflicts with master rate '{to_decimal_rate(cands_name_only[0].rate)}' for tax '{cands_name_only[0].name}'",
                            "placement": obs.placement,
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "TaxMatcher", "phase": "8C"},
                    )

        # ──────────────────────────────────────────────────────────────────────
        # Priority 4: Exact Country + Normalized Tax Name (Without Rate)
        # ──────────────────────────────────────────────────────────────────────
        if norm_country and norm_name:
            cands_by_country = self.store.tax_indexes.by_country.lookup(norm_country)
            name_cands = [t for t in cands_by_country if normalize_name(t.name) == norm_name]

            if len(name_cands) == 1:
                cand = name_cands[0]
                return MasterMatchResult(
                    entity_type="tax",
                    status=MatchStatus.MATCHED,
                    master_id=cand.code,
                    candidates=[cand.to_dict()],
                    method="exact_country_name",
                    evidence_ids=get_evidence_ids(["country", "tax_name"]),
                    matched_fields=["country", "tax_name"],
                    details={
                        "code": cand.code,
                        "rate": str(to_decimal_rate(cand.rate)),
                        "country": cand.country,
                        "placement": obs.placement,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "TaxMatcher", "phase": "8C"},
                )
            elif len(name_cands) > 1:
                return MasterMatchResult(
                    entity_type="tax",
                    status=MatchStatus.AMBIGUOUS,
                    master_id=None,
                    candidates=[t.to_dict() for t in name_cands],
                    method="ambiguous_country_name",
                    evidence_ids=get_evidence_ids(["country", "tax_name"]),
                    matched_fields=["country", "tax_name"],
                    details={
                        "reason": "Multiple taxes share this normalized name in the specified country",
                        "placement": obs.placement,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "TaxMatcher", "phase": "8C"},
                )

        # ──────────────────────────────────────────────────────────────────────
        # Correction 1 & 7: Country + Rate Alone is NOT Sufficient Tax Identity
        # ──────────────────────────────────────────────────────────────────────
        if norm_country and obs_rate_dec is not None and not norm_type and not norm_name:
            cands = [
                t for t in self.store.tax_indexes.by_country.lookup(norm_country)
                if to_decimal_rate(t.rate) == obs_rate_dec
            ]
            return MasterMatchResult(
                entity_type="tax",
                status=MatchStatus.NO_MATCH,
                master_id=None,
                candidates=[t.to_dict() for t in cands],
                method="none",
                evidence_ids=get_evidence_ids(["country", "rate"]),
                matched_fields=[],
                details={
                    "reason": "Country and rate alone are not sufficient tax identity; tax type or tax name required",
                    "placement": obs.placement,
                    "observed_values": observed_values,
                },
                confidence=None,
                provenance={"matcher": "TaxMatcher", "phase": "8C"},
            )

        # ──────────────────────────────────────────────────────────────────────
        # No Match Found
        # ──────────────────────────────────────────────────────────────────────
        return MasterMatchResult(
            entity_type="tax",
            status=MatchStatus.NO_MATCH,
            master_id=None,
            candidates=[],
            method="none",
            evidence_ids=list(obs.evidence_ids),
            matched_fields=[],
            details={
                "reason": "No master tax matched the supplied observed identity evidence",
                "placement": obs.placement,
                "observed_values": observed_values,
            },
            confidence=None,
            provenance={"matcher": "TaxMatcher", "phase": "8C"},
        )
