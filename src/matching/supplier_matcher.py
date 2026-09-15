"""src/matching/supplier_matcher.py — Safe, evidence-aware supplier master-data matcher.

Phase 8B: Supplier + Buyer Master-Data Matching.

Design Principles:
1. Candidate Generation Priority:
   - 1. Exact VAT / registration identifier
   - 2. Exact normalized name + country (or exact normalized name if uniquely identifiable)
   - 3. Exact email
   - 4. Exact bank IBAN
   - 5. Composite multi-signal identity
2. Strong Conflict Safety:
   - Never allow an identifier (such as VAT) to override contradictory independently observed
     identity evidence (e.g. VAT points to Supplier A, but IBAN or Name points to Supplier B).
   - Strong cross-field conflicts must remain AMBIGUOUS or NO_MATCH.
3. Separation of Name-Only vs Name+Country:
   - Name-only is matched only if the candidate is uniquely identifiable across the entire master.
4. No Fabricated Confidence & No Guessing:
   - Status is strictly MATCHED, AMBIGUOUS, or NO_MATCH.
   - `confidence` is strictly None.
5. Evidence & Provenance Preservation:
   - Retains supporting `evidence_ids` and stores original observed values in `details`.
6. Strict Invariance:
   - Ignores PO numbers, amounts, currencies, filenames, and document IDs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedSupplierIdentity,
)
from src.matching.models import SupplierRecord
from src.matching.normalization import (
    normalize_code,
    normalize_iban,
    normalize_identifier,
    normalize_name,
    normalize_text,
)
from src.matching.store import MasterDataStore


class SupplierMatcher:
    """Matches observed document supplier identity against MasterDataStore."""

    def __init__(self, store: MasterDataStore) -> None:
        self.store = store

    def match(
        self,
        observed: Union[ObservedSupplierIdentity, Dict[str, Any]],
    ) -> MasterMatchResult:
        """Match an observed supplier identity against the supplier master data."""
        if isinstance(observed, dict):
            obs = ObservedSupplierIdentity.from_dict(observed)
        elif isinstance(observed, ObservedSupplierIdentity):
            obs = observed
        else:
            raise TypeError(f"observed must be ObservedSupplierIdentity or dict, got {type(observed)}")

        # Track observed values for lossless audit details
        observed_values: Dict[str, Any] = {
            "name": obs.name,
            "vat_id": obs.vat_id,
            "country": obs.country,
            "email": obs.email,
            "bank_iban": obs.bank_iban,
            "address": obs.address,
        }

        # Normalize individual observed fields
        norm_vat = normalize_identifier(obs.vat_id) if obs.vat_id else ""
        norm_name = normalize_name(obs.name) if obs.name else ""
        norm_country = normalize_code(obs.country).upper() if obs.country else ""
        norm_email = normalize_text(obs.email) if obs.email else ""
        norm_iban = normalize_iban(obs.bank_iban) if obs.bank_iban else ""

        # Step 1: Query indexes for each observed signal
        vat_candidates: List[SupplierRecord] = (
            self.store.find_suppliers_by_vat_id(norm_vat) if norm_vat else []
        )
        name_candidates: List[SupplierRecord] = (
            self.store.find_suppliers_by_name(norm_name) if norm_name else []
        )
        email_candidates: List[SupplierRecord] = (
            self.store.supplier_indexes.by_email.lookup(norm_email) if norm_email else []
        )
        iban_candidates: List[SupplierRecord] = (
            self.store.find_suppliers_by_iban(norm_iban) if norm_iban else []
        )

        # Helper to get supporting evidence IDs
        def get_evidence_ids(fields: List[str]) -> List[str]:
            ev_set: Set[str] = set()
            for f in fields:
                if f in obs.field_evidence_ids and obs.field_evidence_ids[f]:
                    ev_set.add(obs.field_evidence_ids[f])
            # If no field-specific IDs were assigned, fall back to global evidence_ids
            if not ev_set and obs.evidence_ids:
                return list(obs.evidence_ids)
            return sorted(ev_set)

        # Step 2: Check for strong cross-field conflicts between independent fields
        # Strong fields: vat_id, bank_iban, email, and name
        observed_signals: List[Tuple[str, List[SupplierRecord]]] = []
        if norm_vat and vat_candidates:
            observed_signals.append(("vat_id", vat_candidates))
        if norm_iban and iban_candidates:
            observed_signals.append(("bank_iban", iban_candidates))
        if norm_email and email_candidates:
            observed_signals.append(("email", email_candidates))
        if norm_name and name_candidates:
            observed_signals.append(("name", name_candidates))

        # Check if any two observed strong signals point to completely disjoint sets of suppliers
        for i in range(len(observed_signals)):
            for j in range(i + 1, len(observed_signals)):
                field_a, cands_a = observed_signals[i]
                field_b, cands_b = observed_signals[j]
                ids_a = {c.supplier_id for c in cands_a}
                ids_b = {c.supplier_id for c in cands_b}
                if not (ids_a & ids_b):
                    # Strong contradiction! (e.g. VAT matches Vendor A, but IBAN matches Vendor B)
                    all_conflicting = list({c.supplier_id: c for c in cands_a + cands_b}.values())
                    return MasterMatchResult(
                        entity_type="supplier",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[c.to_dict() for c in all_conflicting],
                        method="conflicting_identity_evidence",
                        evidence_ids=get_evidence_ids([field_a, field_b]),
                        matched_fields=[field_a, field_b],
                        details={
                            "conflict": f"Contradictory evidence between '{field_a}' and '{field_b}'",
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                    )

        # Step 3: Candidate resolution in order of priority

        # ── Priority 1: Exact VAT / Registration Identifier ────────────────────
        if norm_vat:
            if vat_candidates:
                # Check consistency with observed country if country was provided
                if norm_country:
                    country_consistent = [c for c in vat_candidates if c.country.upper() == norm_country]
                    if not country_consistent:
                        # Country contradicts the VAT record
                        return MasterMatchResult(
                            entity_type="supplier",
                            status=MatchStatus.AMBIGUOUS,
                            master_id=None,
                            candidates=[c.to_dict() for c in vat_candidates],
                            method="conflicting_vat_and_country",
                            evidence_ids=get_evidence_ids(["vat_id", "country"]),
                            matched_fields=["vat_id"],
                            details={
                                "conflict": f"Observed country '{norm_country}' conflicts with VAT country '{vat_candidates[0].country}'",
                                "observed_values": observed_values,
                            },
                            confidence=None,
                            provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                        )
                    vat_candidates = country_consistent

                if len(vat_candidates) == 1:
                    match_rec = vat_candidates[0]
                    matched_f = ["vat_id"]
                    if norm_country and match_rec.country.upper() == norm_country:
                        matched_f.append("country")
                    if norm_name and normalize_name(match_rec.name) == norm_name:
                        matched_f.append("name")

                    return MasterMatchResult(
                        entity_type="supplier",
                        status=MatchStatus.MATCHED,
                        master_id=match_rec.supplier_id,
                        candidates=[match_rec.to_dict()],
                        method="exact_vat_id",
                        evidence_ids=get_evidence_ids(matched_f),
                        matched_fields=matched_f,
                        details={
                            "matched_supplier_id": match_rec.supplier_id,
                            "matched_name": match_rec.name,
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                    )
                elif len(vat_candidates) > 1:
                    # Ambiguous VAT
                    return MasterMatchResult(
                        entity_type="supplier",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[c.to_dict() for c in vat_candidates],
                        method="ambiguous_vat_id",
                        evidence_ids=get_evidence_ids(["vat_id"]),
                        matched_fields=["vat_id"],
                        details={
                            "reason": "Multiple suppliers found with identical VAT ID",
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                    )

        # ── Priority 2: Exact Normalized Name (+ Country) ─────────────────────
        if norm_name:
            if norm_country:
                # Name + Country matching
                cands_with_country = [
                    c for c in name_candidates if c.country.upper() == norm_country
                ]
                if len(cands_with_country) == 1:
                    match_rec = cands_with_country[0]
                    return MasterMatchResult(
                        entity_type="supplier",
                        status=MatchStatus.MATCHED,
                        master_id=match_rec.supplier_id,
                        candidates=[match_rec.to_dict()],
                        method="exact_normalized_name_country",
                        evidence_ids=get_evidence_ids(["name", "country"]),
                        matched_fields=["name", "country"],
                        details={
                            "matched_supplier_id": match_rec.supplier_id,
                            "matched_name": match_rec.name,
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                    )
                elif len(cands_with_country) > 1:
                    return MasterMatchResult(
                        entity_type="supplier",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[c.to_dict() for c in cands_with_country],
                        method="ambiguous_name_country",
                        evidence_ids=get_evidence_ids(["name", "country"]),
                        matched_fields=["name", "country"],
                        details={
                            "reason": "Multiple suppliers found with identical name and country",
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                    )
                elif name_candidates:
                    # Name matched in master data, but not with the observed country
                    return MasterMatchResult(
                        entity_type="supplier",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[c.to_dict() for c in name_candidates],
                        method="conflicting_name_and_country",
                        evidence_ids=get_evidence_ids(["name", "country"]),
                        matched_fields=["name"],
                        details={
                            "conflict": f"Supplier name matched but country '{norm_country}' differed from master country '{name_candidates[0].country}'",
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                    )
            else:
                # Name-only matching: safe ONLY when candidate is uniquely identifiable
                if len(name_candidates) == 1:
                    match_rec = name_candidates[0]
                    return MasterMatchResult(
                        entity_type="supplier",
                        status=MatchStatus.MATCHED,
                        master_id=match_rec.supplier_id,
                        candidates=[match_rec.to_dict()],
                        method="exact_normalized_name",
                        evidence_ids=get_evidence_ids(["name"]),
                        matched_fields=["name"],
                        details={
                            "matched_supplier_id": match_rec.supplier_id,
                            "matched_name": match_rec.name,
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                    )
                elif len(name_candidates) > 1:
                    # Ambiguous name without country disambiguation
                    return MasterMatchResult(
                        entity_type="supplier",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[c.to_dict() for c in name_candidates],
                        method="ambiguous_normalized_name",
                        evidence_ids=get_evidence_ids(["name"]),
                        matched_fields=["name"],
                        details={
                            "reason": "Multiple suppliers found with identical name; country required for disambiguation",
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                    )

        # ── Priority 3: Exact Email ───────────────────────────────────────────
        if norm_email:
            if len(email_candidates) == 1:
                match_rec = email_candidates[0]
                if norm_country and match_rec.country.upper() != norm_country:
                    return MasterMatchResult(
                        entity_type="supplier",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[match_rec.to_dict()],
                        method="conflicting_email_and_country",
                        evidence_ids=get_evidence_ids(["email", "country"]),
                        matched_fields=["email"],
                        details={"conflict": "Observed country conflicts with supplier email record"},
                        confidence=None,
                        provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                    )
                return MasterMatchResult(
                    entity_type="supplier",
                    status=MatchStatus.MATCHED,
                    master_id=match_rec.supplier_id,
                    candidates=[match_rec.to_dict()],
                    method="exact_email",
                    evidence_ids=get_evidence_ids(["email"]),
                    matched_fields=["email"],
                    details={
                        "matched_supplier_id": match_rec.supplier_id,
                        "matched_name": match_rec.name,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                )
            elif len(email_candidates) > 1:
                return MasterMatchResult(
                    entity_type="supplier",
                    status=MatchStatus.AMBIGUOUS,
                    master_id=None,
                    candidates=[c.to_dict() for c in email_candidates],
                    method="ambiguous_email",
                    evidence_ids=get_evidence_ids(["email"]),
                    matched_fields=["email"],
                    details={"reason": "Multiple suppliers share observed email"},
                    confidence=None,
                    provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                )

        # ── Priority 4: Exact Bank IBAN ───────────────────────────────────────
        if norm_iban:
            if len(iban_candidates) == 1:
                match_rec = iban_candidates[0]
                if norm_country and match_rec.country.upper() != norm_country:
                    return MasterMatchResult(
                        entity_type="supplier",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[match_rec.to_dict()],
                        method="conflicting_iban_and_country",
                        evidence_ids=get_evidence_ids(["bank_iban", "country"]),
                        matched_fields=["bank_iban"],
                        details={"conflict": "Observed country conflicts with supplier IBAN country"},
                        confidence=None,
                        provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                    )
                return MasterMatchResult(
                    entity_type="supplier",
                    status=MatchStatus.MATCHED,
                    master_id=match_rec.supplier_id,
                    candidates=[match_rec.to_dict()],
                    method="exact_iban",
                    evidence_ids=get_evidence_ids(["bank_iban"]),
                    matched_fields=["bank_iban"],
                    details={
                        "matched_supplier_id": match_rec.supplier_id,
                        "matched_name": match_rec.name,
                        "observed_values": observed_values,
                    },
                    confidence=None,
                    provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                )
            elif len(iban_candidates) > 1:
                return MasterMatchResult(
                    entity_type="supplier",
                    status=MatchStatus.AMBIGUOUS,
                    master_id=None,
                    candidates=[c.to_dict() for c in iban_candidates],
                    method="ambiguous_iban",
                    evidence_ids=get_evidence_ids(["bank_iban"]),
                    matched_fields=["bank_iban"],
                    details={"reason": "Multiple suppliers share observed IBAN"},
                    confidence=None,
                    provenance={"matcher": "SupplierMatcher", "phase": "8B"},
                )

        # ── No Match Found ───────────────────────────────────────────────────
        return MasterMatchResult(
            entity_type="supplier",
            status=MatchStatus.NO_MATCH,
            master_id=None,
            candidates=[],
            method="none",
            evidence_ids=list(obs.evidence_ids),
            matched_fields=[],
            details={
                "reason": "No master supplier matched observed identity evidence",
                "observed_values": observed_values,
            },
            confidence=None,
            provenance={"matcher": "SupplierMatcher", "phase": "8B"},
        )
