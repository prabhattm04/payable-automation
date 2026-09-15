"""src/matching/po_matcher.py — Safe, evidence-aware purchase-order master-data matcher.

Phase 8D: Purchase Order Master-Data Matching.

Design Principles:
1. PO Number is Primary:
   - Only an explicit, observed PO number can establish a master PO candidate.
   - Absence of an observed PO number strictly defaults to NO_MATCH.
2. Absolute Rejection of PO Substitution:
   - An observed PO number not in master data (e.g. "2287") must NEVER be substituted
     for a master PO (e.g. "PO-EE-2026-0044"), even if amounts, currencies, and suppliers match.
3. Amounts are NEVER Identity:
   - Invoice gross amounts, line amounts, or currencies alone cannot match a PO.
   - The matcher never calculates totals or derives missing unit prices.
4. Independent Confirmation & Disambiguation:
   - Resolved `supplier_id` and currency serve as confirmation and disambiguation constraints.
   - Contradictory supplier/currency evidence prevents MATCHED (master_id=None).
   - AMBIGUOUS is returned when contradictions represent competing valid identities.
   - NO_MATCH is returned when no candidate satisfies all supplied constraints.
5. Strict Match Contract:
   - Status is strictly MATCHED, AMBIGUOUS, or NO_MATCH.
   - `confidence` is strictly None.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union

from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedPOIdentity,
)
from src.matching.models import PurchaseOrderRecord
from src.matching.normalization import (
    normalize_code,
    normalize_currency,
    normalize_name,
)
from src.matching.store import MasterDataStore


class POMatcher:
    """Matches observed document PO identity against MasterDataStore.po_indexes."""

    def __init__(self, store: MasterDataStore) -> None:
        self.store = store

    def match(
        self,
        observed: Union[ObservedPOIdentity, Dict[str, Any]],
    ) -> MasterMatchResult:
        """Match observed PO identity signals against purchase order master data."""
        if isinstance(observed, dict):
            obs = ObservedPOIdentity.from_dict(observed)
        elif isinstance(observed, ObservedPOIdentity):
            obs = observed
        else:
            raise TypeError(f"observed must be ObservedPOIdentity or dict, got {type(observed)}")

        observed_values: Dict[str, Any] = {
            "po_number": obs.po_number,
            "supplier_id": obs.supplier_id,
            "supplier_name": obs.supplier_name,
            "currency": obs.currency,
            "gross_amount": str(obs.gross_amount) if obs.gross_amount is not None else None,
            "lines": [ln.to_dict() for ln in obs.lines],
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

        # ──────────────────────────────────────────────────────────────────────
        # Rule 1: Absence of Observed PO Number -> NO_MATCH
        # Amounts, currencies, suppliers alone cannot identify a PO!
        # ──────────────────────────────────────────────────────────────────────
        if not obs.po_number or not str(obs.po_number).strip():
            return MasterMatchResult(
                entity_type="po",
                status=MatchStatus.NO_MATCH,
                master_id=None,
                candidates=[],
                method="none",
                evidence_ids=list(obs.evidence_ids),
                matched_fields=[],
                details={
                    "reason": "No explicit PO number observed on document; amounts, currency, and suppliers alone cannot identify a PO",
                    "observed_values": observed_values,
                },
                confidence=None,
                provenance={"matcher": "POMatcher", "phase": "8D"},
            )

        clean_po_num = normalize_code(obs.po_number).strip(" \t\n\r\"'.,;:()[]{}")
        norm_supplier_id = normalize_code(obs.supplier_id)
        norm_curr = normalize_currency(obs.currency)

        # ──────────────────────────────────────────────────────────────────────
        # Rule 2: Query PO Candidates by Exact PO Number
        # ──────────────────────────────────────────────────────────────────────
        # Query index by po_number or po_id (with case-insensitive / normalized variations)
        po_cands: List[PurchaseOrderRecord] = self.store.find_pos_by_number(clean_po_num)
        if not po_cands and clean_po_num:
            po_cands = self.store.find_pos_by_number(clean_po_num.upper())
        if not po_cands and clean_po_num:
            # Check by_po_id exact code
            single_by_id = self.store.get_po_by_id(clean_po_num) or self.store.get_po_by_id(clean_po_num.upper())
            if single_by_id:
                po_cands = [single_by_id]

        if not po_cands:
            # Observed PO number does not exist in master data!
            # ABSOLUTE SAFETY: Do NOT replace with a similar PO or amount-matching PO!
            return MasterMatchResult(
                entity_type="po",
                status=MatchStatus.NO_MATCH,
                master_id=None,
                candidates=[],
                method="none",
                evidence_ids=get_evidence_ids(["po_number"]),
                matched_fields=[],
                details={
                    "reason": f"Observed PO number '{obs.po_number}' does not exist in po_master",
                    "observed_values": observed_values,
                },
                confidence=None,
                provenance={"matcher": "POMatcher", "phase": "8D"},
            )

        # ──────────────────────────────────────────────────────────────────────
        # Rule 3: Single Candidate Evaluation & Contradiction Handling
        # ──────────────────────────────────────────────────────────────────────
        if len(po_cands) == 1:
            cand = po_cands[0]
            matched_f = ["po_number"]
            conflicts: List[str] = []

            # Check supplier constraint
            if norm_supplier_id:
                if cand.supplier_id == norm_supplier_id:
                    matched_f.append("supplier_id")
                else:
                    conflicts.append(
                        f"Observed supplier_id '{norm_supplier_id}' conflicts with master PO supplier_id '{cand.supplier_id}'"
                    )

            # Check currency constraint
            if norm_curr:
                if cand.currency.upper() == norm_curr:
                    matched_f.append("currency")
                else:
                    conflicts.append(
                        f"Observed currency '{norm_curr}' conflicts with master PO currency '{cand.currency}'"
                    )

            # Check line evidence if provided
            if obs.lines and cand.po_lines:
                # Check for line description matches
                cand_line_descs = [normalize_name(l.description) for l in cand.po_lines]
                matched_line_any = False
                for obs_ln in obs.lines:
                    if obs_ln.description:
                        if normalize_name(obs_ln.description) in cand_line_descs:
                            matched_line_any = True
                if matched_line_any:
                    matched_f.append("line_evidence")

            # Handle contradictions
            if conflicts:
                # Check if the contradiction represents competing valid identities
                competing_pos: List[PurchaseOrderRecord] = []
                if norm_supplier_id:
                    competing_pos = self.store.find_pos_by_supplier_id(norm_supplier_id)

                all_candidates = list({p.po_id: p for p in [cand] + competing_pos}.values())

                if competing_pos:
                    # Competing valid identities exist in master data -> AMBIGUOUS
                    return MasterMatchResult(
                        entity_type="po",
                        status=MatchStatus.AMBIGUOUS,
                        master_id=None,
                        candidates=[p.to_dict() for p in all_candidates],
                        method="conflicting_po_and_supplier",
                        evidence_ids=get_evidence_ids(["po_number", "supplier_id", "currency"]),
                        matched_fields=["po_number"],
                        details={
                            "conflicts": conflicts,
                            "reason": "Supplied supplier/currency evidence contradicts unique PO candidate; competing master identities exist",
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "POMatcher", "phase": "8D"},
                    )
                else:
                    # No candidate satisfies all supplied constraints -> NO_MATCH
                    return MasterMatchResult(
                        entity_type="po",
                        status=MatchStatus.NO_MATCH,
                        master_id=None,
                        candidates=[cand.to_dict()],
                        method="conflicting_po_constraints",
                        evidence_ids=get_evidence_ids(["po_number", "supplier_id", "currency"]),
                        matched_fields=["po_number"],
                        details={
                            "conflicts": conflicts,
                            "reason": "No master PO satisfies all supplied constraints (PO number contradicts observed supplier/currency)",
                            "observed_values": observed_values,
                        },
                        confidence=None,
                        provenance={"matcher": "POMatcher", "phase": "8D"},
                    )

            # All supplied constraints are consistent!
            # Determine specific method
            if "supplier_id" in matched_f and "currency" in matched_f:
                method = "exact_po_number_supplier_currency"
            elif "supplier_id" in matched_f:
                method = "exact_po_number_supplier"
            elif "currency" in matched_f:
                method = "exact_po_number_currency"
            elif "line_evidence" in matched_f:
                method = "exact_po_number_line_evidence"
            else:
                method = "exact_po_number"

            return MasterMatchResult(
                entity_type="po",
                status=MatchStatus.MATCHED,
                master_id=cand.po_id,
                candidates=[cand.to_dict()],
                method=method,
                evidence_ids=get_evidence_ids(matched_f),
                matched_fields=matched_f,
                details={
                    "po_id": cand.po_id,
                    "po_number": cand.po_number,
                    "supplier_id": cand.supplier_id,
                    "currency": cand.currency,
                    "observed_values": observed_values,
                },
                confidence=None,
                provenance={"matcher": "POMatcher", "phase": "8D"},
            )

        # ──────────────────────────────────────────────────────────────────────
        # Rule 4: Multiple Candidates for Observed PO Number (Disambiguation)
        # ──────────────────────────────────────────────────────────────────────
        matched_f = ["po_number"]
        filtered_cands = list(po_cands)

        # Disambiguate by supplier_id if provided
        if norm_supplier_id:
            matching_sup = [p for p in filtered_cands if p.supplier_id == norm_supplier_id]
            if matching_sup:
                filtered_cands = matching_sup
                matched_f.append("supplier_id")

        # Disambiguate by currency if provided
        if norm_curr:
            matching_curr = [p for p in filtered_cands if p.currency.upper() == norm_curr]
            if matching_curr:
                filtered_cands = matching_curr
                matched_f.append("currency")

        if len(filtered_cands) == 1:
            cand = filtered_cands[0]
            if "supplier_id" in matched_f and "currency" in matched_f:
                method = "exact_po_number_supplier_currency"
            elif "supplier_id" in matched_f:
                method = "exact_po_number_supplier"
            elif "currency" in matched_f:
                method = "exact_po_number_currency"
            else:
                method = "exact_po_number"

            return MasterMatchResult(
                entity_type="po",
                status=MatchStatus.MATCHED,
                master_id=cand.po_id,
                candidates=[cand.to_dict()],
                method=method,
                evidence_ids=get_evidence_ids(matched_f),
                matched_fields=matched_f,
                details={
                    "po_id": cand.po_id,
                    "po_number": cand.po_number,
                    "supplier_id": cand.supplier_id,
                    "currency": cand.currency,
                    "observed_values": observed_values,
                },
                confidence=None,
                provenance={"matcher": "POMatcher", "phase": "8D"},
            )
        elif len(filtered_cands) > 1:
            # Ambiguous duplicate PO numbers without sufficient discriminator
            return MasterMatchResult(
                entity_type="po",
                status=MatchStatus.AMBIGUOUS,
                master_id=None,
                candidates=[p.to_dict() for p in filtered_cands],
                method="ambiguous_po_number",
                evidence_ids=get_evidence_ids(matched_f),
                matched_fields=matched_f,
                details={
                    "reason": f"Multiple master POs share PO number '{clean_po_num}'; supplier_id or currency required to disambiguate",
                    "candidate_po_ids": [p.po_id for p in filtered_cands],
                    "observed_values": observed_values,
                },
                confidence=None,
                provenance={"matcher": "POMatcher", "phase": "8D"},
            )
        else:
            # Filtering eliminated all candidates due to contradiction
            return MasterMatchResult(
                entity_type="po",
                status=MatchStatus.NO_MATCH,
                master_id=None,
                candidates=[p.to_dict() for p in po_cands],
                method="conflicting_po_constraints",
                evidence_ids=get_evidence_ids(["po_number", "supplier_id", "currency"]),
                matched_fields=["po_number"],
                details={
                    "reason": "No master PO satisfies all supplied constraints",
                    "observed_values": observed_values,
                },
                confidence=None,
                provenance={"matcher": "POMatcher", "phase": "8D"},
            )
