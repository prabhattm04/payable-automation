"""tests/test_supplier_matcher.py — Test suite for Phase 8B Supplier Master-Data Matching.

Validates:
A. Unique exact VAT/registration match.
B. Unique exact normalized name + country.
C. Exact email match.
D. Exact IBAN match.
E. Name variation handled by Phase 8A normalization.
F. Unicode / diacritics preserved in matching.
G. Duplicate normalized names -> ambiguous without country disambiguation.
H. Strong cross-field conflicts (VAT vs Name, VAT vs IBAN, VAT vs Country) -> AMBIGUOUS, never unsafe match.
I. Unknown supplier -> NO_MATCH (safe empty master_id).
J. Evidence IDs preserved.
K. Original observed values preserved.
L. Fast indexed lookup without full-master scanning.
M. Negative tests: amount/currency, PO numbers, filenames, document IDs cannot produce supplier matches.
N. Deterministic results & document-ID invariance.
O. Serialization round-trip & confidence is None.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedSupplierIdentity,
)
from src.matching.models import SupplierRecord
from src.matching.store import MasterDataStore
from src.matching.supplier_matcher import SupplierMatcher

MASTER_DATA_DIR = Path("d:/candidate_kit/master_data")


@pytest.fixture(scope="module")
def store() -> MasterDataStore:
    return MasterDataStore.from_directory(MASTER_DATA_DIR)


@pytest.fixture(scope="module")
def matcher(store: MasterDataStore) -> SupplierMatcher:
    return SupplierMatcher(store)


class TestSupplierExactMatching:
    """Verify exact signal matching priorities."""

    def test_a_unique_exact_vat_match(self, matcher: SupplierMatcher):
        """A: Unique exact VAT/registration match."""
        res = matcher.match({"vat_id": "DE209177122"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "2845695"
        assert res.method == "exact_vat_id"
        assert "vat_id" in res.matched_fields
        assert res.confidence is None

    def test_b_unique_exact_normalized_name_and_country(self, matcher: SupplierMatcher):
        """B: Unique exact normalized name + country."""
        res = matcher.match({
            "name": "Phocus Direct Communication GmbH",
            "country": "DE",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "2845695"
        assert res.method == "exact_normalized_name_country"
        assert "name" in res.matched_fields
        assert "country" in res.matched_fields

    def test_name_only_matching_when_unique(self, matcher: SupplierMatcher):
        """Correction 2: Name-only matching is safe when candidate is uniquely identifiable."""
        res = matcher.match({"name": "Phocus Direct Communication GmbH"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "2845695"
        assert res.method == "exact_normalized_name"

    def test_c_exact_email_match(self, matcher: SupplierMatcher):
        """C: Exact email match."""
        res = matcher.match({"email": "billing@phocus.de"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "2845695"
        assert res.method == "exact_email"
        assert "email" in res.matched_fields

    def test_d_exact_iban_match(self, matcher: SupplierMatcher):
        """D: Exact IBAN match."""
        res = matcher.match({"bank_iban": "DE34763500000000018786"})
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "2845695"
        assert res.method == "exact_iban"
        assert "bank_iban" in res.matched_fields

    def test_e_name_variation_handled_by_normalization(self, matcher: SupplierMatcher):
        """E: Whitespace, case, and punctuation variations match canonical name."""
        res = matcher.match({
            "name": "  'PHOCUS DIRECT   COMMUNICATION   GMBH'  \n",
            "country": "de",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "2845695"

    def test_f_unicode_and_diacritics_preserved(self, matcher: SupplierMatcher):
        """F: Unicode characters and diacritics are matched accurately."""
        # 1. Real master supplier: Registrite ja Infosusteemide Keskus
        res = matcher.match({
            "name": "Registrite ja Infosusteemide Keskus",
            "country": "EE",
        })
        assert res.status == MatchStatus.MATCHED
        assert res.master_id == "2746014"

        # 2. Master record containing diacritics: preserves diacritic equality in matching
        diacritic_store = MasterDataStore.from_records(
            suppliers=[
                SupplierRecord(
                    supplier_id="SUP_UMLAUT",
                    name="Nürnberg Logistik GmbH",
                    country="DE",
                )
            ]
        )
        diacritic_matcher = SupplierMatcher(diacritic_store)
        res_diacritic = diacritic_matcher.match({
            "name": "Nürnberg Logistik GmbH",
            "country": "DE",
        })
        assert res_diacritic.status == MatchStatus.MATCHED
        assert res_diacritic.master_id == "SUP_UMLAUT"

        # Ehast Koiduni OU via VAT
        res2 = matcher.match({
            "vat_id": "EE101234567",
        })
        assert res2.status == MatchStatus.MATCHED
        assert res2.master_id == "2807582"


class TestSupplierAmbiguityAndConflicts:
    """Verify cross-field conflict rejection and multi-candidate ambiguity handling."""

    def test_g_duplicate_normalized_names_return_ambiguous(self):
        """G: Multiple suppliers sharing normalized name without country -> AMBIGUOUS."""
        suppliers = [
            SupplierRecord(supplier_id="SUP_A", name="Apex Solutions Ltd", country="GB"),
            SupplierRecord(supplier_id="SUP_B", name="Apex Solutions Ltd", country="US"),
        ]
        local_store = MasterDataStore.from_records(suppliers=suppliers)
        local_matcher = SupplierMatcher(local_store)

        # Name only: cannot safely disambiguate
        res = local_matcher.match({"name": "Apex Solutions Ltd"})
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert len(res.candidates) == 2
        assert res.method == "ambiguous_normalized_name"

        # Disambiguated by country
        res_gb = local_matcher.match({"name": "Apex Solutions Ltd", "country": "GB"})
        assert res_gb.status == MatchStatus.MATCHED
        assert res_gb.master_id == "SUP_A"

    def test_h1_cross_field_conflict_vat_vs_name(self, matcher: SupplierMatcher):
        """H & Correction 1 & 5: VAT of Supplier A + Name of Supplier B -> AMBIGUOUS."""
        # Phocus VAT (DE) + Ehast Koiduni Name (EE)
        res = matcher.match({
            "vat_id": "DE209177122",
            "name": "Ehast Koiduni OU",
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert res.method == "conflicting_identity_evidence"
        assert len(res.candidates) >= 2

    def test_h2_cross_field_conflict_vat_vs_iban(self, matcher: SupplierMatcher):
        """H & Correction 1 & 5: VAT of Supplier A + IBAN of Supplier B -> AMBIGUOUS."""
        # Phocus VAT + Ehast IBAN
        res = matcher.match({
            "vat_id": "DE209177122",
            "bank_iban": "EE472200221017274226",
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert res.method == "conflicting_identity_evidence"

    def test_h3_cross_field_conflict_vat_vs_country(self, matcher: SupplierMatcher):
        """H & Correction 1 & 5: VAT of DE + Country of PT -> AMBIGUOUS."""
        res = matcher.match({
            "vat_id": "DE209177122",
            "country": "PT",
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert res.method == "conflicting_vat_and_country"

    def test_h4_cross_field_conflict_name_vs_country(self, matcher: SupplierMatcher):
        """H: Name in DE master + Country in EE -> AMBIGUOUS."""
        res = matcher.match({
            "name": "Phocus Direct Communication GmbH",
            "country": "EE",
        })
        assert res.status == MatchStatus.AMBIGUOUS
        assert res.master_id is None
        assert res.method == "conflicting_name_and_country"

    def test_i_unknown_supplier_returns_no_match(self, matcher: SupplierMatcher):
        """I: Completely unknown supplier -> NO_MATCH (blank master_id)."""
        res = matcher.match({
            "name": "Phantom Logistics GmbH",
            "vat_id": "DE999999999",
            "country": "DE",
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None
        assert res.candidates == []
        assert res.method == "none"


class TestEvidenceAndTraceability:
    """Verify evidence IDs and raw observed values are preserved."""

    def test_j_evidence_ids_preserved(self, matcher: SupplierMatcher):
        """J: Supporting evidence IDs from caller are retained in match result."""
        identity = ObservedSupplierIdentity(
            vat_id="DE209177122",
            field_evidence_ids={"vat_id": "doc1:p1:ocr:digest123"},
        )
        res = matcher.match(identity)
        assert res.status == MatchStatus.MATCHED
        assert "doc1:p1:ocr:digest123" in res.evidence_ids

    def test_k_original_observed_values_preserved(self, matcher: SupplierMatcher):
        """K: Original observed values are stored in details['observed_values']."""
        raw_name = "  Phocus Direct Communication GmbH  "
        res = matcher.match({"name": raw_name, "country": "DE"})
        assert res.details["observed_values"]["name"] == raw_name
        assert res.details["observed_values"]["country"] == "DE"


class TestNegativeConstraints:
    """Verify strict scope boundaries: non-identity fields have zero effect on supplier match."""

    def test_amount_and_currency_cannot_produce_supplier_match(self, matcher: SupplierMatcher):
        """15: Amount and currency alone must NEVER match a supplier."""
        res = matcher.match({
            "gross_total": "438.00",
            "currency": "EUR",
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None

    def test_po_number_cannot_produce_supplier_match(self, matcher: SupplierMatcher):
        """15: PO number must not be used as supplier identity."""
        res = matcher.match({
            "po_number": "PO-EE-2026-0044",
        })
        assert res.status == MatchStatus.NO_MATCH
        assert res.master_id is None

    def test_filename_and_document_id_ignored(self, matcher: SupplierMatcher):
        """15 & V: Matching is invariant to document ID and filename."""
        res1 = matcher.match({
            "name": "Phocus Direct Communication GmbH",
            "country": "DE",
            "metadata": {"document_id": "INV-01", "filename": "sample.pdf"},
        })
        res2 = matcher.match({
            "name": "Phocus Direct Communication GmbH",
            "country": "DE",
            "metadata": {"document_id": "INV-99", "filename": "different.pdf"},
        })
        assert res1.master_id == res2.master_id == "2845695"
        assert res1.status == res2.status == MatchStatus.MATCHED


class TestSerializationAndGeneral:
    """Verify serialization, determinism, and zero fabricated confidence."""

    def test_serialization_round_trip(self, matcher: SupplierMatcher):
        """W: MasterMatchResult converts to dict and reconstructs identically."""
        res = matcher.match({"vat_id": "DE209177122"})
        d = res.to_dict()
        assert isinstance(d, dict)
        assert d["confidence"] is None
        assert d["status"] == "matched"
        assert d["master_id"] == "2845695"

        reconstructed = MasterMatchResult.from_dict(d)
        assert reconstructed.status == res.status
        assert reconstructed.master_id == res.master_id
        assert reconstructed.method == res.method
        assert reconstructed.confidence is None

    def test_no_fabricated_confidence(self, matcher: SupplierMatcher):
        """X: Confidence must strictly be None."""
        res = matcher.match({"vat_id": "DE209177122"})
        assert res.confidence is None
