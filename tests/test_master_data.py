"""tests/test_master_data.py — Test suite for Phase 8A Master Data Foundation.

Validates:
A. Loading every supplied master-data file.
B. Validating actual record counts.
C. Validating required identifiers/codes.
D. Detecting malformed records.
E. Detecting duplicate keys.
F. Normalization (whitespace, case, Unicode with preserved diacritics, punctuation, conservative codes).
G. Exact lookups return expected candidates.
H. Duplicate normalized names return multiple candidates without overwriting.
I. Missing lookups safely return no candidates.
J. Source codes are preserved exactly as supplied.
K. Source raw records are preserved.
L. Deterministic index rebuilding.
M. Store loads data once and reuses in-memory indexes.
N. Real master-data ambiguity handling (days=0 in payment terms, duplicate location codes in chart of books).
O. Large synthetic dataset demonstration of O(1) dictionary retrieval vs linear scan.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pytest

from src.matching.indexes import (
    ChartOfBooksIndexes,
    MasterIndex,
    PaymentTermIndexes,
    PurchaseOrderIndexes,
    SupplierIndexes,
    TaxIndexes,
)
from src.matching.loaders import (
    MasterDataDuplicateKeyError,
    MasterDataError,
    MasterDataParseError,
    MasterDataValidationError,
    load_chart_of_books,
    load_payment_terms,
    load_po_master,
    load_suppliers,
    load_tax_master,
)
from src.matching.models import (
    BusinessUnitRecord,
    CompanyRecord,
    LocationRecord,
    PaymentTermRecord,
    PurchaseOrderRecord,
    SupplierRecord,
    TaxRecord,
)
from src.matching.normalization import (
    normalize_ascii_folding,
    normalize_code,
    normalize_currency,
    normalize_iban,
    normalize_identifier,
    normalize_name,
    normalize_rate,
    normalize_text,
)
from src.matching.store import MasterDataStore

MASTER_DATA_DIR = Path("d:/candidate_kit/master_data")


# ══════════════════════════════════════════════════════════════════════════
# 1. Real Master Data Loading & Record Counts (Requirements A, B, J, K)
# ══════════════════════════════════════════════════════════════════════════

class TestMasterDataLoading:
    """Verify loading of supplied real master-data files and actual record schemas."""

    def test_load_all_supplied_files(self):
        """A. Load every supplied master-data file from master_data directory."""
        suppliers = load_suppliers(MASTER_DATA_DIR / "suppliers.json")
        companies = load_chart_of_books(MASTER_DATA_DIR / "chart_of_books.json")
        taxes = load_tax_master(MASTER_DATA_DIR / "tax_master.json")
        payment_terms = load_payment_terms(MASTER_DATA_DIR / "payment_terms.json")
        pos = load_po_master(MASTER_DATA_DIR / "po_master.json")

        assert len(suppliers) > 0
        assert len(companies) > 0
        assert len(taxes) > 0
        assert len(payment_terms) > 0
        assert len(pos) > 0

    def test_validate_actual_record_counts(self):
        """B. Validate exact actual record counts from supplied master-data files."""
        suppliers = load_suppliers(MASTER_DATA_DIR / "suppliers.json")
        assert len(suppliers) == 14

        companies = load_chart_of_books(MASTER_DATA_DIR / "chart_of_books.json")
        assert len(companies) == 1
        assert len(companies[0].business_units) == 6
        total_locations = sum(len(bu.locations) for bu in companies[0].business_units)
        assert total_locations == 6

        taxes = load_tax_master(MASTER_DATA_DIR / "tax_master.json")
        assert len(taxes) == 34

        payment_terms = load_payment_terms(MASTER_DATA_DIR / "payment_terms.json")
        assert len(payment_terms) == 10

        pos = load_po_master(MASTER_DATA_DIR / "po_master.json")
        assert len(pos) == 2

    def test_source_raw_records_preserved(self):
        """K. Source raw records are preserved in `raw_record` for full auditability."""
        suppliers = load_suppliers(MASTER_DATA_DIR / "suppliers.json")
        for sup in suppliers:
            assert isinstance(sup.raw_record, dict)
            assert sup.raw_record["supplier_id"] == sup.supplier_id
            assert sup.raw_record["name"] == sup.name

        taxes = load_tax_master(MASTER_DATA_DIR / "tax_master.json")
        for tax in taxes:
            assert isinstance(tax.raw_record, dict)
            assert tax.raw_record["code"] == tax.code

    def test_codes_preserved_exactly_as_supplied(self):
        """J. Codes are preserved exactly as supplied without modification."""
        taxes = load_tax_master(MASTER_DATA_DIR / "tax_master.json")
        tax_codes = {t.code for t in taxes}
        assert "DE_190_VAT" in tax_codes
        assert "TAX024" in tax_codes
        assert "EST_240_VAT" in tax_codes

        pos = load_po_master(MASTER_DATA_DIR / "po_master.json")
        po_ids = {p.po_id for p in pos}
        assert "PO-EE-2026-0044" in po_ids
        assert "PO-GH-2026-0177" in po_ids

        terms = load_payment_terms(MASTER_DATA_DIR / "payment_terms.json")
        term_ids = {t.payment_term_id for t in terms}
        assert "Net_10" in term_ids
        assert "Monthly_in_advance" in term_ids


# ══════════════════════════════════════════════════════════════════════════
# 2. Schema Validation & Malformed Data Detection (Requirements C, D)
# ══════════════════════════════════════════════════════════════════════════

class TestMasterDataValidation:
    """Verify robust rejection of malformed or invalid master data."""

    def test_detect_missing_required_fields_suppliers(self):
        """C & D: Missing required supplier fields raises MasterDataValidationError."""
        # Missing supplier_id
        with pytest.raises(MasterDataValidationError, match="missing required non-empty 'supplier_id'"):
            load_suppliers({"suppliers": [{"name": "Acme", "country": "DE"}]})

        # Empty supplier_id
        with pytest.raises(MasterDataValidationError, match="missing required non-empty 'supplier_id'"):
            load_suppliers({"suppliers": [{"supplier_id": "   ", "name": "Acme", "country": "DE"}]})

        # Missing name
        with pytest.raises(MasterDataValidationError, match="missing required non-empty 'name'"):
            load_suppliers({"suppliers": [{"supplier_id": "SUP1", "name": "", "country": "DE"}]})

        # Missing country
        with pytest.raises(MasterDataValidationError, match="missing required non-empty 'country'"):
            load_suppliers({"suppliers": [{"supplier_id": "SUP1", "name": "Acme", "country": ""}]})

    def test_detect_malformed_tax_master(self):
        """D: Invalid rate types or negative rates raise validation errors."""
        # Missing code
        with pytest.raises(MasterDataValidationError, match="missing required non-empty 'code'"):
            load_tax_master({"taxes": [{"country": "DE", "tax_type": "VAT", "rate": 19, "name": "VAT"}]})

        # Negative rate
        with pytest.raises(MasterDataValidationError, match="rate cannot be negative"):
            load_tax_master({"taxes": [{"code": "T1", "country": "DE", "tax_type": "VAT", "rate": -5, "name": "VAT"}]})

        # Non-numeric rate
        with pytest.raises(MasterDataValidationError, match="rate must be numeric"):
            load_tax_master({"taxes": [{"code": "T1", "country": "DE", "tax_type": "VAT", "rate": "invalid", "name": "VAT"}]})

    def test_detect_malformed_payment_terms(self):
        """D: Payment term validation detects missing IDs and invalid days."""
        # Negative days
        with pytest.raises(MasterDataValidationError, match="cannot be negative"):
            load_payment_terms({"payment_terms": [{"payment_term_id": "Net_minus", "days": -10}]})

        # Non-int days (e.g. float or bool)
        with pytest.raises(MasterDataValidationError, match="'days' must be an integer"):
            load_payment_terms({"payment_terms": [{"payment_term_id": "Net_30", "days": "30"}]})

        with pytest.raises(MasterDataValidationError, match="'days' must be an integer"):
            load_payment_terms({"payment_terms": [{"payment_term_id": "Net_30", "days": True}]})

    def test_detect_malformed_purchase_orders(self):
        """D: PO master validation detects missing required fields."""
        with pytest.raises(MasterDataValidationError, match="missing required 'po_number'"):
            load_po_master({"purchase_orders": [{"po_id": "PO-1", "supplier_id": "S1"}]})

    def test_parse_error_on_invalid_json(self, tmp_path):
        """D: Malformed JSON syntax raises MasterDataParseError."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ unquoted_key: 123 ", encoding="utf-8")

        with pytest.raises(MasterDataParseError):
            load_suppliers(bad_json)

    def test_parse_error_on_missing_file(self):
        """D: Non-existent file raises MasterDataParseError."""
        with pytest.raises(MasterDataParseError, match="file not found"):
            load_suppliers("non_existent_file_path.json")


# ══════════════════════════════════════════════════════════════════════════
# 3. Duplicate Key Detection (Requirement E)
# ══════════════════════════════════════════════════════════════════════════

class TestDuplicateKeyDetection:
    """Verify unique key validation detects duplicates and raises explicit errors."""

    def test_duplicate_supplier_id_raises(self):
        """E: Duplicate supplier_id in master file raises MasterDataDuplicateKeyError."""
        data = {
            "suppliers": [
                {"supplier_id": "SUP100", "name": "Vendor A", "country": "DE"},
                {"supplier_id": "SUP100", "name": "Vendor B", "country": "DE"},
            ]
        }
        with pytest.raises(MasterDataDuplicateKeyError, match="Duplicate supplier_id detected: 'SUP100'"):
            load_suppliers(data)

    def test_duplicate_tax_code_raises(self):
        """E: Duplicate tax code raises MasterDataDuplicateKeyError."""
        data = {
            "taxes": [
                {"code": "TAX_VAT", "country": "DE", "tax_type": "VAT", "rate": 19, "name": "German VAT 19%"},
                {"code": "TAX_VAT", "country": "EE", "tax_type": "VAT", "rate": 22, "name": "Estonian VAT 22%"},
            ]
        }
        with pytest.raises(MasterDataDuplicateKeyError, match="Duplicate tax code detected: 'TAX_VAT'"):
            load_tax_master(data)

    def test_duplicate_payment_term_id_raises(self):
        """E: Duplicate payment_term_id raises MasterDataDuplicateKeyError."""
        data = {
            "payment_terms": [
                {"payment_term_id": "Net_30", "days": 30, "text_aliases": []},
                {"payment_term_id": "Net_30", "days": 30, "text_aliases": []},
            ]
        }
        with pytest.raises(MasterDataDuplicateKeyError, match="Duplicate payment_term_id detected: 'Net_30'"):
            load_payment_terms(data)

    def test_duplicate_po_id_raises(self):
        """E: Duplicate po_id raises MasterDataDuplicateKeyError."""
        data = {
            "purchase_orders": [
                {"po_id": "PO-001", "po_number": "PO-001", "supplier_id": "S1"},
                {"po_id": "PO-001", "po_number": "PO-002", "supplier_id": "S2"},
            ]
        }
        with pytest.raises(MasterDataDuplicateKeyError, match="Duplicate po_id detected: 'PO-001'"):
            load_po_master(data)

    def test_duplicate_company_code_raises(self):
        """E: Duplicate company_code in chart of books raises MasterDataDuplicateKeyError."""
        data = {
            "companies": [
                {"company_code": "BOLT", "company_name": "Bolt A", "business_units": []},
                {"company_code": "BOLT", "company_name": "Bolt B", "business_units": []},
            ]
        }
        with pytest.raises(MasterDataDuplicateKeyError, match="Duplicate company_code detected: 'BOLT'"):
            load_chart_of_books(data)


# ══════════════════════════════════════════════════════════════════════════
# 4. Real Master Data Ambiguity (Correction 4)
# ══════════════════════════════════════════════════════════════════════════

class TestRealMasterDataAmbiguity:
    """Verify that real-world duplicate keys and ambiguous categories return multiple candidates
    without silently overwriting or losing records.
    """

    def test_payment_terms_days_zero_multiple_candidates(self):
        """Correction 4: payment_terms days=0 produces multiple candidates (Immediate and Monthly_in_advance).
        Neither candidate must be silently overwritten.
        """
        store = MasterDataStore.from_directory(MASTER_DATA_DIR)
        candidates = store.find_payment_terms_by_days(0)

        assert len(candidates) == 2
        term_ids = {term.payment_term_id for term in candidates}
        assert "Immediate" in term_ids
        assert "Monthly_in_advance" in term_ids

    def test_chart_of_books_duplicate_location_code_multiple_candidates(self):
        """Correction 4: chart_of_books duplicated location code LOC_EE_001 produces multiple candidates.
        LOC_EE_001 exists under EE001 (Bolt Technology OU) and EE004 (Bolt Holdings OU).
        Neither candidate must be overwritten.
        """
        store = MasterDataStore.from_directory(MASTER_DATA_DIR)
        locations = store.find_locations_by_code("LOC_EE_001")

        assert len(locations) == 2
        bu_codes = {loc.business_unit_code for loc in locations}
        assert "EE001" in bu_codes
        assert "EE004" in bu_codes

        # Composite lookups disambiguate
        loc_ee001 = store.find_locations_by_composite("BOLTGROUP", "EE001", "LOC_EE_001")
        assert len(loc_ee001) == 1
        assert loc_ee001[0].business_unit_code == "EE001"

        loc_ee004 = store.find_locations_by_composite("BOLTGROUP", "EE004", "LOC_EE_001")
        assert len(loc_ee004) == 1
        assert loc_ee004[0].business_unit_code == "EE004"


# ══════════════════════════════════════════════════════════════════════════
# 5. Normalization Helpers (Requirement F, Correction 2)
# ══════════════════════════════════════════════════════════════════════════

class TestNormalization:
    """Verify normalization helpers preserve identity, diacritics, and codes."""

    def test_whitespace_normalization(self):
        """F: Surrounding whitespace and multiple internal whitespaces are collapsed."""
        assert normalize_text("  hello   world  \t\n") == "hello world"
        assert normalize_text("Acme   \t  Logistics   GmbH") == "acme logistics gmbh"

    def test_case_normalization(self):
        """F: casefold handles uppercase, lowercase, and mixed case uniformly."""
        assert normalize_text("BOLT OPERATIONS UK") == "bolt operations uk"
        assert normalize_name("PHOCUS DIRECT COMMUNICATION GMBH") == "phocus direct communication gmbh"

    def test_unicode_preserves_diacritics_in_canonical_representation(self):
        """Correction 2: Canonical normalization preserves diacritics using Unicode NFKC + casefold.
        Do NOT strip accents in canonical normalization.
        """
        # German umlauts and eszett (casefold maps eszett to ss per Unicode standard)
        assert normalize_name("Nürnberg") == "nürnberg"
        assert normalize_text("Lina-Ammon-Straße") == "lina-ammon-strasse"

        # Estonian ä, õ, ü preserved
        assert normalize_name("Pärnu mnt") == "pärnu mnt"
        assert normalize_name("Registrite ja Infosüsteemide Keskus") == "registrite ja infosüsteemide keskus"
        assert normalize_name("Ehast Koiduni OÜ") == "ehast koiduni oü"

        # Secondary accent folding is isolated and explicitly available when needed
        assert normalize_ascii_folding("Nürnberg") == "nurnberg"
        assert normalize_ascii_folding("Pärnu") == "parnu"
        assert normalize_ascii_folding("Ehast Koiduni OÜ") == "ehast koiduni ou"

    def test_name_normalization_peripheral_punctuation(self):
        """F: Trims outer wrapping punctuation while preserving internal hyphens/dots."""
        assert normalize_name(' "Phocus Direct Communication GmbH" ') == "phocus direct communication gmbh"
        assert normalize_name("Verify Now (Pty) Ltd.") == "verify now (pty) ltd"
        assert normalize_name("Freightways Logistics Ltd,") == "freightways logistics ltd"

    def test_identifier_normalization(self):
        """F: Preserves country prefix and alphanumerics while stripping formatting separators."""
        assert normalize_identifier("DE 209 177 122") == "DE209177122"
        assert normalize_identifier("DE-209-177-122") == "DE209177122"
        assert normalize_identifier("GHA-VAT-887766") == "GHAVAT887766"
        assert normalize_identifier("KE-PIN-P051234567X") == "KEPINP051234567X"
        assert normalize_identifier("  pt 501 234 567  ") == "PT501234567"

    def test_conservative_code_normalization(self):
        """F: Codes themselves must NOT be aggressively normalized."""
        # Case, hyphens, and underscores preserved exactly
        assert normalize_code("PO-EE-2026-0044") == "PO-EE-2026-0044"
        assert normalize_code("  DE_190_VAT  ") == "DE_190_VAT"
        assert normalize_code("LOC_EE_001") == "LOC_EE_001"
        assert normalize_code("Net_10") == "Net_10"

    def test_rate_normalization(self):
        """F: Rates normalize to standard float values."""
        assert normalize_rate(19) == 19.0
        assert normalize_rate(2.5) == 2.5
        assert normalize_rate("20%") == 20.0
        assert normalize_rate("8.1") == 8.1
        assert normalize_rate("0") == 0.0


# ══════════════════════════════════════════════════════════════════════════
# 6. Reusable Indexes & Multi-Candidate Resolution (Requirements G, H, I, L)
# ══════════════════════════════════════════════════════════════════════════

class TestIndexLookups:
    """Verify index lookups return correct candidates, handle duplicates, and remain safe on missing keys."""

    def test_exact_lookup_returns_expected_candidates(self):
        """G: Exact lookup returns expected candidate record."""
        store = MasterDataStore.from_directory(MASTER_DATA_DIR)

        # Suppliers
        sup_phocus = store.get_supplier_by_id("2845695")
        assert sup_phocus is not None
        assert sup_phocus.country == "DE"

        by_vat = store.find_suppliers_by_vat_id("DE209177122")
        assert len(by_vat) == 1
        assert by_vat[0].supplier_id == "2845695"

        # Taxes
        tax_de19 = store.get_tax_by_code("DE_190_VAT")
        assert tax_de19 is not None
        assert tax_de19.rate == 19.0

        taxes_de = store.find_taxes_by_country_rate("DE", 19.0)
        assert len(taxes_de) == 1
        assert taxes_de[0].code == "DE_190_VAT"

        # POs
        po = store.get_po_by_id("PO-EE-2026-0044")
        assert po is not None
        assert po.supplier_id == "2807582"

    def test_duplicate_normalized_names_return_multiple_candidates(self):
        """H: Duplicate normalized names return multiple candidates instead of overwriting."""
        suppliers = [
            SupplierRecord(supplier_id="S101", name="Alpha Logistics GmbH", country="DE"),
            SupplierRecord(supplier_id="S102", name="ALPHA LOGISTICS GMBH", country="AT"),
        ]
        indexes = SupplierIndexes.build(suppliers)
        results = indexes.by_normalized_name.lookup("alpha logistics gmbh")

        assert len(results) == 2
        sup_ids = {s.supplier_id for s in results}
        assert "S101" in sup_ids
        assert "S102" in sup_ids

    def test_missing_lookup_returns_no_candidates_safely(self):
        """I: Missing lookups safely return empty list or None without crashing."""
        store = MasterDataStore.from_directory(MASTER_DATA_DIR)

        assert store.get_supplier_by_id("NON_EXISTENT_ID") is None
        assert store.find_suppliers_by_name("Non Existent Company Name") == []
        assert store.find_suppliers_by_vat_id("XX999999999") == []
        assert store.get_tax_by_code("TAX_DOES_NOT_EXIST") is None
        assert store.find_taxes_by_country_rate("ZZ", 99.0) == []
        assert store.get_payment_term_by_id("NET_9999") is None
        assert store.find_payment_terms_by_days(9999) == []
        assert store.get_po_by_id("PO-NON-EXISTENT") is None

    def test_rebuilding_indexes_produces_deterministic_results(self):
        """L: Rebuilding indexes repeatedly produces identical results."""
        store = MasterDataStore.from_directory(MASTER_DATA_DIR)

        first_supplier_keys = set(store.supplier_indexes.by_supplier_id.keys())
        first_tax_keys = set(store.tax_indexes.by_code.keys())
        first_po_keys = set(store.po_indexes.by_po_id.keys())

        # Rebuild twice
        store.rebuild_indexes()
        store.rebuild_indexes()

        assert set(store.supplier_indexes.by_supplier_id.keys()) == first_supplier_keys
        assert set(store.tax_indexes.by_code.keys()) == first_tax_keys
        assert set(store.po_indexes.by_po_id.keys()) == first_po_keys


# ══════════════════════════════════════════════════════════════════════════
# 7. MasterDataStore Singleton / Reuse (Requirement M)
# ══════════════════════════════════════════════════════════════════════════

class TestMasterDataStore:
    """Verify MasterDataStore loads data once and reuses in-memory structures."""

    def test_store_initialization_and_reuse(self):
        """M: Store holds pre-built indexes in memory without repeated file reads."""
        store = MasterDataStore.from_directory(MASTER_DATA_DIR)

        # Queries run against in-memory index structures
        s1 = store.get_supplier_by_id("2845695")
        s2 = store.get_supplier_by_id("2845695")
        assert s1 is s2  # Exact same cached object instance


# ══════════════════════════════════════════════════════════════════════════
# 8. Large Synthetic Dataset Performance Demonstration (Requirement N, Corrections 1 & 3)
# ══════════════════════════════════════════════════════════════════════════

class TestLargeScalePerformance:
    """Demonstrate that exact index lookup uses O(1) hash maps rather than full linear scans.

    Corrections 1 & 3:
    - Scope O(1) lookup guarantees to dictionary/index retrieval, not downstream candidate evaluation.
    - Do not fail on arbitrary wall-clock sub-millisecond thresholds.
    - Demonstrate index lookup performs constant operations compared to linear scan.
    """

    def test_large_scale_indexed_lookup_performance(self):
        """Generate 100,000 synthetic records and demonstrate O(1) dictionary retrieval vs O(N) scan."""
        n_records = 100_000
        synthetic_suppliers: List[SupplierRecord] = []

        for i in range(n_records):
            synthetic_suppliers.append(
                SupplierRecord(
                    supplier_id=f"SUP_{i:07d}",
                    name=f"Synthetic Vendor Corp {i:07d}",
                    country="DE",
                    vat_id=f"DE{i:09d}",
                )
            )

        # Build index once
        t0 = time.perf_counter()
        indexes = SupplierIndexes.build(synthetic_suppliers)
        build_time = time.perf_counter() - t0

        assert len(indexes.by_supplier_id) == n_records

        # Target record placed at the very end of the collection
        target_id = f"SUP_{n_records - 1:07d}"
        target_name = normalize_name(f"Synthetic Vendor Corp {n_records - 1:07d}")

        # Benchmark 1: Indexed O(1) lookup (by code and by normalized name)
        n_lookups = 1_000
        t_start_index = time.perf_counter()
        for _ in range(n_lookups):
            res = indexes.by_supplier_id.get(target_id)
            assert res is not None
        indexed_duration = time.perf_counter() - t_start_index

        # Benchmark 2: Linear scan O(N)
        # Scan once through all records to find target
        t_start_scan = time.perf_counter()
        linear_found = None
        for item in synthetic_suppliers:
            if item.supplier_id == target_id:
                linear_found = item
                break
        linear_duration = time.perf_counter() - t_start_scan

        assert linear_found is not None

        # Averaged indexed lookup time per lookup
        avg_indexed_time = indexed_duration / n_lookups

        print(
            f"\n[Phase 8A Performance Benchmark - {n_records:,} records]:\n"
            f"  Index build time: {build_time:.3f}s\n"
            f"  Single linear scan O(N): {linear_duration * 1000:.3f} ms\n"
            f"  1,000 indexed lookups: {indexed_duration * 1000:.3f} ms\n"
            f"  Average indexed lookup O(1): {avg_indexed_time * 1e6:.2f} microseconds\n"
            f"  Speedup factor: {linear_duration / avg_indexed_time:.1f}x faster"
        )

        # Core assertions:
        # 1. Target record found correctly via index
        assert indexes.by_supplier_id.get(target_id) is synthetic_suppliers[-1]
        assert indexes.by_normalized_name.lookup(target_name)[0] is synthetic_suppliers[-1]

        # 2. Lookup time per operation is orders of magnitude faster than a full scan
        assert avg_indexed_time < linear_duration
