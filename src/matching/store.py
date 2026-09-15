"""src/matching/store.py — MasterDataStore repository abstraction.

Phase 8A: Master-Data Loading and Indexing Foundation.

Design Principles:
1. Single Load & Indexing: Loads master files and builds all indexes once into memory.
2. O(1) Dictionary Index Retrieval: Exposes constant-time lookups via hash-map indexes.
3. Multi-Candidate Safety: Ambiguous or shared normalized keys return candidate lists.
4. Downstream Decoupling: Does not implement matching algorithms or heuristics; provides
   the exact, normalized index foundation for future sub-phases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.matching.indexes import (
    ChartOfBooksIndexes,
    PaymentTermIndexes,
    PurchaseOrderIndexes,
    SupplierIndexes,
    TaxIndexes,
)
from src.matching.loaders import (
    load_chart_of_books,
    load_payment_terms,
    load_po_master,
    load_suppliers,
    load_tax_master,
)
from src.matching.models import (
    CompanyRecord,
    LocationRecord,
    PaymentTermRecord,
    PurchaseOrderRecord,
    SupplierRecord,
    TaxRecord,
)
from src.matching.normalization import (
    normalize_code,
    normalize_iban,
    normalize_identifier,
    normalize_name,
    normalize_rate,
    normalize_text,
)


class MasterDataStore:
    """Central repository holding master records and their precomputed in-memory indexes."""

    def __init__(
        self,
        suppliers: Sequence[SupplierRecord] = (),
        chart_of_books: Sequence[CompanyRecord] = (),
        tax_master: Sequence[TaxRecord] = (),
        payment_terms: Sequence[PaymentTermRecord] = (),
        po_master: Sequence[PurchaseOrderRecord] = (),
    ) -> None:
        self.suppliers: Tuple[SupplierRecord, ...] = tuple(suppliers)
        self.chart_of_books: Tuple[CompanyRecord, ...] = tuple(chart_of_books)
        self.tax_master: Tuple[TaxRecord, ...] = tuple(tax_master)
        self.payment_terms: Tuple[PaymentTermRecord, ...] = tuple(payment_terms)
        self.po_master: Tuple[PurchaseOrderRecord, ...] = tuple(po_master)

        self.supplier_indexes: SupplierIndexes = SupplierIndexes()
        self.chart_of_books_indexes: ChartOfBooksIndexes = ChartOfBooksIndexes()
        self.tax_indexes: TaxIndexes = TaxIndexes()
        self.payment_term_indexes: PaymentTermIndexes = PaymentTermIndexes()
        self.po_indexes: PurchaseOrderIndexes = PurchaseOrderIndexes()

        self.rebuild_indexes()

    def rebuild_indexes(self) -> None:
        """Deterministically build all in-memory indexes over current records."""
        self.supplier_indexes = SupplierIndexes.build(self.suppliers)
        self.chart_of_books_indexes = ChartOfBooksIndexes.build(self.chart_of_books)
        self.tax_indexes = TaxIndexes.build(self.tax_master)
        self.payment_term_indexes = PaymentTermIndexes.build(self.payment_terms)
        self.po_indexes = PurchaseOrderIndexes.build(self.po_master)

    @classmethod
    def from_directory(cls, dir_path: Union[str, Path]) -> MasterDataStore:
        """Load all five master-data files from a directory and construct the store."""
        base = Path(dir_path)
        suppliers = load_suppliers(base / "suppliers.json")
        chart_of_books = load_chart_of_books(base / "chart_of_books.json")
        taxes = load_tax_master(base / "tax_master.json")
        payment_terms = load_payment_terms(base / "payment_terms.json")
        pos = load_po_master(base / "po_master.json")

        return cls(
            suppliers=suppliers,
            chart_of_books=chart_of_books,
            tax_master=taxes,
            payment_terms=payment_terms,
            po_master=pos,
        )

    @classmethod
    def from_records(
        cls,
        suppliers: Sequence[SupplierRecord] = (),
        chart_of_books: Sequence[CompanyRecord] = (),
        tax_master: Sequence[TaxRecord] = (),
        payment_terms: Sequence[PaymentTermRecord] = (),
        po_master: Sequence[PurchaseOrderRecord] = (),
    ) -> MasterDataStore:
        """Construct store from in-memory record sequences."""
        return cls(
            suppliers=suppliers,
            chart_of_books=chart_of_books,
            tax_master=tax_master,
            payment_terms=payment_terms,
            po_master=po_master,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # High-level Safe Lookup Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_supplier_by_id(self, supplier_id: Optional[str]) -> Optional[SupplierRecord]:
        """O(1) exact lookup for supplier by supplier_id code."""
        code = normalize_code(supplier_id)
        return self.supplier_indexes.by_supplier_id.get(code)

    def find_suppliers_by_name(self, name: Optional[str]) -> List[SupplierRecord]:
        """O(1) index lookup returning all suppliers matching normalized name (supports duplicates)."""
        norm_name = normalize_name(name)
        return self.supplier_indexes.by_normalized_name.lookup(norm_name)

    def find_suppliers_by_vat_id(self, vat_id: Optional[str]) -> List[SupplierRecord]:
        """O(1) index lookup returning suppliers matching normalized VAT ID."""
        norm_vat = normalize_identifier(vat_id)
        return self.supplier_indexes.by_vat_id.lookup(norm_vat)

    def find_suppliers_by_iban(self, iban: Optional[str]) -> List[SupplierRecord]:
        """O(1) index lookup returning suppliers matching normalized IBAN."""
        norm_iban = normalize_iban(iban)
        return self.supplier_indexes.by_bank_iban.lookup(norm_iban)

    def get_company_by_code(self, company_code: Optional[str]) -> Optional[CompanyRecord]:
        """O(1) exact lookup for company by company_code."""
        code = normalize_code(company_code)
        return self.chart_of_books_indexes.by_company_code.get(code)

    def find_locations_by_code(self, location_code: Optional[str]) -> List[LocationRecord]:
        """O(1) index lookup returning all locations matching location_code (e.g. across multiple BUs)."""
        code = normalize_code(location_code)
        return self.chart_of_books_indexes.by_location_code.lookup(code)

    def find_locations_by_composite(
        self,
        company_code: Optional[str],
        bu_code: Optional[str],
        location_code: Optional[str],
    ) -> List[LocationRecord]:
        """O(1) composite index lookup for (company_code, bu_code, location_code)."""
        c = normalize_code(company_code)
        b = normalize_code(bu_code)
        loc = normalize_code(location_code)
        if not (c and b and loc):
            return []
        return self.chart_of_books_indexes.by_composite_company_bu_location.lookup((c, b, loc))

    def get_tax_by_code(self, tax_code: Optional[str]) -> Optional[TaxRecord]:
        """O(1) exact lookup for tax record by ERP tax_type_code."""
        code = normalize_code(tax_code)
        return self.tax_indexes.by_code.get(code)

    def find_taxes_by_country_rate(self, country: Optional[str], rate: Any) -> List[TaxRecord]:
        """O(1) composite index lookup for taxes by (country, rate)."""
        c = normalize_code(country).upper()
        r = normalize_rate(rate)
        if not c:
            return []
        return self.tax_indexes.by_country_rate.lookup((c, r))

    def get_payment_term_by_id(self, payment_term_id: Optional[str]) -> Optional[PaymentTermRecord]:
        """O(1) exact lookup for payment term by payment_term_id."""
        code = normalize_code(payment_term_id)
        return self.payment_term_indexes.by_payment_term_id.get(code)

    def find_payment_terms_by_days(self, days: Optional[int]) -> List[PaymentTermRecord]:
        """O(1) index lookup for payment terms by days count (e.g. days=0 yields multiple candidates)."""
        if days is None:
            return []
        return self.payment_term_indexes.by_days.lookup(days)

    def find_payment_terms_by_alias(self, alias: Optional[str]) -> List[PaymentTermRecord]:
        """O(1) index lookup for payment terms by text alias."""
        norm_alias = normalize_text(alias)
        return self.payment_term_indexes.by_normalized_alias.lookup(norm_alias)

    def get_po_by_id(self, po_id: Optional[str]) -> Optional[PurchaseOrderRecord]:
        """O(1) exact lookup for purchase order by po_id."""
        code = normalize_code(po_id)
        return self.po_indexes.by_po_id.get(code)

    def find_pos_by_number(self, po_number: Optional[str]) -> List[PurchaseOrderRecord]:
        """O(1) index lookup for purchase orders by po_number."""
        code = normalize_code(po_number)
        return self.po_indexes.by_po_number.lookup(code)

    def find_pos_by_supplier_id(self, supplier_id: Optional[str]) -> List[PurchaseOrderRecord]:
        """O(1) index lookup for purchase orders by supplier_id."""
        code = normalize_code(supplier_id)
        return self.po_indexes.by_supplier_id.lookup(code)
