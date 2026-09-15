"""src/matching/indexes.py — Reusable in-memory indexes for master data.

Phase 8A: Master-Data Loading and Indexing Foundation.

Design Principles:
1. O(1) Dictionary Lookup: Indexes provide constant-time retrieval via hash maps.
2. Ambiguity & Duplicate Candidate Preservation:
   - Index lookups return candidate lists (`list[V]`).
   - Multiple master records sharing a normalized key (e.g. days=0 in payment terms,
     or duplicate location codes across BUs) are NEVER silently overwritten.
3. Conservative Exact Matching:
   - Codes map directly without modification.
   - Missing lookups safely return empty candidate lists `[]` or `None`.
4. Determinism: Building indexes over the same records produces identical index state.
"""
from __future__ import annotations

from typing import (
    Any,
    Dict,
    Generic,
    ItemsView,
    KeysView,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
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
    normalize_code,
    normalize_iban,
    normalize_identifier,
    normalize_name,
    normalize_rate,
    normalize_text,
)

K = TypeVar("K")
V = TypeVar("V")


class MasterIndex(Generic[K, V]):
    """Generic hash-map index from key `K` to a list of candidate records `list[V]`.
    Guarantees O(1) retrieval time and prevents silent overwriting of duplicate keys.
    """

    def __init__(self) -> None:
        self._index: Dict[K, List[V]] = {}
        self._seen: Dict[K, set[int]] = {}

    def add(self, key: Optional[K], record: V) -> None:
        """Add a record to the index under `key`. Ignores None and empty string keys."""
        if key is None or key == "":
            return
        rec_id = id(record)
        seen_set = self._seen.setdefault(key, set())
        if rec_id not in seen_set:
            seen_set.add(rec_id)
            self._index.setdefault(key, []).append(record)

    def lookup(self, key: Optional[K]) -> List[V]:
        """O(1) dictionary retrieval. Returns a list of candidate records, or [] if key not found."""
        if key is None or key == "":
            return []
        res = self._index.get(key)
        return list(res) if res is not None else []

    def get_unique(self, key: Optional[K]) -> Optional[V]:
        """Return the unique matching record if exactly one candidate exists, else None."""
        candidates = self.lookup(key)
        return candidates[0] if len(candidates) == 1 else None

    def __contains__(self, key: K) -> bool:
        return key in self._index

    def __len__(self) -> int:
        """Number of distinct keys indexed."""
        return len(self._index)

    def total_entries(self) -> int:
        """Total number of record references across all indexed keys."""
        return sum(len(c) for c in self._index.values())

    def keys(self) -> KeysView[K]:
        return self._index.keys()

    def items(self) -> ItemsView[K, List[V]]:
        return self._index.items()


class SupplierIndexes:
    """Precomputed in-memory indexes for supplier master data."""

    def __init__(self) -> None:
        self.by_supplier_id: Dict[str, SupplierRecord] = {}
        self.by_normalized_name: MasterIndex[str, SupplierRecord] = MasterIndex()
        self.by_vat_id: MasterIndex[str, SupplierRecord] = MasterIndex()
        self.by_bank_iban: MasterIndex[str, SupplierRecord] = MasterIndex()
        self.by_email: MasterIndex[str, SupplierRecord] = MasterIndex()
        self.by_country: MasterIndex[str, SupplierRecord] = MasterIndex()

    @classmethod
    def build(cls, suppliers: Sequence[SupplierRecord]) -> SupplierIndexes:
        idx = cls()
        for sup in suppliers:
            # Exact supplier_id code lookup
            code = normalize_code(sup.supplier_id)
            if code:
                idx.by_supplier_id[code] = sup

            # Normalized name index
            norm_name = normalize_name(sup.name)
            if norm_name:
                idx.by_normalized_name.add(norm_name, sup)

            # Normalized VAT ID index (only if populated)
            norm_vat = normalize_identifier(sup.vat_id)
            if norm_vat:
                idx.by_vat_id.add(norm_vat, sup)

            # Normalized IBAN index (only if populated)
            norm_iban = normalize_iban(sup.bank_iban)
            if norm_iban:
                idx.by_bank_iban.add(norm_iban, sup)

            # Normalized email
            norm_email = normalize_text(sup.email)
            if norm_email:
                idx.by_email.add(norm_email, sup)

            # Country index
            country = normalize_code(sup.country).upper()
            if country:
                idx.by_country.add(country, sup)

        return idx


class ChartOfBooksIndexes:
    """Precomputed in-memory indexes for chart of books organizational data."""

    def __init__(self) -> None:
        self.by_company_code: Dict[str, CompanyRecord] = {}
        self.by_bu_code: MasterIndex[str, BusinessUnitRecord] = MasterIndex()
        self.by_location_code: MasterIndex[str, LocationRecord] = MasterIndex()
        self.by_composite_bu_location: MasterIndex[Tuple[str, str], LocationRecord] = MasterIndex()
        self.by_composite_company_bu_location: MasterIndex[Tuple[str, str, str], LocationRecord] = MasterIndex()
        self.by_normalized_bu_name: MasterIndex[str, BusinessUnitRecord] = MasterIndex()
        self.by_normalized_location_name: MasterIndex[str, LocationRecord] = MasterIndex()

    @classmethod
    def build(cls, companies: Sequence[CompanyRecord]) -> ChartOfBooksIndexes:
        idx = cls()
        for comp in companies:
            c_code = normalize_code(comp.company_code)
            if c_code:
                idx.by_company_code[c_code] = comp

            for bu in comp.business_units:
                bu_code = normalize_code(bu.business_unit_code)
                if bu_code:
                    idx.by_bu_code.add(bu_code, bu)

                norm_bu_name = normalize_name(bu.business_unit_name)
                if norm_bu_name:
                    idx.by_normalized_bu_name.add(norm_bu_name, bu)

                for loc in bu.locations:
                    loc_code = normalize_code(loc.location_code)
                    if loc_code:
                        # Note: Same location code (e.g. LOC_EE_001) can belong to multiple BUs!
                        idx.by_location_code.add(loc_code, loc)

                    if bu_code and loc_code:
                        idx.by_composite_bu_location.add((bu_code, loc_code), loc)

                    if c_code and bu_code and loc_code:
                        idx.by_composite_company_bu_location.add((c_code, bu_code, loc_code), loc)

                    norm_loc_name = normalize_name(loc.location_name)
                    if norm_loc_name:
                        idx.by_normalized_location_name.add(norm_loc_name, loc)

        return idx


class TaxIndexes:
    """Precomputed in-memory indexes for tax master data."""

    def __init__(self) -> None:
        self.by_code: Dict[str, TaxRecord] = {}
        self.by_country_rate: MasterIndex[Tuple[str, float], TaxRecord] = MasterIndex()
        self.by_country_type_rate: MasterIndex[Tuple[str, str, float], TaxRecord] = MasterIndex()
        self.by_country: MasterIndex[str, TaxRecord] = MasterIndex()
        self.by_normalized_name: MasterIndex[str, TaxRecord] = MasterIndex()

    @classmethod
    def build(cls, taxes: Sequence[TaxRecord]) -> TaxIndexes:
        idx = cls()
        for tax in taxes:
            code = normalize_code(tax.code)
            if code:
                idx.by_code[code] = tax

            country = normalize_code(tax.country).upper()
            rate = normalize_rate(tax.rate)
            tax_type = normalize_code(tax.tax_type).upper()

            if country:
                idx.by_country.add(country, tax)
                idx.by_country_rate.add((country, rate), tax)
                if tax_type:
                    idx.by_country_type_rate.add((country, tax_type, rate), tax)

            norm_name = normalize_name(tax.name)
            if norm_name:
                idx.by_normalized_name.add(norm_name, tax)

        return idx


class PaymentTermIndexes:
    """Precomputed in-memory indexes for payment terms."""

    def __init__(self) -> None:
        self.by_payment_term_id: Dict[str, PaymentTermRecord] = {}
        self.by_days: MasterIndex[int, PaymentTermRecord] = MasterIndex()
        self.by_normalized_alias: MasterIndex[str, PaymentTermRecord] = MasterIndex()

    @classmethod
    def build(cls, payment_terms: Sequence[PaymentTermRecord]) -> PaymentTermIndexes:
        idx = cls()
        for term in payment_terms:
            term_id = normalize_code(term.payment_term_id)
            if term_id:
                idx.by_payment_term_id[term_id] = term

            # Note: Multiple terms can have the same days count (e.g. days=0 has Immediate and Monthly_in_advance)
            idx.by_days.add(term.days, term)

            for alias in term.text_aliases:
                norm_alias = normalize_text(alias)
                if norm_alias:
                    idx.by_normalized_alias.add(norm_alias, term)

        return idx


class PurchaseOrderIndexes:
    """Precomputed in-memory indexes for purchase orders."""

    def __init__(self) -> None:
        self.by_po_id: Dict[str, PurchaseOrderRecord] = {}
        self.by_po_number: MasterIndex[str, PurchaseOrderRecord] = MasterIndex()
        self.by_supplier_id: MasterIndex[str, PurchaseOrderRecord] = MasterIndex()

    @classmethod
    def build(cls, pos: Sequence[PurchaseOrderRecord]) -> PurchaseOrderIndexes:
        idx = cls()
        for po in pos:
            po_id = normalize_code(po.po_id)
            if po_id:
                idx.by_po_id[po_id] = po

            po_num = normalize_code(po.po_number)
            if po_num:
                idx.by_po_number.add(po_num, po)

            sup_id = normalize_code(po.supplier_id)
            if sup_id:
                idx.by_supplier_id.add(sup_id, po)

        return idx
