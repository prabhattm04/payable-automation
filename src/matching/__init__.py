"""src/matching — Master-data loading, validation, normalization, and indexing foundation.

Phase 8A: Master-Data Loading and Indexing Foundation.
"""
from __future__ import annotations

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
    POLineRecord,
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
from src.matching.match_models import (
    MasterMatchResult,
    MatchStatus,
    ObservedBuyerIdentity,
    ObservedPaymentTermIdentity,
    ObservedPOIdentity,
    ObservedPOLineEvidence,
    ObservedSupplierIdentity,
    ObservedTaxIdentity,
    to_decimal_rate,
)
from src.matching.supplier_matcher import SupplierMatcher
from src.matching.buyer_matcher import BuyerMatcher
from src.matching.tax_matcher import TaxMatcher
from src.matching.payment_terms_matcher import PaymentTermsMatcher
from src.matching.po_matcher import POMatcher

__all__ = [
    # Models
    "BusinessUnitRecord",
    "CompanyRecord",
    "LocationRecord",
    "POLineRecord",
    "PaymentTermRecord",
    "PurchaseOrderRecord",
    "SupplierRecord",
    "TaxRecord",
    # Normalization
    "normalize_ascii_folding",
    "normalize_code",
    "normalize_currency",
    "normalize_iban",
    "normalize_identifier",
    "normalize_name",
    "normalize_rate",
    "normalize_text",
    # Indexes
    "ChartOfBooksIndexes",
    "MasterIndex",
    "PaymentTermIndexes",
    "PurchaseOrderIndexes",
    "SupplierIndexes",
    "TaxIndexes",
    # Loaders & Errors
    "MasterDataDuplicateKeyError",
    "MasterDataError",
    "MasterDataParseError",
    "MasterDataValidationError",
    "load_chart_of_books",
    "load_payment_terms",
    "load_po_master",
    "load_suppliers",
    "load_tax_master",
    # Store
    "MasterDataStore",
    # Matching (Phase 8B, 8C, 8D)
    "BuyerMatcher",
    "MasterMatchResult",
    "MatchStatus",
    "ObservedBuyerIdentity",
    "ObservedPaymentTermIdentity",
    "ObservedPOIdentity",
    "ObservedPOLineEvidence",
    "ObservedSupplierIdentity",
    "ObservedTaxIdentity",
    "POMatcher",
    "PaymentTermsMatcher",
    "SupplierMatcher",
    "TaxMatcher",
    "to_decimal_rate",
]
