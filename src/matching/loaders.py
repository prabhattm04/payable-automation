"""src/matching/loaders.py — Robust loading and schema validation for master data.

Phase 8A: Master-Data Loading and Indexing Foundation.

Design Principles:
1. Strict Validation: Validates types, required fields, and structural shapes.
2. Explicit Error Diagnostics: Raises actionable exceptions on malformed data or duplicate unique keys.
3. No Silent Modification: Malformed data is rejected rather than silently repaired or fabricated.
4. Lossless Provenance: Retains the original raw dictionary for each record.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO, Tuple, Union

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


class MasterDataError(Exception):
    """Base exception for all master-data loading and validation errors."""
    pass


class MasterDataParseError(MasterDataError):
    """Raised when source JSON cannot be parsed or file cannot be read."""
    pass


class MasterDataValidationError(MasterDataError):
    """Raised when master data violates schema, types, or required fields."""
    pass


class MasterDataDuplicateKeyError(MasterDataValidationError):
    """Raised when an identifier that must be globally unique is duplicated."""
    pass


def _read_json_payload(source: Union[str, Path, TextIO, Dict[str, Any]]) -> Dict[str, Any]:
    """Helper to parse a JSON payload from file path, open stream, or direct dictionary."""
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise MasterDataParseError(f"Master-data file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as err:
            raise MasterDataParseError(f"Failed to parse JSON in '{path}': {err}") from err
        except OSError as err:
            raise MasterDataParseError(f"Failed to read file '{path}': {err}") from err
    if hasattr(source, "read"):
        try:
            return json.load(source)
        except json.JSONDecodeError as err:
            raise MasterDataParseError(f"Failed to parse JSON stream: {err}") from err
    raise MasterDataParseError(f"Unsupported source type: {type(source)}")


def load_suppliers(source: Union[str, Path, TextIO, Dict[str, Any]]) -> Tuple[SupplierRecord, ...]:
    """Load, validate, and convert suppliers from JSON source into immutable SupplierRecords."""
    data = _read_json_payload(source)
    if not isinstance(data, dict):
        raise MasterDataValidationError("Suppliers payload root must be a JSON object")

    raw_suppliers = data.get("suppliers")
    if not isinstance(raw_suppliers, list):
        raise MasterDataValidationError("Suppliers payload must contain a 'suppliers' list")

    seen_ids: Set[str] = set()
    records: List[SupplierRecord] = []

    for idx, item in enumerate(raw_suppliers):
        if not isinstance(item, dict):
            raise MasterDataValidationError(f"Supplier item at index {idx} must be an object, got {type(item)}")

        supplier_id = item.get("supplier_id")
        if not isinstance(supplier_id, str) or not supplier_id.strip():
            raise MasterDataValidationError(f"Supplier item at index {idx} missing required non-empty 'supplier_id'")

        clean_id = supplier_id.strip()
        if clean_id in seen_ids:
            raise MasterDataDuplicateKeyError(f"Duplicate supplier_id detected: '{clean_id}' at index {idx}")
        seen_ids.add(clean_id)

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise MasterDataValidationError(f"Supplier '{clean_id}' missing required non-empty 'name'")

        country = item.get("country", "")
        if not isinstance(country, str) or not country.strip():
            raise MasterDataValidationError(f"Supplier '{clean_id}' missing required non-empty 'country'")

        rec = SupplierRecord(
            supplier_id=clean_id,
            name=name.strip(),
            vat_id=str(item.get("vat_id") or "").strip(),
            country=country.strip(),
            email=str(item.get("email") or "").strip(),
            address=str(item.get("address") or "").strip(),
            bank_iban=str(item.get("bank_iban") or "").strip(),
            raw_record=dict(item),
        )
        records.append(rec)

    return tuple(records)


def load_chart_of_books(source: Union[str, Path, TextIO, Dict[str, Any]]) -> Tuple[CompanyRecord, ...]:
    """Load, validate, and convert chart of books organizational structure from JSON source."""
    data = _read_json_payload(source)
    if not isinstance(data, dict):
        raise MasterDataValidationError("Chart of books root must be a JSON object")

    raw_companies = data.get("companies")
    if not isinstance(raw_companies, list):
        raise MasterDataValidationError("Chart of books must contain a 'companies' list")

    seen_company_codes: Set[str] = set()
    companies: List[CompanyRecord] = []

    for c_idx, c_item in enumerate(raw_companies):
        if not isinstance(c_item, dict):
            raise MasterDataValidationError(f"Company entry at index {c_idx} must be a JSON object")

        c_code = c_item.get("company_code")
        if not isinstance(c_code, str) or not c_code.strip():
            raise MasterDataValidationError(f"Company at index {c_idx} missing required 'company_code'")
        clean_c_code = c_code.strip()

        if clean_c_code in seen_company_codes:
            raise MasterDataDuplicateKeyError(f"Duplicate company_code detected: '{clean_c_code}'")
        seen_company_codes.add(clean_c_code)

        c_name = c_item.get("company_name")
        if not isinstance(c_name, str) or not c_name.strip():
            raise MasterDataValidationError(f"Company '{clean_c_code}' missing required 'company_name'")

        raw_bus = c_item.get("business_units")
        if not isinstance(raw_bus, list):
            raise MasterDataValidationError(f"Company '{clean_c_code}' missing 'business_units' list")

        seen_bu_codes: Set[str] = set()
        business_units: List[BusinessUnitRecord] = []

        for b_idx, b_item in enumerate(raw_bus):
            if not isinstance(b_item, dict):
                raise MasterDataValidationError(f"Business unit at index {b_idx} in '{clean_c_code}' must be an object")

            bu_code = b_item.get("business_unit_code")
            if not isinstance(bu_code, str) or not bu_code.strip():
                raise MasterDataValidationError(f"Business unit at index {b_idx} missing 'business_unit_code'")
            clean_bu_code = bu_code.strip()

            if clean_bu_code in seen_bu_codes:
                raise MasterDataDuplicateKeyError(f"Duplicate business_unit_code '{clean_bu_code}' in company '{clean_c_code}'")
            seen_bu_codes.add(clean_bu_code)

            bu_name = b_item.get("business_unit_name")
            if not isinstance(bu_name, str) or not bu_name.strip():
                raise MasterDataValidationError(f"Business unit '{clean_bu_code}' missing 'business_unit_name'")

            raw_locs = b_item.get("locations")
            if not isinstance(raw_locs, list):
                raise MasterDataValidationError(f"Business unit '{clean_bu_code}' missing 'locations' list")

            seen_loc_codes_in_bu: Set[str] = set()
            locations: List[LocationRecord] = []

            for l_idx, l_item in enumerate(raw_locs):
                if not isinstance(l_item, dict):
                    raise MasterDataValidationError(f"Location at index {l_idx} in BU '{clean_bu_code}' must be an object")

                loc_code = l_item.get("location_code")
                if not isinstance(loc_code, str) or not loc_code.strip():
                    raise MasterDataValidationError(f"Location at index {l_idx} in BU '{clean_bu_code}' missing 'location_code'")
                clean_loc_code = loc_code.strip()

                if clean_loc_code in seen_loc_codes_in_bu:
                    raise MasterDataDuplicateKeyError(f"Duplicate location_code '{clean_loc_code}' in BU '{clean_bu_code}'")
                seen_loc_codes_in_bu.add(clean_loc_code)

                loc_name = l_item.get("location_name")
                if not isinstance(loc_name, str) or not loc_name.strip():
                    raise MasterDataValidationError(f"Location '{clean_loc_code}' missing 'location_name'")

                loc_rec = LocationRecord(
                    location_code=clean_loc_code,
                    location_name=loc_name.strip(),
                    invoice_to_address=str(l_item.get("invoice_to_address") or "").strip(),
                    company_code=clean_c_code,
                    business_unit_code=clean_bu_code,
                    raw_record=dict(l_item),
                )
                locations.append(loc_rec)

            bu_rec = BusinessUnitRecord(
                business_unit_code=clean_bu_code,
                business_unit_name=bu_name.strip(),
                company_code=clean_c_code,
                locations=tuple(locations),
                raw_record=dict(b_item),
            )
            business_units.append(bu_rec)

        c_rec = CompanyRecord(
            company_code=clean_c_code,
            company_name=c_name.strip(),
            business_units=tuple(business_units),
            raw_record=dict(c_item),
        )
        companies.append(c_rec)

    return tuple(companies)


def load_tax_master(source: Union[str, Path, TextIO, Dict[str, Any]]) -> Tuple[TaxRecord, ...]:
    """Load, validate, and convert tax reference data from JSON source."""
    data = _read_json_payload(source)
    if not isinstance(data, dict):
        raise MasterDataValidationError("Tax master root must be a JSON object")

    raw_taxes = data.get("taxes")
    if not isinstance(raw_taxes, list):
        raise MasterDataValidationError("Tax master must contain a 'taxes' list")

    seen_codes: Set[str] = set()
    records: List[TaxRecord] = []

    for idx, item in enumerate(raw_taxes):
        if not isinstance(item, dict):
            raise MasterDataValidationError(f"Tax item at index {idx} must be an object")

        code = item.get("code")
        if not isinstance(code, str) or not code.strip():
            raise MasterDataValidationError(f"Tax item at index {idx} missing required non-empty 'code'")
        clean_code = code.strip()

        if clean_code in seen_codes:
            raise MasterDataDuplicateKeyError(f"Duplicate tax code detected: '{clean_code}' at index {idx}")
        seen_codes.add(clean_code)

        country = item.get("country")
        if not isinstance(country, str) or not country.strip():
            raise MasterDataValidationError(f"Tax '{clean_code}' missing required non-empty 'country'")

        tax_type = item.get("tax_type")
        if not isinstance(tax_type, str) or not tax_type.strip():
            raise MasterDataValidationError(f"Tax '{clean_code}' missing required non-empty 'tax_type'")

        rate_val = item.get("rate")
        if rate_val is None or not isinstance(rate_val, (int, float)):
            raise MasterDataValidationError(f"Tax '{clean_code}' rate must be numeric (int/float), got {type(rate_val)}")
        if rate_val < 0:
            raise MasterDataValidationError(f"Tax '{clean_code}' rate cannot be negative: {rate_val}")

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise MasterDataValidationError(f"Tax '{clean_code}' missing required non-empty 'name'")

        rec = TaxRecord(
            code=clean_code,
            country=country.strip().upper(),
            tax_type=tax_type.strip().upper(),
            rate=float(rate_val),
            name=name.strip(),
            raw_record=dict(item),
        )
        records.append(rec)

    return tuple(records)


def load_payment_terms(source: Union[str, Path, TextIO, Dict[str, Any]]) -> Tuple[PaymentTermRecord, ...]:
    """Load, validate, and convert payment terms reference data from JSON source."""
    data = _read_json_payload(source)
    if not isinstance(data, dict):
        raise MasterDataValidationError("Payment terms root must be a JSON object")

    raw_terms = data.get("payment_terms")
    if not isinstance(raw_terms, list):
        raise MasterDataValidationError("Payment terms must contain a 'payment_terms' list")

    seen_ids: Set[str] = set()
    records: List[PaymentTermRecord] = []

    for idx, item in enumerate(raw_terms):
        if not isinstance(item, dict):
            raise MasterDataValidationError(f"Payment term item at index {idx} must be an object")

        term_id = item.get("payment_term_id")
        if not isinstance(term_id, str) or not term_id.strip():
            raise MasterDataValidationError(f"Payment term at index {idx} missing required 'payment_term_id'")
        clean_term_id = term_id.strip()

        if clean_term_id in seen_ids:
            raise MasterDataDuplicateKeyError(f"Duplicate payment_term_id detected: '{clean_term_id}' at index {idx}")
        seen_ids.add(clean_term_id)

        days_val = item.get("days")
        if days_val is None or not isinstance(days_val, int) or isinstance(days_val, bool):
            raise MasterDataValidationError(f"Payment term '{clean_term_id}' 'days' must be an integer, got {type(days_val)}")
        if days_val < 0:
            raise MasterDataValidationError(f"Payment term '{clean_term_id}' 'days' cannot be negative: {days_val}")

        raw_aliases = item.get("text_aliases", [])
        if not isinstance(raw_aliases, list):
            raise MasterDataValidationError(f"Payment term '{clean_term_id}' 'text_aliases' must be a list")

        aliases: List[str] = []
        for a_idx, alias in enumerate(raw_aliases):
            if not isinstance(alias, str) or not alias.strip():
                raise MasterDataValidationError(f"Alias at index {a_idx} in '{clean_term_id}' must be non-empty string")
            aliases.append(alias.strip())

        rec = PaymentTermRecord(
            payment_term_id=clean_term_id,
            days=days_val,
            text_aliases=tuple(aliases),
            raw_record=dict(item),
        )
        records.append(rec)

    return tuple(records)


def load_po_master(source: Union[str, Path, TextIO, Dict[str, Any]]) -> Tuple[PurchaseOrderRecord, ...]:
    """Load, validate, and convert purchase orders from JSON source."""
    data = _read_json_payload(source)
    if not isinstance(data, dict):
        raise MasterDataValidationError("PO master root must be a JSON object")

    raw_pos = data.get("purchase_orders")
    if not isinstance(raw_pos, list):
        raise MasterDataValidationError("PO master must contain a 'purchase_orders' list")

    seen_po_ids: Set[str] = set()
    records: List[PurchaseOrderRecord] = []

    for idx, item in enumerate(raw_pos):
        if not isinstance(item, dict):
            raise MasterDataValidationError(f"PO item at index {idx} must be an object")

        po_id = item.get("po_id")
        if not isinstance(po_id, str) or not po_id.strip():
            raise MasterDataValidationError(f"PO at index {idx} missing required 'po_id'")
        clean_po_id = po_id.strip()

        if clean_po_id in seen_po_ids:
            raise MasterDataDuplicateKeyError(f"Duplicate po_id detected: '{clean_po_id}' at index {idx}")
        seen_po_ids.add(clean_po_id)

        po_number = item.get("po_number")
        if not isinstance(po_number, str) or not po_number.strip():
            raise MasterDataValidationError(f"PO '{clean_po_id}' missing required 'po_number'")

        supplier_id = item.get("supplier_id")
        if not isinstance(supplier_id, str) or not supplier_id.strip():
            raise MasterDataValidationError(f"PO '{clean_po_id}' missing required 'supplier_id'")

        raw_lines = item.get("po_lines", [])
        if not isinstance(raw_lines, list):
            raise MasterDataValidationError(f"PO '{clean_po_id}' 'po_lines' must be a list")

        lines: List[POLineRecord] = []
        for l_idx, l_item in enumerate(raw_lines):
            if not isinstance(l_item, dict):
                raise MasterDataValidationError(f"Line {l_idx} in PO '{clean_po_id}' must be an object")

            line_id = str(l_item.get("line_id") or "").strip()
            desc = str(l_item.get("description") or "").strip()
            qty = float(l_item.get("quantity") or 0.0)
            uom = str(l_item.get("uom") or "").strip()
            price = str(l_item.get("unit_price") or "").strip()

            line_rec = POLineRecord(
                line_id=line_id,
                description=desc,
                quantity=qty,
                uom=uom,
                unit_price=price,
                raw_record=dict(l_item),
            )
            lines.append(line_rec)

        rec = PurchaseOrderRecord(
            po_id=clean_po_id,
            po_number=po_number.strip(),
            supplier_id=supplier_id.strip(),
            currency=str(item.get("currency") or "").strip().upper(),
            po_lines=tuple(lines),
            raw_record=dict(item),
        )
        records.append(rec)

    return tuple(records)
