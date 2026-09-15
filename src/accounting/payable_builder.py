"""src/accounting/payable_builder.py — Final Payable JSON / Autodraft Generation Layer.

Phase 9E: The final deterministic emission layer of the pipeline.
Transforms an approved PayableDecision and its upstream structured accounting data
into the exact payable JSON required by AUTODRAFT_SCHEMA.md and README.md.

Core Architectural Principles:
1. EMIT APPROVED DATA:
   Phase 9E is strictly an emitter. It never infers, repairs, or recalculates data.
   Never balances invoices, invents missing amounts/taxes, or alters prices/quantities.
2. Anti-Bypass Safety Gate:
   Only documents with PayableDecision.status == SAFE_TO_AUTODRAFT may produce payables[].
   HOLD_FOR_REVIEW, UNSAFE_TO_AUTODRAFT, and NOT_PAYABLE never enter payables[].
3. Explicit Source of Truth (Zero Reconstruction Fallbacks):
   FinancialStructure is a required input for document facts. 9E never reconstructs
   a financial structure from ERPReconstruction.
4. Authoritative Semantic Sourcing (Zero Heuristic Defaulting):
   Line item_type is sourced strictly from canonical upstream metadata/role.
   If unavailable, 9E fails safely rather than guessing "SERVICE" or "GOODS".
5. Zero Accounting Arithmetic:
   Header discounts and charges are consumed as pre-aggregated canonical values.
   9E does not perform sums of multiple line or header components.
6. Zero Keyword Heuristic Classification:
   Header charges are classified strictly by their canonical charge_category.
   9E does not perform substring searches on charge names.
7. Zero ERP Calls:
   payable_builder.py never imports erp.py or calls erp_book().
   Verification against the ERP oracle is reserved strictly for test assertions.
8. Controlled Debit Memo Policy:
   AUTODRAFT_SCHEMA.md strictly defines invoice_type as "INVOICE" | "CREDIT_MEMO".
   Any unsupported document type (including DEBIT_MEMO) raises PayableContractError.
9. Deterministic Payloads & Canonical Serialization:
   Payloads use strict canonical key order. Serialization functions guarantee
   exact byte-for-byte output stability.
10. Supporting Document Isolation:
    Supporting groups remain isolated. Secondary currencies (e.g. TRY customs in DU-02)
    never enter the primary payable.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.accounting.decision import PayableDecision, PayableDecisionStatus
from src.accounting.erp_reconstruction import ERPReconstruction
from src.accounting.financial_structure import (
    FinancialStructure,
    NormalizedCharge,
    NormalizedDiscount,
    NormalizedLine,
    NormalizedParty,
    NormalizedPO,
    NormalizedTax,
)
from src.matching.match_models import MasterMatchResult, MatchStatus
from src.understanding.document_facts import InvoiceType, SemanticRole
from src.utils.logging import get_logger

log = get_logger(__name__)

_DECIMAL_PATTERN = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")

_CANONICAL_PAYABLE_KEYS: Tuple[str, ...] = (
    "invoice_number",
    "invoice_date",
    "due_date",
    "invoice_type",
    "currency",
    "supplier",
    "buyer",
    "payment_term_id",
    "po_number",
    "po_id",
    "gross_total",
    "subtotal",
    "total_tax_amount",
    "discount_amount",
    "freight_charges",
    "insurance_charges",
    "extra_charges",
    "excise_duties",
    "taxes",
    "line_items",
)

_CANONICAL_SUPPLIER_KEYS: Tuple[str, ...] = (
    "name",
    "supplier_id",
    "address",
    "vat_id",
)

_CANONICAL_BUYER_KEYS: Tuple[str, ...] = (
    "company_code",
    "business_unit_code",
    "location_code",
)

_CANONICAL_LINE_KEYS: Tuple[str, ...] = (
    "description",
    "item_type",
    "uom",
    "quantity",
    "unit_price",
    "total",
    "discount",
    "discount_percentage",
    "tax_rate",
    "tax_amount",
    "taxes",
)

_CANONICAL_TAX_KEYS: Tuple[str, ...] = (
    "tax_type",
    "tax_name",
    "tax_rate",
    "tax_amount",
    "tax_type_code",
)

_CANONICAL_FILE_OUTPUT_KEYS: Tuple[str, ...] = (
    "file",
    "payables",
    "declined",
)


# ══════════════════════════════════════════════════════════════════════════
# Errors & Result Models
# ══════════════════════════════════════════════════════════════════════════

class PayableContractError(ValueError):
    """Raised when payable generation violates contract constraints or safety requirements."""


@dataclass(frozen=True)
class AutodraftPayableResult:
    """Internal container holding the emitted payable and audit provenance."""
    payable_dict: Dict[str, Any]
    assembly_id: str
    document_id: str
    decision_status: PayableDecisionStatus
    gross_total: str
    currency: str
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FileOutputResult:
    """Internal container holding the per-file output and audit provenance."""
    file_dict: Dict[str, Any]
    file_name: str
    payables_count: int
    declined_count: int
    file_path: Optional[Path] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════

def _format_decimal(val: Optional[Union[Decimal, float, int, str]], default: str = "") -> str:
    """Format a monetary or rate value as a clean dot-decimal string."""
    if val is None or val == "":
        return default
    if isinstance(val, Decimal):
        return f"{val:.2f}"
    try:
        d = Decimal(str(val).strip())
        return f"{d:.2f}"
    except Exception:
        return default


def _format_rate(val: Optional[Union[Decimal, float, int, str]], default: str = "") -> str:
    """Format a tax rate percentage without trailing decimal clutter (e.g. '19' or '19.5')."""
    if val is None or val == "":
        return default
    s = str(val).replace("%", "").strip()
    try:
        d = Decimal(s)
        if d == d.to_integral():
            return str(d.quantize(Decimal("1")))
        return str(d.normalize())
    except Exception:
        return s


# ══════════════════════════════════════════════════════════════════════════
# Structural Contract Validators
# ══════════════════════════════════════════════════════════════════════════

def validate_autodraft_payload(payload: Dict[str, Any]) -> List[str]:
    """Validate that a payable dictionary strictly satisfies AUTODRAFT_SCHEMA.md.
    
    Checks:
    - Exactly required top-level keys present (no missing, no unexpected)
    - Correct data types (strings, lists, dicts; no nulls)
    - Valid enums ('INVOICE' | 'CREDIT_MEMO')
    - Valid dot-decimal formatting on numeric fields
    - Valid nested shapes for supplier, buyer, line_items, taxes
    
    Returns:
        List of error strings (empty if valid).
    """
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["Payload must be a JSON object (dict)"]

    # 1. Top-level keys check
    keys = set(payload.keys())
    expected_keys = set(_CANONICAL_PAYABLE_KEYS)
    missing_keys = expected_keys - keys
    if missing_keys:
        errors.append(f"Missing required top-level keys: {sorted(list(missing_keys))}")
    unexpected_keys = keys - expected_keys
    if unexpected_keys:
        errors.append(f"Unexpected top-level keys: {sorted(list(unexpected_keys))}")

    # 2. String fields & null checks
    str_fields = [
        "invoice_number", "invoice_date", "due_date", "invoice_type", "currency",
        "payment_term_id", "po_number", "po_id", "gross_total", "subtotal",
        "total_tax_amount", "discount_amount", "freight_charges", "insurance_charges",
        "extra_charges", "excise_duties",
    ]
    for fld in str_fields:
        val = payload.get(fld)
        if val is None:
            errors.append(f"Field '{fld}' cannot be null (must be str)")
        elif not isinstance(val, str):
            errors.append(f"Field '{fld}' must be str, got {type(val).__name__}")

    # 3. Enum validation
    inv_type = payload.get("invoice_type")
    if inv_type not in ("INVOICE", "CREDIT_MEMO"):
        errors.append(f"Field 'invoice_type' must be 'INVOICE' or 'CREDIT_MEMO', got '{inv_type}'")

    # 4. Currency check
    curr = payload.get("currency")
    if isinstance(curr, str) and curr and len(curr) != 3:
        errors.append(f"Field 'currency' must be 3-letter ISO code, got '{curr}'")

    # 5. Numeric formatting checks on populated header values
    num_fields = [
        "gross_total", "subtotal", "total_tax_amount", "discount_amount",
        "freight_charges", "insurance_charges", "extra_charges", "excise_duties",
    ]
    for fld in num_fields:
        val = payload.get(fld)
        if isinstance(val, str) and val != "":
            if not _DECIMAL_PATTERN.match(val):
                errors.append(f"Field '{fld}' must be dot-decimal number, got '{val}'")

    # 6. Supplier object validation
    sup = payload.get("supplier")
    if not isinstance(sup, dict):
        errors.append("Field 'supplier' must be a dict")
    else:
        sup_keys = set(sup.keys())
        expected_sup_keys = set(_CANONICAL_SUPPLIER_KEYS)
        if expected_sup_keys - sup_keys:
            errors.append(f"Missing supplier keys: {sorted(list(expected_sup_keys - sup_keys))}")
        if sup_keys - expected_sup_keys:
            errors.append(f"Unexpected supplier keys: {sorted(list(sup_keys - expected_sup_keys))}")
        for sk in _CANONICAL_SUPPLIER_KEYS:
            sval = sup.get(sk)
            if sval is None or not isinstance(sval, str):
                errors.append(f"Supplier field '{sk}' must be str, got {type(sval).__name__}")

    # 7. Buyer object validation
    buy = payload.get("buyer")
    if not isinstance(buy, dict):
        errors.append("Field 'buyer' must be a dict")
    else:
        buy_keys = set(buy.keys())
        expected_buy_keys = set(_CANONICAL_BUYER_KEYS)
        if expected_buy_keys - buy_keys:
            errors.append(f"Missing buyer keys: {sorted(list(expected_buy_keys - buy_keys))}")
        if buy_keys - expected_buy_keys:
            errors.append(f"Unexpected buyer keys: {sorted(list(buy_keys - expected_buy_keys))}")
        for bk in _CANONICAL_BUYER_KEYS:
            bval = buy.get(bk)
            if bval is None or not isinstance(bval, str):
                errors.append(f"Buyer field '{bk}' must be str, got {type(bval).__name__}")

    # 8. Header taxes validation
    taxes = payload.get("taxes")
    if not isinstance(taxes, list):
        errors.append("Field 'taxes' must be a list")
    else:
        for idx, tx in enumerate(taxes):
            if not isinstance(tx, dict):
                errors.append(f"Header taxes[{idx}] must be a dict")
                continue
            tx_keys = set(tx.keys())
            expected_tx_keys = set(_CANONICAL_TAX_KEYS)
            if expected_tx_keys - tx_keys:
                errors.append(f"Header taxes[{idx}] missing keys: {sorted(list(expected_tx_keys - tx_keys))}")
            if tx_keys - expected_tx_keys:
                errors.append(f"Header taxes[{idx}] unexpected keys: {sorted(list(tx_keys - expected_tx_keys))}")
            for tk in _CANONICAL_TAX_KEYS:
                tval = tx.get(tk)
                if tval is None or not isinstance(tval, str):
                    errors.append(f"Header taxes[{idx}] field '{tk}' must be str")
            if tx.get("tax_rate") and not _DECIMAL_PATTERN.match(str(tx.get("tax_rate"))):
                errors.append(f"Header taxes[{idx}] tax_rate must be dot-decimal, got '{tx.get('tax_rate')}'")
            if tx.get("tax_amount") and not _DECIMAL_PATTERN.match(str(tx.get("tax_amount"))):
                errors.append(f"Header taxes[{idx}] tax_amount must be dot-decimal, got '{tx.get('tax_amount')}'")

    # 9. Line items validation
    lines = payload.get("line_items")
    if not isinstance(lines, list):
        errors.append("Field 'line_items' must be a list")
    else:
        for idx, li in enumerate(lines):
            if not isinstance(li, dict):
                errors.append(f"line_items[{idx}] must be a dict")
                continue
            li_keys = set(li.keys())
            expected_li_keys = set(_CANONICAL_LINE_KEYS)
            if expected_li_keys - li_keys:
                errors.append(f"line_items[{idx}] missing keys: {sorted(list(expected_li_keys - li_keys))}")
            if li_keys - expected_li_keys:
                errors.append(f"line_items[{idx}] unexpected keys: {sorted(list(li_keys - expected_li_keys))}")
            
            # Check item_type enum
            itype = li.get("item_type")
            if itype not in ("GOODS", "SERVICE", "FREIGHT", "TAX"):
                errors.append(f"line_items[{idx}] item_type must be GOODS|SERVICE|FREIGHT|TAX, got '{itype}'")

            # Check string fields
            for lk in (
                "description", "item_type", "uom", "quantity", "unit_price", "total",
                "discount", "discount_percentage", "tax_rate", "tax_amount",
            ):
                lval = li.get(lk)
                if lval is None or not isinstance(lval, str):
                    errors.append(f"line_items[{idx}] field '{lk}' must be str")

            # Check dot-decimal formatting
            for lnum_fld in ("quantity", "unit_price", "total", "discount", "discount_percentage", "tax_rate", "tax_amount"):
                lval = li.get(lnum_fld)
                if isinstance(lval, str) and lval != "":
                    if not _DECIMAL_PATTERN.match(lval):
                        errors.append(f"line_items[{idx}] field '{lnum_fld}' must be dot-decimal, got '{lval}'")

            # Check line taxes list
            ltaxes = li.get("taxes")
            if not isinstance(ltaxes, list):
                errors.append(f"line_items[{idx}] taxes must be a list")
            else:
                for tidx, ltx in enumerate(ltaxes):
                    if not isinstance(ltx, dict):
                        errors.append(f"line_items[{idx}] taxes[{tidx}] must be a dict")

    return errors


def validate_file_output(payload: Dict[str, Any]) -> List[str]:
    """Validate that a per-file wrapper dictionary satisfies the assignment contract."""
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["File output payload must be a JSON object (dict)"]

    keys = set(payload.keys())
    expected_keys = set(_CANONICAL_FILE_OUTPUT_KEYS)
    if expected_keys - keys:
        errors.append(f"Missing required file output keys: {sorted(list(expected_keys - keys))}")
    if keys - expected_keys:
        errors.append(f"Unexpected file output keys: {sorted(list(keys - expected_keys))}")

    f_name = payload.get("file")
    if not isinstance(f_name, str) or not f_name.strip():
        errors.append("Field 'file' must be a non-empty str")

    payables = payload.get("payables")
    if not isinstance(payables, list):
        errors.append("Field 'payables' must be a list")
    else:
        for idx, p in enumerate(payables):
            p_errors = validate_autodraft_payload(p)
            for pe in p_errors:
                errors.append(f"payables[{idx}]: {pe}")

    declined = payload.get("declined")
    if not isinstance(declined, list):
        errors.append("Field 'declined' must be a list")
    else:
        for idx, d in enumerate(declined):
            if not isinstance(d, dict):
                errors.append(f"declined[{idx}] must be a dict")
                continue
            d_keys = set(d.keys())
            if not {"doc_type", "reason"}.issubset(d_keys):
                errors.append(f"declined[{idx}] must contain 'doc_type' and 'reason'")
            if not isinstance(d.get("doc_type"), str) or not d.get("doc_type"):
                errors.append(f"declined[{idx}] 'doc_type' must be non-empty str")
            if not isinstance(d.get("reason"), str) or not d.get("reason"):
                errors.append(f"declined[{idx}] 'reason' must be non-empty str")

    return errors


# ══════════════════════════════════════════════════════════════════════════
# Pure In-Memory Builders
# ══════════════════════════════════════════════════════════════════════════

def build_payable(
    decision: PayableDecision,
    financial_structure: FinancialStructure,
    reconstruction: ERPReconstruction,
    supplier_match: Optional[MasterMatchResult] = None,
    buyer_match: Optional[MasterMatchResult] = None,
    po_match: Optional[MasterMatchResult] = None,
    payment_term_match: Optional[MasterMatchResult] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Transform an approved document into an AUTODRAFT_SCHEMA.md payable object.
    
    Zero Fallback / Zero Reconstruction Guarantee:
    - FinancialStructure is mandatory. 9E never reconstructs it from ERPReconstruction.
    - Sourcing for each field is anchored strictly to its authoritative source.
    - Zero formula execution; zero heuristic defaulting for semantic fields.
    
    Args:
        decision: The approved PayableDecision (status must be SAFE_TO_AUTODRAFT).
        financial_structure: Mandatory canonical financial structure.
        reconstruction: Authoritative ERP reconstruction.
        supplier_match: Optional Phase 8B supplier match result.
        buyer_match: Optional Phase 8B buyer match result.
        po_match: Optional Phase 8D PO match result.
        payment_term_match: Optional Phase 8C payment term match result.
        strict: If True, schema validation failures raise PayableContractError.
        
    Returns:
        Deterministic dictionary adhering strictly to AUTODRAFT_SCHEMA.md.
        
    Raises:
        PayableContractError: If decision is not SAFE_TO_AUTODRAFT, mandatory fields
                              are missing, or schema validation fails.
    """
    # ── 1. Anti-Bypass Check ────────────────────────────────────────────────
    if decision.status != PayableDecisionStatus.SAFE_TO_AUTODRAFT:
        raise PayableContractError(
            f"Cannot build payable for document with status '{decision.status.value}'. "
            "Only SAFE_TO_AUTODRAFT documents are approved for payable generation."
        )

    # ── 2. Document & Invoice Type Sourcing ──────────────────────────────────
    doc_type = financial_structure.document_type
    invoice_type_str: str
    if doc_type == InvoiceType.INVOICE:
        invoice_type_str = "INVOICE"
    elif doc_type == InvoiceType.CREDIT_MEMO:
        invoice_type_str = "CREDIT_MEMO"
    else:
        raise PayableContractError(
            f"Unsupported document_type for autodraft schema: '{doc_type.value}'. "
            "AUTODRAFT_SCHEMA.md strictly permits 'INVOICE' or 'CREDIT_MEMO'. "
            "Debit memos and unclassified types are not supported and must fail safely."
        )

    # ── 3. Currency Sourcing ────────────────────────────────────────────────
    # Authoritative source: FinancialStructure.currency (Origin: OBSERVED)
    curr = financial_structure.currency
    if not curr or not str(curr).strip():
        raise PayableContractError(
            "Mandatory field 'currency' is missing from FinancialStructure. "
            "Phase 9E cannot fabricate currency."
        )
    currency_str = str(curr).strip().upper()

    # ── 4. Printed Totals Sourcing ──────────────────────────────────────────
    pt = financial_structure.printed_totals
    if pt is None or pt.gross_total is None:
        raise PayableContractError(
            "Mandatory field 'gross_total' is missing from FinancialStructure.printed_totals. "
            "Phase 9E cannot fabricate gross total."
        )
    # Credit memos and invoices deliver positive magnitudes per contract
    gross_total_str = _format_decimal(abs(pt.gross_total))
    subtotal_str = _format_decimal(abs(pt.subtotal)) if (pt and pt.subtotal is not None) else ""
    total_tax_str = _format_decimal(abs(pt.tax_total)) if (pt and pt.tax_total is not None) else ""

    # ── 5. Posting Line Items (BILLED_LINE only, zero COMPONENT_DETAIL) ───────
    posting_lines: List[NormalizedLine] = [
        ln for ln in financial_structure.lines if ln.semantic_role == SemanticRole.BILLED_LINE
    ]
    if not posting_lines:
        raise PayableContractError(
            "Document has zero posting line items (BILLED_LINE). Cannot emit empty payable."
        )

    line_items_list: List[Dict[str, Any]] = []
    for ln in posting_lines:
        # Strict Authoritative Semantic Sourcing: item_type
        # Must be present in canonical metadata/raw_values; NO HEURISTIC GUESSING!
        itype = ln.metadata.get("item_type") or ln.raw_values.get("item_type")
        if not itype:
            raise PayableContractError(
                f"Mandatory line field 'item_type' is missing for line '{ln.source_line_id}'. "
                "Phase 9E prohibits heuristic defaulting (e.g. guessing SERVICE/GOODS)."
            )
        itype_upper = str(itype).strip().upper()
        if itype_upper not in ("GOODS", "SERVICE", "FREIGHT", "TAX"):
            raise PayableContractError(
                f"Invalid item_type '{itype_upper}' for line '{ln.source_line_id}'. "
                "Must be GOODS, SERVICE, FREIGHT, or TAX."
            )

        uom_str = str(ln.metadata.get("uom") or ln.raw_values.get("uom") or "").strip()
        qty_str = str(ln.quantity) if ln.quantity is not None else ""
        price_str = _format_decimal(abs(ln.unit_price)) if ln.unit_price is not None else ""
        total_str = _format_decimal(abs(ln.amount)) if ln.amount is not None else ""

        # Line discount (magnitude)
        line_disc_amt = ""
        line_disc_pct = ""
        if ln.discounts:
            if len(ln.discounts) > 1:
                raise PayableContractError(
                    f"Line '{ln.source_line_id}' has multiple unaggregated discounts; "
                    "Phase 9E does not perform accounting arithmetic."
                )
            d = ln.discounts[0]
            if d.amount is not None:
                line_disc_amt = _format_decimal(abs(d.amount))
            if d.rate is not None:
                line_disc_pct = _format_rate(d.rate)

        # Line taxes
        line_tax_rate_str = ""
        line_tax_amt_str = ""
        line_taxes_list: List[Dict[str, Any]] = []
        if ln.taxes:
            for tx in ln.taxes:
                tx_rate_s = _format_rate(tx.rate)
                tx_amt_s = _format_decimal(abs(tx.amount)) if tx.amount is not None else ""
                tx_code = str(tx.metadata.get("tax_type_code") or "")
                line_taxes_list.append({
                    "tax_type": tx.tax_type or "",
                    "tax_name": tx.tax_name or "",
                    "tax_rate": tx_rate_s,
                    "tax_amount": tx_amt_s,
                    "tax_type_code": tx_code,
                })
            if len(ln.taxes) == 1:
                line_tax_rate_str = _format_rate(ln.taxes[0].rate)
                line_tax_amt_str = _format_decimal(abs(ln.taxes[0].amount)) if ln.taxes[0].amount is not None else ""

        line_item_dict: Dict[str, Any] = {
            "description": ln.description or "",
            "item_type": itype_upper,
            "uom": uom_str,
            "quantity": qty_str,
            "unit_price": price_str,
            "total": total_str,
            "discount": line_disc_amt,
            "discount_percentage": line_disc_pct,
            "tax_rate": line_tax_rate_str,
            "tax_amount": line_tax_amt_str,
            "taxes": line_taxes_list,
        }
        line_items_list.append(line_item_dict)

    # ── 6. Supplier Master & Observed Identity ──────────────────────────────
    sup_obj = financial_structure.supplier
    sup_name = (sup_obj.name or "") if sup_obj else ""
    sup_addr = (sup_obj.address or "") if sup_obj else ""
    sup_vat = (sup_obj.vat_id or "") if sup_obj else ""

    sup_id = ""
    if supplier_match is not None and supplier_match.status == MatchStatus.MATCHED:
        sup_id = str(supplier_match.master_id or "")

    supplier_dict: Dict[str, Any] = {
        "name": sup_name,
        "supplier_id": sup_id,
        "address": sup_addr,
        "vat_id": sup_vat,
    }

    # ── 7. Buyer Master & Observed Identity ──────────────────────────────────
    # Follow established 9C/8D contract: master matched codes if matched,
    # otherwise authoritative observed codes if explicitly present, else "".
    buy_obj = financial_structure.buyer
    company_code = ""
    business_unit_code = ""
    location_code = ""

    if buyer_match is not None and buyer_match.status == MatchStatus.MATCHED:
        company_code = str(buyer_match.details.get("company_code") or "")
        business_unit_code = str(buyer_match.details.get("business_unit_code") or "")
        location_code = str(buyer_match.details.get("location_code") or "")
    elif buy_obj is not None:
        company_code = buy_obj.company_code or ""
        business_unit_code = buy_obj.business_unit_code or ""
        location_code = buy_obj.location_code or ""

    buyer_dict: Dict[str, Any] = {
        "company_code": company_code,
        "business_unit_code": business_unit_code,
        "location_code": location_code,
    }

    # ── 8. References: Payment Term & Purchase Order ────────────────────────
    term_id = ""
    if payment_term_match is not None and payment_term_match.status == MatchStatus.MATCHED:
        term_id = str(payment_term_match.master_id or "")

    po_num = ""
    if financial_structure.purchase_order and financial_structure.purchase_order.po_number:
        po_num = str(financial_structure.purchase_order.po_number)

    po_master_id = ""
    if po_match is not None and po_match.status == MatchStatus.MATCHED:
        po_master_id = str(po_match.master_id or "")

    # ── 9. Header Discounts (Zero Summing Guarantee) ────────────────────────
    header_disc_str = ""
    if financial_structure.header_discounts:
        if len(financial_structure.header_discounts) > 1:
            raise PayableContractError(
                "Multiple unaggregated header discounts present in FinancialStructure; "
                "Phase 9E does not perform accounting arithmetic."
            )
        hd = financial_structure.header_discounts[0]
        if hd.amount is not None:
            header_disc_str = _format_decimal(abs(hd.amount))

    # ── 10. Header Charges (Strict Canonical Categorization, Zero Summing) ──
    freight_str = ""
    insurance_str = ""
    extra_str = ""
    excise_str = ""

    for chg in financial_structure.header_charges:
        cat = chg.charge_category or chg.metadata.get("charge_category")
        if not cat:
            raise PayableContractError(
                f"NormalizedCharge '{chg.name}' is missing mandatory charge_category. "
                "Phase 9E does not perform heuristic keyword classification."
            )
        cat_lower = cat.lower().strip()
        amt_str = _format_decimal(abs(chg.amount)) if chg.amount is not None else ""

        if cat_lower == "freight":
            if freight_str:
                raise PayableContractError(
                    "Multiple unaggregated freight charges present; Phase 9E does not perform accounting arithmetic."
                )
            freight_str = amt_str
        elif cat_lower == "insurance":
            if insurance_str:
                raise PayableContractError(
                    "Multiple unaggregated insurance charges present; Phase 9E does not perform accounting arithmetic."
                )
            insurance_str = amt_str
        elif cat_lower == "excise":
            if excise_str:
                raise PayableContractError(
                    "Multiple unaggregated excise charges present; Phase 9E does not perform accounting arithmetic."
                )
            excise_str = amt_str
        elif cat_lower == "extra":
            if extra_str:
                raise PayableContractError(
                    "Multiple unaggregated extra charges present; Phase 9E does not perform accounting arithmetic."
                )
            extra_str = amt_str
        else:
            raise PayableContractError(
                f"Unknown charge_category '{cat_lower}' on NormalizedCharge '{chg.name}'."
            )

    # ── 11. Header Taxes ────────────────────────────────────────────────────
    header_taxes_list: List[Dict[str, Any]] = []
    for tx in financial_structure.header_taxes:
        tx_rate_s = _format_rate(tx.rate)
        tx_amt_s = _format_decimal(abs(tx.amount)) if tx.amount is not None else ""
        tx_code = str(tx.metadata.get("tax_type_code") or "")
        header_taxes_list.append({
            "tax_type": tx.tax_type or "",
            "tax_name": tx.tax_name or "",
            "tax_rate": tx_rate_s,
            "tax_amount": tx_amt_s,
            "tax_type_code": tx_code,
        })

    # ── 12. Assemble Canonical Dict (Strict Canonical Key Order) ────────────
    payable_dict: Dict[str, Any] = {
        "invoice_number": financial_structure.invoice_number or "",
        "invoice_date": financial_structure.invoice_date or "",
        "due_date": financial_structure.due_date or "",
        "invoice_type": invoice_type_str,
        "currency": currency_str,
        "supplier": supplier_dict,
        "buyer": buyer_dict,
        "payment_term_id": term_id,
        "po_number": po_num,
        "po_id": po_master_id,
        "gross_total": gross_total_str,
        "subtotal": subtotal_str,
        "total_tax_amount": total_tax_str,
        "discount_amount": header_disc_str,
        "freight_charges": freight_str,
        "insurance_charges": insurance_str,
        "extra_charges": extra_str,
        "excise_duties": excise_str,
        "taxes": header_taxes_list,
        "line_items": line_items_list,
    }

    # ── 13. Final Schema Contract Validation ────────────────────────────────
    errors = validate_autodraft_payload(payable_dict)
    if errors and strict:
        raise PayableContractError(
            f"Payable payload schema validation failed: {'; '.join(errors)}"
        )

    return payable_dict


def build_declined_entry(decision: PayableDecision) -> Dict[str, Any]:
    """Transform a NOT_PAYABLE decision into an AUTODRAFT_SCHEMA.md declined object.
    
    Args:
        decision: The decision (status must be NOT_PAYABLE).
        
    Returns:
        Dictionary with 'doc_type' and 'reason'.
        
    Raises:
        PayableContractError: If decision is not NOT_PAYABLE.
    """
    if decision.status != PayableDecisionStatus.NOT_PAYABLE:
        raise PayableContractError(
            f"Cannot build declined entry for decision with status '{decision.status.value}'. "
            "Only NOT_PAYABLE decisions populate declined[]."
        )
    doc_type = decision.decline_doc_type or "NON_PAYABLE"
    reason = decision.decline_reason or decision.primary_reason or "Document is not an actionable payable"
    return {
        "doc_type": doc_type,
        "reason": reason,
    }


def build_file_output(
    file_name: str,
    payables: Sequence[Dict[str, Any]] = (),
    declined: Sequence[Dict[str, Any]] = (),
    strict: bool = True,
) -> Dict[str, Any]:
    """Construct a complete per-file output wrapper matching the assignment contract.
    
    Args:
        file_name: Base filename of the input PDF (e.g. 'INV-01.pdf').
        payables: Sequence of valid payable dictionaries.
        declined: Sequence of valid declined dictionaries.
        strict: If True, validates structure and raises PayableContractError on error.
        
    Returns:
        Clean dictionary {'file': X.pdf, 'payables': [...], 'declined': [...]}.
    """
    clean_file_name = Path(file_name).name
    payload: Dict[str, Any] = {
        "file": clean_file_name,
        "payables": list(payables),
        "declined": list(declined),
    }
    errors = validate_file_output(payload)
    if errors and strict:
        raise PayableContractError(
            f"File output payload validation failed for '{clean_file_name}': {'; '.join(errors)}"
        )
    return payload


# ══════════════════════════════════════════════════════════════════════════
# Canonical Serialization & File Writing
# ══════════════════════════════════════════════════════════════════════════

def serialize_payable_json(payable_dict: Dict[str, Any]) -> str:
    """Serialize a single payable dictionary to canonical 2-space indented JSON."""
    return json.dumps(payable_dict, indent=2, ensure_ascii=False)


def serialize_file_output_json(file_dict: Dict[str, Any]) -> str:
    """Serialize a per-file wrapper dictionary to canonical 2-space indented JSON."""
    return json.dumps(file_dict, indent=2, ensure_ascii=False)


def write_file_output(
    file_payload: Dict[str, Any],
    output_dir: Union[Path, str] = "output",
    strict: bool = True,
) -> Path:
    """Serialize and write a per-file output payload to <output_dir>/<file_stem>.json.
    
    Args:
        file_payload: Valid per-file dictionary {'file': 'X.pdf', ...}.
        output_dir: Destination directory (defaults to 'output').
        strict: If True, validates payload before writing.
        
    Returns:
        Path of the written JSON file.
    """
    if strict:
        errors = validate_file_output(file_payload)
        if errors:
            raise PayableContractError(
                f"Cannot write invalid file output: {'; '.join(errors)}"
            )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_name = file_payload.get("file", "document.pdf")
    stem = Path(file_name).stem
    target_file = out_path / f"{stem}.json"

    serialized = serialize_file_output_json(file_payload)
    target_file.write_text(serialized, encoding="utf-8")
    log.info("Wrote autodraft output to %s (%d payables, %d declined)", target_file, len(file_payload.get("payables", [])), len(file_payload.get("declined", [])))
    return target_file
