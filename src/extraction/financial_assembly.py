"""src/extraction/financial_assembly.py — Multi-Page Financial Assembly Layer.

Phase 9B-3: Assembles already-consolidated facts (DocumentFacts) across pages
and logical groups into coherent FinancialDocumentAssembly objects.

Core Architectural Principles:
1. Pure Assembly Layer:
   Assembles existing evidence and facts into coherent logical financial documents.
   Does NOT calculate, repair, match, validate accounting, or invent facts.
2. Multi-Page Continuity:
   Continuation pages and groups are associated with their parent logical document
   to produce ONE logical financial assembly, not multiple separate payables.
3. Supporting Document Isolation:
   Supporting groups (packing lists, delivery notes, detailed customs declaration
   sheets in secondary currencies, etc.) are kept strictly separate as references
   via `supporting_group_ids`. Their amounts, currencies, lines, and totals are
   NEVER merged or aggregated into the primary payable's financial facts.
4. Multiple Payables Isolation:
   Independent payable documents in the same PDF (e.g. Invoice A and Invoice B)
   produce separate, distinct assemblies with their respective supporting groups.
5. Line & Role Preservation:
   Semantic roles (BILLED_LINE, COMPONENT_DETAIL, SUBTOTAL, DISCOUNT, CHARGE,
   TAX, TOTAL) and placement (HEADER vs LINE) established upstream remain intact.
   Genuine repeated invoice lines are strictly preserved and never collapsed.
6. Zero Accounting Arithmetic:
   No quantity * price, no tax_rate * base, no sum(lines), no net + tax = gross,
   and no derivation of missing amounts. Missing values remain None.
7. Zero Master Matching:
   No master-data matchers called and no master IDs injected.
8. 100% Deterministic & Offline:
   No OCR, no Vision, no Qwen, no network, and no ERP calls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from src.understanding.document_facts import (
    BuyerIdentityFact,
    ChargeFact,
    DiscountFact,
    DocumentFacts,
    DocumentIdentityFacts,
    FactOrigin,
    FinancialFacts,
    InvoiceType,
    LineFact,
    PartyIdentityFacts,
    Placement,
    POFacts,
    PrintedTotalsFact,
    SemanticRole,
    SupplierIdentityFact,
    TaxFact,
)
from src.understanding.document_grouper import DocumentGroup, GroupingResult
from src.understanding.page_classifier import PageRole, PayableRelevance
from src.utils.logging import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════

def _normalize_tuple_str(val: Any) -> Tuple[str, ...]:
    """Normalize input to a deterministic tuple of unique non-empty strings."""
    if val is None:
        return ()
    if isinstance(val, str):
        s = val.strip()
        return (s,) if s else ()
    if isinstance(val, (list, tuple, set)):
        items: List[str] = []
        for x in val:
            if x is not None:
                sx = str(x).strip()
                if sx and sx not in items:
                    items.append(sx)
        return tuple(items)
    s = str(val).strip()
    return (s,) if s else ()


def _union_evidence(evidence_iter: Sequence[Tuple[str, ...]]) -> Tuple[str, ...]:
    """Return a deterministic, deduplicated tuple of evidence IDs."""
    seen: Set[str] = set()
    result: List[str] = []
    for ev_tuple in evidence_iter:
        for ev in ev_tuple:
            if ev and ev not in seen:
                seen.add(ev)
                result.append(ev)
    return tuple(result)


# ══════════════════════════════════════════════════════════════════════════
# Output Model: FinancialDocumentAssembly
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FinancialDocumentAssembly:
    """Logical financial document assembly comprising primary group(s) and supporting references.
    
    Phase 9B-3 representation:
    - Pure structural/assembly layer upstream of 9B-4 extraction validation and 9C accounting.
    - Minimal, clean wrapper around assembled DocumentFacts, preserving group relations and provenance.
    """
    assembly_id: str
    document_id: str
    primary_group_ids: Tuple[str, ...] = field(default_factory=tuple)
    supporting_group_ids: Tuple[str, ...] = field(default_factory=tuple)
    page_numbers: Tuple[int, ...] = field(default_factory=tuple)
    document_role: PageRole = PageRole.INVOICE
    payable_relevance: PayableRelevance = PayableRelevance.PAYABLE_CANDIDATE
    facts: DocumentFacts = field(default_factory=lambda: DocumentFacts(document_id="empty"))
    group_provenance: Dict[str, Any] = field(default_factory=dict)
    conflicts: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    evidence_ids: Tuple[str, ...] = field(default_factory=tuple)
    assembly_provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p_nums = tuple(sorted(list(dict.fromkeys(int(p) for p in self.page_numbers))))
        prim_groups = _normalize_tuple_str(self.primary_group_ids)
        sup_groups = _normalize_tuple_str(self.supporting_group_ids)
        ev_ids = _normalize_tuple_str(self.evidence_ids)

        d_role = (
            self.document_role
            if isinstance(self.document_role, PageRole)
            else PageRole(str(self.document_role).lower())
        )
        p_rel = (
            self.payable_relevance
            if isinstance(self.payable_relevance, PayableRelevance)
            else PayableRelevance(str(self.payable_relevance).lower())
        )

        object.__setattr__(self, "page_numbers", p_nums)
        object.__setattr__(self, "primary_group_ids", prim_groups)
        object.__setattr__(self, "supporting_group_ids", sup_groups)
        object.__setattr__(self, "evidence_ids", ev_ids)
        object.__setattr__(self, "document_role", d_role)
        object.__setattr__(self, "payable_relevance", p_rel)

    # ── Convenience properties for downstream phases (9B-4 / 9C) ──
    @property
    def invoice_number(self) -> Optional[str]:
        return self.facts.identity.invoice_number

    @property
    def currency(self) -> Optional[str]:
        return self.facts.financials.currency or self.facts.identity.currency

    @property
    def lines(self) -> Tuple[LineFact, ...]:
        return self.facts.financials.lines

    @property
    def taxes(self) -> Tuple[TaxFact, ...]:
        return self.facts.financials.taxes

    @property
    def discounts(self) -> Tuple[DiscountFact, ...]:
        return self.facts.financials.discounts

    @property
    def charges(self) -> Tuple[ChargeFact, ...]:
        return self.facts.financials.charges

    @property
    def printed_totals(self) -> Optional[PrintedTotalsFact]:
        return self.facts.financials.printed_totals

    @property
    def supplier(self) -> SupplierIdentityFact:
        return self.facts.parties.supplier

    @property
    def buyer(self) -> BuyerIdentityFact:
        return self.facts.parties.buyer

    @property
    def po(self) -> POFacts:
        return self.facts.po

    def to_dict(self) -> Dict[str, Any]:
        """Serialize FinancialDocumentAssembly to a JSON-serializable dictionary."""
        return {
            "assembly_id": self.assembly_id,
            "document_id": self.document_id,
            "primary_group_ids": list(self.primary_group_ids),
            "supporting_group_ids": list(self.supporting_group_ids),
            "page_numbers": list(self.page_numbers),
            "document_role": self.document_role.value,
            "payable_relevance": self.payable_relevance.value,
            "facts": self.facts.to_dict(),
            "group_provenance": dict(self.group_provenance),
            "conflicts": [dict(cf) for cf in self.conflicts],
            "evidence_ids": list(self.evidence_ids),
            "assembly_provenance": dict(self.assembly_provenance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FinancialDocumentAssembly:
        """Reconstruct FinancialDocumentAssembly from dictionary."""
        facts_data = data.get("facts") or {}
        facts = (
            facts_data
            if isinstance(facts_data, DocumentFacts)
            else DocumentFacts.from_dict(facts_data)
        )
        return cls(
            assembly_id=data["assembly_id"],
            document_id=data["document_id"],
            primary_group_ids=_normalize_tuple_str(data.get("primary_group_ids")),
            supporting_group_ids=_normalize_tuple_str(data.get("supporting_group_ids")),
            page_numbers=tuple(data.get("page_numbers") or []),
            document_role=PageRole(data.get("document_role", "invoice")),
            payable_relevance=PayableRelevance(data.get("payable_relevance", "payable_candidate")),
            facts=facts,
            group_provenance=dict(data.get("group_provenance") or {}),
            conflicts=tuple(dict(cf) for cf in data.get("conflicts") or []),
            evidence_ids=_normalize_tuple_str(data.get("evidence_ids")),
            assembly_provenance=dict(data.get("assembly_provenance") or {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize FinancialDocumentAssembly to deterministic JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> FinancialDocumentAssembly:
        """Construct FinancialDocumentAssembly from JSON string."""
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# Continuation & Supporting Merge Engine
# ══════════════════════════════════════════════════════════════════════════

def _is_supporting_facts(
    facts: DocumentFacts,
    group: Optional[DocumentGroup] = None,
    primary_facts: Optional[DocumentFacts] = None,
) -> bool:
    """Determine if a DocumentFacts object represents a supporting document."""
    # 1. Explicit supporting role or relevance
    if facts.payable_relevance == PayableRelevance.SUPPORTING:
        return True
    if facts.document_role == PageRole.SUPPORTING_DOCUMENT:
        return True

    # An independent payable candidate with a distinct invoice number is never a supporting document
    if primary_facts is not None:
        p_inv = primary_facts.identity.invoice_number
        f_inv = facts.identity.invoice_number
        if (
            facts.payable_relevance == PayableRelevance.PAYABLE_CANDIDATE
            and f_inv is not None
            and p_inv is not None
            and f_inv.strip().upper() != p_inv.strip().upper()
        ):
            return False

        # 2. Listed in primary supporting_group_ids
        if facts.group_id and facts.group_id in primary_facts.supporting_group_ids:
            return True

        p_curr = primary_facts.financials.currency or primary_facts.identity.currency
        f_curr = facts.financials.currency or facts.identity.currency

        # 3. Dossier detailed sheets: conflicting/secondary currency in same dossier
        if p_curr is not None and f_curr is not None and p_curr.strip().upper() != f_curr.strip().upper():
            return True

        # 4. In a consolidated invoice dossier (e.g. DU-02), if primary group already has 2+ pages
        # or completed pagination, subsequent groups sharing document number or dossier structure
        # that are non-adjacent or follow supporting material are supporting detailed schedules
        if group is not None and primary_facts.page_numbers:
            shared_nums = group.grouping_signals.get("shared_document_numbers", [])
            last_prim = max(primary_facts.page_numbers)
            if shared_nums and min(facts.page_numbers) > last_prim:
                return True



    return False


def _can_join_continuation(
    primary: DocumentFacts,
    candidate: DocumentFacts,
    primary_group: Optional[DocumentGroup] = None,
    candidate_group: Optional[DocumentGroup] = None,
) -> Tuple[bool, str]:
    """Evaluate whether candidate DocumentFacts is a continuation of primary DocumentFacts.
    
    Returns:
        (can_join: bool, reason: str)
    """
    # Boundary Check 1: Supporting documents are never continuation
    if _is_supporting_facts(candidate, candidate_group, primary):
        return False, "Candidate is supporting document"

    # Boundary Check 2: Contradictory explicit invoice numbers separate payables
    p_inv = primary.identity.invoice_number
    c_inv = candidate.identity.invoice_number
    if p_inv and c_inv and p_inv.strip().upper() != c_inv.strip().upper():
        return False, f"Conflicting explicit invoice numbers ({p_inv} vs {c_inv})"

    # Boundary Check 3: Different explicit currencies separate documents
    p_curr = primary.financials.currency or primary.identity.currency
    c_curr = candidate.financials.currency or candidate.identity.currency
    if p_curr and c_curr and p_curr.strip().upper() != c_curr.strip().upper():
        return False, f"Conflicting currencies ({p_curr} vs {c_curr})"

    # Signal A: Explicit continuation page role
    if candidate.document_role == PageRole.CONTINUATION:
        return True, "PageRole continuation marker"

    # Sequential continuity checks between primary and candidate groups
    if primary.page_numbers and candidate.page_numbers:
        last_primary_page = max(primary.page_numbers)
        first_cand_page = min(candidate.page_numbers)

        # A continuation group must immediately follow the primary group
        if first_cand_page == last_primary_page + 1:
            # Signal B: Upstream grouping signals indicate continuation links
            if candidate_group is not None and candidate_group.grouping_signals.get("continuation_links", 0) > 0:
                return True, "Upstream continuation links"

            # Signal C: Matching non-empty invoice number with continuation role or open pagination
            if p_inv and c_inv and p_inv.strip().upper() == c_inv.strip().upper():
                if candidate.document_role == PageRole.CONTINUATION or candidate.payable_relevance == PayableRelevance.AMBIGUOUS:
                    return True, f"Matching invoice number ({p_inv}) on continuation page"

            # Signal D: Sequential page numbering with ambiguous relevance and no new document start
            if (
                candidate.payable_relevance == PayableRelevance.AMBIGUOUS
                and c_inv is None  # Does not declare a new independent invoice
                and candidate.financials.printed_totals is None
            ):
                return True, f"Sequential terms/continuation page (p{last_primary_page} -> p{first_cand_page})"

    return False, "No strong continuation relationship"




def _merge_identity_facts(
    primary: DocumentIdentityFacts,
    cont: DocumentIdentityFacts,
    conflicts: List[Dict[str, Any]],
) -> DocumentIdentityFacts:
    """Merge continuation identity into primary identity, preserving conflicts and evidence."""
    inv_num = primary.invoice_number
    inv_date = primary.invoice_date
    due_date = primary.due_date
    currency = primary.currency
    inv_type = primary.invoice_type

    field_evs = {k: list(v) for k, v in primary.field_evidence_ids.items()}

    # Invoice Number
    if inv_num is None and cont.invoice_number is not None:
        inv_num = cont.invoice_number
        field_evs["invoice_number"] = list(cont.field_evidence_ids.get("invoice_number", ()))
    elif inv_num is not None and cont.invoice_number is not None:
        if inv_num.strip().upper() == cont.invoice_number.strip().upper():
            c_evs = cont.field_evidence_ids.get("invoice_number", ())
            field_evs["invoice_number"] = list(dict.fromkeys(field_evs.get("invoice_number", []) + list(c_evs)))
        else:
            conflicts.append({
                "field": "invoice_number",
                "primary_value": inv_num,
                "continuation_value": cont.invoice_number,
                "reason": "Conflicting invoice number across assembled continuation pages",
            })

    # Invoice Date
    if inv_date is None and cont.invoice_date is not None:
        inv_date = cont.invoice_date
        field_evs["invoice_date"] = list(cont.field_evidence_ids.get("invoice_date", ()))
    elif inv_date is not None and cont.invoice_date is not None:
        if inv_date == cont.invoice_date:
            c_evs = cont.field_evidence_ids.get("invoice_date", ())
            field_evs["invoice_date"] = list(dict.fromkeys(field_evs.get("invoice_date", []) + list(c_evs)))
        else:
            conflicts.append({
                "field": "invoice_date",
                "primary_value": inv_date,
                "continuation_value": cont.invoice_date,
                "reason": "Conflicting invoice date across assembled continuation pages",
            })

    # Due Date
    if due_date is None and cont.due_date is not None:
        due_date = cont.due_date
        field_evs["due_date"] = list(cont.field_evidence_ids.get("due_date", ()))
    elif due_date is not None and cont.due_date is not None:
        if due_date == cont.due_date:
            c_evs = cont.field_evidence_ids.get("due_date", ())
            field_evs["due_date"] = list(dict.fromkeys(field_evs.get("due_date", []) + list(c_evs)))
        else:
            conflicts.append({
                "field": "due_date",
                "primary_value": due_date,
                "continuation_value": cont.due_date,
                "reason": "Conflicting due date across assembled continuation pages",
            })

    # Currency
    if currency is None and cont.currency is not None:
        currency = cont.currency
        field_evs["currency"] = list(cont.field_evidence_ids.get("currency", ()))
    elif currency is not None and cont.currency is not None:
        if currency.strip().upper() == cont.currency.strip().upper():
            c_evs = cont.field_evidence_ids.get("currency", ())
            field_evs["currency"] = list(dict.fromkeys(field_evs.get("currency", []) + list(c_evs)))
        else:
            conflicts.append({
                "field": "currency",
                "primary_value": currency,
                "continuation_value": cont.currency,
                "reason": "Conflicting currency across assembled continuation pages",
            })

    # Invoice Type
    if inv_type == InvoiceType.UNKNOWN and cont.invoice_type != InvoiceType.UNKNOWN:
        inv_type = cont.invoice_type

    all_ev_ids = _union_evidence([primary.evidence_ids, cont.evidence_ids])

    # If all accounting fields are None, avoid passing empty evidence that violates evidence contract
    has_accounting = any(v is not None for v in (inv_num, inv_date, due_date, currency))
    final_ev_ids = all_ev_ids if has_accounting else ()

    return DocumentIdentityFacts(
        invoice_number=inv_num,
        invoice_date=inv_date,
        due_date=due_date,
        invoice_type=inv_type,
        currency=currency,
        evidence_ids=final_ev_ids,
        field_evidence_ids={k: tuple(v) for k, v in field_evs.items() if v},
        origin=FactOrigin.OBSERVED,
        raw_values={**primary.raw_values, **cont.raw_values},
        metadata={**primary.metadata, **cont.metadata},
    )


def _merge_supplier_facts(
    primary: SupplierIdentityFact,
    cont: SupplierIdentityFact,
    conflicts: List[Dict[str, Any]],
) -> SupplierIdentityFact:
    """Merge continuation supplier attributes into primary supplier fact."""
    name = primary.observed_name or cont.observed_name
    vat = primary.vat_id or cont.vat_id
    country = primary.country or cont.country
    email = primary.email or cont.email
    iban = primary.bank_iban or cont.bank_iban
    address = primary.address or cont.address

    # Conflict checks
    if primary.observed_name and cont.observed_name and primary.observed_name.strip().lower() != cont.observed_name.strip().lower():
        conflicts.append({
            "field": "supplier_name",
            "primary_value": primary.observed_name,
            "continuation_value": cont.observed_name,
            "reason": "Conflicting supplier name across assembled continuation pages",
        })
    if primary.bank_iban and cont.bank_iban and primary.bank_iban.strip().upper() != cont.bank_iban.strip().upper():
        conflicts.append({
            "field": "bank_iban",
            "primary_value": primary.bank_iban,
            "continuation_value": cont.bank_iban,
            "reason": "Conflicting bank IBAN across assembled continuation pages",
        })

    field_evs: Dict[str, List[str]] = {k: list(v) for k, v in primary.field_evidence_ids.items()}
    for k, v in cont.field_evidence_ids.items():
        field_evs[k] = list(dict.fromkeys(field_evs.get(k, []) + list(v)))

    all_evs = _union_evidence([primary.evidence_ids, cont.evidence_ids])
    has_accounting = any(v is not None for v in (name, vat, iban, email, address))
    final_evs = all_evs if has_accounting else ()

    return SupplierIdentityFact(
        observed_name=name,
        vat_id=vat,
        country=country,
        email=email,
        bank_iban=iban,
        address=address,
        evidence_ids=final_evs,
        field_evidence_ids={k: tuple(v) for k, v in field_evs.items() if v},
        origin=FactOrigin.OBSERVED,
        raw_values={**primary.raw_values, **cont.raw_values},
        metadata={**primary.metadata, **cont.metadata},
    )


def _merge_buyer_facts(
    primary: BuyerIdentityFact,
    cont: BuyerIdentityFact,
    conflicts: List[Dict[str, Any]],
) -> BuyerIdentityFact:
    """Merge continuation buyer attributes into primary buyer fact."""
    company = primary.observed_company or cont.observed_company
    bu = primary.business_unit or cont.business_unit
    loc = primary.location or cont.location
    c_code = primary.company_code or cont.company_code
    bu_code = primary.business_unit_code or cont.business_unit_code
    loc_code = primary.location_code or cont.location_code
    addr = primary.invoice_to_address or cont.invoice_to_address

    if primary.observed_company and cont.observed_company and primary.observed_company.strip().lower() != cont.observed_company.strip().lower():
        conflicts.append({
            "field": "buyer_company",
            "primary_value": primary.observed_company,
            "continuation_value": cont.observed_company,
            "reason": "Conflicting buyer company across assembled continuation pages",
        })

    field_evs: Dict[str, List[str]] = {k: list(v) for k, v in primary.field_evidence_ids.items()}
    for k, v in cont.field_evidence_ids.items():
        field_evs[k] = list(dict.fromkeys(field_evs.get(k, []) + list(v)))

    all_evs = _union_evidence([primary.evidence_ids, cont.evidence_ids])
    has_accounting = any(v is not None for v in (company, c_code, bu_code, loc_code, addr))
    final_evs = all_evs if has_accounting else ()

    return BuyerIdentityFact(
        observed_company=company,
        business_unit=bu,
        location=loc,
        company_code=c_code,
        business_unit_code=bu_code,
        location_code=loc_code,
        invoice_to_address=addr,
        evidence_ids=final_evs,
        field_evidence_ids={k: tuple(v) for k, v in field_evs.items() if v},
        origin=FactOrigin.OBSERVED,
        raw_values={**primary.raw_values, **cont.raw_values},
        metadata={**primary.metadata, **cont.metadata},
    )


def _merge_po_facts(
    primary: POFacts,
    cont: POFacts,
    conflicts: List[Dict[str, Any]],
) -> POFacts:
    """Merge continuation PO attributes into primary PO fact."""
    po_num = primary.observed_po_number or cont.observed_po_number

    if primary.observed_po_number and cont.observed_po_number and primary.observed_po_number.strip().upper() != cont.observed_po_number.strip().upper():
        conflicts.append({
            "field": "observed_po_number",
            "primary_value": primary.observed_po_number,
            "continuation_value": cont.observed_po_number,
            "reason": "Conflicting PO number across assembled continuation pages",
        })

    field_evs: Dict[str, List[str]] = {k: list(v) for k, v in primary.field_evidence_ids.items()}
    for k, v in cont.field_evidence_ids.items():
        field_evs[k] = list(dict.fromkeys(field_evs.get(k, []) + list(v)))

    all_evs = _union_evidence([primary.evidence_ids, cont.evidence_ids])
    final_evs = all_evs if po_num is not None else ()

    return POFacts(
        observed_po_number=po_num,
        evidence_ids=final_evs,
        field_evidence_ids={k: tuple(v) for k, v in field_evs.items() if v},
        origin=FactOrigin.OBSERVED,
        raw_values={**primary.raw_values, **cont.raw_values},
        metadata={**primary.metadata, **cont.metadata},
    )


def _merge_lines(
    primary_lines: Sequence[LineFact],
    cont_lines: Sequence[LineFact],
) -> Tuple[LineFact, ...]:
    """Assemble line items across continuation pages.
    
    CRITICAL RULES:
    1. Genuine repeated invoice lines (identical description, qty, price, amount)
       from separate rows or separate pages MUST NOT BE COLLAPSED.
    2. Only identical evidence instances (exact same evidence_ids and row context)
       are deduplicated to prevent accidental double-assembly.
    3. Semantic roles (BILLED_LINE, COMPONENT_DETAIL, etc.) are strictly preserved.
    4. NO ARITHMETIC is performed.
    """
    assembled: List[LineFact] = list(primary_lines)
    seen_evidence_sigs: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()

    for ln in primary_lines:
        sig = (ln.evidence_ids, ln.source_row_evidence_ids)
        if sig != ((), ()):
            seen_evidence_sigs.add(sig)

    for ln in cont_lines:
        sig = (ln.evidence_ids, ln.source_row_evidence_ids)
        if sig != ((), ()) and sig in seen_evidence_sigs:
            # Duplicate observation of the exact same row evidence item
            continue
        assembled.append(ln)
        if sig != ((), ()):
            seen_evidence_sigs.add(sig)

    return tuple(assembled)


def _merge_printed_totals(
    primary: Optional[PrintedTotalsFact],
    cont: Optional[PrintedTotalsFact],
    conflicts: List[Dict[str, Any]],
) -> Optional[PrintedTotalsFact]:
    """Assemble printed totals across continuation pages without reconciliation.
    
    If primary has no totals (e.g. Page 1 & 2 have lines, Page 3 has totals),
    continuation totals become the assembly totals.
    If both have printed totals, agreements are merged and disagreements are preserved as conflicts.
    """
    if primary is None and cont is None:
        return None
    if primary is None:
        return cont
    if cont is None:
        return primary

    # Both have totals: check for agreement vs conflict
    gross = primary.gross_total
    subtotal = primary.subtotal
    net = primary.net
    tax_tot = primary.tax_total
    due = primary.amount_due

    field_evs: Dict[str, List[str]] = {k: list(v) for k, v in primary.field_evidence_ids.items()}
    for k, v in cont.field_evidence_ids.items():
        field_evs[k] = list(dict.fromkeys(field_evs.get(k, []) + list(v)))

    # Gross Total Check
    if gross is None and cont.gross_total is not None:
        gross = cont.gross_total
    elif gross is not None and cont.gross_total is not None and gross != cont.gross_total:
        conflicts.append({
            "field": "gross_total",
            "primary_value": str(gross),
            "continuation_value": str(cont.gross_total),
            "reason": "Conflicting printed gross total across assembled continuation pages",
        })

    # Subtotal Check
    if subtotal is None and cont.subtotal is not None:
        subtotal = cont.subtotal
    elif subtotal is not None and cont.subtotal is not None and subtotal != cont.subtotal:
        conflicts.append({
            "field": "subtotal",
            "primary_value": str(subtotal),
            "continuation_value": str(cont.subtotal),
            "reason": "Conflicting printed subtotal across assembled continuation pages",
        })

    # Net Check
    if net is None and cont.net is not None:
        net = cont.net
    elif net is not None and cont.net is not None and net != cont.net:
        conflicts.append({
            "field": "net",
            "primary_value": str(net),
            "continuation_value": str(cont.net),
            "reason": "Conflicting printed net across assembled continuation pages",
        })

    # Tax Total Check
    if tax_tot is None and cont.tax_total is not None:
        tax_tot = cont.tax_total
    elif tax_tot is not None and cont.tax_total is not None and tax_tot != cont.tax_total:
        conflicts.append({
            "field": "tax_total",
            "primary_value": str(tax_tot),
            "continuation_value": str(cont.tax_total),
            "reason": "Conflicting printed tax total across assembled continuation pages",
        })

    # Amount Due Check
    if due is None and cont.amount_due is not None:
        due = cont.amount_due
    elif due is not None and cont.amount_due is not None and due != cont.amount_due:
        conflicts.append({
            "field": "amount_due",
            "primary_value": str(due),
            "continuation_value": str(cont.amount_due),
            "reason": "Conflicting printed amount due across assembled continuation pages",
        })

    all_evs = _union_evidence([primary.evidence_ids, cont.evidence_ids])
    add_totals = {**primary.additional_totals, **cont.additional_totals}

    return PrintedTotalsFact(
        subtotal=subtotal,
        net=net,
        taxable_base=primary.taxable_base or cont.taxable_base,
        tax_total=tax_tot,
        gross_total=gross,
        amount_due=due,
        payment_total=primary.payment_total or cont.payment_total,
        additional_totals=add_totals,
        evidence_ids=all_evs,
        field_evidence_ids={k: tuple(v) for k, v in field_evs.items() if v},
        origin=FactOrigin.OBSERVED,
        raw_values={**primary.raw_values, **cont.raw_values},
        metadata={**primary.metadata, **cont.metadata},
    )


def _merge_continuation_facts(
    primary: DocumentFacts,
    cont: DocumentFacts,
    assembly_conflicts: List[Dict[str, Any]],
) -> DocumentFacts:
    """Merge continuation DocumentFacts into primary DocumentFacts."""
    # 1. Identity
    merged_identity = _merge_identity_facts(primary.identity, cont.identity, assembly_conflicts)

    # 2. Parties
    merged_parties = PartyIdentityFacts(
        supplier=_merge_supplier_facts(primary.parties.supplier, cont.parties.supplier, assembly_conflicts),
        buyer=_merge_buyer_facts(primary.parties.buyer, cont.parties.buyer, assembly_conflicts),
        evidence_ids=_union_evidence([primary.parties.evidence_ids, cont.parties.evidence_ids]),
    )

    # 3. PO
    merged_po = _merge_po_facts(primary.po, cont.po, assembly_conflicts)

    # 4. Financials
    assembled_lines = _merge_lines(primary.financials.lines, cont.financials.lines)

    # Discounts & Charges
    assembled_discounts = tuple(list(primary.financials.discounts) + list(cont.financials.discounts))
    assembled_charges = tuple(list(primary.financials.charges) + list(cont.financials.charges))

    # Taxes (preserving placement: header vs line)
    assembled_taxes = tuple(list(primary.financials.taxes) + list(cont.financials.taxes))

    # Printed Totals
    assembled_totals = _merge_printed_totals(
        primary.financials.printed_totals,
        cont.financials.printed_totals,
        assembly_conflicts,
    )

    merged_financials = FinancialFacts(
        lines=assembled_lines,
        discounts=assembled_discounts,
        charges=assembled_charges,
        taxes=assembled_taxes,
        printed_totals=assembled_totals,
        currency=merged_identity.currency,
        evidence_ids=_union_evidence([primary.financials.evidence_ids, cont.financials.evidence_ids]),
        metadata={**primary.financials.metadata, **cont.financials.metadata},
    )

    # Combined page numbers
    all_pages = tuple(sorted(list(dict.fromkeys(list(primary.page_numbers) + list(cont.page_numbers)))))

    # Combined supporting group IDs
    all_sup_ids = tuple(sorted(list(dict.fromkeys(
        list(primary.supporting_group_ids) + list(cont.supporting_group_ids)
    ))))

    # Combined conflicts
    all_conflicts = tuple(list(primary.conflicting_facts) + list(cont.conflicting_facts) + assembly_conflicts)

    # Combined top-level evidence IDs
    all_evs = _union_evidence([
        primary.evidence_ids,
        cont.evidence_ids,
        merged_identity.evidence_ids,
        merged_parties.evidence_ids,
        merged_po.evidence_ids,
        merged_financials.evidence_ids,
    ])

    return DocumentFacts(
        document_id=primary.document_id,
        group_id=primary.group_id,
        page_numbers=all_pages,
        document_role=primary.document_role,
        payable_relevance=primary.payable_relevance,
        supporting_group_ids=all_sup_ids,
        identity=merged_identity,
        parties=merged_parties,
        po=merged_po,
        financials=merged_financials,
        conflicting_facts=all_conflicts,
        evidence_ids=all_evs,
        metadata={**primary.metadata, **cont.metadata},
        provenance={
            **primary.provenance,
            "assembled_continuation_groups": list(dict.fromkeys(
                primary.provenance.get("assembled_continuation_groups", []) + [cont.group_id]
            )),
        },
    )


# ══════════════════════════════════════════════════════════════════════════
# Main Assembly Engine
# ══════════════════════════════════════════════════════════════════════════

def assemble_financial_documents(
    document_facts: Sequence[DocumentFacts],
    groups: Optional[Sequence[DocumentGroup]] = None,
) -> List[FinancialDocumentAssembly]:
    """Assemble consolidated DocumentFacts into logical FinancialDocumentAssembly objects.
    
    Determines which already-consolidated facts belong to the same logical financial
    document across pages and groups without performing accounting calculations,
    repair, master matching, or OCR/Vision calls.
    
    Parameters:
        document_facts: Sequence of consolidated DocumentFacts from Phase 9B-2.
        groups: Optional sequence of Phase 7C DocumentGroup objects containing
                pagination and continuation grouping signals.
                
    Returns:
        List of typed FinancialDocumentAssembly objects, each representing
        one logical financial payable document.
    """
    if not document_facts:
        return []

    # Map groups by group_id if provided
    groups_by_id: Dict[str, DocumentGroup] = {}
    if groups:
        for g in groups:
            groups_by_id[g.group_id] = g

    # 1. Deterministic Partitioning by physical document_id
    facts_by_doc: Dict[str, List[DocumentFacts]] = {}
    for df in document_facts:
        facts_by_doc.setdefault(df.document_id, []).append(df)

    assemblies: List[FinancialDocumentAssembly] = []

    for doc_id, doc_facts_list in facts_by_doc.items():
        # Sort facts deterministically by first page number, preserving document structure
        sorted_facts = sorted(doc_facts_list, key=lambda f: min(f.page_numbers) if f.page_numbers else 9999)

        # Active payable assembly builders for this document:
        # list of dicts: {"primary_facts": DocumentFacts, "primary_group_ids": list, "supporting_group_ids": list, "conflicts": list, "provenance": dict}
        active_assemblies: List[Dict[str, Any]] = []

        for f in sorted_facts:
            f_group = groups_by_id.get(f.group_id) if f.group_id else None

            # Check if this fact is a supporting document
            curr_primary = (
                active_assemblies[-1]["primary_facts"]
                if active_assemblies else None
            )
            is_sup = _is_supporting_facts(f, f_group, curr_primary)

            if is_sup:
                # Supporting Document: Attach as supporting reference to active primary payable
                if active_assemblies:
                    target_asm = active_assemblies[-1]
                    if f.group_id and f.group_id not in target_asm["supporting_group_ids"]:
                        target_asm["supporting_group_ids"].append(f.group_id)
                    target_asm["group_provenance"][f.group_id or f"p{min(f.page_numbers)}"] = {
                        "role": f.document_role.value,
                        "relevance": f.payable_relevance.value,
                        "relationship": "supporting_isolated",
                        "page_numbers": list(f.page_numbers),
                    }
                else:
                    # Supporting document appearing before any payable -> create independent supporting assembly
                    # (only if no payables exist at all in document)
                    new_sup_asm = {
                        "primary_facts": f,
                        "primary_group_ids": [f.group_id] if f.group_id else [],
                        "supporting_group_ids": [],
                        "conflicts": list(f.conflicting_facts),
                        "group_provenance": {
                            f.group_id or f"p{min(f.page_numbers)}": {
                                "role": f.document_role.value,
                                "relevance": f.payable_relevance.value,
                                "relationship": "standalone_supporting",
                                "page_numbers": list(f.page_numbers),
                            }
                        },
                    }
                    active_assemblies.append(new_sup_asm)
                continue

            # Non-supporting document: Evaluate continuation join vs new payable assembly
            joined = False
            for asm in reversed(active_assemblies):
                # Only join into an active payable candidate (not standalone supporting)
                if asm["primary_facts"].payable_relevance == PayableRelevance.SUPPORTING:
                    continue

                primary_grp = groups_by_id.get(asm["primary_facts"].group_id) if asm["primary_facts"].group_id else None
                can_join, reason = _can_join_continuation(
                    asm["primary_facts"],
                    f,
                    primary_grp,
                    f_group,
                )
                if can_join:
                    # Assemble continuation facts into primary facts
                    step_conflicts: List[Dict[str, Any]] = []
                    merged_facts = _merge_continuation_facts(
                        asm["primary_facts"],
                        f,
                        step_conflicts,
                    )
                    asm["primary_facts"] = merged_facts
                    if f.group_id and f.group_id not in asm["primary_group_ids"]:
                        asm["primary_group_ids"].append(f.group_id)
                    asm["conflicts"].extend(step_conflicts)
                    asm["group_provenance"][f.group_id or f"p{min(f.page_numbers)}"] = {
                        "role": f.document_role.value,
                        "relevance": f.payable_relevance.value,
                        "relationship": "continuation",
                        "reason": reason,
                        "page_numbers": list(f.page_numbers),
                    }
                    joined = True
                    break

            if not joined:
                # Start a new logical FinancialDocumentAssembly
                asm_entry = {
                    "primary_facts": f,
                    "primary_group_ids": [f.group_id] if f.group_id else [],
                    "supporting_group_ids": list(f.supporting_group_ids),
                    "conflicts": list(f.conflicting_facts),
                    "group_provenance": {
                        f.group_id or f"p{min(f.page_numbers)}": {
                            "role": f.document_role.value,
                            "relevance": f.payable_relevance.value,
                            "relationship": "primary_anchor",
                            "page_numbers": list(f.page_numbers),
                        }
                    },
                }
                active_assemblies.append(asm_entry)

        # 2. Finalize active assembly builders into immutable FinancialDocumentAssembly objects
        for idx, asm_data in enumerate(active_assemblies):
            pf: DocumentFacts = asm_data["primary_facts"]
            prim_gids = tuple(asm_data["primary_group_ids"])
            sup_gids = tuple(sorted(list(dict.fromkeys(asm_data["supporting_group_ids"]))))

            # Deterministic assembly ID anchored in document_id and primary group/page
            anchor_key = prim_gids[0] if prim_gids else f"asm_{idx + 1}"
            clean_anchor = anchor_key.split(":")[-1] if ":" in anchor_key else anchor_key
            asm_id = f"{doc_id}:assembly:{clean_anchor}"

            # Update DocumentFacts with finalized supporting groups if not already set
            if sup_gids and pf.supporting_group_ids != sup_gids:
                pf = DocumentFacts(
                    document_id=pf.document_id,
                    group_id=pf.group_id,
                    page_numbers=pf.page_numbers,
                    document_role=pf.document_role,
                    payable_relevance=pf.payable_relevance,
                    supporting_group_ids=sup_gids,
                    identity=pf.identity,
                    parties=pf.parties,
                    po=pf.po,
                    financials=pf.financials,
                    conflicting_facts=pf.conflicting_facts,
                    evidence_ids=pf.evidence_ids,
                    metadata=pf.metadata,
                    provenance=pf.provenance,
                )

            all_ev_ids = _union_evidence([
                pf.evidence_ids,
                *(groups_by_id[gid].evidence_ids for gid in prim_gids if gid in groups_by_id),
            ])

            assembly = FinancialDocumentAssembly(
                assembly_id=asm_id,
                document_id=doc_id,
                primary_group_ids=prim_gids,
                supporting_group_ids=sup_gids,
                page_numbers=pf.page_numbers,
                document_role=pf.document_role,
                payable_relevance=pf.payable_relevance,
                facts=pf,
                group_provenance=asm_data["group_provenance"],
                conflicts=tuple(asm_data["conflicts"]),
                evidence_ids=all_ev_ids,
                assembly_provenance={
                    "assembler": "DeterministicFinancialAssembler",
                    "version": "Phase9B-3-1.0",
                    "primary_group_count": len(prim_gids),
                    "supporting_group_count": len(sup_gids),
                    "total_pages": len(pf.page_numbers),
                    "conflict_count": len(asm_data["conflicts"]),
                },
            )
            assemblies.append(assembly)

    return assemblies
