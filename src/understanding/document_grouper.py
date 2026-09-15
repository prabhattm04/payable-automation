"""src/understanding/document_grouper.py — Multi-Page Document Grouping Layer.

Phase 7C: Determines which pages in the same physical document belong to the same
logical document group, using Phase 7A PageEvidence and Phase 7B PageUnderstanding.

Key Architecture & Principles
-----------------------------
1. Evidence-Based Grouping:
   Operates strictly on PageEvidence (text & geometry) and PageUnderstanding
   (page role & payable relevance). Never reads raw OCR JSON directly.
2. Non-Adjacency Support:
   Does NOT rely on page adjacency alone. Interleaved documents (e.g. Doc A p1,
   Doc B p1, Doc A p2) correctly associate continuation pages with their matching
   logical group via explicit document identity.
3. Strict Order Preservation:
   Page numbers within every group are strictly sorted in ascending sequence.
4. Core Grouping Principle:
   - Strong explicit identity / continuation evidence can JOIN pages.
   - Strong contradictory identity evidence can SEPARATE pages.
   - Weak or ambiguous evidence does NOT force a join.
5. No Fabricated Confidence:
   Deterministic grouping baseline sets `confidence = None` (null in JSON).
6. Explainability:
   Every grouping decision traces:
   group -> grouping_signals -> page_numbers -> evidence_ids.
7. Zero Dataset-Specific Rules:
   No hardcoded filenames, document IDs, sample numbers, or page-number shortcuts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from src.understanding.evidence import Evidence, PageEvidence
from src.understanding.page_classifier import PageRole, PayableRelevance, PageUnderstanding, classify_page
from src.utils.logging import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Document Identity & Pagination Pattern Extractors
# ══════════════════════════════════════════════════════════════════════════

# Common words to exclude from document numbers
_STOP_WORDS: set[str] = {
    "NUMBER", "NO", "NR", "DATE", "INFORMATION", "NOTES", "GROUP",
    "TOTAL", "OICE", "PAGE", "SEITE", "FOLHA", "DETAILS", "CONSOLIDATED",
    "DETAILED", "COMMERCIAL", "INVOICE", "RECHNUNG", "FACTURA", "FACTURE",
    "ARVE", "ITEM", "CODE", "TYPE", "TERMS", "DUE", "VAT", "TAX", "AMOUNT",
}

# 1. Single-block pattern: label + value in the same text block
_DOC_INLINE_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(?:tax\s+)?(?:invoice|rechnung|factura|facture|arve)\s*(?:no\.?|nr\.?|number|#)?\s*[:#\-]?\s*([A-Z0-9\-/]{3,})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:credit\s*(?:memo|note)|gutschrift)\s*(?:no\.?|nr\.?|number|#)?\s*[:#\-]?\s*([A-Z0-9\-/]{3,})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:rechnungs|rechnung)\s*[-:]?\s*(?:nr\.?|nummer)\s*[:#\-]?\s*([A-Z0-9\-/]{3,})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:número|n°|n\.°)\s*(?:da\s+)?(?:fa[ck]tura|doc(?:umento)?)\s*[:#\-]?\s*([A-Z0-9\-/]{3,})\b",
        re.IGNORECASE,
    ),
]

# 2. Split-block label pattern: label alone in block i, value in block i+1
_DOC_LABEL_ONLY_PATTERN = re.compile(
    r"^\s*(?:tax\s+)?(?:invoice|rechnung|factura|facture|arve|credit\s*(?:memo|note)|gutschrift)\s*(?:no\.?|nr\.?|number|#)?\s*[:#\-]?\s*$",
    re.IGNORECASE,
)

# 3. Explicit Pagination Patterns: "Page X of Y", "Seite X von Y", "Folha X de Y"
_PAGINATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bpage\s*(\d+)\s*(?:of|/)\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bseite\s*(\d+)\s*(?:von|/)\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bfolha\s*(\d+)\s*(?:de|/)\s*(\d+)\b", re.IGNORECASE),
]


# ══════════════════════════════════════════════════════════════════════════
# Internal Page Identity Representation
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class _PageIdentity:
    """Internal grouping identity extracted from PageEvidence and PageUnderstanding."""
    page_number: int
    page_role: PageRole
    payable_relevance: PayableRelevance
    document_numbers: set[str]
    doc_number_evidence_ids: list[str]
    cur_page: Optional[int]
    total_pages: Optional[int]
    pagination_evidence_ids: list[str]
    is_continuation_role: bool
    evidence_ids: list[str]


def _extract_page_identity(
    page_ev: PageEvidence,
    understanding: PageUnderstanding,
) -> _PageIdentity:
    """Extract grouping-relevant identity markers from a single page."""
    doc_numbers: set[str] = set()
    doc_num_ev_ids: list[str] = []
    
    cur_p: Optional[int] = None
    tot_p: Optional[int] = None
    pag_ev_ids: list[str] = []

    items = page_ev.items
    num_items = len(items)

    for i, item in enumerate(items):
        text = item.content.strip() if item.content else ""
        if not text:
            continue

        # A. Inline pattern matching (label + number in same block)
        for pat in _DOC_INLINE_PATTERNS:
            m = pat.search(text)
            if m:
                cand = m.group(1).strip().upper()
                if (
                    len(cand) >= 3
                    and cand not in _STOP_WORDS
                    and any(ch.isdigit() for ch in cand)
                ):
                    doc_numbers.add(cand)
                    if item.evidence_id and item.evidence_id not in doc_num_ev_ids:
                        doc_num_ev_ids.append(item.evidence_id)

        # B. Split pattern matching: label in block i, number in block i+1
        if _DOC_LABEL_ONLY_PATTERN.match(text) and (i + 1 < num_items):
            next_item = items[i + 1]
            next_text = next_item.content.strip().upper()
            if (
                3 <= len(next_text) <= 30
                and re.fullmatch(r"[A-Z0-9\-/]+", next_text)
                and any(ch.isdigit() for ch in next_text)
                and next_text not in _STOP_WORDS
            ):
                doc_numbers.add(next_text)
                if item.evidence_id and item.evidence_id not in doc_num_ev_ids:
                    doc_num_ev_ids.append(item.evidence_id)
                if next_item.evidence_id and next_item.evidence_id not in doc_num_ev_ids:
                    doc_num_ev_ids.append(next_item.evidence_id)

        # C. Pagination markers
        for pat in _PAGINATION_PATTERNS:
            m = pat.search(text)
            if m:
                try:
                    c = int(m.group(1))
                    t = int(m.group(2))
                    cur_p = c
                    tot_p = t
                    if item.evidence_id and item.evidence_id not in pag_ev_ids:
                        pag_ev_ids.append(item.evidence_id)
                except (ValueError, IndexError):
                    pass

    # Use pagination from PageUnderstanding signals if not found directly
    if cur_p is None and understanding.signals.get("detected_pagination"):
        parts = str(understanding.signals["detected_pagination"]).split("/")
        if len(parts) == 2:
            try:
                cur_p = int(parts[0])
                tot_p = int(parts[1])
            except ValueError:
                pass

    all_ev_ids = list(dict.fromkeys(
        doc_num_ev_ids + pag_ev_ids + understanding.evidence_ids
    ))

    return _PageIdentity(
        page_number=page_ev.page_number,
        page_role=understanding.page_role,
        payable_relevance=understanding.payable_relevance,
        document_numbers=doc_numbers,
        doc_number_evidence_ids=doc_num_ev_ids,
        cur_page=cur_p,
        total_pages=tot_p,
        pagination_evidence_ids=pag_ev_ids,
        is_continuation_role=(understanding.page_role == PageRole.CONTINUATION),
        evidence_ids=all_ev_ids,
    )


# ══════════════════════════════════════════════════════════════════════════
# Typed Result Models
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class DocumentGroup:
    """A logical document group comprising one or more pages of a physical document."""
    group_id: str
    document_id: str
    page_numbers: list[int]
    evidence_ids: list[str] = field(default_factory=list)
    grouping_signals: dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None  # None for deterministic baseline
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Guarantee strictly sorted, unique page numbers
        self.page_numbers = sorted(list(dict.fromkeys(int(p) for p in self.page_numbers)))
        if not self.page_numbers:
            raise ValueError("DocumentGroup must contain at least one page number")

    def to_dict(self) -> dict[str, Any]:
        """Serialize DocumentGroup to dictionary."""
        return {
            "group_id": self.group_id,
            "document_id": self.document_id,
            "page_numbers": self.page_numbers,
            "evidence_ids": self.evidence_ids,
            "grouping_signals": self.grouping_signals,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentGroup:
        """Construct DocumentGroup from dictionary."""
        return cls(
            group_id=data["group_id"],
            document_id=data["document_id"],
            page_numbers=data["page_numbers"],
            evidence_ids=data.get("evidence_ids", []),
            grouping_signals=data.get("grouping_signals", {}),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            provenance=data.get("provenance", {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> DocumentGroup:
        return cls.from_dict(json.loads(json_str))


@dataclass
class GroupingResult:
    """The complete logical document grouping result for a physical document."""
    document_id: str
    groups: list[DocumentGroup] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[DocumentGroup]:
        return iter(self.groups)

    def __getitem__(self, idx: int) -> DocumentGroup:
        return self.groups[idx]

    def get_group_for_page(self, page_number: int) -> Optional[DocumentGroup]:
        """Return the DocumentGroup containing the specified 1-indexed page number."""
        for g in self.groups:
            if page_number in g.page_numbers:
                return g
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize GroupingResult to dictionary."""
        return {
            "document_id": self.document_id,
            "groups": [g.to_dict() for g in self.groups],
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroupingResult:
        """Construct GroupingResult from dictionary."""
        groups = [DocumentGroup.from_dict(gd) for gd in data.get("groups", [])]
        return cls(
            document_id=data["document_id"],
            groups=groups,
            provenance=data.get("provenance", {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> GroupingResult:
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# Active Logical Group Tracker (Internal)
# ══════════════════════════════════════════════════════════════════════════

class _ActiveGroup:
    """Helper representing an open or completed logical group during scanning."""
    def __init__(self, anchor_page: int, document_id: str) -> None:
        self.anchor_page = anchor_page
        self.document_id = document_id
        self.page_numbers: list[int] = [anchor_page]
        self.document_numbers: set[str] = set()
        self.expected_total_pages: Optional[int] = None
        self.last_seq_page: Optional[int] = None
        self.roles: list[PageRole] = []
        self.evidence_ids: list[str] = []
        self.reasons: list[str] = []
        self.signals: dict[str, Any] = {
            "shared_document_numbers": [],
            "sequential_pagination": False,
            "continuation_links": 0,
        }

    def can_accept_page(self, p: _PageIdentity) -> tuple[bool, int, str]:
        """Evaluate if page p can join this active logical group.
        
        Returns:
            (can_join: bool, score: int, reason: str)
        """
        # Strong contradiction check 1: Disjoint explicit document numbers
        if self.document_numbers and p.document_numbers:
            overlap = self.document_numbers.intersection(p.document_numbers)
            if not overlap:
                return False, 0, "Conflicting explicit document numbers"

        # Strong contradiction check 2: Page p declares Page 1 when group already has Page 1
        if p.cur_page == 1 and 1 in [self.last_seq_page, self.anchor_page]:
            return False, 0, "Page declares new Page 1 start"

        # Strong contradiction check 3: Group already completed its declared total pages
        if (
            self.expected_total_pages is not None
            and len(self.page_numbers) >= self.expected_total_pages
        ):
            return False, 0, f"Group already satisfied declared total pages ({self.expected_total_pages})"

        score = 0
        reasons: list[str] = []

        # ── Strong Join Signal A: Explicit Document Number Match ──
        if self.document_numbers and p.document_numbers:
            overlap = self.document_numbers.intersection(p.document_numbers)
            if overlap:
                score += 10
                reasons.append(f"Matching document number ({', '.join(overlap)})")

        # ── Strong Join Signal B: Sequential Pagination Continuation ──
        if (
            p.cur_page is not None
            and p.total_pages is not None
            and self.expected_total_pages is not None
        ):
            if (
                p.total_pages == self.expected_total_pages
                and self.last_seq_page is not None
                and p.cur_page == self.last_seq_page + 1
            ):
                score += 15
                reasons.append(f"Sequential pagination ({p.cur_page} of {p.total_pages})")

        # ── Strong Join Signal C: Continuation Role with open preceding group ──
        if p.is_continuation_role:
            score += 8
            reasons.append("Page role continuation marker")

        # Threshold for joining: Requires strong evidence (score >= 8)
        if score >= 8:
            return True, score, "; ".join(reasons)

        return False, score, "Insufficient join signals"

    def add_page(self, p: _PageIdentity, reason: str) -> None:
        """Attach page p to this active group."""
        self.page_numbers.append(p.page_number)
        self.document_numbers.update(p.document_numbers)
        if p.total_pages is not None:
            self.expected_total_pages = p.total_pages
        if p.cur_page is not None:
            self.last_seq_page = p.cur_page
        self.roles.append(p.page_role)
        for eid in p.evidence_ids:
            if eid not in self.evidence_ids:
                self.evidence_ids.append(eid)
        if reason:
            self.reasons.append(f"Page {p.page_number}: {reason}")
            if "Sequential pagination" in reason:
                self.signals["sequential_pagination"] = True
            if "Matching document number" in reason:
                self.signals["shared_document_numbers"] = sorted(list(self.document_numbers))
            if "continuation marker" in reason:
                self.signals["continuation_links"] += 1

    def to_document_group(self) -> DocumentGroup:
        """Finalize into immutable typed DocumentGroup with deterministic ID grounded in first page."""
        first_page = min(self.page_numbers)
        group_id = f"{self.document_id}:group:p{first_page}"
        return DocumentGroup(
            group_id=group_id,
            document_id=self.document_id,
            page_numbers=self.page_numbers,
            evidence_ids=self.evidence_ids,
            grouping_signals=self.signals,
            confidence=None,  # Strictly None for deterministic baseline
            provenance={
                "grouper": "DeterministicDocumentGrouper",
                "version": "Phase7C-1.0",
                "anchor_page": first_page,
                "reasons": self.reasons,
            },
        )


# ══════════════════════════════════════════════════════════════════════════
# Main Grouping Engine
# ══════════════════════════════════════════════════════════════════════════

def group_document(
    pages: Sequence[PageEvidence],
    understandings: Optional[Sequence[PageUnderstanding]] = None,
) -> GroupingResult:
    """Partition the pages of a physical document into logical document groups.
    
    CRITICAL SCOPE CONSTRAINTS:
    - Purely structural/grouping layer.
    - Operates strictly on PageEvidence and PageUnderstanding.
    - Does NOT calculate payable totals or perform accounting extraction.
    - Zero dataset-specific constants or filename rules.
    - Invariant to document_id value.
    - Non-adjacent interleaved documents are joined correctly to their matching anchor.
    - Ambiguous evidence is never forcibly merged.
    """
    if not pages:
        return GroupingResult(document_id="empty", groups=[])

    # If understandings not pre-computed, classify each page
    if understandings is None or len(understandings) != len(pages):
        computed_understandings = [classify_page(p) for p in pages]
    else:
        computed_understandings = list(understandings)

    # Invariance check: document_id is taken from first page but never used in matching logic
    doc_id = pages[0].document_id

    # 1. Extract lightweight identity representation per page
    identities: list[_PageIdentity] = [
        _extract_page_identity(page_ev, und)
        for page_ev, und in zip(pages, computed_understandings)
    ]

    active_groups: list[_ActiveGroup] = []

    # 2. Iterate through pages and evaluate join vs separation
    for p in identities:
        best_group: Optional[_ActiveGroup] = None
        best_score = -1
        best_reason = ""

        # Check existing active groups in reverse (preferring closest compatible open group)
        for grp in reversed(active_groups):
            can_join, score, reason = grp.can_accept_page(p)
            if can_join and score > best_score:
                best_group = grp
                best_score = score
                best_reason = reason
                break  # Accept highest-priority matching group

        if best_group is not None:
            # Strong explicit evidence joined page p to existing group
            best_group.add_page(p, best_reason)
        else:
            # Start a new logical document group
            new_grp = _ActiveGroup(anchor_page=p.page_number, document_id=doc_id)
            new_grp.document_numbers.update(p.document_numbers)
            if p.total_pages is not None:
                new_grp.expected_total_pages = p.total_pages
            if p.cur_page is not None:
                new_grp.last_seq_page = p.cur_page
            new_grp.roles.append(p.page_role)
            new_grp.evidence_ids.extend(p.evidence_ids)
            new_grp.reasons.append(
                f"Page {p.page_number}: Starting new logical group (role={p.page_role.value})"
            )
            if p.document_numbers:
                new_grp.signals["shared_document_numbers"] = sorted(list(p.document_numbers))
            active_groups.append(new_grp)

    # 3. Finalize all active groups into DocumentGroup models
    final_groups = [g.to_document_group() for g in active_groups]

    # Ensure groups are ordered by their initial anchor page
    final_groups.sort(key=lambda g: g.page_numbers[0])

    return GroupingResult(
        document_id=doc_id,
        groups=final_groups,
        provenance={
            "grouper": "DeterministicDocumentGrouper",
            "version": "Phase7C-1.0",
            "total_physical_pages": len(pages),
            "logical_group_count": len(final_groups),
        },
    )
