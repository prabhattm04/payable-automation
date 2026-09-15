"""src/understanding/qwen_router.py — Selective Qwen Vision Routing Layer.

Phase 7D: Deterministic routing layer that decides when RapidOCR / native evidence
is sufficient and when a page/group should be escalated to Qwen Vision for
multimodal semantic, layout, or financial disambiguation.

Core Principles
---------------
1. RapidOCR is the default perception layer; Qwen is an escalation layer.
2. Routing decisions are based on combinations of generic signals:
   - OCR quality (mean confidence, low-confidence ratio)
   - Layout complexity (block count, multi-column density, table alignment)
   - Classification ambiguity (unknown role or ambiguous payable relevance)
   - Financial structural complexity (multi-tier taxes, discounts, charges)
   - Grouping ambiguity (unresolved continuation, conflicting identity)
3. Decoupled & Deterministic:
   - `route_page()` is 100% deterministic, network-free, and grounded in evidence.
   - `execute_vision_escalation()` optionally runs the existing Phase 6C provider.
4. No Fabricated Confidence:
   `confidence` is strictly `None` (null in JSON).
5. Zero Dataset-Specific Rules:
   No filename, document_id, or page-number conditionals.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.understanding.evidence import Evidence, PageEvidence, vision_response_to_evidence
from src.understanding.page_classifier import PageRole, PayableRelevance, PageUnderstanding, classify_page
from src.understanding.document_grouper import DocumentGroup
from src.vision.provider import VisionProvider, VisionResponse
from src.utils.logging import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Controlled Decision Vocabulary
# ══════════════════════════════════════════════════════════════════════════

class RouterDecisionType(str, Enum):
    """Controlled routing decision vocabulary."""
    OCR_SUFFICIENT = "ocr_sufficient"
    ROUTE_TO_VISION = "route_to_vision"


# ══════════════════════════════════════════════════════════════════════════
# Typed Result Model
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RoutingDecision:
    """Deterministic routing decision with metrics, explainability, and audit trail."""
    document_id: str
    page_number: int
    decision: RouterDecisionType
    reasons: list[str]
    signals: dict[str, Any]
    evidence_ids: list[str] = field(default_factory=list)
    confidence: Optional[float] = None  # Strictly None for deterministic routing
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert routing decision to clean dictionary."""
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "decision": self.decision.value,
            "reasons": self.reasons,
            "signals": self.signals,
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingDecision:
        """Construct RoutingDecision from dictionary."""
        return cls(
            document_id=data["document_id"],
            page_number=int(data["page_number"]),
            decision=RouterDecisionType(data["decision"]),
            reasons=data.get("reasons", []),
            signals=data.get("signals", {}),
            evidence_ids=data.get("evidence_ids", []),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            provenance=data.get("provenance", {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> RoutingDecision:
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# Signal Extraction Helpers
# ══════════════════════════════════════════════════════════════════════════

_DISCOUNT_PATTERNS = [
    re.compile(r"\b(?:discounts?|descontos?|rabatte?|remises?|rebates?)\b", re.IGNORECASE),
]
_CHARGE_PATTERNS = [
    re.compile(r"\b(?:freight|shipping|transport|delivery\s*charges?|insurance|surcharges?|dut(?:y|ies)|iec|lev(?:y|ies))\b", re.IGNORECASE),
]
_WITHHOLDING_PATTERNS = [
    re.compile(r"\b(?:withholding\s*tax(?:es)?|wht|retention|retenci[oó]n|ภาษีหัก\s*ณ\s*ที่จ่าย)\b", re.IGNORECASE),
]
_TAX_PATTERNS = [
    re.compile(r"\b(?:vat|mwst|iva|tva|gst|sst|tax(?:es)?|käibemaks|ภาษีมูลค่าเพิ่ม)\b", re.IGNORECASE),
]


def _analyze_layout_and_geometry(items: Sequence[Evidence]) -> dict[str, Any]:
    """Analyze spatial layout, columns, and tabular alignment from bounding boxes."""
    block_count = len(items)
    valid_boxes = [e.bbox for e in items if e.bbox and len(e.bbox) == 4]
    
    if not valid_boxes:
        return {
            "block_count": block_count,
            "detected_columns": 1,
            "table_rows": 0,
            "is_complex_layout": False,
        }

    # Group into vertical tracks (columns) based on x0 coordinates
    x_starts = sorted([b[0] for b in valid_boxes])
    min_x, max_x = min(x_starts), max(b[2] for b in valid_boxes)
    page_width = max(max_x - min_x, 100.0)

    # Cluster x-centers into distinct vertical columns (using 8% page width bucket threshold)
    bucket_size = page_width * 0.08
    column_buckets: list[float] = []
    for x in x_starts:
        matched = False
        for c in column_buckets:
            if abs(x - c) <= bucket_size:
                matched = True
                break
        if not matched:
            column_buckets.append(x)
    detected_columns = len(column_buckets)

    # Detect row alignment (multiple items sharing similar y0 within 6 pixels)
    y_starts = [b[1] for b in valid_boxes]
    row_clusters: dict[int, int] = {}
    for y in y_starts:
        bucket = int(round(y / 8.0)) * 8
        row_clusters[bucket] = row_clusters.get(bucket, 0) + 1
    
    # Rows with at least 3 horizontally aligned items
    table_rows = sum(1 for count in row_clusters.values() if count >= 3)

    # Layout complexity rule (combination of block count + column structure + table alignment)
    # A complex layout requires:
    # (High block count >= 80 AND >= 4 columns AND >= 5 table rows)
    # OR (Dense tabular grid: >= 100 blocks AND >= 3 columns)
    is_complex_layout = (
        (block_count >= 80 and detected_columns >= 4 and table_rows >= 5)
        or (block_count >= 100 and detected_columns >= 3)
    )

    return {
        "block_count": block_count,
        "detected_columns": detected_columns,
        "table_rows": table_rows,
        "is_complex_layout": is_complex_layout,
    }


def _analyze_financial_structure(items: Sequence[Evidence]) -> dict[str, Any]:
    """Detect multi-tier structural financial adjustments (taxes, discounts, charges, withholding)."""
    discounts_found = 0
    charges_found = 0
    withholding_found = 0
    tax_mentions = 0
    ev_ids: list[str] = []

    for item in items:
        text = item.content
        if not text:
            continue
        
        hit = False
        for pat in _DISCOUNT_PATTERNS:
            if pat.search(text):
                discounts_found += 1
                hit = True
                break
        for pat in _CHARGE_PATTERNS:
            if pat.search(text):
                charges_found += 1
                hit = True
                break
        for pat in _WITHHOLDING_PATTERNS:
            if pat.search(text):
                withholding_found += 1
                hit = True
                break
        for pat in _TAX_PATTERNS:
            if pat.search(text):
                tax_mentions += 1
                hit = True
                break
        
        if hit and item.evidence_id and item.evidence_id not in ev_ids:
            ev_ids.append(item.evidence_id)

    # Financial structural complexity:
    # Multiple competing adjustment types on the same page (e.g. discounts + charges/excise + tax)
    # OR (simultaneous withholding and standard VAT)
    # OR (discounts + taxes + charges)
    distinct_types = sum([
        1 if discounts_found > 0 else 0,
        1 if charges_found > 0 else 0,
        1 if withholding_found > 0 else 0,
        1 if tax_mentions > 0 else 0,
    ])

    is_complex_financial = (
        distinct_types >= 3
        or (discounts_found > 0 and tax_mentions >= 2 and charges_found > 0)
        or (withholding_found > 0 and discounts_found > 0)
    )

    return {
        "discount_signals": discounts_found,
        "charge_signals": charges_found,
        "withholding_signals": withholding_found,
        "tax_signals": tax_mentions,
        "distinct_financial_adjustment_types": distinct_types,
        "is_complex_financial_structure": is_complex_financial,
        "evidence_ids": ev_ids,
    }


# ══════════════════════════════════════════════════════════════════════════
# Main Deterministic Router Engine
# ══════════════════════════════════════════════════════════════════════════

def route_page(
    page_evidence: PageEvidence,
    page_understanding: Optional[PageUnderstanding] = None,
    group: Optional[DocumentGroup] = None,
) -> RoutingDecision:
    """Determine whether RapidOCR is sufficient or if Qwen Vision is required.
    
    CRITICAL CONSTRAINTS:
    - 100% deterministic and network-free.
    - Zero document ID or filename rules.
    - Confidence is strictly None (uncalibrated rule-based routing).
    - Preserves all supporting evidence IDs for full auditability.
    """
    items = page_evidence.items
    doc_id = page_evidence.document_id
    page_num = page_evidence.page_number

    if page_understanding is None:
        page_understanding = classify_page(page_evidence)

    reasons: list[str] = []
    supporting_evidence_ids: list[str] = []

    def _add_ids(eids: Sequence[str]) -> None:
        for eid in eids:
            if eid and eid not in supporting_evidence_ids:
                supporting_evidence_ids.append(eid)

    # ── 1. OCR Quality Signals ──
    confs = [item.confidence for item in items if item.confidence is not None]
    if confs:
        mean_conf = sum(confs) / len(confs)
        low_conf_count = sum(1 for c in confs if c < 0.70)
        low_conf_ratio = low_conf_count / len(confs)
    else:
        mean_conf = 0.0
        low_conf_count = 0
        low_conf_ratio = 1.0 if items else 0.0

    # Low OCR Quality Trigger: mean confidence < 0.80 or > 30% low confidence blocks
    is_low_ocr_quality = (len(items) > 0 and (mean_conf < 0.80 or low_conf_ratio > 0.30))
    if is_low_ocr_quality:
        reasons.append("low_ocr_quality")
        # Collect evidence IDs of low confidence blocks
        low_conf_ids = [item.evidence_id for item in items if item.confidence is not None and item.confidence < 0.70]
        _add_ids(low_conf_ids[:10])

    # ── 2. Classification Ambiguity Signals (Phase 7B) ──
    is_role_unknown = (page_understanding.page_role == PageRole.UNKNOWN)
    is_relevance_ambiguous = (page_understanding.payable_relevance == PayableRelevance.AMBIGUOUS)
    is_classification_ambiguous = (is_role_unknown or is_relevance_ambiguous)

    if is_classification_ambiguous:
        reasons.append("classification_ambiguous")
        _add_ids(page_understanding.evidence_ids)

    # ── 3. Layout & Geometry Complexity Signals ──
    layout_metrics = _analyze_layout_and_geometry(items)
    is_complex_layout = layout_metrics["is_complex_layout"]

    if is_complex_layout:
        reasons.append("complex_layout")
        _add_ids([item.evidence_id for item in items[:15]])

    # ── 4. Financial Structural Complexity Signals ──
    financial_metrics = _analyze_financial_structure(items)
    is_complex_financial = financial_metrics["is_complex_financial_structure"]

    if is_complex_financial:
        reasons.append("complex_financial_structure")
        _add_ids(financial_metrics["evidence_ids"])

    # ── 5. Grouping Ambiguity Signals (Phase 7C) ──
    is_grouping_ambiguous = False
    if group is not None:
        # Check if page is an orphaned continuation page (declared continuation role but isolated in 1-page group)
        if page_understanding.page_role == PageRole.CONTINUATION and len(group.page_numbers) == 1:
            is_grouping_ambiguous = True
            reasons.append("grouping_ambiguous")
            _add_ids(group.evidence_ids)

    # ── 6. Final Decision Resolution ──
    if reasons:
        decision = RouterDecisionType.ROUTE_TO_VISION
    else:
        decision = RouterDecisionType.OCR_SUFFICIENT
        reasons.append("ocr_sufficient")
        _add_ids(page_understanding.evidence_ids[:5])

    signals: dict[str, Any] = {
        "ocr_mean_confidence": round(mean_conf, 4) if confs else None,
        "ocr_low_confidence_ratio": round(low_conf_ratio, 4),
        "is_low_ocr_quality": is_low_ocr_quality,
        "page_role": page_understanding.page_role.value,
        "payable_relevance": page_understanding.payable_relevance.value,
        "is_classification_ambiguous": is_classification_ambiguous,
        "layout_block_count": layout_metrics["block_count"],
        "layout_detected_columns": layout_metrics["detected_columns"],
        "layout_table_rows": layout_metrics["table_rows"],
        "is_complex_layout": is_complex_layout,
        "is_complex_financial_structure": is_complex_financial,
        "is_grouping_ambiguous": is_grouping_ambiguous,
    }

    provenance = {
        "router": "DeterministicQwenRouter",
        "version": "Phase7D-1.0",
        "strategy": "hybrid_escalation_rules",
    }

    return RoutingDecision(
        document_id=doc_id,
        page_number=page_num,
        decision=decision,
        reasons=reasons,
        signals=signals,
        evidence_ids=supporting_evidence_ids,
        confidence=None,  # Strictly None (unfabricated)
        provenance=provenance,
    )


# ══════════════════════════════════════════════════════════════════════════
# Vision Escalation Execution Helper (Decoupled Runner)
# ══════════════════════════════════════════════════════════════════════════

def execute_vision_escalation(
    decision: RoutingDecision,
    page_evidence: PageEvidence,
    provider: Optional[VisionProvider] = None,
    image_path: Optional[Union[str, Path]] = None,
    prompt: Optional[str] = None,
) -> Optional[Evidence]:
    """Execute multimodal vision inference via Phase 6C provider if routed to vision.
    
    Returns:
        Phase 7A Evidence object with confidence=None, or None if ocr_sufficient.
    """
    if decision.decision != RouterDecisionType.ROUTE_TO_VISION:
        return None

    img_p = image_path or page_evidence.image_path
    if not img_p:
        raise ValueError(f"No image_path available for vision escalation on page {decision.page_number}")

    if provider is None:
        from src.vision.puter_qwen import PuterQwenProvider
        provider = PuterQwenProvider()

    escalation_prompt = prompt or (
        "Analyze this document page as visual evidence. "
        "Describe visible structure, tables, monetary amounts, taxes, discounts, and totals."
    )

    response = provider.analyze_image(image_path=img_p, prompt=escalation_prompt)
    
    # Convert response to Phase 7A Evidence using existing adapter
    ev = vision_response_to_evidence(
        response=response,
        document_id=decision.document_id,
        page_number=decision.page_number,
        file_path=str(img_p),
        extra_metadata={"routing_reasons": decision.reasons},
    )
    return ev
