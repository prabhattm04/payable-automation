"""src/understanding/evidence.py — Unified Evidence Model and Provenance Layer.

Phase 7A: Establishes a generic, production-grade evidence and provenance
abstraction for downstream document understanding.

Key Design Decisions
--------------------
1. Single Source of Truth:
   `EvidenceProvenance` is the canonical owner of `source`, `extraction_method`,
   and `confidence`. `Evidence` exposes these as read-only derived properties.
2. Lossless Geometry:
   Coordinates are preserved as `float` (e.g. `bbox: list[float] | None`,
   `polygon: list[list[float]] | None`) without premature integer truncation.
3. Passive Reading Order:
   `PageEvidence.full_text()` concatenates items strictly in their input sequence
   without 2D coordinate-based heuristic reordering.
4. Passive Native PDF Ingestion:
   Native text extraction is mapped without assigning artificial priority or
   higher confidence over OCR or Vision observations.
5. Conflict Coexistence:
   Conflicting observations (e.g. OCR="608.23" vs Vision="608.28") are preserved
   as distinct evidence items side-by-side without merging or deduplication.
6. Immutable Deterministic IDs:
   Evidence IDs are generated strictly from immutable content and extraction
   parameters (document_id, page_number, source, method, content, geometry, index).
   They never depend on timestamps, file paths, or machine-specific state.
7. Strict Scope:
   No document classification, payable detection, semantic field extraction,
   routing, or ERP validation is implemented here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from src.utils.logging import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Evidence Source Enumeration & Validation
# ══════════════════════════════════════════════════════════════════════════

class EvidenceSource(str, Enum):
    """Explicit, validated source types for document observations."""
    OCR = "ocr"
    VISION = "vision"
    NATIVE_PDF = "native_pdf"
    MANUAL = "manual"
    DERIVED = "derived"


def validate_source(val: Union[str, EvidenceSource]) -> EvidenceSource:
    """Validate and normalize an evidence source.
    
    Raises:
        ValueError: If `val` does not match a supported EvidenceSource.
    """
    if isinstance(val, EvidenceSource):
        return val
    if isinstance(val, str):
        clean_val = val.strip().lower()
        for member in EvidenceSource:
            if member.value == clean_val:
                return member
    valid = [s.value for s in EvidenceSource]
    raise ValueError(f"Invalid evidence source: '{val}'. Supported sources are: {valid}")


# ══════════════════════════════════════════════════════════════════════════
# Deterministic Evidence ID Generation
# ══════════════════════════════════════════════════════════════════════════

def generate_evidence_id(
    document_id: str,
    page_number: int,
    source: Union[EvidenceSource, str],
    extraction_method: str,
    content: str,
    bbox: Optional[Sequence[Union[int, float]]] = None,
    polygon: Optional[Sequence[Sequence[Union[int, float]]]] = None,
    index: Optional[int] = None,
) -> str:
    """Generate a deterministic, collision-resistant identifier for an evidence item.
    
    Deterministic Inputs (strictly immutable):
    -----------------------------------------
    1. document_id: canonical document identifier (e.g., "INV-01.pdf" or "INV-01")
    2. page_number: 1-indexed document page number
    3. source: normalized string value of EvidenceSource (e.g., "ocr", "vision")
    4. extraction_method: model/engine identifier (e.g., "RapidOCR/PP-OCRv5-ONNX-Latin")
    5. content: exact observed text/content string
    6. bbox: formatted coordinates or "none"
    7. polygon: formatted coordinate points or "none"
    8. index: sequential block index on the page or "none"
    
    NOTE: Never incorporates mutable data such as timestamps, file paths, or hostnames.
    """
    src_val = validate_source(source).value
    
    # Format bbox deterministically with 2 decimal places to avoid float jitter
    if bbox is not None:
        bbox_str = ",".join(f"{float(v):.2f}" for v in bbox)
    else:
        bbox_str = "none"
        
    if polygon is not None:
        poly_pts = [f"({float(p[0]):.2f},{float(p[1]):.2f})" for p in polygon]
        poly_str = ";".join(poly_pts)
    else:
        poly_str = "none"
        
    idx_str = str(index) if index is not None else "none"
    
    raw_payload = (
        f"{document_id}|{page_number}|{src_val}|{extraction_method}|"
        f"{content}|{bbox_str}|{poly_str}|{idx_str}"
    )
    
    digest = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()[:12]
    return f"{document_id}:p{page_number}:{src_val}:{digest}"


# ══════════════════════════════════════════════════════════════════════════
# Provenance Model
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceProvenance:
    """Canonical owner of extraction origin, method, and source-level confidence.
    
    Answers:
    - Where did this evidence come from? (`source`, `file_path`, `document_id`)
    - Which page? (`page_number`)
    - Which extraction method/model? (`extraction_method`)
    - What confidence was supplied? (`confidence`)
    """
    source: EvidenceSource
    document_id: str
    page_number: int
    extraction_method: str
    confidence: Optional[float] = None
    file_path: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source = validate_source(self.source)
        self.document_id = str(self.document_id).strip()
        self.page_number = int(self.page_number)
        if self.page_number < 1:
            raise ValueError(f"Page number must be >= 1, got {self.page_number}")
        self.extraction_method = str(self.extraction_method).strip()
        if self.confidence is not None:
            conf = float(self.confidence)
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"Confidence must be within [0.0, 1.0], got {conf}")
            self.confidence = round(conf, 4)

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance to a clean JSON-serializable dictionary."""
        return {
            "source": self.source.value,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceProvenance:
        """Construct EvidenceProvenance from dictionary."""
        return cls(
            source=validate_source(data["source"]),
            document_id=data["document_id"],
            page_number=int(data["page_number"]),
            extraction_method=data["extraction_method"],
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            file_path=data.get("file_path"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> EvidenceProvenance:
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# Core Evidence Model
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Evidence:
    """Unified atomic piece of document evidence with spatial geometry.
    
    Canonical ownership:
    `provenance` canonically owns `source`, `extraction_method`, and `confidence`.
    `Evidence` exposes these as read-only derived properties to prevent drift.
    """
    provenance: EvidenceProvenance
    content: str
    evidence_id: str = ""
    bbox: Optional[list[float]] = None
    polygon: Optional[list[list[float]]] = None
    semantic_role: Optional[str] = None  # Strictly reserved for Phase 7B+ (never auto-populated)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.content = str(self.content)
        
        # Geometry lossless conversion: keep floats as floats
        if self.bbox is not None:
            self.bbox = [float(v) for v in self.bbox]
        if self.polygon is not None:
            self.polygon = [[float(p[0]), float(p[1])] for p in self.polygon]
            
        # Deterministic evidence ID if not provided
        if not self.evidence_id:
            self.evidence_id = generate_evidence_id(
                document_id=self.document_id,
                page_number=self.page_number,
                source=self.source,
                extraction_method=self.extraction_method,
                content=self.content,
                bbox=self.bbox,
                polygon=self.polygon,
                index=self.metadata.get("block_index"),
            )

    # ── Read-only derived properties delegating to canonical provenance ──

    @property
    def source(self) -> EvidenceSource:
        """The source type of this evidence (derived from provenance)."""
        return self.provenance.source

    @property
    def extraction_method(self) -> str:
        """The engine/model that produced this evidence (derived from provenance)."""
        return self.provenance.extraction_method

    @property
    def confidence(self) -> Optional[float]:
        """The confidence score, or None if uncalibrated (derived from provenance)."""
        return self.provenance.confidence

    @property
    def document_id(self) -> str:
        """The document identifier (derived from provenance)."""
        return self.provenance.document_id

    @property
    def page_number(self) -> int:
        """The 1-indexed page number (derived from provenance)."""
        return self.provenance.page_number

    @property
    def text(self) -> str:
        """Convenience alias for `content`."""
        return self.content

    # ── Factory Helper ──

    @classmethod
    def create(
        cls,
        content: str,
        document_id: str,
        page_number: int,
        source: Union[EvidenceSource, str],
        extraction_method: str,
        confidence: Optional[float] = None,
        evidence_id: Optional[str] = None,
        bbox: Optional[Sequence[Union[int, float]]] = None,
        polygon: Optional[Sequence[Sequence[Union[int, float]]]] = None,
        semantic_role: Optional[str] = None,
        file_path: Optional[str] = None,
        timestamp: Optional[str] = None,
        provenance_metadata: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Evidence:
        """Convenience factory to construct Evidence and its canonical EvidenceProvenance."""
        prov = EvidenceProvenance(
            source=validate_source(source),
            document_id=document_id,
            page_number=page_number,
            extraction_method=extraction_method,
            confidence=confidence,
            file_path=file_path,
            timestamp=timestamp,
            metadata=provenance_metadata or {},
        )
        return cls(
            provenance=prov,
            content=content,
            evidence_id=evidence_id or "",
            bbox=[float(v) for v in bbox] if bbox is not None else None,
            polygon=[[float(p[0]), float(p[1])] for p in polygon] if polygon is not None else None,
            semantic_role=semantic_role,
            metadata=metadata or {},
        )

    # ── Serialization ──

    def to_dict(self) -> dict[str, Any]:
        """Serialize Evidence to a structured dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "content": self.content,
            "source": self.source.value,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "polygon": self.polygon,
            "semantic_role": self.semantic_role,
            "provenance": self.provenance.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        """Reconstruct Evidence from a dictionary."""
        prov_dict = data.get("provenance")
        if prov_dict and isinstance(prov_dict, dict):
            provenance = EvidenceProvenance.from_dict(prov_dict)
        else:
            provenance = EvidenceProvenance(
                source=validate_source(data["source"]),
                document_id=data["document_id"],
                page_number=int(data["page_number"]),
                extraction_method=data["extraction_method"],
                confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            )
        return cls(
            provenance=provenance,
            content=data.get("content", ""),
            evidence_id=data.get("evidence_id", ""),
            bbox=[float(v) for v in data["bbox"]] if data.get("bbox") is not None else None,
            polygon=(
                [[float(p[0]), float(p[1])] for p in data["polygon"]]
                if data.get("polygon") is not None
                else None
            ),
            semantic_role=data.get("semantic_role"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> Evidence:
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# Page-Level Evidence Container
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PageEvidence:
    """Aggregated container for multi-source evidence on a single document page."""
    document_id: str
    page_number: int
    items: list[Evidence] = field(default_factory=list)
    image_path: Optional[str] = None
    image_width: Optional[float] = None
    image_height: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_item(self, item: Evidence) -> None:
        """Append an evidence item to the page collection."""
        self.items.append(item)

    def filter_by_source(self, source: Union[EvidenceSource, str]) -> list[Evidence]:
        """Return all evidence items originating from the specified source."""
        target = validate_source(source)
        return [e for e in self.items if e.source == target]

    def full_text(self, separator: str = "\n") -> str:
        """Concatenate observed text of items in their given order.
        
        NOTE: Does not perform coordinate-based reading order or layout reconstruction.
        Reading order reasoning belongs strictly to later document-understanding phases.
        """
        return separator.join(e.content for e in self.items if e.content)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self.items)

    def __getitem__(self, idx: int) -> Evidence:
        return self.items[idx]

    def to_dict(self) -> dict[str, Any]:
        """Convert PageEvidence to dictionary."""
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageEvidence:
        """Reconstruct PageEvidence from dictionary."""
        items = [Evidence.from_dict(item_data) for item_data in data.get("items", [])]
        return cls(
            document_id=data["document_id"],
            page_number=int(data["page_number"]),
            items=items,
            image_path=data.get("image_path"),
            image_width=float(data["image_width"]) if data.get("image_width") is not None else None,
            image_height=float(data["image_height"]) if data.get("image_height") is not None else None,
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> PageEvidence:
        return cls.from_dict(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
# Phase 6B RapidOCR Conversion Adapters
# ══════════════════════════════════════════════════════════════════════════

def ocr_block_to_evidence(
    block: Any,
    document_id: str,
    page_number: int,
    engine: str = "RapidOCR",
    model: str = "PP-OCRv5",
    file_path: Optional[str] = None,
    index: Optional[int] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> Evidence:
    """Convert an OCRBlock or OCR block dictionary into an Evidence object.
    
    Preserves:
    - Text content
    - Calibrated OCR confidence score
    - Bounding box [x1, y1, x2, y2] as floats
    - Polygon points [[x, y], ...] as floats
    - Extraction method: '{engine}/{model}'
    """
    if hasattr(block, "text"):
        content = block.text
        conf = float(block.confidence) if block.confidence is not None else None
        bbox = [float(v) for v in block.bbox] if getattr(block, "bbox", None) is not None else None
        polygon = (
            [[float(pt[0]), float(pt[1])] for pt in block.polygon]
            if getattr(block, "polygon", None) is not None
            else None
        )
    elif isinstance(block, dict):
        content = str(block.get("text", ""))
        conf = float(block["confidence"]) if block.get("confidence") is not None else None
        bbox = [float(v) for v in block["bbox"]] if block.get("bbox") is not None else None
        polygon = (
            [[float(pt[0]), float(pt[1])] for pt in block["polygon"]]
            if block.get("polygon") is not None
            else None
        )
    else:
        raise TypeError(f"Unsupported block type: {type(block)}")

    extraction_method = f"{engine}/{model}" if model else engine
    meta = dict(extra_metadata or {})
    if index is not None:
        meta["block_index"] = index

    provenance = EvidenceProvenance(
        source=EvidenceSource.OCR,
        document_id=document_id,
        page_number=page_number,
        extraction_method=extraction_method,
        confidence=conf,
        file_path=file_path,
        metadata={
            "engine": engine,
            "model": model,
            **meta,
        },
    )

    return Evidence(
        provenance=provenance,
        content=content,
        bbox=bbox,
        polygon=polygon,
        semantic_role=None,
        metadata=meta,
    )


def page_ocr_result_to_evidence(
    result: Any,
    document_id: Optional[str] = None,
) -> PageEvidence:
    """Convert a PageOCRResult or Phase 6B OCR JSON dict into a PageEvidence collection."""
    if hasattr(result, "document"):
        doc_id = document_id or result.document
        page_num = result.page
        img_path = getattr(result, "image_path", None) or getattr(result, "image", None)
        img_w = float(result.image_width) if getattr(result, "image_width", None) is not None else None
        img_h = float(result.image_height) if getattr(result, "image_height", None) is not None else None
        engine = getattr(result, "engine", "RapidOCR")
        model = getattr(result, "model", "PP-OCRv5")
        blocks = getattr(result, "blocks", [])
        meta = {
            "language": getattr(result, "language", None),
            "confidence_mean": getattr(result, "confidence_mean", None),
            "status": getattr(result, "status", None),
        }
    elif isinstance(result, dict):
        doc_id = document_id or result.get("document", "unknown")
        page_num = int(result.get("page", 1))
        img_path = result.get("image") or result.get("image_path")
        img_w = float(result["image_width"]) if result.get("image_width") is not None else None
        img_h = float(result["image_height"]) if result.get("image_height") is not None else None
        engine = result.get("engine", "RapidOCR")
        model = result.get("model", "PP-OCRv5")
        blocks = result.get("blocks", [])
        meta = {
            "language": result.get("language"),
            "confidence_mean": result.get("confidence_mean"),
            "status": result.get("status"),
        }
    else:
        raise TypeError(f"Unsupported OCR result type: {type(result)}")

    items: list[Evidence] = []
    for idx, blk in enumerate(blocks):
        item = ocr_block_to_evidence(
            block=blk,
            document_id=doc_id,
            page_number=page_num,
            engine=engine,
            model=model,
            file_path=img_path,
            index=idx,
        )
        items.append(item)

    return PageEvidence(
        document_id=doc_id,
        page_number=page_num,
        items=items,
        image_path=img_path,
        image_width=img_w,
        image_height=img_h,
        metadata=meta,
    )


def ocr_json_to_page_evidence(path_or_dict: Union[str, Path, dict[str, Any]]) -> PageEvidence:
    """Read a Phase 6B page_NNN.json file or parsed dictionary into PageEvidence."""
    if isinstance(path_or_dict, (str, Path)):
        p = Path(path_or_dict)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(path_or_dict, dict):
        data = path_or_dict
    else:
        raise TypeError(f"Expected str, Path, or dict, got {type(path_or_dict)}")

    return page_ocr_result_to_evidence(data)


# ══════════════════════════════════════════════════════════════════════════
# Phase 6C Vision Provider Conversion Adapters
# ══════════════════════════════════════════════════════════════════════════

def vision_response_to_evidence(
    response: Any,
    document_id: str,
    page_number: int,
    file_path: Optional[str] = None,
    content: Optional[str] = None,
    bbox: Optional[Sequence[Union[int, float]]] = None,
    polygon: Optional[Sequence[Sequence[Union[int, float]]]] = None,
    index: Optional[int] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> Evidence:
    """Convert a VisionResponse object into an Evidence object.
    
    CRITICAL RULE (Requirement 6):
    Vision/Qwen observations do not provide calibrated confidence scores.
    `confidence` is strictly set to `None` (null in JSON). No fabricated score.
    """
    if hasattr(response, "content"):
        raw_content = response.content or ""
        provider = getattr(response, "provider", "vision")
        model = getattr(response, "model", "")
        success = getattr(response, "success", True)
        latency = getattr(response, "latency_seconds", 0.0)
        resp_meta = getattr(response, "metadata", {}) or {}
    elif isinstance(response, dict):
        raw_content = response.get("content") or ""
        provider = response.get("provider", "vision")
        model = response.get("model", "")
        success = response.get("success", True)
        latency = response.get("latency_seconds", 0.0)
        resp_meta = response.get("metadata", {}) or {}
    else:
        raise TypeError(f"Unsupported Vision response type: {type(response)}")

    observed_content = content if content is not None else raw_content
    extraction_method = f"{provider}/{model}" if (provider and model) else (provider or model or "vision")

    meta = {
        "provider": provider,
        "model": model,
        "success": success,
        "latency_seconds": latency,
        **resp_meta,
        **(extra_metadata or {}),
    }
    if index is not None:
        meta["block_index"] = index

    provenance = EvidenceProvenance(
        source=EvidenceSource.VISION,
        document_id=document_id,
        page_number=page_number,
        extraction_method=extraction_method,
        confidence=None,  # Null calibrated confidence
        file_path=file_path,
        metadata=meta,
    )

    return Evidence(
        provenance=provenance,
        content=observed_content,
        bbox=[float(v) for v in bbox] if bbox is not None else None,
        polygon=[[float(p[0]), float(p[1])] for p in polygon] if polygon is not None else None,
        semantic_role=None,
        metadata=meta,
    )


def vision_observation_to_evidence(
    content: str,
    document_id: str,
    page_number: int,
    model: str = "qwen3-vl-plus",
    provider: str = "Puter",
    file_path: Optional[str] = None,
    bbox: Optional[Sequence[Union[int, float]]] = None,
    polygon: Optional[Sequence[Sequence[Union[int, float]]]] = None,
    index: Optional[int] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> Evidence:
    """Direct helper to create Vision Evidence from observed text with null confidence."""
    extraction_method = f"{provider}/{model}" if (provider and model) else (provider or model or "vision")
    meta = {
        "provider": provider,
        "model": model,
        **(extra_metadata or {}),
    }
    if index is not None:
        meta["block_index"] = index

    provenance = EvidenceProvenance(
        source=EvidenceSource.VISION,
        document_id=document_id,
        page_number=page_number,
        extraction_method=extraction_method,
        confidence=None,  # Null calibrated confidence
        file_path=file_path,
        metadata=meta,
    )

    return Evidence(
        provenance=provenance,
        content=content,
        bbox=[float(v) for v in bbox] if bbox is not None else None,
        polygon=[[float(p[0]), float(p[1])] for p in polygon] if polygon is not None else None,
        semantic_role=None,
        metadata=meta,
    )


# ══════════════════════════════════════════════════════════════════════════
# Phase 5/6 Native PDF Conversion Adapter
# ══════════════════════════════════════════════════════════════════════════

def native_page_text_to_evidence(
    page_text: Any,
    document_id: str,
    file_path: Optional[str] = None,
) -> list[Evidence]:
    """Convert native PageText into Evidence objects.
    
    CRITICAL RULE (Design Adjustment 4):
    Remains a purely passive adapter. Does NOT assign higher trust or artificial
    confidence to native PDF text over OCR or Vision. Confidence is set to None.
    """
    if hasattr(page_text, "raw_text"):
        raw_text = page_text.raw_text
        page_num = page_text.page_number
        method = getattr(page_text, "extraction_method", "fitz/native") or "fitz/native"
        char_count = getattr(page_text, "character_count", len(raw_text))
        fonts = getattr(page_text, "fonts_detected", [])
    elif isinstance(page_text, dict):
        raw_text = page_text.get("raw_text", "")
        page_num = int(page_text.get("page_number", 1))
        method = page_text.get("extraction_method", "fitz/native")
        char_count = page_text.get("character_count", len(raw_text))
        fonts = page_text.get("fonts_detected", [])
    else:
        raise TypeError(f"Unsupported PageText type: {type(page_text)}")

    if not raw_text or not raw_text.strip():
        return []

    provenance = EvidenceProvenance(
        source=EvidenceSource.NATIVE_PDF,
        document_id=document_id,
        page_number=page_num,
        extraction_method=method,
        confidence=None,  # Passive adapter: uncalibrated confidence
        file_path=file_path,
        metadata={
            "character_count": char_count,
            "fonts_detected": fonts,
        },
    )

    ev = Evidence(
        provenance=provenance,
        content=raw_text,
        bbox=None,
        polygon=None,
        semantic_role=None,
        metadata={
            "character_count": char_count,
            "fonts_detected": fonts,
        },
    )
    return [ev]
