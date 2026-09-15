"""src/extraction/ocr.py — Reusable OCR layer using PP-OCRv5.

Responsibilities
----------------
* Provide engine-independent OCR interfaces (OCRProvider).
* Implement PP_OCRv5Provider using PaddleOCR.
* Preserve raw OCR evidence: text, per-block confidence, bounding boxes,
  4-point polygons, image dimensions, and page identity.
* Provide page-level exception isolation (failure on one page never aborts the run).
* Calculate page-level confidence statistics (mean, min, max).
* Serialize raw OCR results to artifacts/ocr/<stem>/page_NNN.json.
* Generate a complete dataset manifest at artifacts/ocr/ocr_manifest.json.
* Strictly avoid document understanding, classification, financial interpretation,
  or filename-specific language routing rules.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Union

from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

DEFAULT_ENGINE_NAME = "PaddleOCR"
DEFAULT_MODEL_NAME = "PP-OCRv5"
DEFAULT_LANGUAGE = "latin"


# ══════════════════════════════════════════════════════════════════════════
# Helper: Extract image dimensions without heavyweight dependencies
# ══════════════════════════════════════════════════════════════════════════

def get_image_dimensions(image_path: Union[str, Path]) -> tuple[int, int]:
    """Return (width, height) in pixels from PNG header or PIL if available.
    
    Falls back to (0, 0) if unreadable.
    """
    path = Path(image_path)
    if not path.exists():
        return (0, 0)
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        pass

    # Fast PNG header parser as fallback (PNG IHDR is at byte offset 16-24)
    try:
        with open(path, "rb") as f:
            header = f.read(24)
            if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
                import struct
                w, h = struct.unpack(">II", header[16:24])
                return (int(w), int(h))
    except Exception:
        pass

    return (0, 0)


# ══════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class OCRBlock:
    """A single recognized text region and its geometry.
    
    Preserves:
    - text: raw recognized text string (Unicode preserved)
    - confidence: float score in [0.0, 1.0]
    - bbox: axis-aligned bounding box [x1, y1, x2, y2]
    - polygon: 4-point polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    text: str
    confidence: float
    bbox: list[int]
    polygon: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "bbox": [int(v) for v in self.bbox],
            "polygon": [[int(pt[0]), int(pt[1])] for pt in self.polygon],
        }


@dataclass
class PageOCRResult:
    """Raw OCR evidence and metadata for one document page.
    
    Preserves page identity, raw text, per-block geometry, and confidence stats.
    """
    document: str                    # source document stem or filename (e.g. "HLD-01.pdf" or "HLD-01")
    page: int                        # 1-indexed page number
    image_path: str                  # path to rendered source image
    image_width: int                 # page width in pixels
    image_height: int                # page height in pixels
    engine: str = DEFAULT_ENGINE_NAME
    model: str = DEFAULT_MODEL_NAME
    language: str = DEFAULT_LANGUAGE
    text: str = ""                   # full joined text, newline-separated
    confidence_mean: float = 0.0     # mean confidence across detected blocks
    confidence_min: float = 0.0      # min confidence across detected blocks
    confidence_max: float = 0.0      # max confidence across detected blocks
    blocks: list[OCRBlock] = field(default_factory=list)
    runtime_seconds: float = 0.0
    status: str = "success"          # "success" | "failed"
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching Phase 6B schema."""
        return {
            "document": self.document,
            "page": self.page,
            "image": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "engine": self.engine,
            "model": self.model,
            "language": self.language,
            "text": self.text,
            "confidence": round(float(self.confidence_mean), 4),
            "confidence_mean": round(float(self.confidence_mean), 4),
            "confidence_min": round(float(self.confidence_min), 4),
            "confidence_max": round(float(self.confidence_max), 4),
            "blocks": [b.to_dict() for b in self.blocks],
            "runtime_seconds": round(float(self.runtime_seconds), 3),
            "status": self.status,
            "error": self.error,
        }


@dataclass
class DocumentOCRResult:
    """Ordered collection of PageOCRResult objects for a multi-page document.
    
    Strictly an ordered container — no classification, grouping, or semantic interpretation.
    """
    document: str
    pages: list[PageOCRResult] = field(default_factory=list)
    total_runtime_seconds: float = 0.0
    status: str = "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "page_count": len(self.pages),
            "total_runtime_seconds": round(float(self.total_runtime_seconds), 3),
            "status": self.status,
            "pages": [p.to_dict() for p in self.pages],
        }


# ══════════════════════════════════════════════════════════════════════════
# Abstract OCR Provider Interface
# ══════════════════════════════════════════════════════════════════════════

class OCRProvider(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def ocr_page(
        self,
        image_path: Union[str, Path],
        document_id: Optional[str] = None,
        page_number: Optional[int] = None,
        language: str = DEFAULT_LANGUAGE,
        **kwargs: Any,
    ) -> PageOCRResult:
        """Perform OCR on a single rendered page image."""
        pass

    def ocr_document(
        self,
        page_image_paths: list[Union[str, Path]],
        document_id: Optional[str] = None,
        language: str = DEFAULT_LANGUAGE,
        **kwargs: Any,
    ) -> DocumentOCRResult:
        """Perform OCR sequentially on a list of page images for a document.
        
        Preserves page ordering. Does not perform semantic analysis or grouping.
        """
        start_time = time.perf_counter()
        doc_name = document_id or "unknown"
        page_results: list[PageOCRResult] = []
        overall_status = "success"

        # Sort paths by page number if discernable
        sorted_paths = sorted(page_image_paths, key=_extract_page_sort_key)

        for idx, img_path in enumerate(sorted_paths, start=1):
            extracted_doc, extracted_page = _infer_doc_and_page_from_path(img_path)
            cur_doc = doc_name if doc_name != "unknown" else extracted_doc
            cur_page = extracted_page if extracted_page is not None else idx

            res = self.ocr_page(
                image_path=img_path,
                document_id=cur_doc,
                page_number=cur_page,
                language=language,
                **kwargs,
            )
            if res.status != "success":
                overall_status = "partial_failure" if overall_status == "success" else overall_status
            page_results.append(res)

        total_runtime = time.perf_counter() - start_time
        return DocumentOCRResult(
            document=doc_name,
            pages=page_results,
            total_runtime_seconds=total_runtime,
            status=overall_status,
        )


# ══════════════════════════════════════════════════════════════════════════
# RapidOCR Provider Implementation (ONNX Runtime)
# ══════════════════════════════════════════════════════════════════════════

class RapidOCRProvider(OCRProvider):
    """RapidOCR implementation using ONNX Runtime.
    
    Provides fast, local, lightweight execution of PP-OCR ONNX models.
    Supports dynamic language configuration (e.g. "latin", "thai", "default")
    by resolving dedicated PP-OCRv5 ONNX recognition models and dictionaries.
    Preserves text, per-block confidence, bounding boxes, and polygon geometry.
    Page-level exception isolation ensures one page failure never aborts the run.
    """

    MODEL_DIR = Path("models") / "ocr"

    def __init__(
        self,
        config_path: Optional[str] = None,
        det_model_path: Optional[str] = None,
        rec_model_path: Optional[str] = None,
        rec_keys_path: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.config_path = config_path
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.rec_keys_path = rec_keys_path
        self.model_name = model_name
        self.extra_kwargs = kwargs
        self._engines: dict[str, Any] = {}

    def _resolve_model_paths(self, language: str) -> tuple[Optional[str], Optional[str], str]:
        """Resolve recognition model and dictionary paths for a given language."""
        if self.rec_model_path and self.rec_keys_path:
            return (self.rec_model_path, self.rec_keys_path, self.model_name or "Custom-ONNX")

        lang_key = language.lower().strip()
        if lang_key in ("latin", "en", "english", "german", "de", "portuguese", "pt", "estonian", "et"):
            target_sub = "latin"
            model_tag = "PP-OCRv5-ONNX-Latin"
        elif lang_key in ("thai", "th"):
            target_sub = "thai"
            model_tag = "PP-OCRv5-ONNX-Thai"
        else:
            target_sub = None
            model_tag = "PP-OCRv4-ONNX-Default"

        if target_sub:
            candidate_model = self.MODEL_DIR / target_sub / "rec.onnx"
            candidate_dict = self.MODEL_DIR / target_sub / "dict.txt"
            if candidate_model.exists() and candidate_dict.exists():
                return (str(candidate_model), str(candidate_dict), model_tag)

        return (None, None, "PP-OCRv4-ONNX-Default")

    def _get_engine(self, language: str) -> tuple[Any, str]:
        rec_model, rec_keys, model_tag = self._resolve_model_paths(language)
        cache_key = f"{rec_model}_{rec_keys}"

        if cache_key not in self._engines:
            try:
                from rapidocr_onnxruntime import RapidOCR
                kwargs: dict[str, Any] = dict(self.extra_kwargs)
                if self.config_path:
                    kwargs["config_path"] = self.config_path
                if self.det_model_path:
                    kwargs["det_model_path"] = self.det_model_path
                if rec_model:
                    kwargs["rec_model_path"] = rec_model
                if rec_keys:
                    kwargs["rec_keys_path"] = rec_keys

                log.info("Initializing RapidOCR engine for language='%s' (model=%s, rec_keys=%s)", language, rec_model, rec_keys)
                self._engines[cache_key] = (RapidOCR(**kwargs), model_tag)
            except Exception as e:
                log.error("Failed to initialize RapidOCR engine for language '%s': %s", language, e)
                raise

        return self._engines[cache_key]

    def ocr_page(
        self,
        image_path: Union[str, Path],
        document_id: Optional[str] = None,
        page_number: Optional[int] = None,
        language: str = DEFAULT_LANGUAGE,
        **kwargs: Any,
    ) -> PageOCRResult:
        """Run RapidOCR on a single page image file."""
        start_time = time.perf_counter()
        img_p = Path(image_path)

        inferred_doc, inferred_page = _infer_doc_and_page_from_path(img_p)
        doc = document_id if document_id is not None else inferred_doc
        page_num = page_number if page_number is not None else (inferred_page or 1)

        if not img_p.exists():
            runtime = time.perf_counter() - start_time
            log.warning("Image file does not exist: %s", img_p)
            return PageOCRResult(
                document=doc,
                page=page_num,
                image_path=str(img_p),
                image_width=0,
                image_height=0,
                engine="RapidOCR",
                model=self.model_name or "PP-OCR",
                language=language,
                text="",
                confidence_mean=0.0,
                confidence_min=0.0,
                confidence_max=0.0,
                blocks=[],
                runtime_seconds=runtime,
                status="failed",
                error=f"File not found: {img_p}",
            )

        width, height = get_image_dimensions(img_p)

        try:
            engine, active_model_tag = self._get_engine(language)
            raw_result, elapse = engine(str(img_p))

            blocks: list[OCRBlock] = []
            text_lines: list[str] = []
            confidences: list[float] = []

            # RapidOCR returns list of [poly_coords, text, score]
            if raw_result:
                for item in raw_result:
                    if not item or len(item) < 3:
                        continue
                    poly_coords, recognized_text, score = item[0], str(item[1]), float(item[2])

                    polygon: list[list[int]] = []
                    xs: list[int] = []
                    ys: list[int] = []
                    for pt in poly_coords:
                        px, py = int(round(pt[0])), int(round(pt[1]))
                        polygon.append([px, py])
                        xs.append(px)
                        ys.append(py)

                    bbox = [min(xs), min(ys), max(xs), max(ys)] if xs and ys else [0, 0, 0, 0]

                    blocks.append(
                        OCRBlock(
                            text=recognized_text,
                            confidence=score,
                            bbox=bbox,
                            polygon=polygon,
                        )
                    )
                    text_lines.append(recognized_text)
                    confidences.append(score)

            full_text = "\n".join(text_lines)
            if confidences:
                mean_conf = sum(confidences) / len(confidences)
                min_conf = min(confidences)
                max_conf = max(confidences)
            else:
                mean_conf = 0.0
                min_conf = 0.0
                max_conf = 0.0

            runtime = time.perf_counter() - start_time
            return PageOCRResult(
                document=doc,
                page=page_num,
                image_path=str(img_p),
                image_width=width,
                image_height=height,
                engine="RapidOCR",
                model=self.model_name or active_model_tag,
                language=language,
                text=full_text,
                confidence_mean=mean_conf,
                confidence_min=min_conf,
                confidence_max=max_conf,
                blocks=blocks,
                runtime_seconds=runtime,
                status="success",
                error=None,
            )

        except Exception as e:
            runtime = time.perf_counter() - start_time
            err_msg = f"{type(e).__name__}: {str(e)}"
            log.error("RapidOCR failed for %s page %d: %s", doc, page_num, err_msg)
            return PageOCRResult(
                document=doc,
                page=page_num,
                image_path=str(img_p),
                image_width=width,
                image_height=height,
                engine="RapidOCR",
                model=self.model_name,
                language=language,
                text="",
                confidence_mean=0.0,
                confidence_min=0.0,
                confidence_max=0.0,
                blocks=[],
                runtime_seconds=runtime,
                status="failed",
                error=err_msg,
            )


# ══════════════════════════════════════════════════════════════════════════
# PP-OCRv5 Provider Implementation
# ══════════════════════════════════════════════════════════════════════════

class PP_OCRv5Provider(OCRProvider):
    """PP-OCRv5 implementation using PaddleOCR.
    
    Supports language configurations (e.g. "latin", "thai", "en", etc.).
    Caches model instances per language configuration to avoid redundant reloads.
    Provides exception isolation so a failing image returns a structured error.
    """

    # Language mapping for PaddleOCR backend
    # PaddleOCR supports 'latin', 'en', 'th', 'german', 'ch', etc.
    LANG_MAP = {
        "latin": "latin",
        "en": "en",
        "english": "en",
        "thai": "th",
        "th": "th",
        "german": "german",
        "de": "german",
        "portuguese": "latin",
        "pt": "latin",
        "estonian": "latin",
        "et": "latin",
    }

    def __init__(
        self,
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        ocr_version: str = "PP-OCRv4",  # or PP-OCRv5 if supported by installed version
    ) -> None:
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.ocr_version = ocr_version
        self._models: dict[str, Any] = {}

    def _get_paddleocr_instance(self, language: str) -> Any:
        """Lazily initialize and cache PaddleOCR instance for a given language."""
        target_lang = self.LANG_MAP.get(language.lower(), language.lower())

        cache_key = f"{target_lang}_{self.use_angle_cls}_{self.use_gpu}"
        if cache_key in self._models:
            return self._models[cache_key]

        log.info("Initializing PaddleOCR engine for language='%s' (target='%s')", language, target_lang)
        try:
            from paddleocr import PaddleOCR
            # Disable unnecessary logging output from paddle
            logging.getLogger("ppocr").setLevel(logging.WARNING)

            kwargs: dict[str, Any] = {
                "use_angle_cls": self.use_angle_cls,
                "lang": target_lang,
                "show_log": False,
            }
            # Attempt to configure version if parameter is accepted
            try:
                ocr_inst = PaddleOCR(ocr_version=self.ocr_version, **kwargs)
            except TypeError:
                ocr_inst = PaddleOCR(**kwargs)

            self._models[cache_key] = ocr_inst
            return ocr_inst
        except Exception as e:
            log.error("Failed to initialize PaddleOCR for language '%s': %s", language, e)
            raise

    def ocr_page(
        self,
        image_path: Union[str, Path],
        document_id: Optional[str] = None,
        page_number: Optional[int] = None,
        language: str = DEFAULT_LANGUAGE,
        **kwargs: Any,
    ) -> PageOCRResult:
        """Run PP-OCRv5 on a single page image file.
        
        Returns PageOCRResult with full text, per-block confidence, bounding boxes,
        and polygon geometry. Catches and records any exception without crashing.
        """
        start_time = time.perf_counter()
        img_p = Path(image_path)

        # Infer identity if not explicitly passed
        inferred_doc, inferred_page = _infer_doc_and_page_from_path(img_p)
        doc = document_id if document_id is not None else inferred_doc
        page_num = page_number if page_number is not None else (inferred_page or 1)

        # Image existence check
        if not img_p.exists():
            runtime = time.perf_counter() - start_time
            log.warning("Image file does not exist: %s", img_p)
            return PageOCRResult(
                document=doc,
                page=page_num,
                image_path=str(img_p),
                image_width=0,
                image_height=0,
                engine=DEFAULT_ENGINE_NAME,
                model=DEFAULT_MODEL_NAME,
                language=language,
                text="",
                confidence_mean=0.0,
                confidence_min=0.0,
                confidence_max=0.0,
                blocks=[],
                runtime_seconds=runtime,
                status="failed",
                error=f"File not found: {img_p}",
            )

        # Read dimensions
        width, height = get_image_dimensions(img_p)

        try:
            ocr_engine = self._get_paddleocr_instance(language)
            # Run inference
            raw_result = ocr_engine.ocr(str(img_p), cls=self.use_angle_cls)

            # Parse results
            blocks: list[OCRBlock] = []
            text_lines: list[str] = []
            confidences: list[float] = []

            # PaddleOCR result structure is typically:
            # [ [ [ [x1,y1],[x2,y2],[x3,y3],[x4,y4] ], (text, confidence) ], ... ]
            if raw_result and len(raw_result) > 0 and raw_result[0] is not None:
                for item in raw_result[0]:
                    if not item or len(item) < 2:
                        continue
                    poly_coords, text_info = item[0], item[1]
                    if not text_info or len(text_info) < 2:
                        continue

                    recognized_text = str(text_info[0])
                    conf = float(text_info[1])

                    # Polygon points
                    polygon: list[list[int]] = []
                    xs: list[int] = []
                    ys: list[int] = []
                    for pt in poly_coords:
                        px, py = int(round(pt[0])), int(round(pt[1]))
                        polygon.append([px, py])
                        xs.append(px)
                        ys.append(py)

                    # Bounding box [min_x, min_y, max_x, max_y]
                    bbox = [min(xs), min(ys), max(xs), max(ys)] if xs and ys else [0, 0, 0, 0]

                    blocks.append(
                        OCRBlock(
                            text=recognized_text,
                            confidence=conf,
                            bbox=bbox,
                            polygon=polygon,
                        )
                    )
                    text_lines.append(recognized_text)
                    confidences.append(conf)

            # Compute stats
            full_text = "\n".join(text_lines)
            if confidences:
                mean_conf = sum(confidences) / len(confidences)
                min_conf = min(confidences)
                max_conf = max(confidences)
            else:
                mean_conf = 0.0
                min_conf = 0.0
                max_conf = 0.0

            runtime = time.perf_counter() - start_time
            return PageOCRResult(
                document=doc,
                page=page_num,
                image_path=str(img_p),
                image_width=width,
                image_height=height,
                engine=DEFAULT_ENGINE_NAME,
                model=DEFAULT_MODEL_NAME,
                language=language,
                text=full_text,
                confidence_mean=mean_conf,
                confidence_min=min_conf,
                confidence_max=max_conf,
                blocks=blocks,
                runtime_seconds=runtime,
                status="success",
                error=None,
            )

        except Exception as e:
            runtime = time.perf_counter() - start_time
            err_msg = f"{type(e).__name__}: {str(e)}"
            log.error("OCR failed for %s page %d: %s", doc, page_num, err_msg)
            return PageOCRResult(
                document=doc,
                page=page_num,
                image_path=str(img_p),
                image_width=width,
                image_height=height,
                engine=DEFAULT_ENGINE_NAME,
                model=DEFAULT_MODEL_NAME,
                language=language,
                text="",
                confidence_mean=0.0,
                confidence_min=0.0,
                confidence_max=0.0,
                blocks=[],
                runtime_seconds=runtime,
                status="failed",
                error=err_msg,
            )


# ══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════

def _infer_doc_and_page_from_path(img_path: Union[str, Path]) -> tuple[str, Optional[int]]:
    """Extract document stem and page number from standard artifact paths.
    
    e.g. artifacts/rendered_pages/HLD-01/page_001.png -> ("HLD-01.pdf", 1)
    """
    p = Path(img_path)
    parent_stem = p.parent.name
    # Check page_NNN.png pattern
    match = re.match(r"^page_?(\d+)", p.stem, re.IGNORECASE)
    page_num = int(match.group(1)) if match else None

    # If parent is a known document stem (e.g. INV-01, HLD-01, DU-02)
    doc_name = f"{parent_stem}.pdf" if parent_stem and parent_stem != "rendered_pages" else p.stem
    return (doc_name, page_num)


def _extract_page_sort_key(p: Union[str, Path]) -> int:
    """Sort key based on page number in filename."""
    stem = Path(p).stem
    match = re.search(r"\d+", stem)
    return int(match.group(0)) if match else 0


# ══════════════════════════════════════════════════════════════════════════
# Dataset Execution & Manifest Generation
# ══════════════════════════════════════════════════════════════════════════

def save_page_ocr_json(result: PageOCRResult, output_dir: Union[str, Path]) -> Path:
    """Save PageOCRResult to artifacts/ocr/<stem>/page_NNN.json."""
    out_dir = Path(output_dir)
    # Extract document stem (e.g. "HLD-01.pdf" -> "HLD-01")
    doc_stem = Path(result.document).stem
    doc_output_dir = out_dir / doc_stem
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    page_filename = f"page_{result.page:03d}.json"
    json_path = doc_output_dir / page_filename

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    return json_path


def run_ocr_dataset(
    rendered_pages_dir: Union[str, Path],
    output_dir: Union[str, Path],
    provider: Optional[OCRProvider] = None,
    language: str = DEFAULT_LANGUAGE,
    doc_language_map: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Execute OCR across all rendered pages in rendered_pages_dir.
    
    Scans rendered_pages_dir/<stem>/page_NNN.png.
    Saves JSON artifacts in output_dir/<stem>/page_NNN.json.
    Writes dataset manifest to output_dir/ocr_manifest.json.
    One page failure does not abort the run.
    """
    start_total = time.perf_counter()
    input_p = Path(rendered_pages_dir)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if provider is None:
        provider = RapidOCRProvider()

    doc_lang_config = doc_language_map or {}

    # Discover document folders and images
    doc_dirs = [d for d in input_p.iterdir() if d.is_dir()]
    if not doc_dirs:
        doc_dirs = [input_p]

    all_page_paths: list[Path] = []
    for d in sorted(doc_dirs, key=lambda x: x.name):
        pages = sorted(d.glob("page_*.png"), key=_extract_page_sort_key)
        all_page_paths.extend(pages)

    log.info("Discovered %d pages across %d document directories", len(all_page_paths), len(doc_dirs))

    pages_attempted = 0
    pages_succeeded = 0
    pages_failed = 0
    page_runtimes: list[float] = []
    page_confidences: list[float] = []
    failures: list[dict[str, Any]] = []
    models_used: set[str] = set()
    languages_used: set[str] = set()

    manifest_docs: dict[str, dict[str, Any]] = {}

    for img_path in all_page_paths:
        pages_attempted += 1
        doc_name, page_num = _infer_doc_and_page_from_path(img_path)
        stem = Path(doc_name).stem

        # Use document-level run configuration if explicitly provided; otherwise default language
        target_lang = doc_lang_config.get(stem, language)

        log.info("[%d/%d] Processing %s page %s (lang=%s)", pages_attempted, len(all_page_paths), stem, page_num, target_lang)
        res = provider.ocr_page(
            image_path=img_path,
            document_id=doc_name,
            page_number=page_num,
            language=target_lang,
        )

        save_page_ocr_json(res, out_p)

        page_runtimes.append(res.runtime_seconds)
        models_used.add(res.model)
        languages_used.add(res.language)

        if res.status == "success":
            pages_succeeded += 1
            if res.confidence_mean > 0:
                page_confidences.append(res.confidence_mean)
        else:
            pages_failed += 1
            failures.append({
                "document": res.document,
                "page": res.page,
                "image": str(img_path),
                "error": res.error,
            })

        # Track per-document stats
        if stem not in manifest_docs:
            manifest_docs[stem] = {
                "document": doc_name,
                "language_configuration": target_lang,
                "pages_attempted": 0,
                "pages_succeeded": 0,
                "pages_failed": 0,
                "total_runtime_seconds": 0.0,
                "pages": [],
            }
        manifest_docs[stem]["pages_attempted"] += 1
        if res.status == "success":
            manifest_docs[stem]["pages_succeeded"] += 1
        else:
            manifest_docs[stem]["pages_failed"] += 1
        manifest_docs[stem]["total_runtime_seconds"] = round(
            manifest_docs[stem]["total_runtime_seconds"] + res.runtime_seconds, 3
        )
        manifest_docs[stem]["pages"].append({
            "page": res.page,
            "status": res.status,
            "model": res.model,
            "language": res.language,
            "confidence_mean": round(res.confidence_mean, 4),
            "blocks_count": len(res.blocks),
            "runtime_seconds": round(res.runtime_seconds, 3),
            "error": res.error,
        })

    total_duration = time.perf_counter() - start_total
    avg_runtime = (sum(page_runtimes) / len(page_runtimes)) if page_runtimes else 0.0
    min_runtime = min(page_runtimes) if page_runtimes else 0.0
    max_runtime = max(page_runtimes) if page_runtimes else 0.0

    # Calculate total size of generated OCR JSONs
    total_json_bytes = sum(f.stat().st_size for f in out_p.rglob("*.json") if f.name != "ocr_manifest.json")

    manifest = {
        "manifest_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_documents": len(manifest_docs),
        "total_pages": len(all_page_paths),
        "pages_attempted": pages_attempted,
        "pages_succeeded": pages_succeeded,
        "pages_failed": pages_failed,
        "ocr_engine": getattr(provider, "engine", "RapidOCR"),
        "ocr_models_used": sorted(list(models_used)),
        "language_configurations": sorted(list(languages_used)),
        "default_language": language,
        "preprocessing_configuration": "none (baseline 200 DPI rendered PNG)",
        "total_runtime_seconds": round(total_duration, 3),
        "average_runtime_per_page_seconds": round(avg_runtime, 3),
        "min_runtime_per_page_seconds": round(min_runtime, 3),
        "max_runtime_per_page_seconds": round(max_runtime, 3),
        "output_size_bytes": total_json_bytes,
        "output_size_mb": round(total_json_bytes / (1024 * 1024), 2),
        "confidence_statistics": {
            "mean": round(sum(page_confidences) / len(page_confidences), 4) if page_confidences else 0.0,
            "min": round(min(page_confidences), 4) if page_confidences else 0.0,
            "max": round(max(page_confidences), 4) if page_confidences else 0.0,
        },
        "failures": failures,
        "documents": manifest_docs,
    }

    manifest_path = out_p / "ocr_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log.info(
        "OCR dataset run complete: %d/%d pages succeeded (%d failed) in %.2fs (manifest: %s)",
        pages_succeeded,
        pages_attempted,
        pages_failed,
        total_duration,
        manifest_path,
    )
    return manifest


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6B OCR Ingestion using PP-OCRv5 (RapidOCR)")
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/rendered_pages",
        help="Path to rendered page directory (default: artifacts/rendered_pages)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/ocr",
        help="Path to OCR output directory (default: artifacts/ocr)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Default OCR language configuration ('latin', 'thai', etc.)",
    )
    parser.add_argument(
        "--doc-lang-map",
        type=str,
        default=None,
        help="JSON string or path to JSON file specifying per-document language configuration",
    )
    args = parser.parse_args()

    configure_logging()
    provider = RapidOCRProvider()

    doc_lang_map: Optional[dict[str, str]] = None
    if args.doc_lang_map:
        if Path(args.doc_lang_map).exists():
            with open(args.doc_lang_map, "r", encoding="utf-8") as f:
                doc_lang_map = json.load(f)
        else:
            doc_lang_map = json.loads(args.doc_lang_map)

    run_ocr_dataset(
        rendered_pages_dir=args.input,
        output_dir=args.output,
        provider=provider,
        language=args.language,
        doc_language_map=doc_lang_map,
    )


if __name__ == "__main__":
    main()
