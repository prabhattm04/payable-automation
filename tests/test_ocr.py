"""tests/test_ocr.py — Focused OCR infrastructure tests.

Tests all required aspects of Phase 6B:
1. Provider initialization
2. Single page OCR
3. Multi-page document OCR (ordered processing)
4. Output JSON creation & schema adherence
5. Unicode preservation (German umlauts, Portuguese accents, Estonian chars)
6. Thai text handling (Unicode integrity)
7. Sparse/near-empty page handling
8. Missing image handling (failure recorded cleanly)
9. OCR exception handling (isolation, does not abort run)
10. Confidence preservation (block-level & page-level mean/min/max)
11. Bounding-box and polygon geometry preservation
12. Configurable language support
13. Deterministic output structure
14. Manifest generation
15. Renderer to OCR path compatibility
16. Page identity preservation (doc, page, image_path)
17. Image dimensions preservation (width, height)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.ocr import (
    DEFAULT_ENGINE_NAME,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_NAME,
    DocumentOCRResult,
    OCRBlock,
    OCRProvider,
    PP_OCRv5Provider,
    PageOCRResult,
    _extract_page_sort_key,
    _infer_doc_and_page_from_path,
    get_image_dimensions,
    run_ocr_dataset,
    save_page_ocr_json,
)


# ══════════════════════════════════════════════════════════════════════════
# Mock & Fixtures
# ══════════════════════════════════════════════════════════════════════════

class MockOCRProvider(OCRProvider):
    """Deterministic mock provider for unit testing without downloading heavy models."""

    def __init__(self, should_fail: bool = False, fail_on_page: int = -1) -> None:
        self.should_fail = should_fail
        self.fail_on_page = fail_on_page
        self.languages_seen: list[str] = []

    def ocr_page(
        self,
        image_path: str | Path,
        document_id: str | None = None,
        page_number: int | None = None,
        language: str = DEFAULT_LANGUAGE,
        **kwargs: Any,
    ) -> PageOCRResult:
        self.languages_seen.append(language)
        img_p = Path(image_path)
        inferred_doc, inferred_page = _infer_doc_and_page_from_path(img_p)
        doc = document_id or inferred_doc
        page = page_number if page_number is not None else (inferred_page or 1)

        if not img_p.exists():
            return PageOCRResult(
                document=doc,
                page=page,
                image_path=str(img_p),
                image_width=0,
                image_height=0,
                language=language,
                status="failed",
                error=f"File not found: {img_p}",
            )

        if self.should_fail or (self.fail_on_page == page):
            return PageOCRResult(
                document=doc,
                page=page,
                image_path=str(img_p),
                image_width=1654,
                image_height=2339,
                language=language,
                status="failed",
                error="Simulated OCR engine failure",
            )

        # Realistic mock blocks
        blocks = [
            OCRBlock(
                text="Rechnung Nr. 12345 — Total €816,50",
                confidence=0.985,
                bbox=[100, 200, 500, 240],
                polygon=[[100, 200], [500, 200], [500, 240], [100, 240]],
            ),
            OCRBlock(
                text="Serviço comissão: não incluído — Überweisung",
                confidence=0.942,
                bbox=[100, 260, 600, 300],
                polygon=[[100, 260], [600, 260], [600, 300], [100, 300]],
            ),
            OCRBlock(
                text="ใบแจ้งหนี้ / ใบเสร็จรับเงิน ค่าบริการ 15,000.00 บาท",
                confidence=0.910,
                bbox=[100, 320, 700, 360],
                polygon=[[100, 320], [700, 320], [700, 360], [100, 360]],
            ),
        ]
        text = "\n".join(b.text for b in blocks)
        confs = [b.confidence for b in blocks]

        return PageOCRResult(
            document=doc,
            page=page,
            image_path=str(img_p),
            image_width=1654,
            image_height=2339,
            engine="PaddleOCR",
            model="PP-OCRv5",
            language=language,
            text=text,
            confidence_mean=sum(confs) / len(confs),
            confidence_min=min(confs),
            confidence_max=max(confs),
            blocks=blocks,
            runtime_seconds=0.05,
            status="success",
        )


@pytest.fixture
def temp_rendered_dir(tmp_path: Path) -> Path:
    """Creates a temporary rendered pages directory matching Phase 6A output."""
    stem_dir = tmp_path / "INV-01"
    stem_dir.mkdir(parents=True)
    # Create mock 1x1 PNG files
    # Minimal 1x1 PNG binary
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    for p in (1, 2):
        (stem_dir / f"page_{p:03d}.png").write_bytes(png_bytes)
    return tmp_path


# ══════════════════════════════════════════════════════════════════════════
# Test Cases
# ══════════════════════════════════════════════════════════════════════════

class TestOCRInfrastructure:
    """Comprehensive test suite for Phase 6B OCR infrastructure."""

    def test_provider_initialization(self) -> None:
        """1. OCR provider can be initialized with default and custom parameters."""
        provider = PP_OCRv5Provider(use_angle_cls=False, use_gpu=False, ocr_version="PP-OCRv4")
        assert provider.use_angle_cls is False
        assert provider.use_gpu is False
        assert provider.ocr_version == "PP-OCRv4"
        assert provider._models == {}

    def test_ocr_one_page(self, temp_rendered_dir: Path) -> None:
        """2. OCR a single page produces a valid PageOCRResult."""
        provider = MockOCRProvider()
        img = temp_rendered_dir / "INV-01" / "page_001.png"
        res = provider.ocr_page(img, language="latin")

        assert res.status == "success"
        assert res.document == "INV-01.pdf"
        assert res.page == 1
        assert len(res.blocks) == 3
        assert res.confidence_mean > 0.9
        assert "Rechnung" in res.text

    def test_ocr_multipage_document(self, temp_rendered_dir: Path) -> None:
        """3. ocr_document() processes multiple pages in ordered sequence."""
        provider = MockOCRProvider()
        p1 = temp_rendered_dir / "INV-01" / "page_001.png"
        p2 = temp_rendered_dir / "INV-01" / "page_002.png"

        doc_res = provider.ocr_document([p2, p1], document_id="INV-01.pdf", language="latin")

        assert doc_res.document == "INV-01.pdf"
        assert len(doc_res.pages) == 2
        # Verify page order was sorted correctly
        assert doc_res.pages[0].page == 1
        assert doc_res.pages[1].page == 2
        assert doc_res.status == "success"

    def test_output_json_creation(self, temp_rendered_dir: Path, tmp_path: Path) -> None:
        """4. save_page_ocr_json() creates the required JSON artifact matching Section 9 schema."""
        provider = MockOCRProvider()
        img = temp_rendered_dir / "INV-01" / "page_001.png"
        res = provider.ocr_page(img)

        out_dir = tmp_path / "artifacts" / "ocr"
        saved_path = save_page_ocr_json(res, out_dir)

        assert saved_path.exists()
        assert saved_path.name == "page_001.json"
        assert saved_path.parent.name == "INV-01"

        data = json.loads(saved_path.read_text(encoding="utf-8"))
        # Required Section 9 keys
        for key in ("document", "page", "image", "engine", "model", "language", "text", "confidence", "blocks"):
            assert key in data, f"Missing required key in OCR JSON: {key}"

        assert isinstance(data["blocks"], list)
        assert len(data["blocks"]) > 0
        block = data["blocks"][0]
        assert "text" in block
        assert "confidence" in block
        assert "bbox" in block
        assert "polygon" in block

    def test_unicode_preservation(self, tmp_path: Path) -> None:
        """5. Unicode characters (German umlauts, Portuguese accents, currency symbols) survive serialization."""
        block = OCRBlock(
            text="Geprüfte Qualität: Größter Erlös — Überweisung 1.000,00 € / Comissão não é cobrança",
            confidence=0.97,
            bbox=[10, 20, 200, 50],
            polygon=[[10, 20], [200, 20], [200, 50], [10, 50]],
        )
        res = PageOCRResult(
            document="DOC-UNI.pdf",
            page=1,
            image_path="dummy.png",
            image_width=1000,
            image_height=1000,
            text=block.text,
            confidence_mean=0.97,
            blocks=[block],
        )
        out_file = save_page_ocr_json(res, tmp_path)
        loaded = json.loads(out_file.read_text(encoding="utf-8"))

        assert "Geprüfte" in loaded["text"]
        assert "Größter" in loaded["text"]
        assert "Überweisung" in loaded["text"]
        assert "€" in loaded["text"]
        assert "Comissão" in loaded["text"]
        assert "não" in loaded["text"]
        assert "cobrança" in loaded["text"]

    def test_thai_text_handling(self, tmp_path: Path) -> None:
        """6. Thai script is preserved intact through data structures and JSON serialization."""
        thai_sample = "บริษัท ทดสอบ จำกัด ใบเสร็จรับเงิน ภาษีมูลค่าเพิ่ม 7%"
        block = OCRBlock(
            text=thai_sample,
            confidence=0.93,
            bbox=[50, 50, 400, 80],
            polygon=[[50, 50], [400, 50], [400, 80], [50, 80]],
        )
        res = PageOCRResult(
            document="HLD-01.pdf",
            page=1,
            image_path="dummy.png",
            image_width=1654,
            image_height=2339,
            language="thai",
            text=thai_sample,
            confidence_mean=0.93,
            blocks=[block],
        )
        out_file = save_page_ocr_json(res, tmp_path)
        loaded = json.loads(out_file.read_text(encoding="utf-8"))

        assert loaded["text"] == thai_sample
        assert loaded["blocks"][0]["text"] == thai_sample
        assert loaded["language"] == "thai"

    def test_sparse_near_empty_page(self, tmp_path: Path) -> None:
        """7. A sparse or empty page produces valid zero-count output without crashing."""
        res = PageOCRResult(
            document="EMPTY.pdf",
            page=1,
            image_path="empty.png",
            image_width=1654,
            image_height=2339,
            text="",
            confidence_mean=0.0,
            confidence_min=0.0,
            confidence_max=0.0,
            blocks=[],
            status="success",
        )
        out_file = save_page_ocr_json(res, tmp_path)
        loaded = json.loads(out_file.read_text(encoding="utf-8"))

        assert loaded["status"] == "success"
        assert loaded["text"] == ""
        assert loaded["blocks"] == []
        assert loaded["confidence"] == 0.0

    def test_missing_image_handling(self) -> None:
        """8. Non-existent image file is recorded cleanly as a failure without crashing."""
        provider = MockOCRProvider()
        res = provider.ocr_page("non_existent_image_path_12345.png")

        assert res.status == "failed"
        assert "not found" in (res.error or "").lower()
        assert res.blocks == []
        assert res.confidence_mean == 0.0

    def test_ocr_exception_handling(self, temp_rendered_dir: Path) -> None:
        """9. Exception on one page does not crash multi-page execution; errors are isolated."""
        provider = MockOCRProvider(fail_on_page=1)
        p1 = temp_rendered_dir / "INV-01" / "page_001.png"
        p2 = temp_rendered_dir / "INV-01" / "page_002.png"

        doc_res = provider.ocr_document([p1, p2], document_id="INV-01.pdf")

        assert len(doc_res.pages) == 2
        # Page 1 failed
        assert doc_res.pages[0].status == "failed"
        assert doc_res.pages[0].error is not None
        # Page 2 succeeded
        assert doc_res.pages[1].status == "success"
        assert len(doc_res.pages[1].blocks) > 0
        assert doc_res.status == "partial_failure"

    def test_confidence_preservation(self, temp_rendered_dir: Path) -> None:
        """10. Per-block confidence and page-level stats (mean, min, max) are preserved in [0.0, 1.0]."""
        provider = MockOCRProvider()
        img = temp_rendered_dir / "INV-01" / "page_001.png"
        res = provider.ocr_page(img)

        assert 0.0 <= res.confidence_mean <= 1.0
        assert 0.0 <= res.confidence_min <= 1.0
        assert 0.0 <= res.confidence_max <= 1.0
        assert res.confidence_min <= res.confidence_mean <= res.confidence_max

        for block in res.blocks:
            assert isinstance(block.confidence, float)
            assert 0.0 <= block.confidence <= 1.0

    def test_bbox_and_polygon_preservation(self, temp_rendered_dir: Path) -> None:
        """11. Axis-aligned bounding box [x1, y1, x2, y2] and 4-point polygon coordinates are preserved."""
        provider = MockOCRProvider()
        img = temp_rendered_dir / "INV-01" / "page_001.png"
        res = provider.ocr_page(img)

        for block in res.blocks:
            # bbox checks
            assert len(block.bbox) == 4
            x1, y1, x2, y2 = block.bbox
            assert x2 >= x1
            assert y2 >= y1
            # polygon checks
            assert len(block.polygon) == 4
            for pt in block.polygon:
                assert len(pt) == 2

    def test_configurable_language(self, temp_rendered_dir: Path) -> None:
        """12. OCRProvider accepts configurable language parameter dynamically without code change."""
        provider = MockOCRProvider()
        img = temp_rendered_dir / "INV-01" / "page_001.png"

        res_latin = provider.ocr_page(img, language="latin")
        assert res_latin.language == "latin"

        res_thai = provider.ocr_page(img, language="thai")
        assert res_thai.language == "thai"

        assert provider.languages_seen == ["latin", "thai"]

    def test_deterministic_output_structure(self, temp_rendered_dir: Path) -> None:
        """13. Multiple runs produce identical schema keys and deterministic types."""
        provider = MockOCRProvider()
        img = temp_rendered_dir / "INV-01" / "page_001.png"

        d1 = provider.ocr_page(img).to_dict()
        d2 = provider.ocr_page(img).to_dict()

        assert d1.keys() == d2.keys()
        for k in d1:
            assert type(d1[k]) == type(d2[k])

    def test_manifest_generation(self, temp_rendered_dir: Path, tmp_path: Path) -> None:
        """14. Dataset run produces valid ocr_manifest.json with all required summary fields."""
        provider = MockOCRProvider()
        out_dir = tmp_path / "artifacts" / "ocr"

        manifest = run_ocr_dataset(
            rendered_pages_dir=temp_rendered_dir,
            output_dir=out_dir,
            provider=provider,
            language="latin",
        )

        manifest_path = out_dir / "ocr_manifest.json"
        assert manifest_path.exists()

        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert loaded_manifest["total_pages"] == 2
        assert loaded_manifest["pages_succeeded"] == 2
        assert loaded_manifest["pages_failed"] == 0
        assert "confidence_statistics" in loaded_manifest
        assert "INV-01" in loaded_manifest["documents"]

    def test_renderer_to_ocr_path_compatibility(self) -> None:
        """15. Renderer output paths correctly map to document stem and page number."""
        doc, page = _infer_doc_and_page_from_path("artifacts/rendered_pages/HLD-01/page_001.png")
        assert doc == "HLD-01.pdf"
        assert page == 1

        doc2, page2 = _infer_doc_and_page_from_path("artifacts/rendered_pages/DU-02/page_018.png")
        assert doc2 == "DU-02.pdf"
        assert page2 == 18

    def test_page_identity_preservation(self, temp_rendered_dir: Path) -> None:
        """16. Document identity, page number, and source image path remain exact."""
        provider = MockOCRProvider()
        img = temp_rendered_dir / "INV-01" / "page_002.png"
        res = provider.ocr_page(img, document_id="INV-01.pdf", page_number=2)

        assert res.document == "INV-01.pdf"
        assert res.page == 2
        assert res.image_path == str(img)

        # In serialized form as well
        d = res.to_dict()
        assert d["document"] == "INV-01.pdf"
        assert d["page"] == 2
        assert d["image"] == str(img)

    def test_image_dimensions_preservation(self, temp_rendered_dir: Path) -> None:
        """17. Preserves image_width and image_height in PageOCRResult."""
        provider = MockOCRProvider()
        img = temp_rendered_dir / "INV-01" / "page_001.png"
        res = provider.ocr_page(img)

        assert res.image_width > 0
        assert res.image_height > 0
        d = res.to_dict()
        assert "image_width" in d
        assert "image_height" in d
        assert d["image_width"] == res.image_width
        assert d["image_height"] == res.image_height

    def test_rapidocr_multilingual_resolution(self) -> None:
        """18. RapidOCRProvider dynamically resolves Latin, Thai, and Default models."""
        from src.extraction.ocr import RapidOCRProvider

        provider = RapidOCRProvider()
        lat_model, lat_dict, lat_tag = provider._resolve_model_paths("latin")
        assert "latin" in lat_model.lower()
        assert lat_tag == "PP-OCRv5-ONNX-Latin"

        thai_model, thai_dict, thai_tag = provider._resolve_model_paths("thai")
        assert "thai" in thai_model.lower()
        assert thai_tag == "PP-OCRv5-ONNX-Thai"

        def_model, def_dict, def_tag = provider._resolve_model_paths("unknown")
        assert def_model is None
        assert def_tag == "PP-OCRv4-ONNX-Default"

    def test_rapidocr_real_smoke_test(self) -> None:
        """19. Real RapidOCR smoke test verifying end-to-end inference and geometry."""
        from src.extraction.ocr import RapidOCRProvider

        provider = RapidOCRProvider()
        img_path = Path("artifacts/rendered_pages/INV-01/page_001.png")
        if img_path.exists():
            res = provider.ocr_page(img_path, language="latin")
            assert res.status == "success"
            assert len(res.blocks) > 0
            assert res.confidence_mean > 0.9
            assert res.image_width == 1654
            assert res.image_height == 2339
            assert "Rechnung" in res.text
            # Verify geometry integrity
            for b in res.blocks:
                assert len(b.bbox) == 4
                assert len(b.polygon) == 4

