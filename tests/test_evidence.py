"""tests/test_evidence.py — Comprehensive test suite for Phase 7A Evidence & Provenance."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.understanding.evidence import (
    Evidence,
    EvidenceProvenance,
    EvidenceSource,
    PageEvidence,
    generate_evidence_id,
    native_page_text_to_evidence,
    ocr_block_to_evidence,
    ocr_json_to_page_evidence,
    page_ocr_result_to_evidence,
    validate_source,
    vision_observation_to_evidence,
    vision_response_to_evidence,
)
from src.extraction.ocr import OCRBlock, PageOCRResult
from src.vision.provider import VisionResponse
from src.pdf.text_extractor import PageText


# ══════════════════════════════════════════════════════════════════════════
# 1. Source Validation & Enumeration
# ══════════════════════════════════════════════════════════════════════════

class TestEvidenceSourceValidation:
    def test_all_supported_sources_valid(self) -> None:
        """Supported sources (ocr, vision, native_pdf, manual, derived) validate cleanly."""
        expected = ["ocr", "vision", "native_pdf", "manual", "derived"]
        for src_str in expected:
            res = validate_source(src_str)
            assert isinstance(res, EvidenceSource)
            assert res.value == src_str

    def test_case_insensitivity_and_whitespace(self) -> None:
        """Source validator is case-insensitive and trims whitespace."""
        assert validate_source("  OCR  ") == EvidenceSource.OCR
        assert validate_source("Vision") == EvidenceSource.VISION
        assert validate_source("NATIVE_PDF") == EvidenceSource.NATIVE_PDF

    def test_invalid_source_raises_value_error(self) -> None:
        """Unsupported sources raise ValueError with clear error message."""
        with pytest.raises(ValueError, match="Invalid evidence source: 'unknown'"):
            validate_source("unknown")

        with pytest.raises(ValueError, match="Invalid evidence source"):
            validate_source("llm_generated")


# ══════════════════════════════════════════════════════════════════════════
# 2. Provenance Single Source of Truth & Canonical Ownership
# ══════════════════════════════════════════════════════════════════════════

class TestProvenanceCanonicalOwnership:
    def test_evidence_derived_properties_delegate_to_provenance(self) -> None:
        """Evidence exposes source, extraction_method, confidence, document_id, page_number
        as read-only derived properties delegating to EvidenceProvenance."""
        prov = EvidenceProvenance(
            source=EvidenceSource.OCR,
            document_id="INV-01.pdf",
            page_number=2,
            extraction_method="RapidOCR/PP-OCRv5-ONNX-Latin",
            confidence=0.985,
            file_path="artifacts/rendered_pages/INV-01/page_002.png",
        )
        ev = Evidence(provenance=prov, content="Invoice Total: $500.00")

        # Canonical delegation
        assert ev.source == EvidenceSource.OCR
        assert ev.extraction_method == "RapidOCR/PP-OCRv5-ONNX-Latin"
        assert ev.confidence == 0.985
        assert ev.document_id == "INV-01.pdf"
        assert ev.page_number == 2
        assert ev.text == "Invoice Total: $500.00"

    def test_derived_properties_are_read_only(self) -> None:
        """Attempting to assign directly to derived properties on Evidence raises AttributeError."""
        prov = EvidenceProvenance(
            source=EvidenceSource.OCR,
            document_id="INV-01.pdf",
            page_number=1,
            extraction_method="RapidOCR",
            confidence=0.95,
        )
        ev = Evidence(provenance=prov, content="Line Item")

        with pytest.raises(AttributeError):
            ev.source = EvidenceSource.VISION  # type: ignore

        with pytest.raises(AttributeError):
            ev.extraction_method = "other"  # type: ignore

        with pytest.raises(AttributeError):
            ev.confidence = 0.5  # type: ignore

        with pytest.raises(AttributeError):
            ev.document_id = "other"  # type: ignore

        with pytest.raises(AttributeError):
            ev.page_number = 99  # type: ignore

    def test_provenance_answers_all_core_questions(self) -> None:
        """Provenance contains explicit fields answering where, which page, what method, what confidence."""
        prov = EvidenceProvenance(
            source=EvidenceSource.VISION,
            document_id="HLD-03.pdf",
            page_number=1,
            extraction_method="Puter/qwen/qwen3-vl-plus-2025-12-19",
            confidence=None,
            file_path="artifacts/rendered_pages/HLD-03/page_001.png",
            metadata={"latency_seconds": 12.34},
        )
        # 1. Where did it come from?
        assert prov.source == EvidenceSource.VISION
        assert prov.file_path == "artifacts/rendered_pages/HLD-03/page_001.png"
        assert prov.document_id == "HLD-03.pdf"
        # 2. Which page?
        assert prov.page_number == 1
        # 3. Which extraction method / model?
        assert prov.extraction_method == "Puter/qwen/qwen3-vl-plus-2025-12-19"
        # 4. What confidence was supplied?
        assert prov.confidence is None


# ══════════════════════════════════════════════════════════════════════════
# 3. Lossless Geometry Preservation
# ══════════════════════════════════════════════════════════════════════════

class TestLosslessGeometry:
    def test_float_coordinates_preserved_without_integer_coercion(self) -> None:
        """Float coordinates are preserved with full floating-point precision."""
        prov = EvidenceProvenance(
            source=EvidenceSource.OCR,
            document_id="DOC-01",
            page_number=1,
            extraction_method="CustomOCR",
            confidence=0.99,
        )
        bbox = [10.55, 20.75, 150.125, 200.875]
        poly = [
            [10.55, 20.75],
            [150.125, 20.75],
            [150.125, 200.875],
            [10.55, 200.875],
        ]
        ev = Evidence(
            provenance=prov,
            content="Subtotal",
            bbox=bbox,
            polygon=poly,
        )
        assert ev.bbox == [10.55, 20.75, 150.125, 200.875]
        assert ev.polygon == poly

    def test_integer_coordinates_safely_cast_to_floats(self) -> None:
        """Integer coordinates (from e.g. RapidOCR) are converted to float representation."""
        prov = EvidenceProvenance(
            source=EvidenceSource.OCR,
            document_id="DOC-01",
            page_number=1,
            extraction_method="RapidOCR",
            confidence=0.99,
        )
        ev = Evidence(
            provenance=prov,
            content="Sample",
            bbox=[100, 200, 300, 400],
        )
        assert ev.bbox == [100.0, 200.0, 300.0, 400.0]


# ══════════════════════════════════════════════════════════════════════════
# 4. Deterministic Evidence ID Generation
# ══════════════════════════════════════════════════════════════════════════

class TestDeterministicEvidenceIDs:
    def test_deterministic_id_reproducibility(self) -> None:
        """Identical immutable inputs always yield the exact same evidence ID."""
        id1 = generate_evidence_id(
            document_id="INV-01.pdf",
            page_number=1,
            source=EvidenceSource.OCR,
            extraction_method="RapidOCR/PP-OCRv5",
            content="Rechnung Nr.: 852566",
            bbox=[156.0, 411.0, 416.0, 446.0],
            index=0,
        )
        id2 = generate_evidence_id(
            document_id="INV-01.pdf",
            page_number=1,
            source="ocr",
            extraction_method="RapidOCR/PP-OCRv5",
            content="Rechnung Nr.: 852566",
            bbox=[156, 411, 416, 446],
            index=0,
        )
        assert id1 == id2
        assert id1.startswith("INV-01.pdf:p1:ocr:")

    def test_id_independent_of_file_paths_or_timestamps(self) -> None:
        """Evidence IDs do not incorporate file paths, timestamps, or system state."""
        prov1 = EvidenceProvenance(
            source=EvidenceSource.OCR,
            document_id="INV-01.pdf",
            page_number=1,
            extraction_method="RapidOCR",
            file_path="C:/Users/prabh/path_a/page_001.png",
            timestamp="2026-09-14T12:00:00Z",
        )
        prov2 = EvidenceProvenance(
            source=EvidenceSource.OCR,
            document_id="INV-01.pdf",
            page_number=1,
            extraction_method="RapidOCR",
            file_path="/var/data/different_machine/page_001.png",
            timestamp="2026-09-14T16:30:00Z",
        )
        ev1 = Evidence(provenance=prov1, content="Total", bbox=[100, 200, 300, 400])
        ev2 = Evidence(provenance=prov2, content="Total", bbox=[100, 200, 300, 400])
        assert ev1.evidence_id == ev2.evidence_id

    def test_different_geometry_or_content_produces_distinct_ids(self) -> None:
        """Distinct text or distinct bbox produces distinct IDs."""
        id_a = generate_evidence_id(
            document_id="DOC-01",
            page_number=1,
            source=EvidenceSource.OCR,
            extraction_method="M",
            content="Alpha",
            bbox=[10, 10, 50, 50],
        )
        id_b = generate_evidence_id(
            document_id="DOC-01",
            page_number=1,
            source=EvidenceSource.OCR,
            extraction_method="M",
            content="Beta",
            bbox=[10, 10, 50, 50],
        )
        id_c = generate_evidence_id(
            document_id="DOC-01",
            page_number=1,
            source=EvidenceSource.OCR,
            extraction_method="M",
            content="Alpha",
            bbox=[10, 10, 60, 60],
        )
        assert id_a != id_b
        assert id_a != id_c


# ══════════════════════════════════════════════════════════════════════════
# 5. Conversion Adapters: OCR, Vision, and Native PDF
# ══════════════════════════════════════════════════════════════════════════

class TestConversionAdapters:
    def test_ocr_block_to_evidence(self) -> None:
        """Converts Phase 6B OCRBlock preserving all fields and geometry."""
        block = OCRBlock(
            text="Northwind Operations OÜ",
            confidence=0.9843,
            bbox=[156, 411, 416, 446],
            polygon=[[156, 414], [416, 411], [416, 443], [156, 446]],
        )
        ev = ocr_block_to_evidence(
            block=block,
            document_id="INV-01.pdf",
            page_number=1,
            engine="RapidOCR",
            model="PP-OCRv5-ONNX-Latin",
            file_path="artifacts/rendered_pages/INV-01/page_001.png",
            index=0,
        )
        assert ev.content == "Northwind Operations OÜ"
        assert ev.source == EvidenceSource.OCR
        assert ev.extraction_method == "RapidOCR/PP-OCRv5-ONNX-Latin"
        assert ev.confidence == 0.9843
        assert ev.bbox == [156.0, 411.0, 416.0, 446.0]
        assert ev.polygon == [[156.0, 414.0], [416.0, 411.0], [416.0, 443.0], [156.0, 446.0]]
        assert ev.semantic_role is None
        assert ev.provenance.file_path == "artifacts/rendered_pages/INV-01/page_001.png"

    def test_page_ocr_result_to_evidence(self) -> None:
        """Converts PageOCRResult into PageEvidence."""
        b1 = OCRBlock(text="Line 1", confidence=0.99, bbox=[0, 0, 10, 10], polygon=[[0,0],[10,0],[10,10],[0,10]])
        b2 = OCRBlock(text="Line 2", confidence=0.92, bbox=[0, 20, 10, 30], polygon=[[0,20],[10,20],[10,30],[0,30]])
        res = PageOCRResult(
            document="DOC-01",
            page=1,
            image_path="test.png",
            image_width=800,
            image_height=1000,
            blocks=[b1, b2],
            confidence_mean=0.955,
        )
        page_ev = page_ocr_result_to_evidence(res)
        assert page_ev.document_id == "DOC-01"
        assert page_ev.page_number == 1
        assert len(page_ev) == 2
        assert page_ev.image_width == 800.0
        assert page_ev.image_height == 1000.0
        assert page_ev[0].content == "Line 1"
        assert page_ev[1].content == "Line 2"

    def test_vision_response_to_evidence_strictly_null_confidence(self) -> None:
        """Vision responses must set confidence strictly to None (no fabricated scores)."""
        resp = VisionResponse(
            success=True,
            content="Invoice Number: INV-2026-001\nTotal: 608.28 EUR",
            model="qwen/qwen3-vl-plus-2025-12-19",
            provider="Puter",
            latency_seconds=3.456,
            metadata={"finish_reason": "stop"},
        )
        ev = vision_response_to_evidence(
            response=resp,
            document_id="INV-02.pdf",
            page_number=1,
            file_path="artifacts/rendered_pages/INV-02/page_001.png",
        )
        assert ev.source == EvidenceSource.VISION
        assert ev.extraction_method == "Puter/qwen/qwen3-vl-plus-2025-12-19"
        assert ev.confidence is None  # Requirement 6 & Design Adjustment 1
        assert ev.content == "Invoice Number: INV-2026-001\nTotal: 608.28 EUR"
        assert ev.provenance.confidence is None
        assert ev.provenance.metadata["latency_seconds"] == 3.456

    def test_vision_observation_helper_null_confidence(self) -> None:
        """vision_observation_to_evidence sets confidence to None."""
        ev = vision_observation_to_evidence(
            content="Payment Terms: 30 days net",
            document_id="INV-03.pdf",
            page_number=1,
            model="qwen3-vl-plus",
            provider="Puter",
        )
        assert ev.source == EvidenceSource.VISION
        assert ev.confidence is None
        assert ev.content == "Payment Terms: 30 days net"

    def test_native_page_text_to_evidence_passive(self) -> None:
        """native_page_text_to_evidence is a passive adapter with uncalibrated (None) confidence."""
        pt = PageText(
            page_number=1,
            raw_text="Native PDF Extracted Heading\nAccount: 12345",
            character_count=42,
            word_count=5,
            native_text_available=True,
            extraction_method="native",
            fonts_detected=["Helvetica-Bold"],
            has_images=False,
            width_pt=595.0,
            height_pt=842.0,
        )
        items = native_page_text_to_evidence(pt, document_id="DOC-NATIVE.pdf")
        assert len(items) == 1
        ev = items[0]
        assert ev.source == EvidenceSource.NATIVE_PDF
        assert ev.confidence is None  # Design Adjustment 4: passive adapter, no artificial trust
        assert "Native PDF Extracted Heading" in ev.content
        assert ev.extraction_method == "native"
        assert ev.metadata["character_count"] == 42


# ══════════════════════════════════════════════════════════════════════════
# 6. Reading Order & Coexistence of Conflicting Evidence
# ══════════════════════════════════════════════════════════════════════════

class TestReadingOrderAndConflictCoexistence:
    def test_page_evidence_full_text_concatenates_in_given_order(self) -> None:
        """PageEvidence.full_text() strictly concatenates items in their existing sequence
        without spatial sorting or 2D heuristic reading order."""
        page_ev = PageEvidence(document_id="DOC-01", page_number=1)
        # Add item that is spatially lower first
        e1 = Evidence.create(
            content="Bottom Footer",
            document_id="DOC-01",
            page_number=1,
            source=EvidenceSource.OCR,
            extraction_method="OCR",
            bbox=[100, 900, 200, 950],
        )
        # Add item that is spatially higher second
        e2 = Evidence.create(
            content="Top Header",
            document_id="DOC-01",
            page_number=1,
            source=EvidenceSource.OCR,
            extraction_method="OCR",
            bbox=[100, 50, 200, 100],
        )
        page_ev.add_item(e1)
        page_ev.add_item(e2)

        # Must preserve exact insertion order, not re-sort by y-coordinate
        assert page_ev.full_text(separator="\n") == "Bottom Footer\nTop Header"

    def test_conflicting_observations_coexist_independently(self) -> None:
        """Design Adjustment 5:
        Conflicting observations from different sources (OCR vs Vision) can coexist.
        Example: OCR = '608.23', Vision = '608.28'.
        The evidence layer must NOT resolve, merge, repair, rank, or reject the conflict.
        """
        page_ev = PageEvidence(document_id="INV-CONFLICT.pdf", page_number=1)

        ocr_ev = Evidence.create(
            content="608.23",
            document_id="INV-CONFLICT.pdf",
            page_number=1,
            source=EvidenceSource.OCR,
            extraction_method="RapidOCR/PP-OCRv5-ONNX-Latin",
            confidence=0.975,
            bbox=[350.0, 720.0, 420.0, 740.0],
        )
        vision_ev = Evidence.create(
            content="608.28",
            document_id="INV-CONFLICT.pdf",
            page_number=1,
            source=EvidenceSource.VISION,
            extraction_method="Puter/qwen3-vl-plus",
            confidence=None,
            bbox=[348.0, 718.0, 422.0, 742.0],
        )

        page_ev.add_item(ocr_ev)
        page_ev.add_item(vision_ev)

        # Both items preserved independently
        assert len(page_ev) == 2
        assert page_ev[0].content == "608.23"
        assert page_ev[0].source == EvidenceSource.OCR
        assert page_ev[0].confidence == 0.975

        assert page_ev[1].content == "608.28"
        assert page_ev[1].source == EvidenceSource.VISION
        assert page_ev[1].confidence is None

        # Filter helpers work cleanly
        ocr_items = page_ev.filter_by_source(EvidenceSource.OCR)
        vision_items = page_ev.filter_by_source(EvidenceSource.VISION)
        assert len(ocr_items) == 1 and ocr_items[0].content == "608.23"
        assert len(vision_items) == 1 and vision_items[0].content == "608.28"


# ══════════════════════════════════════════════════════════════════════════
# 7. Serialization & Deserialization Round-Trip
# ══════════════════════════════════════════════════════════════════════════

class TestSerializationRoundTrip:
    def test_evidence_json_round_trip(self) -> None:
        """Evidence -> to_dict() -> json.dumps -> json.loads -> from_dict() round trip."""
        original = Evidence.create(
            content="Total: 1,234.56 EUR",
            document_id="INV-01.pdf",
            page_number=1,
            source=EvidenceSource.OCR,
            extraction_method="RapidOCR/PP-OCRv5",
            confidence=0.9876,
            bbox=[100.5, 200.25, 300.75, 250.0],
            polygon=[[100.5, 200.25], [300.75, 200.25], [300.75, 250.0], [100.5, 250.0]],
            semantic_role=None,
            file_path="artifacts/rendered_pages/INV-01/page_001.png",
            metadata={"test_key": "test_value"},
        )
        json_str = original.to_json()
        loaded = Evidence.from_json(json_str)

        assert loaded.evidence_id == original.evidence_id
        assert loaded.content == original.content
        assert loaded.source == original.source
        assert loaded.extraction_method == original.extraction_method
        assert loaded.confidence == original.confidence
        assert loaded.bbox == original.bbox
        assert loaded.polygon == original.polygon
        assert loaded.semantic_role == original.semantic_role
        assert loaded.metadata == original.metadata
        assert loaded.provenance.file_path == original.provenance.file_path

    def test_null_confidence_serializes_as_explicit_null(self) -> None:
        """When confidence is None, JSON serialization outputs 'confidence': null."""
        ev = Evidence.create(
            content="Visual description",
            document_id="DOC-01",
            page_number=1,
            source=EvidenceSource.VISION,
            extraction_method="Puter/qwen3-vl-plus",
            confidence=None,
        )
        json_str = ev.to_json()
        raw_dict = json.loads(json_str)
        assert raw_dict["confidence"] is None
        assert raw_dict["provenance"]["confidence"] is None

    def test_page_evidence_json_round_trip(self) -> None:
        """PageEvidence round trip preserves all items, dimensions, and metadata."""
        page_ev = PageEvidence(
            document_id="INV-01.pdf",
            page_number=1,
            image_path="page_001.png",
            image_width=1654.0,
            image_height=2339.0,
            metadata={"lang": "de"},
        )
        page_ev.add_item(
            Evidence.create(
                content="Line 1",
                document_id="INV-01.pdf",
                page_number=1,
                source=EvidenceSource.OCR,
                extraction_method="RapidOCR",
                confidence=0.99,
            )
        )
        json_str = page_ev.to_json()
        loaded_page = PageEvidence.from_json(json_str)

        assert loaded_page.document_id == "INV-01.pdf"
        assert loaded_page.page_number == 1
        assert loaded_page.image_width == 1654.0
        assert loaded_page.image_height == 2339.0
        assert len(loaded_page) == 1
        assert loaded_page[0].content == "Line 1"

    def test_missing_optional_fields_handled_cleanly(self) -> None:
        """Evidence without bbox, polygon, metadata, or semantic_role handles defaults gracefully."""
        minimal_dict = {
            "evidence_id": "MIN-01",
            "document_id": "DOC-MIN",
            "page_number": 1,
            "content": "Minimal text",
            "source": "manual",
            "extraction_method": "user_input",
            "confidence": None,
        }
        ev = Evidence.from_dict(minimal_dict)
        assert ev.evidence_id == "MIN-01"
        assert ev.content == "Minimal text"
        assert ev.bbox is None
        assert ev.polygon is None
        assert ev.semantic_role is None
        assert ev.metadata == {}
        assert ev.provenance.source == EvidenceSource.MANUAL


# ══════════════════════════════════════════════════════════════════════════
# 8. Live Artifact Ingestion
# ══════════════════════════════════════════════════════════════════════════

class TestLiveArtifactIngestion:
    def test_ingest_real_ocr_artifact(self) -> None:
        """Loads and verifies actual Phase 6B artifact artifacts/ocr/INV-01/page_001.json."""
        artifact_path = Path("artifacts/ocr/INV-01/page_001.json")
        if not artifact_path.exists():
            pytest.skip(f"Artifact {artifact_path} not found in environment.")

        page_ev = ocr_json_to_page_evidence(artifact_path)
        assert page_ev.document_id == "INV-01.pdf"
        assert page_ev.page_number == 1
        assert page_ev.image_width == 1654.0
        assert page_ev.image_height == 2339.0
        assert len(page_ev) > 10  # INV-01 has dozens of OCR blocks

        # Verify first block
        first = page_ev[0]
        assert first.source == EvidenceSource.OCR
        assert first.extraction_method == "RapidOCR/PP-OCRv5-ONNX-Latin"
        assert first.content == "Northwind Operations OÜ"
        assert first.confidence == pytest.approx(0.9843, abs=1e-4)
        assert first.bbox == [156.0, 411.0, 416.0, 446.0]
        assert len(first.polygon) == 4
        assert first.evidence_id.startswith("INV-01.pdf:p1:ocr:")
