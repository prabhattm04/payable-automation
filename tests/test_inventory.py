"""tests/test_inventory.py — Phase 5 inventory pipeline tests.

Tests are based on actual file presence and behaviour, NOT filename numbering.
Fixtures use a temporary directory with minimal in-memory PDFs created via PyMuPDF.

Test categories
---------------
1. PDF discovery (loader.py)
2. Missing / non-existent directory handling
3. Native text extraction (text_extractor.py)
4. Image-only page detection
5. Page count accuracy
6. Inventory generation (inspector.py)
7. One failed PDF does not abort the run

Run with:
    pytest tests/test_inventory.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make sure the project root is on the path when running from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────────────────────────────────────────────────────
# Helpers: create minimal PDFs in memory for testing
# ──────────────────────────────────────────────────────────────────────────

def _make_text_pdf(path: Path, text: str = "Invoice Number: 12345\nTotal: 100.00 EUR") -> None:
    """Create a minimal single-page text PDF using PyMuPDF."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), text)
    doc.save(str(path))
    doc.close()


def _make_image_pdf(path: Path) -> None:
    """Create a minimal PDF page with only an image (no native text)."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Draw a rectangle as a stand-in for an image (no text)
    page.draw_rect(fitz.Rect(50, 50, 400, 400), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    doc.save(str(path))
    doc.close()


def _make_multi_page_pdf(path: Path, n_pages: int = 3) -> None:
    """Create a multi-page text PDF."""
    import fitz
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((50, 100), f"Page {i+1}\nInvoice\nTotal: {(i+1)*100}.00 EUR")
    doc.save(str(path))
    doc.close()


def _make_corrupt_file(path: Path) -> None:
    """Write garbage bytes that are not a valid PDF."""
    path.write_bytes(b"NOT A PDF FILE %PDF")


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def docs_dir(tmp_path: Path):
    """A temporary directory with a mix of PDFs."""
    _make_text_pdf(tmp_path / "text_doc.pdf", "Invoice Number: TEST-001\nTotal: 250.00 EUR")
    _make_text_pdf(tmp_path / "another_invoice.pdf", "Rechnung Nr. 42\nMwSt: 19%\nGesamt: 119.00 EUR")
    _make_image_pdf(tmp_path / "image_only.pdf")
    _make_multi_page_pdf(tmp_path / "multi_page.pdf", n_pages=4)
    return tmp_path


@pytest.fixture()
def mixed_dir(tmp_path: Path):
    """Directory with one corrupt PDF and two valid ones."""
    _make_text_pdf(tmp_path / "valid1.pdf")
    _make_corrupt_file(tmp_path / "corrupt.pdf")
    _make_text_pdf(tmp_path / "valid2.pdf")
    return tmp_path


# ──────────────────────────────────────────────────────────────────────────
# 1. PDF discovery
# ──────────────────────────────────────────────────────────────────────────

class TestPdfDiscovery:
    def test_discovers_all_pdfs(self, docs_dir: Path):
        from src.pdf.loader import discover_pdfs
        found = discover_pdfs(docs_dir)
        assert len(found) == 4

    def test_sorted_deterministically(self, docs_dir: Path):
        from src.pdf.loader import discover_pdfs
        found1 = discover_pdfs(docs_dir)
        found2 = discover_pdfs(docs_dir)
        assert [f.filename for f in found1] == [f.filename for f in found2]

    def test_filenames_are_actual_files(self, docs_dir: Path):
        from src.pdf.loader import discover_pdfs
        found = discover_pdfs(docs_dir)
        for f in found:
            assert f.path.exists(), f"{f.filename} path does not exist"
            assert f.path.is_file()

    def test_file_size_is_positive(self, docs_dir: Path):
        from src.pdf.loader import discover_pdfs
        found = discover_pdfs(docs_dir)
        for f in found:
            assert f.file_size_bytes > 0, f"{f.filename} has zero size"

    def test_no_filename_numbering_assumption(self, tmp_path: Path):
        """Filenames with gaps must all be discovered (no range-based assumptions)."""
        from src.pdf.loader import discover_pdfs
        # Create PDFs with non-sequential names
        _make_text_pdf(tmp_path / "DOC-01.pdf")
        _make_text_pdf(tmp_path / "DOC-05.pdf")  # gap: 02,03,04 missing
        _make_text_pdf(tmp_path / "DOC-99.pdf")
        found = discover_pdfs(tmp_path)
        names = {f.filename for f in found}
        assert "DOC-01.pdf" in names
        assert "DOC-05.pdf" in names
        assert "DOC-99.pdf" in names
        assert len(found) == 3  # only what exists


# ──────────────────────────────────────────────────────────────────────────
# 2. Missing / non-existent directory handling
# ──────────────────────────────────────────────────────────────────────────

class TestMissingDirectory:
    def test_nonexistent_raises(self):
        from src.pdf.loader import discover_pdfs
        with pytest.raises(FileNotFoundError):
            discover_pdfs("/does/not/exist/ever/phase5test")

    def test_file_instead_of_dir_raises(self, tmp_path: Path):
        from src.pdf.loader import discover_pdfs
        f = tmp_path / "notadir.pdf"
        f.write_bytes(b"dummy")
        with pytest.raises(NotADirectoryError):
            discover_pdfs(f)

    def test_empty_dir_returns_empty_list(self, tmp_path: Path):
        from src.pdf.loader import discover_pdfs
        found = discover_pdfs(tmp_path)
        assert found == []


# ──────────────────────────────────────────────────────────────────────────
# 3. Native text extraction
# ──────────────────────────────────────────────────────────────────────────

class TestNativeTextExtraction:
    def test_text_pdf_has_text(self, tmp_path: Path):
        from src.pdf.text_extractor import extract_document_text
        p = tmp_path / "t.pdf"
        _make_text_pdf(p, "Hello Invoice Total: 500 EUR")
        doc = extract_document_text(p)
        assert doc.load_error == ""
        assert doc.total_characters > 0
        assert doc.total_words > 0
        assert doc.pages[0].native_text_available is True

    def test_text_contains_inserted_content(self, tmp_path: Path):
        from src.pdf.text_extractor import extract_document_text
        content = "UNIQUE_STRING_XYZ_987"
        p = tmp_path / "t.pdf"
        _make_text_pdf(p, content)
        doc = extract_document_text(p)
        assert content in doc.full_text

    def test_extraction_method_is_native(self, tmp_path: Path):
        from src.pdf.text_extractor import extract_document_text
        p = tmp_path / "t.pdf"
        _make_text_pdf(p)
        doc = extract_document_text(p)
        assert doc.pages[0].extraction_method == "native"


# ──────────────────────────────────────────────────────────────────────────
# 4. Image-only page detection
# ──────────────────────────────────────────────────────────────────────────

class TestImageOnlyDetection:
    def test_image_page_has_no_native_text(self, tmp_path: Path):
        from src.pdf.text_extractor import extract_document_text
        p = tmp_path / "img.pdf"
        _make_image_pdf(p)
        doc = extract_document_text(p)
        assert doc.load_error == ""
        # PyMuPDF drawn rectangles produce no text — character count < threshold
        assert doc.pages[0].character_count < 20
        assert doc.pages[0].native_text_available is False

    def test_image_page_extraction_method_is_none(self, tmp_path: Path):
        from src.pdf.text_extractor import extract_document_text
        p = tmp_path / "img.pdf"
        _make_image_pdf(p)
        doc = extract_document_text(p)
        assert doc.pages[0].extraction_method == "none"

    def test_text_coverage_zero_for_image_pdf(self, tmp_path: Path):
        from src.pdf.text_extractor import extract_document_text
        p = tmp_path / "img.pdf"
        _make_image_pdf(p)
        doc = extract_document_text(p)
        assert doc.text_coverage == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 5. Page count accuracy
# ──────────────────────────────────────────────────────────────────────────

class TestPageCount:
    @pytest.mark.parametrize("n_pages", [1, 2, 5, 10])
    def test_page_count_is_accurate(self, tmp_path: Path, n_pages: int):
        from src.pdf.text_extractor import extract_document_text
        p = tmp_path / f"multi_{n_pages}.pdf"
        _make_multi_page_pdf(p, n_pages=n_pages)
        doc = extract_document_text(p)
        assert doc.page_count == n_pages
        assert len(doc.pages) == n_pages

    def test_page_numbers_are_one_indexed(self, tmp_path: Path):
        from src.pdf.text_extractor import extract_document_text
        p = tmp_path / "two.pdf"
        _make_multi_page_pdf(p, n_pages=2)
        doc = extract_document_text(p)
        assert doc.pages[0].page_number == 1
        assert doc.pages[1].page_number == 2


# ──────────────────────────────────────────────────────────────────────────
# 6. Inventory generation
# ──────────────────────────────────────────────────────────────────────────

class TestInventoryGeneration:
    def test_inventory_produces_json(self, docs_dir: Path, tmp_path: Path):
        from src.inventory.inspector import run_inventory
        out = tmp_path / "out"
        run_inventory(str(docs_dir), str(out))
        assert (out / "document_inventory.json").exists()

    def test_inventory_produces_csv(self, docs_dir: Path, tmp_path: Path):
        from src.inventory.inspector import run_inventory
        out = tmp_path / "out"
        run_inventory(str(docs_dir), str(out))
        assert (out / "document_inventory.csv").exists()

    def test_inventory_produces_markdown(self, docs_dir: Path, tmp_path: Path):
        from src.inventory.inspector import run_inventory
        out = tmp_path / "out"
        run_inventory(str(docs_dir), str(out))
        assert (out / "document_inventory.md").exists()

    def test_json_has_expected_document_count(self, docs_dir: Path, tmp_path: Path):
        from src.inventory.inspector import run_inventory
        out = tmp_path / "out"
        records = run_inventory(str(docs_dir), str(out))
        assert len(records) == 4  # fixtures create 4 PDFs

    def test_json_schema_structure(self, docs_dir: Path, tmp_path: Path):
        from src.inventory.inspector import run_inventory
        out = tmp_path / "out"
        run_inventory(str(docs_dir), str(out))
        with open(out / "document_inventory.json", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "_schema" in data
        assert "summary" in data
        assert "documents" in data
        for doc in data["documents"]:
            assert "filename" in doc
            assert "file_size_bytes" in doc
            assert "page_count" in doc
            assert "text" in doc
            assert "pages" in doc
            assert "classification" in doc

    def test_page_records_have_required_fields(self, docs_dir: Path, tmp_path: Path):
        from src.inventory.inspector import run_inventory
        out = tmp_path / "out"
        records = run_inventory(str(docs_dir), str(out))
        for rec in records:
            for pg in rec.get("pages", []):
                assert "page_number" in pg
                assert "native_text_available" in pg
                assert "character_count" in pg
                assert "word_count" in pg


# ──────────────────────────────────────────────────────────────────────────
# 7. One corrupt PDF must NOT stop the run
# ──────────────────────────────────────────────────────────────────────────

class TestRobustness:
    def test_corrupt_pdf_does_not_abort(self, mixed_dir: Path, tmp_path: Path):
        """The two valid PDFs must still be processed despite the corrupt one."""
        from src.inventory.inspector import run_inventory
        out = tmp_path / "out"
        records = run_inventory(str(mixed_dir), str(out))
        # All 3 files should produce records
        assert len(records) == 3

    def test_corrupt_pdf_has_load_error(self, mixed_dir: Path, tmp_path: Path):
        from src.inventory.inspector import run_inventory
        out = tmp_path / "out"
        records = run_inventory(str(mixed_dir), str(out))
        corrupt_records = [r for r in records if r.get("load_error")]
        assert len(corrupt_records) >= 1

    def test_valid_pdfs_fully_processed_after_corrupt(self, mixed_dir: Path, tmp_path: Path):
        from src.inventory.inspector import run_inventory
        out = tmp_path / "out"
        records = run_inventory(str(mixed_dir), str(out))
        valid = [r for r in records if not r.get("load_error")]
        assert len(valid) == 2
        for r in valid:
            assert r["page_count"] > 0
