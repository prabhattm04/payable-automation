"""tests/test_renderer.py — Phase 6A renderer tests.

All 10 required test categories:
1.  Single-page PDF rendering.
2.  Multi-page PDF rendering.
3.  Correct page count.
4.  Correct page ordering.
5.  Output image exists on disk.
6.  Output image dimensions are valid.
7.  Configurable DPI.
8.  Invalid PDF handling.
9.  Missing file handling.
10. Rendering one page without rendering the entire PDF.

Run with:
    pytest tests/test_renderer.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────
# Shared PDF fixture helpers (reused from test_inventory.py pattern)
# ──────────────────────────────────────────────────────────────────────────

def _make_text_pdf(path: Path, n_pages: int = 1, text: str = "Test Invoice 12345") -> None:
    """Create a minimal PDF with *n_pages* text pages using PyMuPDF."""
    import fitz
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=595, height=842)   # A4 portrait
        page.insert_text((50, 100), f"Page {i+1}\n{text}")
    doc.save(str(path))
    doc.close()


def _make_corrupt_file(path: Path) -> None:
    """Write bytes that are not a valid PDF."""
    path.write_bytes(b"%%NOT_A_REAL_PDF_FILE%%")


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def single_page_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "single.pdf"
    _make_text_pdf(p, n_pages=1)
    return p


@pytest.fixture()
def multi_page_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "multi.pdf"
    _make_text_pdf(p, n_pages=5)
    return p


@pytest.fixture()
def corrupt_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "corrupt.pdf"
    _make_corrupt_file(p)
    return p


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "rendered"
    out.mkdir()
    return out


# ══════════════════════════════════════════════════════════════════════════
# 1. Single-page PDF rendering
# ══════════════════════════════════════════════════════════════════════════

class TestSinglePageRendering:
    def test_render_page_returns_rendered_page(self, single_page_pdf: Path):
        from src.pdf.renderer import render_page, RenderedPage
        rp = render_page(single_page_pdf, page_number=1)
        assert isinstance(rp, RenderedPage)

    def test_rendered_page_has_bytes(self, single_page_pdf: Path):
        from src.pdf.renderer import render_page
        rp = render_page(single_page_pdf, page_number=1)
        assert len(rp.data) > 0

    def test_rendered_page_format_is_png(self, single_page_pdf: Path):
        from src.pdf.renderer import render_page
        rp = render_page(single_page_pdf, page_number=1)
        assert rp.format == "PNG"
        # Verify PNG magic bytes
        assert rp.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_save_page_creates_file(self, single_page_pdf: Path, output_dir: Path):
        from src.pdf.renderer import save_page
        result = save_page(single_page_pdf, page_number=1, output_dir=output_dir)
        assert result.success is True
        assert Path(result.image_path).exists()

    def test_save_page_deterministic_name(self, single_page_pdf: Path, output_dir: Path):
        from src.pdf.renderer import save_page
        result = save_page(single_page_pdf, page_number=1, output_dir=output_dir)
        name = Path(result.image_path).name
        assert name == "page_001.png"


# ══════════════════════════════════════════════════════════════════════════
# 2. Multi-page PDF rendering
# ══════════════════════════════════════════════════════════════════════════

class TestMultiPageRendering:
    def test_render_pdf_produces_correct_file_count(
        self, multi_page_pdf: Path, output_dir: Path
    ):
        from src.pdf.renderer import render_pdf
        result = render_pdf(multi_page_pdf, output_dir=output_dir)
        assert result.successful_pages == 5
        assert result.failed_pages == 0

    def test_render_pdf_all_images_exist(
        self, multi_page_pdf: Path, output_dir: Path
    ):
        from src.pdf.renderer import render_pdf
        result = render_pdf(multi_page_pdf, output_dir=output_dir)
        for page_result in result.pages:
            assert page_result.success is True
            assert Path(page_result.image_path).exists()


# ══════════════════════════════════════════════════════════════════════════
# 3. Correct page count
# ══════════════════════════════════════════════════════════════════════════

class TestPageCount:
    @pytest.mark.parametrize("n", [1, 2, 4, 7])
    def test_page_count_matches_pdf(self, tmp_path: Path, output_dir: Path, n: int):
        from src.pdf.renderer import render_pdf
        p = tmp_path / f"doc_{n}.pdf"
        _make_text_pdf(p, n_pages=n)
        result = render_pdf(p, output_dir=output_dir)
        assert result.page_count == n
        assert len(result.pages) == n

    def test_document_render_result_page_count(
        self, multi_page_pdf: Path, output_dir: Path
    ):
        from src.pdf.renderer import render_pdf
        result = render_pdf(multi_page_pdf, output_dir=output_dir)
        assert result.page_count == 5
        assert result.successful_pages == 5


# ══════════════════════════════════════════════════════════════════════════
# 4. Correct page ordering
# ══════════════════════════════════════════════════════════════════════════

class TestPageOrdering:
    def test_pages_are_one_indexed(
        self, multi_page_pdf: Path, output_dir: Path
    ):
        from src.pdf.renderer import render_pdf
        result = render_pdf(multi_page_pdf, output_dir=output_dir)
        page_nums = [p.page_number for p in result.pages]
        assert page_nums == list(range(1, 6))

    def test_filenames_are_ordered(
        self, multi_page_pdf: Path, output_dir: Path
    ):
        from src.pdf.renderer import render_pdf
        result = render_pdf(multi_page_pdf, output_dir=output_dir)
        names = sorted(Path(p.image_path).name for p in result.pages if p.success)
        assert names == [f"page_{i:03d}.png" for i in range(1, 6)]

    def test_page_number_in_result_matches_filename(
        self, multi_page_pdf: Path, output_dir: Path
    ):
        from src.pdf.renderer import render_pdf
        result = render_pdf(multi_page_pdf, output_dir=output_dir)
        for pr in result.pages:
            expected_name = f"page_{pr.page_number:03d}.png"
            assert Path(pr.image_path).name == expected_name


# ══════════════════════════════════════════════════════════════════════════
# 5. Output image exists on disk
# ══════════════════════════════════════════════════════════════════════════

class TestOutputExists:
    def test_output_file_exists(self, single_page_pdf: Path, output_dir: Path):
        from src.pdf.renderer import save_page
        result = save_page(single_page_pdf, page_number=1, output_dir=output_dir)
        assert Path(result.image_path).exists()
        assert Path(result.image_path).is_file()

    def test_output_directory_created(self, single_page_pdf: Path, tmp_path: Path):
        from src.pdf.renderer import save_page
        # Use a deeply nested output dir that doesn't exist yet
        nested = tmp_path / "a" / "b" / "c"
        result = save_page(single_page_pdf, page_number=1, output_dir=nested)
        assert result.success is True
        assert Path(result.image_path).exists()

    def test_stem_subdirectory_created(self, single_page_pdf: Path, output_dir: Path):
        from src.pdf.renderer import save_page
        result = save_page(single_page_pdf, page_number=1, output_dir=output_dir)
        # The image should be inside a subdirectory named after the PDF stem
        assert Path(result.image_path).parent.name == single_page_pdf.stem


# ══════════════════════════════════════════════════════════════════════════
# 6. Output image dimensions are valid
# ══════════════════════════════════════════════════════════════════════════

class TestImageDimensions:
    def test_dimensions_positive(self, single_page_pdf: Path):
        from src.pdf.renderer import render_page
        rp = render_page(single_page_pdf, page_number=1)
        assert rp.width_px > 0
        assert rp.height_px > 0

    def test_dimensions_reasonable_at_200dpi(self, single_page_pdf: Path):
        """A4 at 200 DPI ≈ 1654 × 2339 px; letter at 200 DPI ≈ 1700 × 2200 px.
        Accept any dimension > 100 px to stay fixture-agnostic."""
        from src.pdf.renderer import render_page
        rp = render_page(single_page_pdf, page_number=1, dpi=200)
        assert rp.width_px > 100
        assert rp.height_px > 100

    def test_page_render_result_has_dimensions(
        self, single_page_pdf: Path, output_dir: Path
    ):
        from src.pdf.renderer import save_page
        result = save_page(single_page_pdf, page_number=1, output_dir=output_dir)
        assert result.width_px > 0
        assert result.height_px > 0


# ══════════════════════════════════════════════════════════════════════════
# 7. Configurable DPI
# ══════════════════════════════════════════════════════════════════════════

class TestConfigurableDPI:
    def test_higher_dpi_produces_larger_image(self, single_page_pdf: Path):
        from src.pdf.renderer import render_page
        rp_low = render_page(single_page_pdf, page_number=1, dpi=72)
        rp_high = render_page(single_page_pdf, page_number=1, dpi=300)
        # Pixel dimensions must scale proportionally
        assert rp_high.width_px > rp_low.width_px
        assert rp_high.height_px > rp_low.height_px

    def test_dpi_recorded_in_result(self, single_page_pdf: Path):
        from src.pdf.renderer import render_page
        rp = render_page(single_page_pdf, page_number=1, dpi=150)
        assert rp.dpi == 150

    def test_dpi_recorded_in_page_render_result(
        self, single_page_pdf: Path, output_dir: Path
    ):
        from src.pdf.renderer import save_page
        result = save_page(single_page_pdf, page_number=1, output_dir=output_dir, dpi=300)
        assert result.dpi == 300

    def test_pixel_dimensions_scale_with_dpi(self, single_page_pdf: Path):
        """Width at 2× DPI should be approximately 2× the pixels."""
        from src.pdf.renderer import render_page
        rp_100 = render_page(single_page_pdf, page_number=1, dpi=100)
        rp_200 = render_page(single_page_pdf, page_number=1, dpi=200)
        ratio = rp_200.width_px / rp_100.width_px
        assert 1.9 <= ratio <= 2.1, f"Expected ~2.0 ratio, got {ratio:.2f}"


# ══════════════════════════════════════════════════════════════════════════
# 8. Invalid PDF handling
# ══════════════════════════════════════════════════════════════════════════

class TestInvalidPdfHandling:
    def test_corrupt_pdf_save_page_returns_failure(
        self, corrupt_pdf: Path, output_dir: Path
    ):
        from src.pdf.renderer import save_page
        result = save_page(corrupt_pdf, page_number=1, output_dir=output_dir)
        assert result.success is False
        assert result.error != ""

    def test_corrupt_pdf_render_pdf_returns_failure_not_exception(
        self, corrupt_pdf: Path, output_dir: Path
    ):
        from src.pdf.renderer import render_pdf
        # Must not raise — should return a result with failed_pages > 0
        result = render_pdf(corrupt_pdf, output_dir=output_dir)
        assert result.failed_pages > 0

    def test_corrupt_pdf_does_not_abort_dataset(
        self, corrupt_pdf: Path, output_dir: Path, single_page_pdf: Path, tmp_path: Path
    ):
        from src.pdf.renderer import render_dataset
        # Mix corrupt + valid in a temp docs dir
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        import shutil
        shutil.copy(corrupt_pdf, docs_dir / "corrupt.pdf")
        shutil.copy(single_page_pdf, docs_dir / "valid.pdf")

        results = render_dataset(docs_dir, output_dir)
        assert len(results) == 2
        # The valid PDF must succeed
        valid = next(r for r in results if r.source_file == "valid.pdf")
        assert valid.successful_pages == 1

    def test_out_of_range_page_raises_value_error(self, single_page_pdf: Path):
        from src.pdf.renderer import render_page
        with pytest.raises(ValueError):
            render_page(single_page_pdf, page_number=99)

    def test_zero_page_number_raises_value_error(self, single_page_pdf: Path):
        from src.pdf.renderer import render_page
        with pytest.raises(ValueError):
            render_page(single_page_pdf, page_number=0)


# ══════════════════════════════════════════════════════════════════════════
# 9. Missing file handling
# ══════════════════════════════════════════════════════════════════════════

class TestMissingFileHandling:
    def test_missing_pdf_raises_file_not_found(self):
        from src.pdf.renderer import render_page
        with pytest.raises(FileNotFoundError):
            render_page("/nonexistent/path/to/doc.pdf", page_number=1)

    def test_missing_pdf_save_page_returns_failure(self, output_dir: Path):
        from src.pdf.renderer import save_page
        result = save_page("/nonexistent/doc.pdf", page_number=1, output_dir=output_dir)
        assert result.success is False
        assert result.error != ""

    def test_missing_dir_render_dataset_raises(self, output_dir: Path):
        from src.pdf.renderer import render_dataset
        from src.pdf.loader import discover_pdfs
        with pytest.raises(FileNotFoundError):
            render_dataset("/nonexistent/docs", output_dir)


# ══════════════════════════════════════════════════════════════════════════
# 10. Rendering one page without rendering the entire PDF
# ══════════════════════════════════════════════════════════════════════════

class TestSinglePageWithoutFullRender:
    def test_render_page_does_not_create_files(self, multi_page_pdf: Path, tmp_path: Path):
        """render_page() is in-memory only — no disk side effects."""
        from src.pdf.renderer import render_page
        before = list(tmp_path.rglob("*.png"))
        render_page(multi_page_pdf, page_number=3)
        after = list(tmp_path.rglob("*.png"))
        assert before == after  # no files created

    def test_can_render_middle_page_only(self, multi_page_pdf: Path, output_dir: Path):
        from src.pdf.renderer import save_page
        # Render only page 3 of a 5-page PDF
        result = save_page(multi_page_pdf, page_number=3, output_dir=output_dir)
        assert result.success is True
        assert result.page_number == 3
        assert Path(result.image_path).name == "page_003.png"
        # Pages 1, 2, 4, 5 must NOT exist
        stem_dir = Path(result.image_path).parent
        existing = {f.name for f in stem_dir.iterdir()}
        assert "page_001.png" not in existing
        assert "page_002.png" not in existing
        assert "page_003.png" in existing
        assert "page_004.png" not in existing

    def test_render_page_correct_number_recorded(self, multi_page_pdf: Path):
        from src.pdf.renderer import render_page
        rp = render_page(multi_page_pdf, page_number=4)
        assert rp.page_number == 4

    def test_save_page_custom_stem(
        self, single_page_pdf: Path, output_dir: Path
    ):
        from src.pdf.renderer import save_page
        result = save_page(
            single_page_pdf, page_number=1,
            output_dir=output_dir, stem="custom_stem"
        )
        assert result.success is True
        assert Path(result.image_path).parent.name == "custom_stem"


# ══════════════════════════════════════════════════════════════════════════
# Manifest tests
# ══════════════════════════════════════════════════════════════════════════

class TestRenderManifest:
    def test_manifest_written_after_dataset_render(
        self, tmp_path: Path, output_dir: Path
    ):
        from src.pdf.renderer import render_dataset
        docs = tmp_path / "docs"
        docs.mkdir()
        _make_text_pdf(docs / "doc_a.pdf", n_pages=2)
        _make_text_pdf(docs / "doc_b.pdf", n_pages=1)

        render_dataset(docs, output_dir)
        manifest_path = output_dir / "render_manifest.json"
        assert manifest_path.exists()

    def test_manifest_schema(self, tmp_path: Path, output_dir: Path):
        from src.pdf.renderer import render_dataset
        docs = tmp_path / "docs"
        docs.mkdir()
        _make_text_pdf(docs / "doc_a.pdf", n_pages=2)

        render_dataset(docs, output_dir)
        with open(output_dir / "render_manifest.json", encoding="utf-8") as fh:
            manifest = json.load(fh)

        assert "summary" in manifest
        assert "documents" in manifest
        s = manifest["summary"]
        assert s["total_pdfs"] == 1
        assert s["total_pages_attempted"] == 2
        assert s["total_pages_ok"] == 2
        assert s["total_pages_failed"] == 0
