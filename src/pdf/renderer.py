"""src/pdf/renderer.py — PDF page rendering via PyMuPDF.

Responsibilities
----------------
* Render individual PDF pages to in-memory pixel maps  (RenderedPage).
* Save rendered pages to disk with deterministic filenames           (save_page).
* Render an entire PDF to a directory, page by page                 (render_pdf).
* Provide structured metadata for every rendered/saved page          (PageRenderResult).
* Expose a simple CLI:
      python -m src.pdf.renderer --documents <dir> --output <dir> [--dpi N]

Architecture
------------
The clean boundary is:

    PDF
     ↓
    PDFRenderer  (this module)
     ↓
    RenderedPage  — in-memory image bytes (DPI, dimensions, format)
     ↓
    OCR / Vision  (future phases — NOT implemented here)

Callers that only need in-memory bytes use render_page().
Callers that need disk files use save_page() or render_pdf().

Design decisions
----------------
* Default DPI = 200.  Rationale: sufficient OCR/Vision quality for
  letter/A4 pages (~1650×2125 px) while keeping per-page PNG ≈ 200–500 KB.
  300 DPI would treble the file size with marginal OCR benefit.
* PNG output: lossless, no JPEG artefacts, preferred by OCR engines.
* Page numbering is always 1-indexed, matching human expectation and
  the PDF page convention used throughout this project.
* Rotation: PyMuPDF applies the PDF's internal /Rotate matrix automatically,
  so page orientation is correct without extra handling.
* One page failure ≠ abort.  Errors are captured in PageRenderResult.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.utils.logging import configure_logging, get_logger
from src.pdf.loader import discover_pdfs

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_DPI: int = 200
DEFAULT_FORMAT: str = "PNG"

# Minimum dimension (px) for a rendered page to be considered non-empty.
# A blank page rendered at 200 DPI will have very small file size; this
# guard catches completely empty pixmaps that indicate a rendering problem.
MIN_PIXEL_DIMENSION: int = 10


# ══════════════════════════════════════════════════════════════════════════
# Public dataclasses
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RenderedPage:
    """An in-memory rendered page — the boundary object between the renderer
    and downstream OCR / Vision components.

    This dataclass is the clean abstraction that downstream callers consume.
    They should NOT need to know whether the source was a text PDF or a
    scanned image PDF.
    """
    page_number: int        # 1-indexed
    width_px: int
    height_px: int
    dpi: int
    format: str             # "PNG" | "JPEG"
    data: bytes             # raw image bytes


@dataclass
class PageRenderResult:
    """Metadata for one attempted page render (succeeded or failed).

    This is what the pipeline uses for provenance, logging, and
    downstream consumer routing.
    """
    source_file: str        # source PDF filename (basename)
    page_number: int        # 1-indexed
    image_path: str         # absolute path to the saved image ("" if failed/not saved)
    width_px: int           # 0 if failed
    height_px: int          # 0 if failed
    dpi: int
    format: str
    success: bool
    error: str              # non-empty if success=False
    render_seconds: float   # time taken to render this page

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentRenderResult:
    """Aggregated render results for one PDF."""
    source_file: str
    source_path: str
    output_dir: str
    page_count: int
    pages: list[PageRenderResult] = field(default_factory=list)

    @property
    def successful_pages(self) -> int:
        return sum(1 for p in self.pages if p.success)

    @property
    def failed_pages(self) -> int:
        return sum(1 for p in self.pages if not p.success)

    @property
    def total_bytes(self) -> int:
        total = 0
        for p in self.pages:
            if p.success and p.image_path:
                try:
                    total += Path(p.image_path).stat().st_size
                except OSError:
                    pass
        return total

    @property
    def total_render_seconds(self) -> float:
        return sum(p.render_seconds for p in self.pages)


# ══════════════════════════════════════════════════════════════════════════
# Core rendering functions
# ══════════════════════════════════════════════════════════════════════════

def render_page(
    pdf_path: str | Path,
    page_number: int,
    dpi: int = DEFAULT_DPI,
    fmt: str = DEFAULT_FORMAT,
) -> RenderedPage:
    """Render a single page of *pdf_path* to an in-memory RenderedPage.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    page_number:
        1-indexed page number.
    dpi:
        Render resolution in dots per inch.  Default: 200.
    fmt:
        Output image format: "PNG" (lossless) or "JPEG" (lossy, smaller).

    Returns
    -------
    RenderedPage
        Contains raw image bytes and dimension metadata.

    Raises
    ------
    FileNotFoundError:
        If *pdf_path* does not exist.
    ValueError:
        If *page_number* is out of range (< 1 or > page count).
    RuntimeError:
        If PyMuPDF cannot open the file (corrupt PDF, etc.).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (fitz) is required. Install with: pip install pymupdf"
        ) from exc

    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Cannot open {path.name}: {exc}") from exc

    try:
        n = len(doc)
        if page_number < 1 or page_number > n:
            raise ValueError(
                f"page_number {page_number} is out of range for "
                f"{path.name} (has {n} page(s))"
            )

        page = doc[page_number - 1]

        # PyMuPDF applies the PDF's /Rotate annotation automatically when
        # building the Matrix, so we get the page in its intended orientation.
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img_fmt = fmt.upper()
        if img_fmt == "JPEG":
            data = pix.tobytes("jpeg")
        else:
            data = pix.tobytes("png")

        log.debug(
            "Rendered %s page %d/%d at %d DPI → %dx%d px (%s, %.1f KB)",
            path.name, page_number, n, dpi,
            pix.width, pix.height, img_fmt, len(data) / 1024,
        )

        return RenderedPage(
            page_number=page_number,
            width_px=pix.width,
            height_px=pix.height,
            dpi=dpi,
            format=img_fmt,
            data=data,
        )
    finally:
        doc.close()


def save_page(
    pdf_path: str | Path,
    page_number: int,
    output_dir: str | Path,
    dpi: int = DEFAULT_DPI,
    fmt: str = DEFAULT_FORMAT,
    stem: Optional[str] = None,
) -> PageRenderResult:
    """Render one page and save it to *output_dir* with a deterministic name.

    Output filename pattern:
        <stem>/page_<NNN>.<ext>
    where NNN is zero-padded to at least 3 digits.

    Example:
        output_dir / "INV-23" / "page_004.png"

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    page_number:
        1-indexed page number to render and save.
    output_dir:
        Directory under which the <stem>/ subdirectory will be created.
    dpi:
        Render resolution.
    fmt:
        "PNG" or "JPEG".
    stem:
        Override the subdirectory name.  Defaults to the PDF filename stem
        (e.g. "INV-23" for "INV-23.pdf").

    Returns
    -------
    PageRenderResult
        Success or failure metadata.
    """
    path = Path(pdf_path).resolve()
    doc_stem = stem if stem is not None else path.stem
    out_root = Path(output_dir).resolve()

    page_dir = out_root / doc_stem
    page_dir.mkdir(parents=True, exist_ok=True)

    ext = "jpg" if fmt.upper() == "JPEG" else "png"
    image_name = f"page_{page_number:03d}.{ext}"
    image_path = page_dir / image_name

    t0 = time.perf_counter()
    try:
        rp = render_page(pdf_path=path, page_number=page_number, dpi=dpi, fmt=fmt)
        elapsed = time.perf_counter() - t0

        image_path.write_bytes(rp.data)

        result = PageRenderResult(
            source_file=path.name,
            page_number=page_number,
            image_path=str(image_path),
            width_px=rp.width_px,
            height_px=rp.height_px,
            dpi=dpi,
            format=rp.format,
            success=True,
            error="",
            render_seconds=round(elapsed, 4),
        )
        log.debug(
            "Saved %s p%d → %s (%dx%d, %.1f KB)",
            path.name, page_number, image_name,
            rp.width_px, rp.height_px, len(rp.data) / 1024,
        )
        return result

    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        log.warning(
            "Failed to render %s page %d: %s",
            path.name, page_number, exc,
        )
        return PageRenderResult(
            source_file=path.name,
            page_number=page_number,
            image_path="",
            width_px=0,
            height_px=0,
            dpi=dpi,
            format=fmt.upper(),
            success=False,
            error=str(exc),
            render_seconds=round(elapsed, 4),
        )


def render_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = DEFAULT_DPI,
    fmt: str = DEFAULT_FORMAT,
    stem: Optional[str] = None,
) -> DocumentRenderResult:
    """Render all pages of *pdf_path* and save them to *output_dir*.

    A single page failure does NOT abort rendering of remaining pages.
    All results (success and failure) are captured in the returned
    DocumentRenderResult.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    output_dir:
        Root directory for rendered output.  A subdirectory named after
        the PDF stem is created automatically.
    dpi:
        Render resolution.
    fmt:
        "PNG" or "JPEG".
    stem:
        Override the output subdirectory name.

    Returns
    -------
    DocumentRenderResult
        Aggregated metadata for the entire document.
    """
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF (fitz) is required.") from exc

    path = Path(pdf_path).resolve()
    doc_stem = stem if stem is not None else path.stem

    # Determine page count without rendering
    try:
        doc = fitz.open(str(path))
        page_count = len(doc)
        doc.close()
    except Exception as exc:  # noqa: BLE001
        log.error("Cannot open %s for rendering: %s", path.name, exc)
        return DocumentRenderResult(
            source_file=path.name,
            source_path=str(path),
            output_dir=str(Path(output_dir).resolve() / doc_stem),
            page_count=0,
            pages=[
                PageRenderResult(
                    source_file=path.name,
                    page_number=0,
                    image_path="",
                    width_px=0,
                    height_px=0,
                    dpi=dpi,
                    format=fmt.upper(),
                    success=False,
                    error=str(exc),
                    render_seconds=0.0,
                )
            ],
        )

    log.info("Rendering %s: %d page(s) at %d DPI", path.name, page_count, dpi)

    page_results: list[PageRenderResult] = []
    for pg in range(1, page_count + 1):
        result = save_page(
            pdf_path=path,
            page_number=pg,
            output_dir=output_dir,
            dpi=dpi,
            fmt=fmt,
            stem=doc_stem,
        )
        page_results.append(result)

    doc_result = DocumentRenderResult(
        source_file=path.name,
        source_path=str(path),
        output_dir=str(Path(output_dir).resolve() / doc_stem),
        page_count=page_count,
        pages=page_results,
    )

    log.info(
        "%s done: %d/%d pages OK, %.2fs total",
        path.name,
        doc_result.successful_pages,
        page_count,
        doc_result.total_render_seconds,
    )
    return doc_result


# ══════════════════════════════════════════════════════════════════════════
# Dataset-level rendering pipeline
# ══════════════════════════════════════════════════════════════════════════

def render_dataset(
    documents_dir: str | Path,
    output_dir: str | Path,
    dpi: int = DEFAULT_DPI,
    fmt: str = DEFAULT_FORMAT,
) -> list[DocumentRenderResult]:
    """Render every PDF in *documents_dir* to *output_dir*.

    One corrupt / unreadable PDF does NOT abort the run.

    Returns
    -------
    list[DocumentRenderResult]
        One entry per discovered PDF.
    """
    pdf_files = discover_pdfs(documents_dir)

    if not pdf_files:
        log.warning("No PDF files found in %s", documents_dir)
        return []

    total = len(pdf_files)
    log.info(
        "=== Phase 6A — Dataset Rendering ===\n"
        "  documents: %s\n  output:    %s\n  dpi: %d  format: %s",
        Path(documents_dir).resolve(), Path(output_dir).resolve(), dpi, fmt,
    )

    results: list[DocumentRenderResult] = []
    t_start = time.perf_counter()

    for i, pdf_file in enumerate(pdf_files, 1):
        log.info("[%d/%d] %s", i, total, pdf_file.filename)
        try:
            doc_result = render_pdf(
                pdf_path=pdf_file.path,
                output_dir=output_dir,
                dpi=dpi,
                fmt=fmt,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Unexpected error rendering %s: %s", pdf_file.filename, exc)
            log.debug(traceback.format_exc())
            doc_result = DocumentRenderResult(
                source_file=pdf_file.filename,
                source_path=str(pdf_file.path),
                output_dir="",
                page_count=0,
                pages=[
                    PageRenderResult(
                        source_file=pdf_file.filename,
                        page_number=0,
                        image_path="",
                        width_px=0, height_px=0,
                        dpi=dpi, format=fmt.upper(),
                        success=False,
                        error=str(exc),
                        render_seconds=0.0,
                    )
                ],
            )
        results.append(doc_result)

    wall_time = time.perf_counter() - t_start
    total_pages_attempted = sum(r.page_count for r in results)
    total_ok = sum(r.successful_pages for r in results)
    total_fail = sum(r.failed_pages for r in results)
    total_bytes = sum(r.total_bytes for r in results)

    log.info(
        "=== Rendering complete ===\n"
        "  PDFs processed : %d\n"
        "  Pages attempted: %d\n"
        "  Successful     : %d\n"
        "  Failed         : %d\n"
        "  Total output   : %.1f MB\n"
        "  Wall time      : %.2f s\n"
        "  Avg/page       : %.3f s",
        total,
        total_pages_attempted,
        total_ok,
        total_fail,
        total_bytes / 1_048_576,
        wall_time,
        wall_time / max(total_pages_attempted, 1),
    )

    # Write a machine-readable summary alongside the rendered images
    _write_render_manifest(results, output_dir, wall_time, dpi, fmt)

    return results


def _write_render_manifest(
    results: list[DocumentRenderResult],
    output_dir: str | Path,
    wall_seconds: float,
    dpi: int,
    fmt: str,
) -> None:
    """Write render_manifest.json summarising the dataset render run."""
    from datetime import datetime, timezone

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    total_pages = sum(r.page_count for r in results)
    total_ok = sum(r.successful_pages for r in results)
    total_fail = sum(r.failed_pages for r in results)
    total_bytes = sum(r.total_bytes for r in results)

    # Per-document summary rows (no raw image bytes — just metadata)
    doc_summaries = []
    for r in results:
        doc_summaries.append({
            "source_file": r.source_file,
            "page_count": r.page_count,
            "successful_pages": r.successful_pages,
            "failed_pages": r.failed_pages,
            "output_dir": r.output_dir,
            "total_render_seconds": round(r.total_render_seconds, 4),
            "total_bytes": r.total_bytes,
            "pages": [p.as_dict() for p in r.pages],
        })

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dpi": dpi,
        "format": fmt,
        "summary": {
            "total_pdfs": len(results),
            "total_pages_attempted": total_pages,
            "total_pages_ok": total_ok,
            "total_pages_failed": total_fail,
            "total_output_bytes": total_bytes,
            "total_output_mb": round(total_bytes / 1_048_576, 2),
            "wall_seconds": round(wall_seconds, 3),
            "avg_seconds_per_page": round(
                wall_seconds / max(total_pages, 1), 4
            ),
        },
        "documents": doc_summaries,
    }

    manifest_path = out / "render_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    log.info("Render manifest written: %s", manifest_path)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pdf.renderer",
        description=(
            "Phase 6A — PDF Page Renderer\n"
            "Renders every page of every PDF under --documents to --output as PNG images.\n"
            "Does NOT perform OCR or Vision processing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--documents",
        default="documents",
        metavar="DIR",
        help="Root directory containing PDF files (default: documents/)",
    )
    parser.add_argument(
        "--output",
        default="artifacts/rendered_pages",
        metavar="DIR",
        help="Root output directory for rendered pages (default: artifacts/rendered_pages/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        metavar="N",
        help=f"Render resolution in DPI (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--format",
        choices=["PNG", "JPEG"],
        default=DEFAULT_FORMAT,
        dest="fmt",
        help=f"Output image format (default: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="FILE",
        help="Optional log file path.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(level=level, log_file=args.log_file)

    render_dataset(
        documents_dir=args.documents,
        output_dir=args.output,
        dpi=args.dpi,
        fmt=args.fmt,
    )


if __name__ == "__main__":
    main()
