"""analysis/compare_dpi.py — 200 DPI vs 300 DPI experiment on dense page INV-23."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import fitz  # PyMuPDF
from src.extraction.ocr import RapidOCRProvider

def render_page_at_dpi(pdf_path: str, page_number: int, dpi: int, output_path: str) -> tuple[int, int]:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(output_path)
    w, h = pix.width, pix.height
    doc.close()
    return (w, h)

def main() -> None:
    pdf_path = "documents/INV-23.pdf"
    p200 = "artifacts/rendered_pages/INV-23/page_001.png"
    p300 = "artifacts/scratch/INV-23_page_001_300dpi.png"
    Path(p300).parent.mkdir(parents=True, exist_ok=True)

    print("Rendering INV-23 page 1 at 300 DPI...")
    w300, h300 = render_page_at_dpi(pdf_path, 1, 300, p300)
    print(f"300 DPI image: {w300}x{h300} px (file: {p300})")

    provider = RapidOCRProvider()

    # OCR 200 DPI
    t0 = time.perf_counter()
    r200 = provider.ocr_page(p200, language="latin")
    el200 = time.perf_counter() - t0

    # OCR 300 DPI
    t0 = time.perf_counter()
    r300 = provider.ocr_page(p300, language="latin")
    el300 = time.perf_counter() - t0

    comparison = {
        "page": "INV-23 page 1",
        "200_dpi": {
            "image_dimensions": f"{r200.image_width}x{r200.image_height}",
            "file_size_kb": round(Path(p200).stat().st_size / 1024, 1),
            "runtime_seconds": round(r200.runtime_seconds, 3),
            "blocks_count": len(r200.blocks),
            "confidence_mean": round(r200.confidence_mean, 4),
            "confidence_min": round(r200.confidence_min, 4),
            "confidence_max": round(r200.confidence_max, 4),
            "text_length": len(r200.text),
        },
        "300_dpi": {
            "image_dimensions": f"{w300}x{h300}",
            "file_size_kb": round(Path(p300).stat().st_size / 1024, 1),
            "runtime_seconds": round(r300.runtime_seconds, 3),
            "blocks_count": len(r300.blocks),
            "confidence_mean": round(r300.confidence_mean, 4),
            "confidence_min": round(r300.confidence_min, 4),
            "confidence_max": round(r300.confidence_max, 4),
            "text_length": len(r300.text),
        }
    }

    print("\nComparison Results:")
    print(json.dumps(comparison, indent=2))

    # Clean up scratch 300dpi image
    out_file = Path("analysis") / "dpi_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved comparison to {out_file}")

if __name__ == "__main__":
    main()
