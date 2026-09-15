"""analysis/spot_check_rapidocr.py — Checkpoint evaluation of RapidOCR on representative pages."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

from src.extraction.ocr import RapidOCRProvider, save_page_ocr_json

PAGES_TO_TEST = [
    ("INV-01", "artifacts/rendered_pages/INV-01/page_001.png", "German invoice (clean table, umlauts)"),
    ("HLD-01", "artifacts/rendered_pages/HLD-01/page_001.png", "Thai invoice (Thai + English, tax)"),
    ("DU-02", "artifacts/rendered_pages/DU-02/page_001.png", "Dense multilingual invoice (complex table)"),
    ("HLD-03", "artifacts/rendered_pages/HLD-03/page_001.png", "Landscape Portuguese invoice"),
]

def main() -> None:
    print("=" * 80)
    print("PHASE 6B — RAPIDOCR CHECKPOINT EVALUATION")
    print("=" * 80)

    provider = RapidOCRProvider()

    for stem, img_path, description in PAGES_TO_TEST:
        print(f"\n[{stem}] — {description}")
        print(f"Image: {img_path}")
        
        t0 = time.perf_counter()
        result = provider.ocr_page(img_path)
        elapsed = time.perf_counter() - t0

        print(f"Status: {result.status}")
        print(f"Runtime: {result.runtime_seconds:.3f}s (total call: {elapsed:.3f}s)")
        print(f"Image Dimensions: {result.image_width}x{result.image_height} px")
        print(f"Blocks Detected: {len(result.blocks)}")
        print(f"Confidence — Mean: {result.confidence_mean:.4f}, Min: {result.confidence_min:.4f}, Max: {result.confidence_max:.4f}")

        # Save artifact
        out_path = save_page_ocr_json(result, "artifacts/ocr")
        print(f"Saved artifact: {out_path}")

        # Display first 8 blocks
        print("Sample Recognized Blocks:")
        for idx, block in enumerate(result.blocks[:8], 1):
            print(f"  {idx:02d}. [conf={block.confidence:.3f}, bbox={block.bbox}] {block.text}")

        # Display text snippet
        lines = [b.text for b in result.blocks]
        print(f"\nFull Text Lines Count: {len(lines)}")
        print("First 15 lines of raw text:")
        for line in lines[:15]:
            print(f"   | {line}")

        # Check for specific language characters
        text = result.text
        has_umlauts = any(c in text for c in "äöüÄÖÜß")
        has_pt_accents = any(c in text for c in "ãõçéêíóúáà")
        has_thai = any('\u0e00' <= c <= '\u0e7f' for c in text)

        print(f"Language Check -> Umlauts: {has_umlauts}, Portuguese Accents: {has_pt_accents}, Thai Script: {has_thai}")
        print("-" * 80)

if __name__ == "__main__":
    main()
