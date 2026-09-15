"""analysis/investigate_multilingual_models.py — Phase 6B Checkpoint 2 Investigation.

Compares RapidOCR default model against dedicated PP-OCRv5 Latin and Thai models on:
1. HLD-01/page_001.png (Thai vs Default baseline + mismatch)
2. INV-01/page_001.png (Latin vs Default baseline + mismatch)
3. HLD-03/page_001.png (Latin vs Default baseline for Portuguese accents)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 console output
sys.stdout.reconfigure(encoding="utf-8")

from src.extraction.ocr import RapidOCRProvider


def run_page_eval(
    provider: RapidOCRProvider,
    image_path: str,
    language: str,
    desc: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    res = provider.ocr_page(image_path, language=language)
    elapsed = time.perf_counter() - t0

    thai_chars = sum(1 for c in res.text if "\u0e00" <= c <= "\u0e7f")
    german_umlauts = sum(1 for c in res.text if c in "äöüÄÖÜß")
    pt_accents = sum(1 for c in res.text if c in "ãõçéêíóúáà")
    currency_symbols = sum(1 for c in res.text if c in "€$£฿")

    return {
        "description": desc,
        "image_path": image_path,
        "language_config": language,
        "model_used": res.model,
        "status": res.status,
        "runtime_seconds": round(res.runtime_seconds, 3),
        "total_elapsed_seconds": round(elapsed, 3),
        "blocks_count": len(res.blocks),
        "confidence_mean": round(res.confidence_mean, 4),
        "confidence_min": round(res.confidence_min, 4),
        "confidence_max": round(res.confidence_max, 4),
        "metrics": {
            "thai_characters_detected": thai_chars,
            "german_umlauts_detected": german_umlauts,
            "portuguese_accents_detected": pt_accents,
            "currency_symbols_detected": currency_symbols,
        },
        "sample_lines": [b.text for b in res.blocks[:15]],
        "raw_text": res.text,
    }


def main() -> None:
    print("=" * 80)
    print("PHASE 6B — CHECKPOINT 2: RAPIDOCR MULTILINGUAL MODEL INVESTIGATION")
    print("=" * 80)

    provider = RapidOCRProvider()
    experiments: list[dict[str, Any]] = []

    # 1. HLD-01: Default vs Thai vs Latin Mismatch
    print("\n[TEST 1] HLD-01 (Thai + English Invoice)")
    print("--------------------------------------------------------------------------------")
    
    print("Running HLD-01 with DEFAULT model...")
    exp_hld_def = run_page_eval(
        provider,
        "artifacts/rendered_pages/HLD-01/page_001.png",
        language="default",
        desc="HLD-01 (Default bundled PP-OCRv4 model)",
    )
    experiments.append(exp_hld_def)
    print(f"  Default: {exp_hld_def['blocks_count']} blocks, conf={exp_hld_def['confidence_mean']:.4f}, Thai chars={exp_hld_def['metrics']['thai_characters_detected']}")

    print("Running HLD-01 with THAI PP-OCRv5 model...")
    exp_hld_thai = run_page_eval(
        provider,
        "artifacts/rendered_pages/HLD-01/page_001.png",
        language="thai",
        desc="HLD-01 (Dedicated Thai PP-OCRv5 model)",
    )
    experiments.append(exp_hld_thai)
    print(f"  Thai model: {exp_hld_thai['blocks_count']} blocks, conf={exp_hld_thai['confidence_mean']:.4f}, Thai chars={exp_hld_thai['metrics']['thai_characters_detected']}")

    print("Running HLD-01 with LATIN model (mismatch experiment)...")
    exp_hld_lat = run_page_eval(
        provider,
        "artifacts/rendered_pages/HLD-01/page_001.png",
        language="latin",
        desc="HLD-01 (Latin model mismatch)",
    )
    experiments.append(exp_hld_lat)
    print(f"  Latin mismatch: {exp_hld_lat['blocks_count']} blocks, conf={exp_hld_lat['confidence_mean']:.4f}, Thai chars={exp_hld_lat['metrics']['thai_characters_detected']}")

    # 2. INV-01: Default vs Latin vs Thai Mismatch
    print("\n[TEST 2] INV-01 (German Invoice)")
    print("--------------------------------------------------------------------------------")

    print("Running INV-01 with DEFAULT model...")
    exp_inv_def = run_page_eval(
        provider,
        "artifacts/rendered_pages/INV-01/page_001.png",
        language="default",
        desc="INV-01 (Default bundled PP-OCRv4 model)",
    )
    experiments.append(exp_inv_def)
    print(f"  Default: {exp_inv_def['blocks_count']} blocks, conf={exp_inv_def['confidence_mean']:.4f}, Umlauts={exp_inv_def['metrics']['german_umlauts_detected']}")

    print("Running INV-01 with LATIN PP-OCRv5 model...")
    exp_inv_lat = run_page_eval(
        provider,
        "artifacts/rendered_pages/INV-01/page_001.png",
        language="latin",
        desc="INV-01 (Dedicated Latin PP-OCRv5 model)",
    )
    experiments.append(exp_inv_lat)
    print(f"  Latin model: {exp_inv_lat['blocks_count']} blocks, conf={exp_inv_lat['confidence_mean']:.4f}, Umlauts={exp_inv_lat['metrics']['german_umlauts_detected']}")

    print("Running INV-01 with THAI model (mismatch experiment)...")
    exp_inv_thai = run_page_eval(
        provider,
        "artifacts/rendered_pages/INV-01/page_001.png",
        language="thai",
        desc="INV-01 (Thai model mismatch)",
    )
    experiments.append(exp_inv_thai)
    print(f"  Thai mismatch: {exp_inv_thai['blocks_count']} blocks, conf={exp_inv_thai['confidence_mean']:.4f}, Umlauts={exp_inv_thai['metrics']['german_umlauts_detected']}")

    # 3. HLD-03: Default vs Latin for Portuguese Accents
    print("\n[TEST 3] HLD-03 (Portuguese Landscape Invoice)")
    print("--------------------------------------------------------------------------------")

    print("Running HLD-03 with DEFAULT model...")
    exp_hld03_def = run_page_eval(
        provider,
        "artifacts/rendered_pages/HLD-03/page_001.png",
        language="default",
        desc="HLD-03 (Default bundled PP-OCRv4 model)",
    )
    experiments.append(exp_hld03_def)
    print(f"  Default: {exp_hld03_def['blocks_count']} blocks, conf={exp_hld03_def['confidence_mean']:.4f}, PT Accents={exp_hld03_def['metrics']['portuguese_accents_detected']}")

    print("Running HLD-03 with LATIN PP-OCRv5 model...")
    exp_hld03_lat = run_page_eval(
        provider,
        "artifacts/rendered_pages/HLD-03/page_001.png",
        language="latin",
        desc="HLD-03 (Dedicated Latin PP-OCRv5 model)",
    )
    experiments.append(exp_hld03_lat)
    print(f"  Latin model: {exp_hld03_lat['blocks_count']} blocks, conf={exp_hld03_lat['confidence_mean']:.4f}, PT Accents={exp_hld03_lat['metrics']['portuguese_accents_detected']}")

    # Save to JSON
    out_file = Path("analysis") / "multilingual_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False)
    print(f"\nAll benchmark results saved to: {out_file}")

    # Print Detailed Side-by-Side Excerpts
    print("\n" + "=" * 80)
    print("DETAILED SIDE-BY-SIDE EXCERPTS")
    print("=" * 80)

    print("\n--- HLD-01: Default Model vs Dedicated Thai Model ---")
    print("DEFAULT:")
    for l in exp_hld_def["sample_lines"][:8]:
        print(f"  {l}")
    print("\nTHAI MODEL:")
    for l in exp_hld_thai["sample_lines"][:8]:
        print(f"  {l}")

    print("\n--- INV-01: Default Model vs Dedicated Latin Model ---")
    print("DEFAULT:")
    for l in [b for b in exp_inv_def["sample_lines"] if any(w in b for w in ["Northwind", "Projekt", "Rechnung", "€", "standort"] or "852566" in b)][:6]:
        print(f"  {l}")
    print("\nLATIN MODEL:")
    for l in [b for b in exp_inv_lat["sample_lines"] if any(w in b for w in ["Northwind", "Projekt", "Rechnung", "€", "Standort"] or "852566" in b)][:6]:
        print(f"  {l}")

    print("\n--- HLD-03: Default Model vs Dedicated Latin Model (Portuguese) ---")
    print("DEFAULT:")
    for l in exp_hld03_def["sample_lines"][:6]:
        print(f"  {l}")
    print("\nLATIN MODEL:")
    for l in exp_hld03_lat["sample_lines"][:6]:
        print(f"  {l}")


if __name__ == "__main__":
    main()
