"""analysis/phase6c/run_benchmark.py — Phase 6C-2 Controlled Vision Benchmark Runner.

Executes the standardized visual evidence benchmark on the 5 designated pages:
1. INV-01 page 1 (artifacts/rendered_pages/INV-01/page_001.png)
2. HLD-01 page 1 (artifacts/rendered_pages/HLD-01/page_001.png)
3. DU-02 page 1 (artifacts/rendered_pages/DU-02/page_001.png)
4. DU-02 page 5 (artifacts/rendered_pages/DU-02/page_005.png)
5. HLD-03 page 1 (artifacts/rendered_pages/HLD-03/page_001.png) — complex tax/discount page

Produces:
analysis/phase6c/benchmark/<stem>_p<N>.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.puter_qwen import PuterQwenProvider

BENCHMARK_DIR = PROJECT_ROOT / "analysis" / "phase6c" / "benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_VERSION = "phase6c-v1"
BENCHMARK_PROMPT = """Analyze this document page as visual evidence.

Determine:

1. What type of document/page this appears to be.
2. Whether the page appears to be part of a payable/invoice document,
   supporting documentation, or non-payable material.
3. Identify the major sections and their spatial relationships.
4. Describe any tables, including their columns, rows, and how values
   align with labels.
5. Identify monetary amounts and explain which labels/sections they
   appear to belong to.
6. Identify taxes, discounts, charges, and totals if visibly present,
   and describe their relationship to the surrounding values.
7. If this appears to be a continuation/supporting page, explain what
   evidence suggests that.
8. Identify any invoice/order/reference numbers and where they appear.

Return observations grounded only in visible evidence.

Do not invent missing values.
Do not calculate values that are not explicitly needed to describe
the visible structure.
Do not assume a document is payable merely because it contains
currency amounts.
Do not produce final accounting JSON."""

BENCHMARK_PAGES = [
    {
        "document": "INV-01",
        "page": 1,
        "image_path": "artifacts/rendered_pages/INV-01/page_001.png",
        "output_filename": "INV-01_p1.json",
        "rationale": "Single-page German commercial invoice baseline",
    },
    {
        "document": "HLD-01",
        "page": 1,
        "image_path": "artifacts/rendered_pages/HLD-01/page_001.png",
        "output_filename": "HLD-01_p1.json",
        "rationale": "Bilingual Thai/English service invoice with withholding tax",
    },
    {
        "document": "DU-02",
        "page": 1,
        "image_path": "artifacts/rendered_pages/DU-02/page_001.png",
        "output_filename": "DU-02_p1.json",
        "rationale": "Dense multi-column customs consolidated invoice",
    },
    {
        "document": "DU-02",
        "page": 5,
        "image_path": "artifacts/rendered_pages/DU-02/page_005.png",
        "output_filename": "DU-02_p5.json",
        "rationale": "Sparse continuation/shipping packing list page with whitespace",
    },
    {
        "document": "HLD-03",
        "page": 1,
        "image_path": "artifacts/rendered_pages/HLD-03/page_001.png",
        "output_filename": "HLD-03_p1.json",
        "rationale": "Complex tax/discount representative page (landscape Portuguese invoice with promotional discounts, excise duty IEC, alcohol tax, and VAT)",
    },
]


def run_benchmark() -> None:
    print("=" * 80)
    print("PHASE 6C-2: CONTROLLED VISION BENCHMARK")
    print(f"Target Model:   {PuterQwenProvider().model_name}")
    print(f"Prompt Version: {PROMPT_VERSION}")
    print(f"Pages to test:  {len(BENCHMARK_PAGES)}")
    print("=" * 80)

    provider = PuterQwenProvider()
    if not provider.has_api_key:
        print("[ERROR] PUTER_API_KEY is not set. Aborting benchmark.")
        sys.exit(1)

    results = []

    for idx, item in enumerate(BENCHMARK_PAGES, start=1):
        doc_stem = item["document"]
        page_num = item["page"]
        rel_img_path = item["image_path"]
        abs_img_path = PROJECT_ROOT / rel_img_path
        out_file = BENCHMARK_DIR / item["output_filename"]

        print(f"\n[{idx}/{len(BENCHMARK_PAGES)}] Running {doc_stem} page {page_num}...")
        print(f"    Image: {rel_img_path}")
        print(f"    Rationale: {item['rationale']}")

        if not abs_img_path.exists():
            print(f"    [ERROR] Image file does not exist: {abs_img_path}")
            continue

        # Execute single vision request
        t0 = time.perf_counter()
        resp = provider.analyze_image(abs_img_path, BENCHMARK_PROMPT, timeout=120.0)
        elapsed = time.perf_counter() - t0

        status_str = "success" if resp.success else "failed"

        output_payload = {
            "document": doc_stem,
            "page": page_num,
            "image_path": rel_img_path,
            "model": provider.model_name,
            "prompt_version": PROMPT_VERSION,
            "latency_seconds": round(resp.latency_seconds if resp.latency_seconds > 0 else elapsed, 4),
            "status": status_str,
            "response": resp.content if resp.success else resp.error,
        }

        # Also store non-secret metadata (token usage, finish reason)
        if resp.metadata:
            output_payload["metadata"] = {
                "finish_reason": resp.metadata.get("finish_reason"),
                "usage": resp.metadata.get("usage"),
                "image_size_bytes": resp.metadata.get("image_size_bytes"),
            }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)

        print(f"    Status:  {status_str.upper()}")
        print(f"    Latency: {output_payload['latency_seconds']:.3f}s")
        if resp.success and resp.content:
            print(f"    Chars:   {len(resp.content)}")
            usage = resp.metadata.get("usage", {})
            print(f"    Tokens:  prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}")
        else:
            print(f"    Error:   {resp.error}")

        results.append(output_payload)

        # Brief pacing delay between pages
        if idx < len(BENCHMARK_PAGES):
            time.sleep(2)

    print("\n" + "=" * 80)
    print("BENCHMARK RUN COMPLETED")
    print(f"Total pages processed: {len(results)}")
    successes = sum(1 for r in results if r["status"] == "success")
    print(f"Successes: {successes} / {len(results)}")
    print(f"Outputs written to: {BENCHMARK_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
