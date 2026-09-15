"""analysis/phase6c/run_smoke_test.py — Phase 6C connectivity and smoke tests.

Executes:
1. Text-only connectivity test -> analysis/phase6c/puter_connectivity_test.json
2. Single-image smoke test (if step 1 succeeds) -> analysis/phase6c/puter_qwen_smoke_test.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.puter_qwen import PuterQwenProvider

ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "phase6c"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

CONNECTIVITY_FILE = ANALYSIS_DIR / "puter_connectivity_test.json"
SMOKE_TEST_FILE = ANALYSIS_DIR / "puter_qwen_smoke_test.json"

TEXT_TEST_PROMPT = "Respond with exactly: PUTER_QWEN_TEST_OK"

IMAGE_TEST_PROMPT = """Analyze this document visually. Describe:
the apparent document type,
the major sections,
the spatial relationship between labels and values,
whether there is a table and how its rows/columns are organized.
Do not calculate totals.
Do not invent values.
Do not produce final payable JSON.
Do not make accounting decisions."""

IMAGE_PATH = PROJECT_ROOT / "artifacts" / "rendered_pages" / "INV-01" / "page_001.png"


def run_tests() -> dict:
    provider = PuterQwenProvider()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Text-Only Connectivity Test
    # ──────────────────────────────────────────────────────────────────────────
    print(f"[*] Starting Step 4: Text-only connectivity test...")
    print(f"    Provider: {provider.provider_name}")
    print(f"    Model:    {provider.model_name}")
    print(f"    API Key present: {provider.has_api_key}")

    text_resp = provider.complete_text(TEXT_TEST_PROMPT)

    connectivity_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider.provider_name,
        "model": provider.model_name,
        "prompt": TEXT_TEST_PROMPT,
        "success": text_resp.success,
        "latency_seconds": round(text_resp.latency_seconds, 4),
        "response_content": text_resp.content,
        "error": text_resp.error,
        "metadata": text_resp.metadata,
    }

    with open(CONNECTIVITY_FILE, "w", encoding="utf-8") as f:
        json.dump(connectivity_data, f, indent=2)
    print(f"[+] Connectivity test result written to: {CONNECTIVITY_FILE}")
    print(f"    Success: {text_resp.success} (Latency: {text_resp.latency_seconds:.3f}s)")
    if text_resp.error:
        print(f"    Error: {text_resp.error}")
    else:
        print(f"    Content: {text_resp.content}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Single Image Test (ONLY if Step 4 succeeds)
    # ──────────────────────────────────────────────────────────────────────────
    smoke_data = None
    if text_resp.success:
        print(f"\n[*] Starting Step 5: Single image smoke test on {IMAGE_PATH.name}...")
        img_resp = provider.analyze_image(IMAGE_PATH, IMAGE_TEST_PROMPT)

        content_len = len(img_resp.content) if img_resp.content else 0
        smoke_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider.provider_name,
            "model": provider.model_name,
            "image_path": str(IMAGE_PATH.relative_to(PROJECT_ROOT)),
            "prompt": IMAGE_TEST_PROMPT,
            "success": img_resp.success,
            "latency_seconds": round(img_resp.latency_seconds, 4),
            "response_length_chars": content_len,
            "response_content": img_resp.content,
            "error": img_resp.error,
            "metadata": img_resp.metadata,
        }

        with open(SMOKE_TEST_FILE, "w", encoding="utf-8") as f:
            json.dump(smoke_data, f, indent=2)
        print(f"[+] Single image test result written to: {SMOKE_TEST_FILE}")
        print(f"    Success: {img_resp.success} (Latency: {img_resp.latency_seconds:.3f}s, Chars: {content_len})")
        if img_resp.error:
            print(f"    Error: {img_resp.error}")
    else:
        print("\n[!] Step 5 SKIPPED because text-only connectivity test did not succeed.")

    return {
        "text_success": text_resp.success,
        "image_success": smoke_data["success"] if smoke_data else False,
    }


if __name__ == "__main__":
    run_tests()
