"""tests/test_vision.py — Tests for Vision Provider Abstraction (Phase 6C)."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.vision.provider import VisionProvider, VisionResponse
from src.vision.puter_qwen import PuterQwenProvider, _resolve_image_path


class TestVisionProviderContract:
    def test_vision_response_to_dict_does_not_leak_secrets(self):
        resp = VisionResponse(
            success=True,
            content="Document is an invoice.",
            model="qwen/qwen3-vl-plus-2025-12-19",
            provider="Puter",
            latency_seconds=1.23456,
            metadata={"finish_reason": "stop"},
        )
        d = resp.to_dict()
        assert d["success"] is True
        assert d["content"] == "Document is an invoice."
        assert d["model"] == "qwen/qwen3-vl-plus-2025-12-19"
        assert d["provider"] == "Puter"
        assert d["latency_seconds"] == 1.2346
        assert "api_key" not in d
        assert "key" not in d

    def test_puter_provider_initialization_without_key(self):
        provider = PuterQwenProvider(api_key="")
        assert provider.provider_name == "Puter"
        assert provider.model_name == "qwen/qwen3-vl-plus-2025-12-19"
        assert provider.has_api_key is False

    def test_complete_text_fails_gracefully_without_key(self):
        provider = PuterQwenProvider(api_key="")
        resp = provider.complete_text("test prompt")
        assert resp.success is False
        assert "PUTER_API_KEY is not configured" in resp.error
        assert resp.latency_seconds == 0.0

    def test_analyze_image_fails_gracefully_without_key(self, tmp_path):
        dummy_img = tmp_path / "test.png"
        dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        provider = PuterQwenProvider(api_key="")
        resp = provider.analyze_image(dummy_img, "analyze")
        assert resp.success is False
        assert "PUTER_API_KEY is not configured" in resp.error

    def test_resolve_image_path_hyphen_underscore(self, tmp_path):
        p = tmp_path / "page_001.png"
        p.write_bytes(b"data")

        hyphen_req = tmp_path / "page-001.png"
        resolved = _resolve_image_path(hyphen_req)
        assert resolved == p
        assert resolved.exists()

    @patch("requests.post")
    def test_complete_text_success_mock(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "result": {
                "message": {"content": "PUTER_QWEN_TEST_OK", "role": "assistant"},
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 19, "completion_tokens": 8, "total_tokens": 27},
            },
        }
        mock_post.return_value = mock_resp

        provider = PuterQwenProvider(api_key="mock-key")
        resp = provider.complete_text("Respond with exactly: PUTER_QWEN_TEST_OK")

        assert resp.success is True
        assert resp.content == "PUTER_QWEN_TEST_OK"
        assert resp.model == "qwen/qwen3-vl-plus-2025-12-19"
        assert resp.metadata["finish_reason"] == "stop"
        assert resp.metadata["usage"]["total_tokens"] == 27

    @patch("requests.post")
    def test_analyze_image_success_mock(self, mock_post, tmp_path):
        dummy_img = tmp_path / "page_001.png"
        dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "result": {
                "message": {"content": "Commercial Invoice", "role": "assistant"},
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 250, "completion_tokens": 40, "total_tokens": 290},
            },
        }
        mock_post.return_value = mock_resp

        provider = PuterQwenProvider(api_key="mock-key")
        resp = provider.analyze_image(dummy_img, "Analyze this document visually.")

        assert resp.success is True
        assert resp.content == "Commercial Invoice"
        assert resp.metadata["image_size_bytes"] > 0
