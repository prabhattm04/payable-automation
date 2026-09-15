"""src/vision/puter_qwen.py — Puter hosted inference provider for Qwen Vision.

Uses Puter's official HTTP driver API (powering puter.ai.chat) to query Qwen vision
models (e.g., qwen/qwen3-vl-plus-2025-12-19) with local rendered PDF page images.
Compatible with standard OpenAI-format multimodal messages.
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from dotenv import load_dotenv

from src.utils.logging import get_logger
from src.vision.provider import VisionProvider, VisionResponse

log = get_logger(__name__)

DEFAULT_PUTER_MODEL = "qwen/qwen3-vl-plus-2025-12-19"
DEFAULT_PUTER_ENDPOINT = "https://api.puter.com/drivers/call"


def _resolve_image_path(image_path: Union[str, Path]) -> Path:
    """Resolve an image path, handling alternate naming conventions (e.g. page-001 vs page_001)."""
    p = Path(image_path)
    if p.exists():
        return p
    alt_name = p.name.replace("-", "_") if "-" in p.name else p.name.replace("_", "-")
    alt_path = p.parent / alt_name
    if alt_path.exists():
        return alt_path
    return p


class PuterQwenProvider(VisionProvider):
    """Hosted inference provider leveraging Puter for Qwen Vision."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_PUTER_MODEL,
        endpoint: str = DEFAULT_PUTER_ENDPOINT,
    ) -> None:
        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            load_dotenv()
            self._api_key = os.environ.get("PUTER_API_KEY", "").strip()
        self._model = model
        self._endpoint = endpoint

    @property
    def provider_name(self) -> str:
        return "Puter"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    def _build_headers(self) -> Dict[str, str]:
        if not self._api_key:
            raise ValueError(
                "PUTER_API_KEY is not configured. "
                "Set PUTER_API_KEY in your shell environment or in a .env file."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "BookablePayable-Vision/1.0",
        }

    def _call_driver(self, messages: list, timeout: float = 60.0) -> VisionResponse:
        """Call Puter's chat completion driver with standard message payload."""
        start_time = time.perf_counter()
        if not self._api_key:
            return VisionResponse(
                success=False,
                model=self._model,
                provider=self.provider_name,
                latency_seconds=0.0,
                error="PUTER_API_KEY is not configured. Set PUTER_API_KEY in .env or environment.",
            )

        payload = {
            "interface": "puter-chat-completion",
            "method": "complete",
            "args": {
                "model": self._model,
                "messages": messages,
            },
        }

        try:
            headers = self._build_headers()
            response = requests.post(
                self._endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start_time

            if response.status_code != 200:
                err_text = response.text
                if self._api_key and self._api_key in err_text:
                    err_text = err_text.replace(self._api_key, "[REDACTED_API_KEY]")
                return VisionResponse(
                    success=False,
                    model=self._model,
                    provider=self.provider_name,
                    latency_seconds=elapsed,
                    error=f"HTTP {response.status_code}: {err_text}",
                )

            data = response.json()
            result = data.get("result", {})
            message = result.get("message", {})
            content = message.get("content", "")
            finish_reason = result.get("finish_reason")
            usage = result.get("usage", {})

            return VisionResponse(
                success=True,
                content=content,
                model=self._model,
                provider=self.provider_name,
                latency_seconds=elapsed,
                metadata={
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "endpoint": self._endpoint,
                },
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            err_msg = str(exc)
            if self._api_key and self._api_key in err_msg:
                err_msg = err_msg.replace(self._api_key, "[REDACTED_API_KEY]")
            log.error("Puter request failed: %s", err_msg)
            return VisionResponse(
                success=False,
                model=self._model,
                provider=self.provider_name,
                latency_seconds=elapsed,
                error=err_msg,
            )

    def complete_text(self, prompt: str, timeout: float = 30.0) -> VisionResponse:
        """Send a text-only prompt to verify connectivity, endpoint, and authentication."""
        messages = [{"role": "user", "content": prompt}]
        return self._call_driver(messages, timeout=timeout)

    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        timeout: float = 60.0,
    ) -> VisionResponse:
        """Analyze a local rendered image with Qwen Vision via Puter."""
        resolved_path = _resolve_image_path(image_path)
        if not resolved_path.exists():
            return VisionResponse(
                success=False,
                model=self._model,
                provider=self.provider_name,
                latency_seconds=0.0,
                error=f"Image file does not exist: {resolved_path}",
            )

        try:
            with open(resolved_path, "rb") as f:
                img_bytes = f.read()
            b64_data = base64.b64encode(img_bytes).decode("utf-8")

            ext = resolved_path.suffix.lower().lstrip(".")
            mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "webp") else "image/png"
            data_url = f"data:{mime};base64,{b64_data}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            },
                        },
                    ],
                }
            ]
            response = self._call_driver(messages, timeout=timeout)
            if response.metadata is not None:
                response.metadata["image_path"] = str(resolved_path)
                response.metadata["image_size_bytes"] = len(img_bytes)
            return response
        except Exception as exc:
            err_msg = str(exc)
            if self._api_key and self._api_key in err_msg:
                err_msg = err_msg.replace(self._api_key, "[REDACTED_API_KEY]")
            return VisionResponse(
                success=False,
                model=self._model,
                provider=self.provider_name,
                latency_seconds=0.0,
                error=err_msg,
            )
