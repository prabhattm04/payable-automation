"""src/vision/provider.py — Generic Vision Provider Interface.

Defines the abstract contract and response data structures for multimodal
vision model inference in the Bookable Payable pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class VisionResponse:
    """Standardized response from a vision model request."""
    success: bool
    content: Optional[str] = None
    model: str = ""
    provider: str = ""
    latency_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to a serializable dictionary, guaranteeing no secrets."""
        return {
            "success": self.success,
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "latency_seconds": round(self.latency_seconds, 4),
            "error": self.error,
            "metadata": self.metadata,
        }


class VisionProvider(ABC):
    """Abstract interface for hosted or local vision-language model providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The display name of the inference provider."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The identifier of the active vision model."""
        ...

    @abstractmethod
    def complete_text(self, prompt: str, timeout: float = 30.0) -> VisionResponse:
        """Send a text-only prompt to the model (useful for connectivity verification)."""
        ...

    @abstractmethod
    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        timeout: float = 60.0,
    ) -> VisionResponse:
        """Analyze a local image file with a prompt and return the model's response."""
        ...
