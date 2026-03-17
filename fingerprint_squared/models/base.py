"""
Base class for Vision-Language Model interfaces.

Provides a unified interface for interacting with different VLM providers,
enabling consistent evaluation across models.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from PIL import Image


@dataclass
class VLMResponse:
    """Standardized response from a VLM."""

    text: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    raw_response: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class VLMRequest:
    """Standardized request to a VLM."""

    prompt: str
    images: List[Union[str, Image.Image]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    stop_sequences: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Model interfaces.

    All VLM adapters should inherit from this class and implement
    the required methods.

    Attributes:
        model_name: Name/identifier of the model
        provider: Provider name (openai, anthropic, etc.)
        api_key: API key for authentication
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts

    Example:
        >>> class MyVLM(BaseVLM):
        ...     async def generate(self, request):
        ...         # Implementation
        ...         pass
    """

    def __init__(
        self,
        model_name: str,
        provider: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs,
    ):
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_config = kwargs

    @abstractmethod
    async def generate(self, request: VLMRequest) -> VLMResponse:
        """
        Generate a response from the VLM.

        Args:
            request: VLMRequest with prompt and optional images

        Returns:
            VLMResponse with generated text
        """
        pass

    async def generate_batch(
        self,
        requests: List[VLMRequest],
        max_concurrent: int = 5,
    ) -> List[VLMResponse]:
        """
        Generate responses for multiple requests concurrently.

        Args:
            requests: List of VLMRequests
            max_concurrent: Maximum concurrent requests

        Returns:
            List of VLMResponses
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_generate(request: VLMRequest) -> VLMResponse:
            async with semaphore:
                return await self.generate(request)

        tasks = [limited_generate(req) for req in requests]
        return await asyncio.gather(*tasks)

    def generate_sync(self, request: VLMRequest) -> VLMResponse:
        """Synchronous wrapper for generate."""
        return asyncio.run(self.generate(request))

    def generate_batch_sync(
        self,
        requests: List[VLMRequest],
        max_concurrent: int = 5,
    ) -> List[VLMResponse]:
        """Synchronous wrapper for generate_batch."""
        return asyncio.run(self.generate_batch(requests, max_concurrent))

    @abstractmethod
    def encode_image(self, image: Union[str, Image.Image]) -> str:
        """
        Encode an image for API submission.

        Args:
            image: Image path or PIL Image

        Returns:
            Encoded image string (base64 or URL)
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, provider={self.provider})"
