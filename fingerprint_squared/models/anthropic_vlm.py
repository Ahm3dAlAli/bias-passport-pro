"""Anthropic Claude Vision interface."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from fingerprint_squared.models.base import BaseVLM, VLMRequest, VLMResponse
from fingerprint_squared.utils.io import image_to_base64


class AnthropicVLM(BaseVLM):
    """
    Anthropic Claude Vision interface.

    Supports Claude 3 Opus, Sonnet, and Haiku models with vision capabilities.

    Example:
        >>> vlm = AnthropicVLM(model_name="claude-3-opus-20240229")
        >>> response = vlm.generate_sync(VLMRequest(
        ...     prompt="Describe this image",
        ...     images=["image.jpg"]
        ... ))
    """

    SUPPORTED_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
        "claude-sonnet-4-20250514",
        "claude-opus-4-5-20251101",
    ]

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            provider="anthropic",
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")

            self._client = AsyncAnthropic(
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate a response from Claude."""
        client = self._get_client()
        start_time = time.perf_counter()

        try:
            # Build content with images and text
            content = []

            # Add images
            for image in request.images:
                encoded, media_type = self.encode_image(image)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": encoded,
                    },
                })

            # Add text prompt
            content.append({
                "type": "text",
                "text": request.prompt,
            })

            # Make API call
            response = await client.messages.create(
                model=self.model_name,
                max_tokens=request.max_tokens,
                system=request.system_prompt or "",
                messages=[{
                    "role": "user",
                    "content": content,
                }],
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences or [],
            )

            latency = (time.perf_counter() - start_time) * 1000

            # Extract text from response
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            return VLMResponse(
                text=text,
                model=self.model_name,
                provider=self.provider,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                latency_ms=latency,
                raw_response=response,
                metadata={
                    "stop_reason": response.stop_reason,
                    "model": response.model,
                },
            )

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return VLMResponse(
                text="",
                model=self.model_name,
                provider=self.provider,
                latency_ms=latency,
                error=str(e),
            )

    def encode_image(self, image: Union[str, Path, Image.Image]) -> tuple[str, str]:
        """
        Encode image to base64 for Anthropic API.

        Returns:
            Tuple of (base64_data, media_type)
        """
        if isinstance(image, str):
            image = Path(image)

        if isinstance(image, Path):
            with open(image, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            suffix = image.suffix.lower()
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "image/jpeg")

            return image_data, media_type

        if isinstance(image, Image.Image):
            b64 = image_to_base64(image)
            return b64, "image/png"

        raise ValueError(f"Unsupported image type: {type(image)}")
