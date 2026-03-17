"""OpenAI Vision-Language Model interface."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from fingerprint_squared.models.base import BaseVLM, VLMRequest, VLMResponse
from fingerprint_squared.utils.io import image_to_base64


class OpenAIVLM(BaseVLM):
    """
    OpenAI GPT-4 Vision interface.

    Supports GPT-4V, GPT-4o, and other vision-capable OpenAI models.

    Example:
        >>> vlm = OpenAIVLM(model_name="gpt-4o")
        >>> response = vlm.generate_sync(VLMRequest(
        ...     prompt="Describe this image",
        ...     images=["image.jpg"]
        ... ))
    """

    SUPPORTED_MODELS = [
        "gpt-4-vision-preview",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini",
    ]

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            provider="openai",
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        self.organization = organization
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("Please install openai: pip install openai")

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                timeout=self.timeout,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate a response from GPT-4V."""
        client = self._get_client()
        start_time = time.perf_counter()

        try:
            # Build messages
            messages = []

            if request.system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt,
                })

            # Build user content with text and images
            content = []

            # Add images first
            for image in request.images:
                encoded = self.encode_image(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": encoded,
                        "detail": "high",
                    },
                })

            # Add text prompt
            content.append({
                "type": "text",
                "text": request.prompt,
            })

            messages.append({
                "role": "user",
                "content": content,
            })

            # Make API call
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop_sequences,
            )

            latency = (time.perf_counter() - start_time) * 1000

            return VLMResponse(
                text=response.choices[0].message.content or "",
                model=self.model_name,
                provider=self.provider,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                latency_ms=latency,
                raw_response=response,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
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

    def encode_image(self, image: Union[str, Path, Image.Image]) -> str:
        """Encode image to base64 data URL."""
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                return image
            image = Path(image)

        if isinstance(image, Path):
            with open(image, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            suffix = image.suffix.lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "image/jpeg")
            return f"data:{mime_type};base64,{image_data}"

        if isinstance(image, Image.Image):
            b64 = image_to_base64(image)
            return f"data:image/png;base64,{b64}"

        raise ValueError(f"Unsupported image type: {type(image)}")
