"""
OpenRouter VLM Adapter

Access proprietary models (GPT-4o, Claude, Gemini) through a unified API.
Single API key for all providers - perfect for comparison studies.

https://openrouter.ai
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from fingerprint_squared.models.base import BaseVLM, VLMRequest, VLMResponse


# OpenRouter model IDs for VLMs
OPENROUTER_MODELS = {
    # OpenAI
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4-turbo": "openai/gpt-4-turbo",

    # Anthropic
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
    "claude-3-haiku": "anthropic/claude-3-haiku-20240307",
    "claude-3-opus": "anthropic/claude-3-opus-20240229",

    # Google
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-1.5-pro": "google/gemini-pro-1.5",
    "gemini-1.5-flash": "google/gemini-flash-1.5",

    # Meta (via API providers)
    "llama-3.2-90b-vision": "meta-llama/llama-3.2-90b-vision-instruct",
    "llama-3.2-11b-vision": "meta-llama/llama-3.2-11b-vision-instruct",

    # Qwen (via API providers)
    "qwen-2.5-vl-72b": "qwen/qwen-2.5-vl-72b-instruct",
    "qwen-2-vl-7b": "qwen/qwen-2-vl-7b-instruct",
}


class OpenRouterVLM(BaseVLM):
    """
    OpenRouter VLM adapter for proprietary model access.

    Unified API for GPT-4o, Claude, Gemini, etc.
    Single API key, pay-per-use, great for research.

    Example:
        >>> vlm = OpenRouterVLM(model="gpt-4o")
        >>> response = await vlm.generate(request)

        # Or use full model ID
        >>> vlm = OpenRouterVLM(model="anthropic/claude-3.5-sonnet")
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        site_url: str = "https://fingerprint-squared.github.io",
        site_name: str = "Fingerprint Squared",
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        # Resolve model shorthand to full ID
        self.model_id = OPENROUTER_MODELS.get(model, model)
        self.model_short = model

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter. Get one at https://openrouter.ai"
            )

        self.site_url = site_url
        self.site_name = site_name
        self.max_retries = max_retries
        self.timeout = timeout

        self._client = None

    async def _get_client(self):
        """Get or create async HTTP client."""
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError("Please install httpx: pip install httpx")

            self._client = httpx.AsyncClient(timeout=self.timeout)

        return self._client

    def _encode_image(self, image_source: Union[str, Image.Image]) -> str:
        """Encode image to base64 data URL."""
        if isinstance(image_source, str):
            # URL - use directly
            if image_source.startswith(("http://", "https://")):
                return image_source

            # File path - load and encode
            with open(image_source, "rb") as f:
                image_data = f.read()

            # Detect format
            ext = Path(image_source).suffix.lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(ext, "image/jpeg")

        elif isinstance(image_source, Image.Image):
            # PIL Image - encode to JPEG
            buffer = BytesIO()
            image_source.save(buffer, format="JPEG", quality=95)
            image_data = buffer.getvalue()
            mime_type = "image/jpeg"
        else:
            raise ValueError(f"Unsupported image source: {type(image_source)}")

        b64 = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    async def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate response via OpenRouter API."""
        client = await self._get_client()
        start_time = time.time()

        try:
            # Encode image
            image_url = self._encode_image(request.image)

            # Build messages
            messages = []

            if request.system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt
                })

            # User message with image
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": request.prompt
                    }
                ]
            })

            # Request payload
            payload = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": request.max_tokens or 512,
                "temperature": request.temperature or 0.7,
            }

            # Headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
                "Content-Type": "application/json",
            }

            # Make request with retries
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        self.BASE_URL,
                        json=payload,
                        headers=headers,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        text = data["choices"][0]["message"]["content"]
                        usage = data.get("usage", {})

                        latency = (time.time() - start_time) * 1000

                        return VLMResponse(
                            text=text.strip(),
                            model=self.model_id,
                            latency_ms=latency,
                            tokens_used=usage.get("total_tokens"),
                            metadata={
                                "provider": "openrouter",
                                "usage": usage,
                            }
                        )

                    elif response.status_code == 429:
                        # Rate limited - wait and retry
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue

                    else:
                        last_error = f"HTTP {response.status_code}: {response.text}"

                except Exception as e:
                    last_error = str(e)
                    await asyncio.sleep(1)

            # All retries failed
            return VLMResponse(
                text="",
                model=self.model_id,
                latency_ms=(time.time() - start_time) * 1000,
                error=last_error or "Unknown error",
            )

        except Exception as e:
            return VLMResponse(
                text="",
                model=self.model_id,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def model_name(self) -> str:
        return self.model_short

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List available models."""
        return OPENROUTER_MODELS.copy()


class MultiProviderVLM:
    """
    Convenience class to create VLMs from multiple providers.

    Automatically routes to the right adapter based on model string.

    Example:
        >>> vlm = MultiProviderVLM.create("openrouter:gpt-4o")
        >>> vlm = MultiProviderVLM.create("qwen:Qwen2.5-VL-7B")
        >>> vlm = MultiProviderVLM.create("internvl:InternVL3-8B")
    """

    @staticmethod
    def create(
        model_spec: str,
        api_key: Optional[str] = None,
        device: str = "auto",
        **kwargs
    ) -> BaseVLM:
        """
        Create VLM from model specification.

        Format: "provider:model" or just "model" (defaults to openrouter)

        Providers:
        - openrouter: GPT-4o, Claude, Gemini via OpenRouter
        - openai: Direct OpenAI API
        - anthropic: Direct Anthropic API
        - qwen: Qwen2.5-VL, Qwen3-VL (local HuggingFace)
        - internvl: InternVL3 (local HuggingFace)
        - llama: LLaMA 3.2 Vision (local HuggingFace)
        - smol: SmolVLM 2B (local HuggingFace)
        """
        if ":" in model_spec:
            provider, model = model_spec.split(":", 1)
        else:
            # Default to openrouter for known proprietary models
            if model_spec.lower() in ["gpt-4o", "gpt-4o-mini", "claude", "gemini"]:
                provider = "openrouter"
                model = model_spec
            else:
                provider = "openrouter"
                model = model_spec

        provider = provider.lower()

        if provider == "openrouter":
            return OpenRouterVLM(model=model, api_key=api_key, **kwargs)

        elif provider == "openai":
            from fingerprint_squared.models.openai_vlm import OpenAIVLM
            return OpenAIVLM(model=model, api_key=api_key)

        elif provider == "anthropic":
            from fingerprint_squared.models.anthropic_vlm import AnthropicVLM
            return AnthropicVLM(model=model, api_key=api_key)

        elif provider in ["qwen", "qwen2.5", "qwen25"]:
            from fingerprint_squared.models.qwen_vlm import Qwen25VLM
            model_id = model if "/" in model else f"Qwen/{model}"
            return Qwen25VLM(model_id=model_id, device=device, **kwargs)

        elif provider in ["qwen3"]:
            from fingerprint_squared.models.qwen_vlm import Qwen3VLM
            model_id = model if "/" in model else f"Qwen/{model}"
            return Qwen3VLM(model_id=model_id, device=device, **kwargs)

        elif provider == "internvl":
            from fingerprint_squared.models.internvl_vlm import InternVL3VLM
            model_id = model if "/" in model else f"OpenGVLab/{model}"
            return InternVL3VLM(model_id=model_id, device=device, **kwargs)

        elif provider == "llama":
            from fingerprint_squared.models.llama_vision_vlm import LlamaVisionVLM
            model_id = model if "/" in model else f"meta-llama/{model}"
            return LlamaVisionVLM(model_id=model_id, device=device, hf_token=api_key, **kwargs)

        elif provider == "smol":
            from fingerprint_squared.models.llama_vision_vlm import SmolVLM
            return SmolVLM(device=device, **kwargs)

        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: openrouter, openai, anthropic, qwen, qwen3, internvl, llama, smol"
            )
