"""Google Gemini Vision interface."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from fingerprint_squared.models.base import BaseVLM, VLMRequest, VLMResponse
from fingerprint_squared.utils.io import image_to_base64


class GoogleVLM(BaseVLM):
    """
    Google Gemini Vision interface.

    Supports Gemini Pro Vision and Gemini Ultra Vision models.

    Example:
        >>> vlm = GoogleVLM(model_name="gemini-pro-vision")
        >>> response = vlm.generate_sync(VLMRequest(
        ...     prompt="Describe this image",
        ...     images=["image.jpg"]
        ... ))
    """

    SUPPORTED_MODELS = [
        "gemini-pro-vision",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-ultra-vision",
    ]

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            provider="google",
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        self._model = None

    def _get_model(self):
        """Lazy initialization of Gemini model."""
        if self._model is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "Please install google-generativeai: pip install google-generativeai"
                )

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)

        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate a response from Gemini."""
        import asyncio

        model = self._get_model()
        start_time = time.perf_counter()

        try:
            # Build content parts
            parts = []

            # Add images
            for image in request.images:
                img_data = self.encode_image(image)
                parts.append(img_data)

            # Add text prompt
            parts.append(request.prompt)

            # Generation config
            generation_config = {
                "max_output_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            }

            if request.stop_sequences:
                generation_config["stop_sequences"] = request.stop_sequences

            # Run in executor since google API is sync
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    parts,
                    generation_config=generation_config,
                ),
            )

            latency = (time.perf_counter() - start_time) * 1000

            # Extract text
            text = response.text if hasattr(response, "text") else ""

            # Get usage if available
            usage = {}
            if hasattr(response, "usage_metadata"):
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                }

            return VLMResponse(
                text=text,
                model=self.model_name,
                provider=self.provider,
                usage=usage,
                latency_ms=latency,
                raw_response=response,
                metadata={
                    "finish_reason": getattr(response.candidates[0], "finish_reason", None) if response.candidates else None,
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

    def encode_image(self, image: Union[str, Path, Image.Image]) -> Any:
        """Encode image for Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Please install google-generativeai: pip install google-generativeai"
            )

        if isinstance(image, str):
            image = Path(image)

        if isinstance(image, Path):
            pil_image = Image.open(image)
            return pil_image

        if isinstance(image, Image.Image):
            return image

        raise ValueError(f"Unsupported image type: {type(image)}")
