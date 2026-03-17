"""
LLaMA 3.2 Vision Adapter

Meta's SOTA VLM - Western-trained, gated model.
Provides interesting contrast to Qwen (Chinese-trained).

Different training data geography = different bias fingerprint?
"""

from __future__ import annotations

import time
from io import BytesIO
from typing import Optional, Union

from PIL import Image

from fingerprint_squared.models.base import BaseVLM, VLMRequest, VLMResponse


class LlamaVisionVLM(BaseVLM):
    """
    LLaMA 3.2 Vision adapter.

    Key research angle: Compare Western-trained (Meta) vs Chinese-trained (Qwen)
    bias fingerprints. Does training data geography affect bias patterns?

    Note: Requires HuggingFace access token for gated model.

    Example:
        >>> vlm = LlamaVisionVLM(
        ...     model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        ...     hf_token="your_token"
        ... )
        >>> response = await vlm.generate(request)
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        hf_token: Optional[str] = None,
        use_flash_attention: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.hf_token = hf_token
        self.use_flash_attention = use_flash_attention

        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load LLaMA Vision model."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import MllamaForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("Please install transformers>=4.45.0 and torch")

        dtype = getattr(torch, self.torch_dtype)

        self._model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            token=self.hf_token,
            attn_implementation="flash_attention_2" if self.use_flash_attention else "eager",
        )

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=self.hf_token,
        )

    def _load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                import requests
                return Image.open(BytesIO(requests.get(image_source).content)).convert("RGB")
            return Image.open(image_source).convert("RGB")
        raise ValueError(f"Unsupported image source: {type(image_source)}")

    async def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate response using LLaMA Vision."""
        import torch

        self._load_model()
        start_time = time.time()

        try:
            image = self._load_image(request.image)

            # Build messages in LLaMA format
            messages = []

            if request.system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt
                })

            # LLaMA Vision uses {"type": "image"} without image data in message
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": request.prompt}
                ]
            })

            # Apply chat template
            input_text = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            # Process inputs
            inputs = self._processor(
                images=image,
                text=input_text,
                return_tensors="pt"
            ).to(self._model.device)

            # Generate
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens or 512,
                    temperature=request.temperature or 0.7,
                    do_sample=(request.temperature or 0.7) > 0,
                )

            # Decode only generated tokens
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            response_text = self._processor.decode(
                generated_ids,
                skip_special_tokens=True
            )

            latency = (time.time() - start_time) * 1000

            return VLMResponse(
                text=response_text.strip(),
                model=self.model_id,
                latency_ms=latency,
                tokens_used=len(generated_ids),
            )

        except Exception as e:
            return VLMResponse(
                text="",
                model=self.model_id,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    @property
    def model_name(self) -> str:
        return self.model_id.split("/")[-1]


class SmolVLM(BaseVLM):
    """
    SmolVLM adapter - Tiny 2B model.

    Research question: Does model size correlate with bias?
    Use as "lightweight comparison point."

    Apache 2.0 licensed, very fast.
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-Instruct",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype

        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load SmolVLM."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except ImportError:
            raise ImportError("Please install transformers and torch")

        dtype = getattr(torch, self.torch_dtype)

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

    def _load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """Load image."""
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                import requests
                return Image.open(BytesIO(requests.get(image_source).content)).convert("RGB")
            return Image.open(image_source).convert("RGB")
        raise ValueError(f"Unsupported image source: {type(image_source)}")

    async def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate response using SmolVLM."""
        import torch

        self._load_model()
        start_time = time.time()

        try:
            image = self._load_image(request.image)

            # SmolVLM message format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": request.prompt}
                ]
            }]

            prompt = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            inputs = self._processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            ).to(self._model.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens or 512,
                    temperature=request.temperature or 0.7,
                    do_sample=(request.temperature or 0.7) > 0,
                )

            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            response_text = self._processor.decode(
                generated_ids,
                skip_special_tokens=True
            )

            latency = (time.time() - start_time) * 1000

            return VLMResponse(
                text=response_text.strip(),
                model=self.model_id,
                latency_ms=latency,
                tokens_used=len(generated_ids),
            )

        except Exception as e:
            return VLMResponse(
                text="",
                model=self.model_id,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    @property
    def model_name(self) -> str:
        return "SmolVLM-2B"
