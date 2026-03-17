"""
Qwen2.5-VL and Qwen3-VL Adapters

Current open-source SOTA VLMs with:
- Dynamic resolution support
- Multilingual capabilities
- Qwen3's "thinking mode" for reasoning analysis
"""

from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from fingerprint_squared.models.base import BaseVLM, VLMRequest, VLMResponse


class Qwen25VLM(BaseVLM):
    """
    Qwen2.5-VL adapter - Current open-source SOTA.

    Features:
    - Dynamic resolution (handles any image size efficiently)
    - Strong instruction following
    - Multilingual (critical for diverse demographics)

    Example:
        >>> vlm = Qwen25VLM(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
        >>> response = await vlm.generate(request)
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        use_flash_attention: bool = True,
        max_memory: Optional[Dict[int, str]] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_flash_attention = use_flash_attention
        self.max_memory = max_memory

        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load model and processor."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "Please install transformers and torch: "
                "pip install transformers torch"
            )

        # Determine dtype
        if self.torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = getattr(torch, self.torch_dtype)

        # Load model
        attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            attn_implementation=attn_impl,
            max_memory=self.max_memory,
        )

        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def _load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """Load image from path, URL, or PIL Image."""
        if isinstance(image_source, Image.Image):
            return image_source

        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                import requests
                response = requests.get(image_source)
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                return Image.open(image_source).convert("RGB")

        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    async def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate response for image + prompt."""
        import torch

        self._load_model()
        start_time = time.time()

        try:
            # Load image
            image = self._load_image(request.image)

            # Build messages in Qwen format
            messages = []

            if request.system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt
                })

            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.prompt}
                ]
            })

            # Process with Qwen's vision utils
            try:
                from qwen_vl_utils import process_vision_info
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self._model.device)
            except ImportError:
                # Fallback without qwen_vl_utils
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=text,
                    images=image,
                    return_tensors="pt"
                ).to(self._model.device)

            # Generate
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens or 512,
                    temperature=request.temperature or 0.7,
                    do_sample=request.temperature > 0,
                )

            # Decode - only new tokens
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            response_text = self._processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
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


class Qwen3VLM(BaseVLM):
    """
    Qwen3-VL adapter with THINKING MODE support.

    The thinking mode is novel for bias research:
    - Captures chain-of-thought reasoning
    - Reveals *why* the model makes biased inferences
    - "We show reasoning chains invoke stereotypes before biased outputs"

    Example:
        >>> vlm = Qwen3VLM(model_id="Qwen/Qwen3-VL-8B-Instruct", enable_thinking=True)
        >>> response = await vlm.generate(request)
        >>> print(response.thinking)  # The reasoning chain
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        enable_thinking: bool = True,  # Novel: capture reasoning
        use_flash_attention: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.enable_thinking = enable_thinking
        self.use_flash_attention = use_flash_attention

        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load model."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("Please install transformers and torch")

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Qwen3-VL uses same architecture as Qwen2.5-VL
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            attn_implementation="flash_attention_2" if self.use_flash_attention else "eager",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def _load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image_source, Image.Image):
            return image_source
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                import requests
                return Image.open(BytesIO(requests.get(image_source).content)).convert("RGB")
            return Image.open(image_source).convert("RGB")
        raise ValueError(f"Unsupported image source: {type(image_source)}")

    async def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate with optional thinking mode."""
        import torch

        self._load_model()
        start_time = time.time()

        try:
            image = self._load_image(request.image)

            # Build prompt - optionally enable thinking
            prompt = request.prompt
            if self.enable_thinking:
                # Qwen3's thinking mode trigger
                prompt = f"/think\n{prompt}"

            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})

            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            })

            # Process
            try:
                from qwen_vl_utils import process_vision_info
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _ = process_vision_info(messages)
                inputs = self._processor(
                    text=[text], images=image_inputs, return_tensors="pt"
                ).to(self._model.device)
            except ImportError:
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=text, images=image, return_tensors="pt"
                ).to(self._model.device)

            # Generate
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens or 1024,  # Longer for thinking
                    temperature=request.temperature or 0.7,
                    do_sample=request.temperature > 0,
                )

            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            full_response = self._processor.decode(
                generated_ids, skip_special_tokens=True
            )

            # Parse thinking vs answer
            thinking = None
            answer = full_response

            if self.enable_thinking and "<think>" in full_response:
                # Extract thinking block
                parts = full_response.split("</think>")
                if len(parts) == 2:
                    thinking = parts[0].replace("<think>", "").strip()
                    answer = parts[1].strip()

            latency = (time.time() - start_time) * 1000

            response = VLMResponse(
                text=answer,
                model=self.model_id,
                latency_ms=latency,
                tokens_used=len(generated_ids),
            )

            # Attach thinking for bias analysis
            if thinking:
                response.metadata = {"thinking": thinking}

            return response

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
