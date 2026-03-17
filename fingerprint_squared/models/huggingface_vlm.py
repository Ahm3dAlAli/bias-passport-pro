"""HuggingFace Vision-Language Model interface."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from fingerprint_squared.models.base import BaseVLM, VLMRequest, VLMResponse


class HuggingFaceVLM(BaseVLM):
    """
    HuggingFace Transformers VLM interface.

    Supports local models like LLaVA, BLIP-2, InstructBLIP, Qwen2.5-VL, and other
    vision-language models available on HuggingFace.

    Example:
        >>> vlm = HuggingFaceVLM(model_name="llava-hf/llava-1.5-7b-hf")
        >>> response = vlm.generate_sync(VLMRequest(
        ...     prompt="Describe this image",
        ...     images=["image.jpg"]
        ... ))
    """

    SUPPORTED_MODELS = [
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-1.5-13b-hf",
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        "Salesforce/blip2-opt-2.7b",
        "Salesforce/instructblip-vicuna-7b",
        "microsoft/Florence-2-large",
        "Qwen/Qwen-VL-Chat",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "HuggingFaceTB/SmolVLM-Instruct",
        "openbmb/MiniCPM-V-2_5",
    ]

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        api_key: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: str = "float16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        timeout: float = 120.0,
        max_retries: int = 1,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            provider="huggingface",
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None

    def _load_model(self):
        """Load model and processor."""
        if self._model is not None:
            return

        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers accelerate"
            )

        model_lower = self.model_name.lower()

        # Qwen2.5-VL and Qwen2-VL models
        if "qwen2.5-vl" in model_lower or "qwen2-vl" in model_lower:
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            except ImportError:
                raise ImportError(
                    "Please install transformers>=4.45.0 for Qwen2-VL support: "
                    "pip install transformers>=4.45.0 accelerate qwen-vl-utils"
                )

            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            self._is_qwen2_vl = True

        # LLaVA models (including v1.6)
        elif "llava" in model_lower:
            from transformers import LlavaForConditionalGeneration, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
            )
            self._is_qwen2_vl = False

        # SmolVLM models
        elif "smolvlm" in model_lower:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            self._is_qwen2_vl = False

        # BLIP models
        elif "blip" in model_lower:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration

            self._processor = Blip2Processor.from_pretrained(self.model_name)
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
            self._is_qwen2_vl = False

        else:
            # Generic loading
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            self._is_qwen2_vl = False

        if self.device == "cuda" and not (self.load_in_8bit or self.load_in_4bit):
            # Skip .to() for models with device_map="auto"
            if not hasattr(self._model, 'hf_device_map'):
                self._model = self._model.to(self.device)

    async def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate a response from the local model."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, request)

    def _generate_sync(self, request: VLMRequest) -> VLMResponse:
        """Synchronous generation."""
        self._load_model()
        start_time = time.perf_counter()

        try:
            # Load images
            images = [self.encode_image(img) for img in request.images]
            model_lower = self.model_name.lower()

            # Handle Qwen2-VL models with their special message format
            if getattr(self, '_is_qwen2_vl', False):
                return self._generate_qwen2_vl(request, images, start_time)

            # Prepare prompt based on model type
            if "llava" in model_lower:
                # LLaVA format
                prompt = f"USER: <image>\n{request.prompt}\nASSISTANT:"
            elif "smolvlm" in model_lower:
                # SmolVLM format
                prompt = f"<image>\n{request.prompt}"
            else:
                prompt = request.prompt

            # Process inputs
            if images:
                inputs = self._processor(
                    text=prompt,
                    images=images[0] if len(images) == 1 else images,
                    return_tensors="pt",
                ).to(self.device)
            else:
                inputs = self._processor(
                    text=prompt,
                    return_tensors="pt",
                ).to(self.device)

            # Generate
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature if request.temperature > 0 else None,
                    top_p=request.top_p,
                    do_sample=request.temperature > 0,
                )

            # Decode
            generated_text = self._processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )[0]

            # Extract response (remove prompt)
            if "ASSISTANT:" in generated_text:
                generated_text = generated_text.split("ASSISTANT:")[-1].strip()

            latency = (time.perf_counter() - start_time) * 1000

            # Calculate token counts safely
            input_tokens = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0
            output_tokens = output_ids.shape[1] if len(output_ids.shape) > 1 else len(output_ids)

            return VLMResponse(
                text=generated_text,
                model=self.model_name,
                provider=self.provider,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": max(0, output_tokens - input_tokens),
                    "total_tokens": output_tokens,
                },
                latency_ms=latency,
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

    def _generate_qwen2_vl(self, request: VLMRequest, images: List[Image.Image], start_time: float) -> VLMResponse:
        """Generate response using Qwen2-VL model format."""
        try:
            # Qwen2-VL uses a chat format with messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[0]} if images else None,
                        {"type": "text", "text": request.prompt},
                    ],
                }
            ]
            # Filter out None content
            messages[0]["content"] = [c for c in messages[0]["content"] if c is not None]

            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            if images:
                inputs = self._processor(
                    text=[text],
                    images=images,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
            else:
                inputs = self._processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

            # Generate
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature if request.temperature > 0 else None,
                    top_p=request.top_p,
                    do_sample=request.temperature > 0,
                )

            # Decode only the generated part
            generated_ids = [
                output_ids[i][len(inputs.input_ids[i]):]
                for i in range(len(output_ids))
            ]
            generated_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            latency = (time.perf_counter() - start_time) * 1000

            return VLMResponse(
                text=generated_text.strip(),
                model=self.model_name,
                provider=self.provider,
                usage={
                    "prompt_tokens": inputs.input_ids.shape[1],
                    "completion_tokens": output_ids.shape[1] - inputs.input_ids.shape[1],
                    "total_tokens": output_ids.shape[1],
                },
                latency_ms=latency,
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

    def encode_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load and return PIL Image."""
        if isinstance(image, str):
            image = Path(image)

        if isinstance(image, Path):
            return Image.open(image).convert("RGB")

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        raise ValueError(f"Unsupported image type: {type(image)}")
