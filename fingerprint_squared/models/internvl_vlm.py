"""
InternVL3 Adapter

Different training lineage than Qwen - enables comparison of
bias fingerprints across model families.

Strong on OCR/scene understanding, different RLHF approach.
"""

from __future__ import annotations

import math
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from fingerprint_squared.models.base import BaseVLM, VLMRequest, VLMResponse


class InternVL3VLM(BaseVLM):
    """
    InternVL3 adapter.

    Key differentiator: Different training data and RLHF than Qwen family.
    This enables studying whether bias fingerprints correlate with
    training methodology.

    Example:
        >>> vlm = InternVL3VLM(model_id="OpenGVLab/InternVL3-8B")
        >>> response = await vlm.generate(request)
    """

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL3-8B",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_tiles: int = 12,  # InternVL's dynamic tiling
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_tiles = max_tiles

        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load InternVL3 model."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers and torch")

        dtype = getattr(torch, self.torch_dtype)

        self._model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,  # InternVL requires this
            low_cpu_mem_usage=True,
        ).eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

    def _load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """Load image from path or PIL."""
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                import requests
                return Image.open(BytesIO(requests.get(image_source).content)).convert("RGB")
            return Image.open(image_source).convert("RGB")
        raise ValueError(f"Unsupported image source: {type(image_source)}")

    def _build_transform(self, input_size: int = 448):
        """Build InternVL image transform."""
        try:
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
        except ImportError:
            raise ImportError("Please install torchvision")

        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find closest aspect ratio for dynamic tiling."""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height

        for ratio in target_ratios:
            target_aspect = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448):
        """InternVL's dynamic resolution preprocessing."""
        import torch

        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Generate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find best ratio
        target_aspect = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # Resize and split into tiles
        target_width = image_size * target_aspect[0]
        target_height = image_size * target_aspect[1]
        blocks = target_aspect[0] * target_aspect[1]

        resized = image.resize((target_width, target_height))
        processed_images = []

        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            processed_images.append(resized.crop(box))

        # Add thumbnail
        processed_images.append(image.resize((image_size, image_size)))

        return processed_images

    def _load_image_tensors(self, image: Image.Image) -> "torch.Tensor":
        """Process image into tensor tiles."""
        import torch

        transform = self._build_transform()
        images = self._dynamic_preprocess(image, max_num=self.max_tiles)
        pixel_values = [transform(img) for img in images]
        return torch.stack(pixel_values)

    async def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate response using InternVL3."""
        import torch

        self._load_model()
        start_time = time.time()

        try:
            image = self._load_image(request.image)
            pixel_values = self._load_image_tensors(image).to(
                dtype=self._model.dtype,
                device=self._model.device
            )

            # Build prompt with system
            prompt = request.prompt
            if request.system_prompt:
                prompt = f"{request.system_prompt}\n\n{prompt}"

            # InternVL uses <image> token
            prompt_with_image = f"<image>\n{prompt}"

            # Generation config
            generation_config = {
                "max_new_tokens": request.max_tokens or 512,
                "temperature": request.temperature or 0.7,
                "do_sample": (request.temperature or 0.7) > 0,
            }

            # Generate using InternVL's chat method
            with torch.no_grad():
                response_text = self._model.chat(
                    self._tokenizer,
                    pixel_values,
                    prompt_with_image,
                    generation_config,
                )

            latency = (time.time() - start_time) * 1000

            return VLMResponse(
                text=response_text.strip(),
                model=self.model_id,
                latency_ms=latency,
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
