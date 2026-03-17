"""VLM model interfaces and adapters."""

from fingerprint_squared.models.base import BaseVLM, VLMRequest, VLMResponse
from fingerprint_squared.models.openai_vlm import OpenAIVLM
from fingerprint_squared.models.anthropic_vlm import AnthropicVLM
from fingerprint_squared.models.google_vlm import GoogleVLM
from fingerprint_squared.models.huggingface_vlm import HuggingFaceVLM
from fingerprint_squared.models.registry import ModelRegistry, get_model

# SOTA Open Source VLMs
from fingerprint_squared.models.qwen_vlm import Qwen25VLM, Qwen3VLM
from fingerprint_squared.models.internvl_vlm import InternVL3VLM
from fingerprint_squared.models.llama_vision_vlm import LlamaVisionVLM, SmolVLM

# OpenRouter for proprietary models
from fingerprint_squared.models.openrouter_vlm import (
    OpenRouterVLM,
    MultiProviderVLM,
    OPENROUTER_MODELS,
)

__all__ = [
    # Base
    "BaseVLM",
    "VLMRequest",
    "VLMResponse",

    # Direct API adapters
    "OpenAIVLM",
    "AnthropicVLM",
    "GoogleVLM",
    "HuggingFaceVLM",

    # SOTA Open Source
    "Qwen25VLM",
    "Qwen3VLM",
    "InternVL3VLM",
    "LlamaVisionVLM",
    "SmolVLM",

    # OpenRouter (unified proprietary access)
    "OpenRouterVLM",
    "MultiProviderVLM",
    "OPENROUTER_MODELS",

    # Registry
    "ModelRegistry",
    "get_model",
]
