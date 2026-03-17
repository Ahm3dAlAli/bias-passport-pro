"""Model registry for VLM interfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from fingerprint_squared.models.base import BaseVLM
from fingerprint_squared.models.openai_vlm import OpenAIVLM
from fingerprint_squared.models.anthropic_vlm import AnthropicVLM
from fingerprint_squared.models.google_vlm import GoogleVLM
from fingerprint_squared.models.huggingface_vlm import HuggingFaceVLM


class ModelRegistry:
    """
    Registry for VLM model interfaces.

    Provides a central registry for discovering and instantiating
    different VLM providers.

    Example:
        >>> registry = ModelRegistry()
        >>> model = registry.get_model("gpt-4o")
        >>> print(model.provider)  # "openai"
    """

    # Default model mappings
    MODEL_MAPPINGS: Dict[str, Dict[str, Any]] = {
        # OpenAI models
        "gpt-4-vision": {"provider": "openai", "model_name": "gpt-4-vision-preview"},
        "gpt-4-vision-preview": {"provider": "openai", "model_name": "gpt-4-vision-preview"},
        "gpt-4-turbo": {"provider": "openai", "model_name": "gpt-4-turbo"},
        "gpt-4o": {"provider": "openai", "model_name": "gpt-4o"},
        "gpt-4o-mini": {"provider": "openai", "model_name": "gpt-4o-mini"},

        # Anthropic models
        "claude-3-opus": {"provider": "anthropic", "model_name": "claude-3-opus-20240229"},
        "claude-3-sonnet": {"provider": "anthropic", "model_name": "claude-3-sonnet-20240229"},
        "claude-3-haiku": {"provider": "anthropic", "model_name": "claude-3-haiku-20240307"},
        "claude-3.5-sonnet": {"provider": "anthropic", "model_name": "claude-3-5-sonnet-20241022"},
        "claude-4-sonnet": {"provider": "anthropic", "model_name": "claude-sonnet-4-20250514"},
        "claude-4.5-opus": {"provider": "anthropic", "model_name": "claude-opus-4-5-20251101"},

        # Google models
        "gemini-pro-vision": {"provider": "google", "model_name": "gemini-pro-vision"},
        "gemini-1.5-pro": {"provider": "google", "model_name": "gemini-1.5-pro"},
        "gemini-1.5-flash": {"provider": "google", "model_name": "gemini-1.5-flash"},

        # HuggingFace models
        "llava-1.5-7b": {"provider": "huggingface", "model_name": "llava-hf/llava-1.5-7b-hf"},
        "llava-1.5-13b": {"provider": "huggingface", "model_name": "llava-hf/llava-1.5-13b-hf"},
        "blip2": {"provider": "huggingface", "model_name": "Salesforce/blip2-opt-2.7b"},
        "instructblip": {"provider": "huggingface", "model_name": "Salesforce/instructblip-vicuna-7b"},
    }

    PROVIDER_CLASSES: Dict[str, Type[BaseVLM]] = {
        "openai": OpenAIVLM,
        "anthropic": AnthropicVLM,
        "google": GoogleVLM,
        "huggingface": HuggingFaceVLM,
    }

    def __init__(self):
        self._custom_mappings: Dict[str, Dict[str, Any]] = {}

    def register_model(
        self,
        alias: str,
        provider: str,
        model_name: str,
        **kwargs,
    ) -> None:
        """
        Register a custom model mapping.

        Args:
            alias: Short alias for the model
            provider: Provider name
            model_name: Full model name/ID
            **kwargs: Additional model configuration
        """
        self._custom_mappings[alias] = {
            "provider": provider,
            "model_name": model_name,
            **kwargs,
        }

    def get_model(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> BaseVLM:
        """
        Get a model instance by ID.

        Args:
            model_id: Model alias or full model name
            api_key: Optional API key
            **kwargs: Additional model configuration

        Returns:
            Instantiated VLM model
        """
        # Check custom mappings first
        if model_id in self._custom_mappings:
            config = self._custom_mappings[model_id].copy()
        elif model_id in self.MODEL_MAPPINGS:
            config = self.MODEL_MAPPINGS[model_id].copy()
        else:
            # Try to infer provider from model name
            config = self._infer_config(model_id)

        provider = config.pop("provider")
        model_name = config.pop("model_name", model_id)

        # Merge with kwargs
        config.update(kwargs)

        # Get provider class
        if provider not in self.PROVIDER_CLASSES:
            raise ValueError(f"Unknown provider: {provider}")

        model_class = self.PROVIDER_CLASSES[provider]

        return model_class(
            model_name=model_name,
            api_key=api_key,
            **config,
        )

    def _infer_config(self, model_id: str) -> Dict[str, Any]:
        """Infer provider configuration from model ID."""
        model_lower = model_id.lower()

        if "gpt" in model_lower:
            return {"provider": "openai", "model_name": model_id}
        elif "claude" in model_lower:
            return {"provider": "anthropic", "model_name": model_id}
        elif "gemini" in model_lower:
            return {"provider": "google", "model_name": model_id}
        elif "/" in model_id:  # HuggingFace format
            return {"provider": "huggingface", "model_name": model_id}
        else:
            raise ValueError(
                f"Cannot infer provider for model: {model_id}. "
                "Please specify provider explicitly."
            )

    def list_models(self) -> List[str]:
        """List all available model aliases."""
        return list(set(self.MODEL_MAPPINGS.keys()) | set(self._custom_mappings.keys()))

    def list_providers(self) -> List[str]:
        """List all available providers."""
        return list(self.PROVIDER_CLASSES.keys())

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_id in self._custom_mappings:
            return self._custom_mappings[model_id]
        elif model_id in self.MODEL_MAPPINGS:
            return self.MODEL_MAPPINGS[model_id]
        else:
            return self._infer_config(model_id)


# Global registry instance
_registry = ModelRegistry()


def get_model(
    model_id: str,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseVLM:
    """
    Get a model instance from the global registry.

    Args:
        model_id: Model alias or full model name
        api_key: Optional API key
        **kwargs: Additional model configuration

    Returns:
        Instantiated VLM model
    """
    return _registry.get_model(model_id, api_key=api_key, **kwargs)


def register_model(
    alias: str,
    provider: str,
    model_name: str,
    **kwargs,
) -> None:
    """Register a custom model in the global registry."""
    _registry.register_model(alias, provider, model_name, **kwargs)


def list_models() -> List[str]:
    """List all available models."""
    return _registry.list_models()
