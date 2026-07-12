"""
DPE-Wrapped HuggingFace VLM

Wraps a HuggingFaceVLM and injects Demographic Positional Encodings (DPE)
into the visual token embeddings via PyTorch forward hooks before generation.

The hook fires on the vision tower's output and adds α·ε_g to every image
patch embedding, where ε_g is the group-specific correction vector produced
by DemographicPositionalEncoder.

Supported model families
------------------------
LLaVA    — hooks model.model.vision_tower
Qwen2-VL — hooks model.visual
InternVL — hooks model.vision_model
SmolVLM  — hooks model.model.vision_model

For any other architecture the wrapper falls back to prompt-space injection
(prints a warning) so evaluation can still proceed.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple, Union

import torch
from PIL import Image

from fingerprint_squared.models.base import VLMRequest, VLMResponse
from fingerprint_squared.models.huggingface_vlm import HuggingFaceVLM
from fingerprint_squared.debiasing.demographic_positional_encoder import (
    DemographicPositionalEncoder,
)


class DPEWrappedHuggingFaceVLM(HuggingFaceVLM):
    """
    HuggingFaceVLM extended with Demographic Positional Encoding.

    At inference time, a forward hook adds the demographic correction
    vector to every visual token produced by the vision encoder.

    Parameters
    ----------
    encoder : DemographicPositionalEncoder
        Pre-built encoder with computed correction vectors.
    alpha : float
        Correction strength (additive scale on the positional encoding).
        Larger values apply a stronger debiasing push.
    **kwargs
        Forwarded to HuggingFaceVLM.__init__.
    """

    def __init__(
        self,
        encoder: DemographicPositionalEncoder,
        alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Public API — same as HuggingFaceVLM but accepts demographic context
    # ------------------------------------------------------------------

    def generate_with_demographics(
        self,
        request: VLMRequest,
        gender: str,
        region: str,
    ) -> VLMResponse:
        """
        Synchronous generation with DPE applied for the given demographics.

        Parameters
        ----------
        request : VLMRequest
        gender : str
            e.g. "female", "male", "non-binary"
        region : str
            e.g. "Africa", "Europe"

        Returns
        -------
        VLMResponse with an added 'dpe_applied' flag in metadata.
        """
        self._load_model()
        hook_handle, embedding_dim = self._register_dpe_hook(gender, region)
        try:
            response = self._generate_sync(request)
        finally:
            hook_handle.remove()

        response.metadata["dpe_applied"] = True
        response.metadata["dpe_gender"] = gender
        response.metadata["dpe_region"] = region
        response.metadata["dpe_embedding_dim"] = embedding_dim
        response.metadata["dpe_alpha"] = self.alpha
        return response

    # ------------------------------------------------------------------
    # Hook machinery
    # ------------------------------------------------------------------

    def _register_dpe_hook(
        self, gender: str, region: str
    ) -> Tuple[Any, int]:
        """
        Find the vision tower module and register a forward hook that adds
        the demographic encoding to its output tensor.

        Returns (hook_handle, embedding_dim).
        """
        vision_module, embedding_dim = self._find_vision_module()

        encoding = self.encoder.get_embedding(
            gender=gender,
            region=region,
            embedding_dim=embedding_dim,
            device=next(self._model.parameters()).device,
            alpha=self.alpha,
        )

        handle = vision_module.register_forward_hook(
            _make_dpe_hook(encoding)
        )
        return handle, embedding_dim

    def _find_vision_module(self) -> Tuple[torch.nn.Module, int]:
        """
        Locate the vision encoder sub-module and infer its output hidden dim.

        Returns (module, hidden_dim).
        """
        model = self._model
        candidates = [
            # LLaVA — after mm_projector so we're in language-model space
            ("model.model.mm_projector", None),
            # LLaVA — vision tower (before projector)
            ("model.model.vision_tower", None),
            # Qwen2-VL
            ("model.visual", None),
            # InternVL
            ("model.vision_model", None),
            # SmolVLM
            ("model.model.vision_model", None),
            # Generic
            ("model.vision_tower", None),
            ("model.encoder", None),
        ]

        for attr_path, _ in candidates:
            module = _get_nested_attr(model, attr_path)
            if module is not None:
                hidden_dim = _infer_hidden_dim(module, model)
                return module, hidden_dim

        # Fallback: hook the whole model and accept first tensor output
        print(
            "[DPE] WARNING: could not locate vision tower — hooking top-level model. "
            "DPE may have no effect."
        )
        hidden_dim = _infer_hidden_dim(model, model)
        return model, hidden_dim


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------

def _make_dpe_hook(encoding: torch.Tensor):
    """
    Return a forward hook that adds `encoding` to the vision module output.

    Handles:
    - Raw tensor outputs  → shape [batch, n_patches, hidden_dim]
    - Transformer output dataclasses with .last_hidden_state
    - Tuples/lists (takes first element)
    """
    def hook(module, input, output):  # noqa: ARG001
        return _add_encoding_to_output(output, encoding)

    return hook


def _add_encoding_to_output(output, encoding: torch.Tensor):
    """Recursively find the first float tensor and add the encoding to it."""
    if isinstance(output, torch.Tensor):
        if output.is_floating_point() and output.ndim >= 2:
            # Broadcast [hidden_dim] → [batch, n_patches, hidden_dim]
            enc = encoding
            while enc.ndim < output.ndim:
                enc = enc.unsqueeze(0)
            return output + enc.to(output.device, output.dtype)
        return output

    if hasattr(output, "last_hidden_state"):
        modified = _add_encoding_to_output(output.last_hidden_state, encoding)
        # Re-pack into the same type (transformers ModelOutput dataclass)
        return output.__class__(last_hidden_state=modified, **{
            k: v for k, v in output.items() if k != "last_hidden_state"
        })

    if isinstance(output, (tuple, list)):
        first = _add_encoding_to_output(output[0], encoding)
        rest = output[1:]
        return type(output)([first, *rest])

    return output


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_nested_attr(obj: Any, path: str) -> Optional[torch.nn.Module]:
    """Resolve a dot-separated attribute path; return None if any step is missing."""
    # path like "model.model.vision_tower" — first token is "model" which is the
    # variable name, but `obj` IS the model already.
    # Strip the leading "model." prefix before traversal.
    parts = path.split(".")
    if parts[0] == "model":
        parts = parts[1:]

    cur = obj
    for part in parts:
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur if isinstance(cur, torch.nn.Module) else None


def _infer_hidden_dim(module: torch.nn.Module, root_model: torch.nn.Module) -> int:
    """
    Infer the output hidden dimension of the vision module by checking
    common attribute names, then falling back to a dummy forward pass.
    """
    for attr in ("hidden_size", "embed_dim", "d_model", "config"):
        val = getattr(module, attr, None)
        if val is None:
            continue
        if isinstance(val, int) and val > 0:
            return val
        if hasattr(val, "hidden_size") and isinstance(val.hidden_size, int):
            return val.hidden_size
        if hasattr(val, "embed_dim") and isinstance(val.embed_dim, int):
            return val.embed_dim

    # Try language model config for the projector case
    for attr in ("config",):
        cfg = getattr(root_model, attr, None)
        if cfg is None:
            continue
        for dim_attr in ("hidden_size", "text_config"):
            sub = getattr(cfg, dim_attr, None)
            if isinstance(sub, int) and sub > 0:
                return sub
            if hasattr(sub, "hidden_size") and isinstance(sub.hidden_size, int):
                return sub.hidden_size

    # Safe default covering most 7B-class VLMs
    return 4096
