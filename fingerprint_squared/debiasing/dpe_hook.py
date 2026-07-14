"""
DPE forward-hook controller.

Injects Demographic Positional Encodings into a VLM's visual token embeddings
at inference time, independent of how the model was loaded. Works with any
`torch.nn.Module` VLM — including the per-model client classes in
`scripts/run_fhibe_benchmark.py` (LlavaClient, InternVLClient, IDEFICSClient, …).

Usage
-----
    controller = DPEHookController(encoder, alpha=1.5)
    module, path = find_vision_module(client.model)
    handle = module.register_forward_hook(controller)
    ...
    controller.set_demographics("female", "Africa")
    out = client.generate(pil_image, prompt)   # hook fires, adds ε_g
    controller.clear()
    ...
    handle.remove()

The encoding dimension is inferred from the hooked module's output tensor on the
fly, so it adapts to whichever vision encoder the model uses (CLIP 1024, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from fingerprint_squared.debiasing.demographic_positional_encoder import (
    DemographicPositionalEncoder,
)


# Candidate attribute paths to the vision encoder, by model family.
VISION_MODULE_PATHS = [
    "vision_tower",          # LLaVA / LLaVA-Next
    "vision_model",          # InternVL, Idefics2 (top-level), SigLIP-based
    "model.vision_model",    # Idefics2 / Idefics3 (nested)
    "model.vision_tower",    # some LLaVA packagings
    "visual",                # Qwen2-VL
    "model.visual",          # Qwen2-VL (nested)
    "vpm",                   # MiniCPM-V
]


def find_vision_module(model: torch.nn.Module) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    """
    Locate the vision-encoder submodule of a loaded VLM.

    Returns (module, dotted_path) or (None, None) if nothing matched.
    """
    for path in VISION_MODULE_PATHS:
        cur: Any = model
        ok = True
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                ok = False
                break
        if ok and isinstance(cur, torch.nn.Module):
            return cur, path
    return None, None


class DPEHookController:
    """
    A stateful forward hook. Register it once on the vision module; before each
    generation set the current demographic group, and the hook adds the
    group-specific encoding to every visual token. Set no group (clear) to pass
    the output through untouched (recovers the baseline).
    """

    def __init__(self, encoder: DemographicPositionalEncoder, alpha: float = 1.0):
        self.encoder = encoder
        self.alpha = alpha
        self._gender: Optional[str] = None
        self._region: Optional[str] = None
        self._cache: Dict[Tuple[str, str, int], torch.Tensor] = {}
        self.n_applied = 0        # diagnostic: how many tokens-tensors we modified
        self.last_norm = 0.0      # diagnostic: L2 norm of the last encoding added

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def set_demographics(self, gender: str, region: str) -> None:
        self._gender = gender or "unknown"
        self._region = region or "unknown"

    def clear(self) -> None:
        self._gender = None
        self._region = None

    @property
    def active(self) -> bool:
        return self._gender is not None

    # ------------------------------------------------------------------
    # Hook entry point
    # ------------------------------------------------------------------

    def __call__(self, module: torch.nn.Module, inputs: Any, output: Any) -> Any:
        if self._gender is None:
            return output
        return self._add_encoding(output)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _encoding_for(self, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (self._gender, self._region, dim)
        enc = self._cache.get(key)
        if enc is None:
            enc = self.encoder.get_embedding(
                gender=self._gender,
                region=self._region,
                embedding_dim=dim,
                alpha=self.alpha,
            )
            self._cache[key] = enc
        return enc.to(device=device, dtype=dtype)

    def _add_encoding(self, output: Any) -> Any:
        # Raw tensor of visual tokens: [..., dim]
        if isinstance(output, torch.Tensor):
            if output.is_floating_point() and output.ndim >= 2:
                dim = output.shape[-1]
                enc = self._encoding_for(dim, output.device, output.dtype)
                self.last_norm = float(torch.linalg.vector_norm(enc).item())
                self.n_applied += 1
                while enc.ndim < output.ndim:
                    enc = enc.unsqueeze(0)
                return output + enc
            return output

        # transformers ModelOutput with .last_hidden_state
        if hasattr(output, "last_hidden_state"):
            modified = self._add_encoding(output.last_hidden_state)
            try:
                output.last_hidden_state = modified
                return output
            except Exception:
                return output

        # Tuple/list: modify the first tensor element (hidden states)
        if isinstance(output, (tuple, list)):
            if len(output) == 0:
                return output
            first = self._add_encoding(output[0])
            return type(output)([first, *list(output[1:])])

        return output
