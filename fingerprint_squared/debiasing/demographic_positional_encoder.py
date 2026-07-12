"""
Demographic Positional Encoder (DPE)

Derives per-demographic-group bias correction vectors from baseline judge_scores,
then projects them into a VLM's visual embedding space as additive positional offsets.

Algorithm
---------
1. Load judge_scores grouped by (gender_presentation, jurisdiction_region).
2. Compute grand mean μ and per-group mean μ_g across
   (valence, stereotype_alignment, confidence).
3. Correction vector: δ_g = μ - μ_g   (push biased group toward parity).
4. Project to embedding space: ε_g = W @ δ_g, where W ∈ R^{d_model × 3} is a
   fixed orthogonal random matrix (seeded, so reproducible across runs).
5. At inference: visual_tokens += α * ε_g   (α = correction strength).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


SCORE_COLS = ("valence", "stereotype_alignment", "confidence")
DEMO_COLS = ("gender_presentation", "jurisdiction_region")


@dataclass
class DemographicGroup:
    gender: str
    region: str

    @property
    def key(self) -> str:
        return f"{self.gender}|{self.region}"

    @classmethod
    def from_key(cls, key: str) -> "DemographicGroup":
        gender, region = key.split("|", 1)
        return cls(gender=gender, region=region)


@dataclass
class GroupBiasStats:
    group: DemographicGroup
    mean_scores: np.ndarray       # shape (3,) — valence, stereo, conf
    correction_vector: np.ndarray # shape (3,) — δ_g = μ - μ_g
    n_samples: int


class DemographicPositionalEncoder:
    """
    Computes and applies demographic-conditioned positional encodings to
    VLM visual token embeddings at inference time.

    Parameters
    ----------
    group_stats : dict mapping DemographicGroup.key → GroupBiasStats
    projection_seed : int
        Seed for the fixed orthogonal projection matrix W.
    alpha : float
        Default correction strength (can be overridden per call).
    """

    def __init__(
        self,
        group_stats: Dict[str, GroupBiasStats],
        grand_mean: np.ndarray,
        projection_seed: int = 42,
        alpha: float = 1.0,
    ):
        self.group_stats = group_stats
        self.grand_mean = grand_mean
        self.projection_seed = projection_seed
        self.alpha = alpha
        self._projection_cache: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_sqlite(
        cls,
        db_paths: List[str],
        alpha: float = 1.0,
        projection_seed: int = 42,
        min_samples: int = 30,
    ) -> "DemographicPositionalEncoder":
        """
        Build encoder from one or more benchmark result databases.

        The databases must have a `judge_scores` table with columns:
        valence, stereotype_alignment, confidence, gender_presentation,
        jurisdiction_region.
        """
        rows: List[dict] = []
        for db_path in db_paths:
            rows.extend(_load_judge_scores(db_path))

        if not rows:
            raise ValueError(
                f"No judge_scores rows found in: {db_paths}"
            )

        return cls._from_rows(
            rows, alpha=alpha, projection_seed=projection_seed, min_samples=min_samples
        )

    @classmethod
    def _from_rows(
        cls,
        rows: List[dict],
        alpha: float,
        projection_seed: int,
        min_samples: int,
    ) -> "DemographicPositionalEncoder":
        # Aggregate scores per demographic group
        from collections import defaultdict

        buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
        all_scores: List[np.ndarray] = []

        for row in rows:
            gender = row.get("gender_presentation") or "unknown"
            region = row.get("jurisdiction_region") or "unknown"

            scores = np.array([
                row.get("valence") or 0.0,
                row.get("stereotype_alignment") or 0.0,
                row.get("confidence") or 0.0,
            ], dtype=np.float32)

            group = DemographicGroup(gender=gender, region=region)
            buckets[group.key].append(scores)
            all_scores.append(scores)

        grand_mean = np.mean(all_scores, axis=0)

        group_stats: Dict[str, GroupBiasStats] = {}
        for key, score_list in buckets.items():
            if len(score_list) < min_samples:
                continue
            group_mean = np.mean(score_list, axis=0)
            correction = grand_mean - group_mean  # δ_g
            group_stats[key] = GroupBiasStats(
                group=DemographicGroup.from_key(key),
                mean_scores=group_mean,
                correction_vector=correction,
                n_samples=len(score_list),
            )

        return cls(
            group_stats=group_stats,
            grand_mean=grand_mean,
            projection_seed=projection_seed,
            alpha=alpha,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_correction_vector(
        self, gender: str, region: str
    ) -> Optional[np.ndarray]:
        """Return the 3-dim bias correction vector δ_g for a demographic group."""
        key = DemographicGroup(gender=gender, region=region).key
        stats = self.group_stats.get(key)
        if stats is None:
            # Fall back to gender-only match
            for k, s in self.group_stats.items():
                if k.startswith(f"{gender}|"):
                    return s.correction_vector
            return None
        return stats.correction_vector

    def get_embedding(
        self,
        gender: str,
        region: str,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Return a [embedding_dim] tensor that should be broadcast-added to
        all visual token embeddings for a given demographic group.

        Parameters
        ----------
        gender : str
            e.g. "female", "male", "non-binary"
        region : str
            e.g. "Africa", "Europe", "Asia"
        embedding_dim : int
            Hidden dimension of the VLM's visual embedding space.
        device : torch.device, optional
        alpha : float, optional
            Override the default correction strength.

        Returns
        -------
        torch.Tensor of shape [embedding_dim]
        """
        correction = self.get_correction_vector(gender, region)
        if correction is None:
            return torch.zeros(embedding_dim, device=device)

        W = self._get_projection_matrix(embedding_dim)
        # Project 3-dim correction into embedding space
        correction_tensor = torch.tensor(correction, dtype=W.dtype)
        encoding = W @ correction_tensor  # [embedding_dim]

        strength = alpha if alpha is not None else self.alpha
        encoding = encoding * strength

        if device is not None:
            encoding = encoding.to(device)
        return encoding

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, dict]:
        """Return per-group bias stats as a plain dict for logging."""
        out = {}
        for key, stats in self.group_stats.items():
            out[key] = {
                "n_samples": stats.n_samples,
                "mean_valence": float(stats.mean_scores[0]),
                "mean_stereotype": float(stats.mean_scores[1]),
                "mean_confidence": float(stats.mean_scores[2]),
                "correction_valence": float(stats.correction_vector[0]),
                "correction_stereotype": float(stats.correction_vector[1]),
                "correction_confidence": float(stats.correction_vector[2]),
            }
        return out

    def top_biased_groups(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return the n groups with largest L2 correction magnitude."""
        ranked = sorted(
            self.group_stats.items(),
            key=lambda kv: float(np.linalg.norm(kv[1].correction_vector)),
            reverse=True,
        )
        return [(k, float(np.linalg.norm(v.correction_vector))) for k, v in ranked[:n]]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_projection_matrix(self, embedding_dim: int) -> torch.Tensor:
        """
        Return a fixed orthogonal projection W ∈ R^{embedding_dim × 3}.

        Same seed → same W, so the encoding is deterministic across runs.
        The matrix is cached per embedding_dim.
        """
        if embedding_dim in self._projection_cache:
            return self._projection_cache[embedding_dim]

        rng = torch.Generator()
        rng.manual_seed(self.projection_seed)
        # Draw a tall random matrix and orthogonalise its columns
        raw = torch.randn(embedding_dim, 3, generator=rng)
        Q, _ = torch.linalg.qr(raw)  # Q: [embedding_dim, 3]
        W = Q  # orthonormal columns

        self._projection_cache[embedding_dim] = W
        return W


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_judge_scores(db_path: str) -> List[dict]:
    """Load all scored rows from a benchmark result database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT valence, stereotype_alignment, confidence,
                   gender_presentation, jurisdiction_region
            FROM judge_scores
            WHERE valence IS NOT NULL
              AND stereotype_alignment IS NOT NULL
              AND confidence IS NOT NULL
            """
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            f"Failed to read judge_scores from {db_path}: {exc}"
        ) from exc
    finally:
        conn.close()
