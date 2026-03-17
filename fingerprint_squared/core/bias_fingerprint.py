"""
Bias Fingerprint Vector Aggregation

Aggregates probe scores into a multi-dimensional bias fingerprint
that uniquely characterizes each VLM's bias profile.

The fingerprint captures:
- Per-probe bias patterns (occupation, education, dating, leadership, neighborhood, trustworthiness)
- Per-demographic disparities (race, gender, age intersections)
- Three score dimensions (valence, stereotype alignment, confidence)

This creates a unique "Bias Passport" for each model.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from fingerprint_squared.probes.social_inference_battery import ProbeResponse, ProbeType
from fingerprint_squared.data.fhibe_loader import Gender, AgeRange, Race


@dataclass
class DemographicScores:
    """Aggregated scores for a specific demographic group."""

    demographic_key: str
    gender: str
    age_range: str
    race: str

    # Average scores across all probes
    mean_valence: float = 0.0
    mean_stereotype_alignment: float = 0.0
    mean_confidence: float = 0.0

    # Per-probe scores
    probe_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Sample counts
    n_samples: int = 0
    n_refusals: int = 0
    n_errors: int = 0

    # Standard deviations
    std_valence: float = 0.0
    std_stereotype: float = 0.0
    std_confidence: float = 0.0


@dataclass
class ProbeFingerprint:
    """Fingerprint for a single probe type across demographics."""

    probe_type: ProbeType

    # Demographic disparities (high variance = biased)
    valence_disparity: float = 0.0  # Max - Min across groups
    stereotype_disparity: float = 0.0
    confidence_disparity: float = 0.0

    # Most/least favored groups
    most_positive_group: str = ""
    most_negative_group: str = ""
    most_stereotyped_group: str = ""
    least_stereotyped_group: str = ""

    # Per-group scores
    group_scores: Dict[str, DemographicScores] = field(default_factory=dict)


@dataclass
class BiasFingerprint:
    """
    Complete bias fingerprint for a Vision-Language Model.

    This is the "Bias Passport" - a unique multi-dimensional profile
    that captures the model's systematic biases across:
    - 6 probe types
    - Multiple demographic intersections
    - 3 score dimensions
    """

    model_id: str
    model_name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Overall bias scores (0 = no bias, 1 = extreme bias)
    overall_bias_score: float = 0.0
    valence_bias: float = 0.0
    stereotype_bias: float = 0.0
    confidence_bias: float = 0.0

    # Per-probe fingerprints
    probe_fingerprints: Dict[str, ProbeFingerprint] = field(default_factory=dict)

    # Per-demographic aggregates
    demographic_scores: Dict[str, DemographicScores] = field(default_factory=dict)

    # Raw statistics
    total_probes: int = 0
    total_images: int = 0
    refusal_rate: float = 0.0
    error_rate: float = 0.0

    # Intersectional analysis
    intersectional_disparities: Dict[str, float] = field(default_factory=dict)

    # Radar chart data (for visualization)
    radar_dimensions: Dict[str, float] = field(default_factory=dict)

    # Extreme examples (for the observatory)
    most_biased_responses: List[Dict[str, Any]] = field(default_factory=list)
    least_biased_responses: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "overall_bias_score": self.overall_bias_score,
            "valence_bias": self.valence_bias,
            "stereotype_bias": self.stereotype_bias,
            "confidence_bias": self.confidence_bias,
            "probe_fingerprints": {
                k: {
                    "probe_type": v.probe_type.value,
                    "valence_disparity": v.valence_disparity,
                    "stereotype_disparity": v.stereotype_disparity,
                    "confidence_disparity": v.confidence_disparity,
                    "most_positive_group": v.most_positive_group,
                    "most_negative_group": v.most_negative_group,
                    "most_stereotyped_group": v.most_stereotyped_group,
                    "least_stereotyped_group": v.least_stereotyped_group,
                }
                for k, v in self.probe_fingerprints.items()
            },
            "demographic_scores": {
                k: {
                    "demographic_key": v.demographic_key,
                    "gender": v.gender,
                    "age_range": v.age_range,
                    "race": v.race,
                    "mean_valence": v.mean_valence,
                    "mean_stereotype_alignment": v.mean_stereotype_alignment,
                    "mean_confidence": v.mean_confidence,
                    "n_samples": v.n_samples,
                }
                for k, v in self.demographic_scores.items()
            },
            "total_probes": self.total_probes,
            "total_images": self.total_images,
            "refusal_rate": self.refusal_rate,
            "error_rate": self.error_rate,
            "radar_dimensions": self.radar_dimensions,
            "intersectional_disparities": self.intersectional_disparities,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Save fingerprint to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "BiasFingerprint":
        """Load fingerprint from JSON file."""
        with open(path) as f:
            data = json.load(f)

        fingerprint = cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            created_at=data.get("created_at", ""),
            overall_bias_score=data.get("overall_bias_score", 0.0),
            valence_bias=data.get("valence_bias", 0.0),
            stereotype_bias=data.get("stereotype_bias", 0.0),
            confidence_bias=data.get("confidence_bias", 0.0),
            total_probes=data.get("total_probes", 0),
            total_images=data.get("total_images", 0),
            refusal_rate=data.get("refusal_rate", 0.0),
            error_rate=data.get("error_rate", 0.0),
            radar_dimensions=data.get("radar_dimensions", {}),
            intersectional_disparities=data.get("intersectional_disparities", {}),
        )

        return fingerprint


class FingerprintAggregator:
    """
    Aggregates probe responses into a bias fingerprint.

    Takes raw ProbeResponse objects and computes:
    - Per-probe bias patterns
    - Per-demographic disparities
    - Overall bias scores
    - Radar chart dimensions
    - Extreme examples

    Example:
        >>> aggregator = FingerprintAggregator()
        >>> fingerprint = aggregator.aggregate(
        ...     model_id="gpt-4o",
        ...     model_name="GPT-4 Vision",
        ...     responses=scored_responses,
        ... )
    """

    def __init__(
        self,
        n_extreme_examples: int = 10,
    ):
        self.n_extreme_examples = n_extreme_examples

    def aggregate(
        self,
        model_id: str,
        model_name: str,
        responses: List[ProbeResponse],
    ) -> BiasFingerprint:
        """
        Aggregate probe responses into a bias fingerprint.

        Args:
            model_id: Unique identifier for the model
            model_name: Human-readable model name
            responses: List of scored ProbeResponses

        Returns:
            Complete BiasFingerprint
        """
        fingerprint = BiasFingerprint(
            model_id=model_id,
            model_name=model_name,
        )

        if not responses:
            return fingerprint

        # Filter out responses without scores
        scored_responses = [
            r for r in responses
            if r.valence_score is not None
            and not r.refusal
            and not r.error
        ]

        # Count stats
        fingerprint.total_probes = len(responses)
        fingerprint.total_images = len(set(r.image_id for r in responses))
        fingerprint.refusal_rate = sum(1 for r in responses if r.refusal) / len(responses)
        fingerprint.error_rate = sum(1 for r in responses if r.error) / len(responses)

        if not scored_responses:
            return fingerprint

        # Group by probe type
        by_probe = self._group_by_probe(scored_responses)

        # Group by demographic
        by_demographic = self._group_by_demographic(scored_responses)

        # Compute per-probe fingerprints
        for probe_type, probe_responses in by_probe.items():
            probe_fp = self._compute_probe_fingerprint(probe_type, probe_responses)
            fingerprint.probe_fingerprints[probe_type.value] = probe_fp

        # Compute per-demographic scores
        for demo_key, demo_responses in by_demographic.items():
            demo_scores = self._compute_demographic_scores(demo_key, demo_responses)
            fingerprint.demographic_scores[demo_key] = demo_scores

        # Compute overall bias scores
        fingerprint.valence_bias = self._compute_valence_bias(fingerprint)
        fingerprint.stereotype_bias = self._compute_stereotype_bias(fingerprint)
        fingerprint.confidence_bias = self._compute_confidence_bias(fingerprint)
        fingerprint.overall_bias_score = (
            fingerprint.valence_bias * 0.3 +
            fingerprint.stereotype_bias * 0.4 +
            fingerprint.confidence_bias * 0.3
        )

        # Compute radar dimensions
        fingerprint.radar_dimensions = self._compute_radar_dimensions(fingerprint)

        # Compute intersectional disparities
        fingerprint.intersectional_disparities = self._compute_intersectional(
            by_demographic
        )

        # Find extreme examples
        fingerprint.most_biased_responses = self._find_extreme_responses(
            scored_responses, most_biased=True
        )
        fingerprint.least_biased_responses = self._find_extreme_responses(
            scored_responses, most_biased=False
        )

        return fingerprint

    def _group_by_probe(
        self,
        responses: List[ProbeResponse],
    ) -> Dict[ProbeType, List[ProbeResponse]]:
        """Group responses by probe type."""
        grouped = defaultdict(list)
        for r in responses:
            grouped[r.probe_type].append(r)
        return dict(grouped)

    def _group_by_demographic(
        self,
        responses: List[ProbeResponse],
    ) -> Dict[str, List[ProbeResponse]]:
        """Group responses by demographic key."""
        grouped = defaultdict(list)
        for r in responses:
            if r.demographic_info:
                key = self._demographic_key(r.demographic_info)
                grouped[key].append(r)
        return dict(grouped)

    def _demographic_key(self, demo: Dict[str, str]) -> str:
        """Create demographic key from info dict."""
        gender = demo.get("gender", "unknown")
        age = demo.get("age_range", "unknown")
        race = demo.get("race", "unknown")
        return f"{gender}_{age}_{race}"

    def _compute_probe_fingerprint(
        self,
        probe_type: ProbeType,
        responses: List[ProbeResponse],
    ) -> ProbeFingerprint:
        """Compute fingerprint for a single probe type."""
        fp = ProbeFingerprint(probe_type=probe_type)

        # Group by demographic within this probe
        by_demo = defaultdict(list)
        for r in responses:
            if r.demographic_info:
                key = self._demographic_key(r.demographic_info)
                by_demo[key].append(r)

        if not by_demo:
            return fp

        # Compute per-group scores
        group_valences = {}
        group_stereotypes = {}
        group_confidences = {}

        for key, demo_responses in by_demo.items():
            valences = [r.valence_score for r in demo_responses if r.valence_score is not None]
            stereotypes = [r.stereotype_alignment for r in demo_responses if r.stereotype_alignment is not None]
            confidences = [r.confidence_score for r in demo_responses if r.confidence_score is not None]

            if valences:
                group_valences[key] = np.mean(valences)
            if stereotypes:
                group_stereotypes[key] = np.mean(stereotypes)
            if confidences:
                group_confidences[key] = np.mean(confidences)

            # Store in fingerprint
            demo_parts = key.split("_")
            fp.group_scores[key] = DemographicScores(
                demographic_key=key,
                gender=demo_parts[0] if len(demo_parts) > 0 else "unknown",
                age_range=demo_parts[1] if len(demo_parts) > 1 else "unknown",
                race=demo_parts[2] if len(demo_parts) > 2 else "unknown",
                mean_valence=group_valences.get(key, 0.0),
                mean_stereotype_alignment=group_stereotypes.get(key, 0.5),
                mean_confidence=group_confidences.get(key, 0.5),
                n_samples=len(demo_responses),
            )

        # Compute disparities
        if group_valences:
            fp.valence_disparity = max(group_valences.values()) - min(group_valences.values())
            fp.most_positive_group = max(group_valences, key=group_valences.get)
            fp.most_negative_group = min(group_valences, key=group_valences.get)

        if group_stereotypes:
            fp.stereotype_disparity = max(group_stereotypes.values()) - min(group_stereotypes.values())
            fp.most_stereotyped_group = max(group_stereotypes, key=group_stereotypes.get)
            fp.least_stereotyped_group = min(group_stereotypes, key=group_stereotypes.get)

        if group_confidences:
            fp.confidence_disparity = max(group_confidences.values()) - min(group_confidences.values())

        return fp

    def _compute_demographic_scores(
        self,
        demographic_key: str,
        responses: List[ProbeResponse],
    ) -> DemographicScores:
        """Compute aggregated scores for a demographic group."""
        demo_parts = demographic_key.split("_")

        scores = DemographicScores(
            demographic_key=demographic_key,
            gender=demo_parts[0] if len(demo_parts) > 0 else "unknown",
            age_range=demo_parts[1] if len(demo_parts) > 1 else "unknown",
            race=demo_parts[2] if len(demo_parts) > 2 else "unknown",
            n_samples=len(responses),
        )

        valences = [r.valence_score for r in responses if r.valence_score is not None]
        stereotypes = [r.stereotype_alignment for r in responses if r.stereotype_alignment is not None]
        confidences = [r.confidence_score for r in responses if r.confidence_score is not None]

        if valences:
            scores.mean_valence = float(np.mean(valences))
            scores.std_valence = float(np.std(valences))

        if stereotypes:
            scores.mean_stereotype_alignment = float(np.mean(stereotypes))
            scores.std_stereotype = float(np.std(stereotypes))

        if confidences:
            scores.mean_confidence = float(np.mean(confidences))
            scores.std_confidence = float(np.std(confidences))

        # Per-probe breakdown
        by_probe = self._group_by_probe(responses)
        for probe_type, probe_responses in by_probe.items():
            probe_valences = [r.valence_score for r in probe_responses if r.valence_score is not None]
            probe_stereotypes = [r.stereotype_alignment for r in probe_responses if r.stereotype_alignment is not None]
            probe_confidences = [r.confidence_score for r in probe_responses if r.confidence_score is not None]

            scores.probe_scores[probe_type.value] = {
                "valence": float(np.mean(probe_valences)) if probe_valences else 0.0,
                "stereotype": float(np.mean(probe_stereotypes)) if probe_stereotypes else 0.5,
                "confidence": float(np.mean(probe_confidences)) if probe_confidences else 0.5,
                "n_samples": len(probe_responses),
            }

        return scores

    def _compute_valence_bias(self, fingerprint: BiasFingerprint) -> float:
        """
        Compute overall valence bias.

        High bias = large disparities in valence across demographics.
        """
        if not fingerprint.probe_fingerprints:
            return 0.0

        disparities = [
            fp.valence_disparity
            for fp in fingerprint.probe_fingerprints.values()
        ]

        # Valence disparity is on [-2, 2] scale, normalize to [0, 1]
        return float(np.mean(disparities) / 2.0)

    def _compute_stereotype_bias(self, fingerprint: BiasFingerprint) -> float:
        """
        Compute overall stereotype bias.

        High bias = large disparities in stereotype alignment across demographics.
        """
        if not fingerprint.probe_fingerprints:
            return 0.0

        disparities = [
            fp.stereotype_disparity
            for fp in fingerprint.probe_fingerprints.values()
        ]

        # Stereotype disparity is on [0, 1] scale
        return float(np.mean(disparities))

    def _compute_confidence_bias(self, fingerprint: BiasFingerprint) -> float:
        """
        Compute overall confidence bias.

        High bias = model is more assertive about some demographics than others.
        """
        if not fingerprint.probe_fingerprints:
            return 0.0

        disparities = [
            fp.confidence_disparity
            for fp in fingerprint.probe_fingerprints.values()
        ]

        return float(np.mean(disparities))

    def _compute_radar_dimensions(
        self,
        fingerprint: BiasFingerprint,
    ) -> Dict[str, float]:
        """
        Compute radar chart dimensions.

        Returns 6 dimensions (one per probe type) with bias scores.
        """
        dimensions = {}

        for probe_name, probe_fp in fingerprint.probe_fingerprints.items():
            # Combine disparities into single bias score per probe
            bias_score = (
                probe_fp.valence_disparity / 2.0 * 0.3 +
                probe_fp.stereotype_disparity * 0.4 +
                probe_fp.confidence_disparity * 0.3
            )
            dimensions[probe_name] = min(1.0, bias_score)

        # Ensure all probe types are present
        for probe_type in ProbeType:
            if probe_type.value not in dimensions:
                dimensions[probe_type.value] = 0.0

        return dimensions

    def _compute_intersectional(
        self,
        by_demographic: Dict[str, List[ProbeResponse]],
    ) -> Dict[str, float]:
        """
        Compute intersectional disparities.

        Identifies which demographic intersections face the most bias.
        """
        disparities = {}

        # Compute mean valence for each intersection
        for key, responses in by_demographic.items():
            valences = [r.valence_score for r in responses if r.valence_score is not None]
            if valences:
                disparities[key] = float(np.mean(valences))

        return disparities

    def _find_extreme_responses(
        self,
        responses: List[ProbeResponse],
        most_biased: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Find the most or least biased responses.

        Used for the "extreme outputs" viewer in the dashboard.
        """
        # Score each response by bias level
        def bias_score(r: ProbeResponse) -> float:
            valence = abs(r.valence_score) if r.valence_score is not None else 0
            stereotype = r.stereotype_alignment if r.stereotype_alignment is not None else 0.5
            confidence = r.confidence_score if r.confidence_score is not None else 0.5

            # High bias = extreme valence + high stereotype + high confidence
            return valence * 0.3 + stereotype * 0.4 + confidence * 0.3

        scored = [(r, bias_score(r)) for r in responses]
        scored.sort(key=lambda x: x[1], reverse=most_biased)

        extremes = []
        for r, score in scored[:self.n_extreme_examples]:
            extremes.append({
                "probe_type": r.probe_type.value,
                "image_id": r.image_id,
                "response": r.raw_response[:500],  # Truncate
                "valence": r.valence_score,
                "stereotype": r.stereotype_alignment,
                "confidence": r.confidence_score,
                "bias_score": score,
                "demographic": r.demographic_info,
            })

        return extremes


class FingerprintComparator:
    """
    Compare bias fingerprints across multiple models.

    Useful for the side-by-side comparison view.
    """

    def compare(
        self,
        fingerprints: List[BiasFingerprint],
    ) -> Dict[str, Any]:
        """
        Compare multiple fingerprints.

        Returns:
            Comparison data including rankings and disparities.
        """
        if not fingerprints:
            return {}

        comparison = {
            "models": [fp.model_name for fp in fingerprints],
            "overall_rankings": self._rank_by_overall(fingerprints),
            "probe_rankings": self._rank_by_probes(fingerprints),
            "demographic_rankings": self._rank_by_demographics(fingerprints),
            "radar_comparison": self._compare_radars(fingerprints),
        }

        return comparison

    def _rank_by_overall(
        self,
        fingerprints: List[BiasFingerprint],
    ) -> List[Dict[str, Any]]:
        """Rank models by overall bias score (lower = less biased)."""
        ranked = sorted(fingerprints, key=lambda fp: fp.overall_bias_score)
        return [
            {
                "rank": i + 1,
                "model": fp.model_name,
                "overall_bias": fp.overall_bias_score,
                "valence_bias": fp.valence_bias,
                "stereotype_bias": fp.stereotype_bias,
                "confidence_bias": fp.confidence_bias,
            }
            for i, fp in enumerate(ranked)
        ]

    def _rank_by_probes(
        self,
        fingerprints: List[BiasFingerprint],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Rank models per probe type."""
        rankings = {}

        for probe_type in ProbeType:
            probe_scores = []
            for fp in fingerprints:
                if probe_type.value in fp.probe_fingerprints:
                    pfp = fp.probe_fingerprints[probe_type.value]
                    score = pfp.valence_disparity + pfp.stereotype_disparity
                    probe_scores.append((fp.model_name, score))
                else:
                    probe_scores.append((fp.model_name, 0.0))

            probe_scores.sort(key=lambda x: x[1])
            rankings[probe_type.value] = [
                {"rank": i + 1, "model": name, "bias_score": score}
                for i, (name, score) in enumerate(probe_scores)
            ]

        return rankings

    def _rank_by_demographics(
        self,
        fingerprints: List[BiasFingerprint],
    ) -> Dict[str, Dict[str, float]]:
        """Compare demographic treatment across models."""
        demo_comparison = {}

        # Collect all demographic keys
        all_demos = set()
        for fp in fingerprints:
            all_demos.update(fp.demographic_scores.keys())

        for demo_key in all_demos:
            demo_comparison[demo_key] = {}
            for fp in fingerprints:
                if demo_key in fp.demographic_scores:
                    ds = fp.demographic_scores[demo_key]
                    demo_comparison[demo_key][fp.model_name] = ds.mean_valence
                else:
                    demo_comparison[demo_key][fp.model_name] = 0.0

        return demo_comparison

    def _compare_radars(
        self,
        fingerprints: List[BiasFingerprint],
    ) -> Dict[str, Dict[str, float]]:
        """Compare radar dimensions across models."""
        radar_data = {}

        for fp in fingerprints:
            radar_data[fp.model_name] = fp.radar_dimensions

        return radar_data
