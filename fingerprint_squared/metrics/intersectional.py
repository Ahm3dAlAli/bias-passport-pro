"""
Intersectional bias analysis for VLM evaluation.

This module implements intersectional fairness analysis, measuring how
biases compound when multiple protected attributes intersect.

References:
    - Crenshaw, "Demarginalizing the Intersection of Race and Sex" (1989)
    - Buolamwini & Gebru, "Gender Shades" (2018)
    - Wang et al., "Towards Intersectionality in Machine Learning" (2022)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats


@dataclass
class IntersectionalGroup:
    """Represents an intersectional demographic group."""

    attributes: Dict[str, str]  # e.g., {"gender": "female", "race": "black"}
    sample_count: int = 0
    performance_metric: Optional[float] = None
    fairness_metric: Optional[float] = None

    @property
    def name(self) -> str:
        """Get readable name for the group."""
        return " + ".join(f"{k}={v}" for k, v in sorted(self.attributes.items()))

    @property
    def dimensionality(self) -> int:
        """Number of attributes in this intersection."""
        return len(self.attributes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attributes": self.attributes,
            "name": self.name,
            "sample_count": self.sample_count,
            "performance_metric": self.performance_metric,
            "fairness_metric": self.fairness_metric,
            "dimensionality": self.dimensionality,
        }


@dataclass
class IntersectionalAnalysisResult:
    """Results of intersectional bias analysis."""

    groups: List[IntersectionalGroup]
    worst_performing_group: Optional[IntersectionalGroup]
    best_performing_group: Optional[IntersectionalGroup]
    intersectional_gap: float  # Gap between best and worst
    amplification_scores: Dict[str, float]  # How much bias amplifies at intersections
    interaction_effects: Dict[str, float]  # Interaction between attributes
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "worst_performing_group": self.worst_performing_group.to_dict() if self.worst_performing_group else None,
            "best_performing_group": self.best_performing_group.to_dict() if self.best_performing_group else None,
            "intersectional_gap": self.intersectional_gap,
            "amplification_scores": self.amplification_scores,
            "interaction_effects": self.interaction_effects,
            "details": self.details,
        }


class IntersectionalAnalyzer:
    """
    Analyzer for intersectional bias in VLM outputs.

    Examines how biases compound when multiple protected attributes
    intersect, following the intersectionality framework.

    Attributes:
        protected_attributes: List of attribute names to analyze
        min_group_size: Minimum samples required for group analysis

    Example:
        >>> analyzer = IntersectionalAnalyzer(["gender", "race", "age"])
        >>> data = [
        ...     {"gender": "female", "race": "black", "score": 0.7},
        ...     {"gender": "male", "race": "white", "score": 0.9},
        ... ]
        >>> result = analyzer.analyze(data, metric_key="score")
    """

    def __init__(
        self,
        protected_attributes: List[str],
        min_group_size: int = 30,
        max_intersection_depth: int = 3,
    ):
        self.protected_attributes = protected_attributes
        self.min_group_size = min_group_size
        self.max_intersection_depth = max_intersection_depth

    def analyze(
        self,
        data: List[Dict[str, Any]],
        metric_key: str = "score",
        higher_is_better: bool = True,
    ) -> IntersectionalAnalysisResult:
        """
        Perform intersectional analysis on data.

        Args:
            data: List of samples with attributes and metrics
            metric_key: Key for the metric to analyze
            higher_is_better: Whether higher metric values are better

        Returns:
            IntersectionalAnalysisResult with comprehensive analysis
        """
        # Build intersectional groups
        groups = self._build_groups(data, metric_key)

        if not groups:
            return IntersectionalAnalysisResult(
                groups=[],
                worst_performing_group=None,
                best_performing_group=None,
                intersectional_gap=0.0,
                amplification_scores={},
                interaction_effects={},
            )

        # Find best and worst performing groups
        valid_groups = [g for g in groups if g.performance_metric is not None]

        if not valid_groups:
            return IntersectionalAnalysisResult(
                groups=groups,
                worst_performing_group=None,
                best_performing_group=None,
                intersectional_gap=0.0,
                amplification_scores={},
                interaction_effects={},
            )

        if higher_is_better:
            worst_group = min(valid_groups, key=lambda g: g.performance_metric)
            best_group = max(valid_groups, key=lambda g: g.performance_metric)
        else:
            worst_group = max(valid_groups, key=lambda g: g.performance_metric)
            best_group = min(valid_groups, key=lambda g: g.performance_metric)

        intersectional_gap = abs(
            best_group.performance_metric - worst_group.performance_metric
        )

        # Compute amplification scores
        amplification_scores = self._compute_amplification(
            groups, data, metric_key, higher_is_better
        )

        # Compute interaction effects
        interaction_effects = self._compute_interactions(
            groups, data, metric_key
        )

        return IntersectionalAnalysisResult(
            groups=groups,
            worst_performing_group=worst_group,
            best_performing_group=best_group,
            intersectional_gap=intersectional_gap,
            amplification_scores=amplification_scores,
            interaction_effects=interaction_effects,
            details={
                "total_samples": len(data),
                "n_groups_analyzed": len(valid_groups),
                "attributes_analyzed": self.protected_attributes,
            },
        )

    def _build_groups(
        self,
        data: List[Dict[str, Any]],
        metric_key: str,
    ) -> List[IntersectionalGroup]:
        """Build all intersectional groups from data."""
        groups = []
        group_data: Dict[str, List[float]] = {}

        # Generate all possible intersections up to max depth
        for depth in range(1, min(self.max_intersection_depth, len(self.protected_attributes)) + 1):
            for attr_combo in combinations(self.protected_attributes, depth):
                # Find all unique value combinations for these attributes
                value_combos = self._get_value_combinations(data, list(attr_combo))

                for values in value_combos:
                    attrs = dict(zip(attr_combo, values))
                    group_key = str(sorted(attrs.items()))

                    # Collect metrics for this group
                    metrics = []
                    for sample in data:
                        if all(sample.get(k) == v for k, v in attrs.items()):
                            if metric_key in sample and sample[metric_key] is not None:
                                metrics.append(sample[metric_key])

                    if len(metrics) >= self.min_group_size:
                        group = IntersectionalGroup(
                            attributes=attrs,
                            sample_count=len(metrics),
                            performance_metric=np.mean(metrics),
                        )
                        groups.append(group)
                        group_data[group_key] = metrics

        return groups

    def _get_value_combinations(
        self,
        data: List[Dict[str, Any]],
        attributes: List[str],
    ) -> List[Tuple]:
        """Get all unique value combinations for given attributes."""
        seen = set()
        combinations_list = []

        for sample in data:
            values = tuple(sample.get(attr) for attr in attributes)
            if None not in values and values not in seen:
                seen.add(values)
                combinations_list.append(values)

        return combinations_list

    def _compute_amplification(
        self,
        groups: List[IntersectionalGroup],
        data: List[Dict[str, Any]],
        metric_key: str,
        higher_is_better: bool,
    ) -> Dict[str, float]:
        """
        Compute bias amplification at intersections.

        Measures how much worse the intersectional group performs
        compared to what we'd expect from the marginal groups.
        """
        amplification = {}

        # Get marginal performance for each attribute value
        marginal_performance = {}
        for attr in self.protected_attributes:
            values = set(sample.get(attr) for sample in data if sample.get(attr) is not None)
            for value in values:
                metrics = [
                    sample[metric_key]
                    for sample in data
                    if sample.get(attr) == value and metric_key in sample
                ]
                if metrics:
                    marginal_performance[(attr, value)] = np.mean(metrics)

        # Compute amplification for multi-attribute groups
        for group in groups:
            if group.dimensionality < 2:
                continue

            # Expected performance from marginals (assuming independence)
            overall_mean = np.mean([
                sample[metric_key]
                for sample in data
                if metric_key in sample and sample[metric_key] is not None
            ])

            expected_deviations = []
            for attr, value in group.attributes.items():
                if (attr, value) in marginal_performance:
                    marginal = marginal_performance[(attr, value)]
                    expected_deviations.append(marginal - overall_mean)

            if expected_deviations and group.performance_metric is not None:
                # Expected under independence
                expected_perf = overall_mean + sum(expected_deviations)
                actual_perf = group.performance_metric

                # Amplification: how much worse than expected
                if higher_is_better:
                    amp = expected_perf - actual_perf
                else:
                    amp = actual_perf - expected_perf

                amplification[group.name] = amp

        return amplification

    def _compute_interactions(
        self,
        groups: List[IntersectionalGroup],
        data: List[Dict[str, Any]],
        metric_key: str,
    ) -> Dict[str, float]:
        """
        Compute statistical interaction effects between attributes.

        Uses a simple interaction analysis to detect non-additive effects.
        """
        interactions = {}

        # For each pair of attributes
        for attr1, attr2 in combinations(self.protected_attributes, 2):
            # Get all 2x2 (or higher) combinations
            values1 = list(set(s.get(attr1) for s in data if s.get(attr1) is not None))
            values2 = list(set(s.get(attr2) for s in data if s.get(attr2) is not None))

            if len(values1) < 2 or len(values2) < 2:
                continue

            # Build matrix of means
            means = {}
            for v1 in values1:
                for v2 in values2:
                    metrics = [
                        s[metric_key]
                        for s in data
                        if s.get(attr1) == v1 and s.get(attr2) == v2
                        and metric_key in s and s[metric_key] is not None
                    ]
                    if len(metrics) >= 5:  # Minimum for interaction
                        means[(v1, v2)] = np.mean(metrics)

            # Compute interaction effect (simplified)
            if len(means) >= 4:
                # Take first 2x2 submatrix
                v1_pair = values1[:2]
                v2_pair = values2[:2]

                try:
                    a = means.get((v1_pair[0], v2_pair[0]), 0)
                    b = means.get((v1_pair[0], v2_pair[1]), 0)
                    c = means.get((v1_pair[1], v2_pair[0]), 0)
                    d = means.get((v1_pair[1], v2_pair[1]), 0)

                    # Interaction effect: (a-b) - (c-d)
                    interaction = abs((a - b) - (c - d))
                    interactions[f"{attr1} x {attr2}"] = interaction
                except KeyError:
                    pass

        return interactions

    def subgroup_analysis(
        self,
        data: List[Dict[str, Any]],
        metric_key: str = "score",
        target_attributes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Detailed analysis of a specific subgroup.

        Args:
            data: Full dataset
            metric_key: Metric to analyze
            target_attributes: Specific subgroup attributes to analyze

        Returns:
            Detailed subgroup analysis
        """
        if target_attributes is None:
            target_attributes = {}

        # Filter data to subgroup
        subgroup_data = [
            s for s in data
            if all(s.get(k) == v for k, v in target_attributes.items())
        ]

        if not subgroup_data:
            return {"error": "No data for specified subgroup"}

        # Get metrics
        metrics = [
            s[metric_key]
            for s in subgroup_data
            if metric_key in s and s[metric_key] is not None
        ]

        if not metrics:
            return {"error": "No valid metrics for subgroup"}

        # Compare to overall population
        all_metrics = [
            s[metric_key]
            for s in data
            if metric_key in s and s[metric_key] is not None
        ]

        # Statistical comparison
        if len(metrics) >= 5 and len(all_metrics) >= 5:
            statistic, p_value = stats.mannwhitneyu(
                metrics, all_metrics, alternative='two-sided'
            )
        else:
            statistic, p_value = None, None

        return {
            "subgroup": target_attributes,
            "sample_size": len(subgroup_data),
            "metric_mean": np.mean(metrics),
            "metric_std": np.std(metrics),
            "metric_median": np.median(metrics),
            "population_mean": np.mean(all_metrics),
            "difference_from_population": np.mean(metrics) - np.mean(all_metrics),
            "statistical_test": {
                "test": "mann_whitney_u",
                "statistic": statistic,
                "p_value": p_value,
            },
            "percentile_in_population": stats.percentileofscore(all_metrics, np.mean(metrics)),
        }

    def generate_disparity_matrix(
        self,
        data: List[Dict[str, Any]],
        attribute1: str,
        attribute2: str,
        metric_key: str = "score",
    ) -> Dict[str, Any]:
        """
        Generate a disparity matrix between two attributes.

        Args:
            data: Dataset
            attribute1: First attribute for rows
            attribute2: Second attribute for columns
            metric_key: Metric to analyze

        Returns:
            Disparity matrix with statistical tests
        """
        values1 = sorted(set(s.get(attribute1) for s in data if s.get(attribute1) is not None))
        values2 = sorted(set(s.get(attribute2) for s in data if s.get(attribute2) is not None))

        matrix = {}
        for v1 in values1:
            matrix[v1] = {}
            for v2 in values2:
                metrics = [
                    s[metric_key]
                    for s in data
                    if s.get(attribute1) == v1 and s.get(attribute2) == v2
                    and metric_key in s and s[metric_key] is not None
                ]
                if metrics:
                    matrix[v1][v2] = {
                        "mean": np.mean(metrics),
                        "std": np.std(metrics),
                        "count": len(metrics),
                    }
                else:
                    matrix[v1][v2] = None

        # Find disparities
        all_means = [
            cell["mean"]
            for row in matrix.values()
            for cell in row.values()
            if cell is not None
        ]

        if all_means:
            max_mean = max(all_means)
            min_mean = min(all_means)
            disparity_range = max_mean - min_mean
        else:
            disparity_range = 0

        return {
            "matrix": matrix,
            "row_attribute": attribute1,
            "column_attribute": attribute2,
            "row_values": values1,
            "column_values": values2,
            "disparity_range": disparity_range,
            "overall_mean": np.mean(all_means) if all_means else None,
        }
