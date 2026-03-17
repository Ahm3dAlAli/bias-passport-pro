"""
Core fairness metrics for VLM evaluation.

This module implements a comprehensive suite of fairness metrics adapted for
Vision-Language Models, drawing from established fairness literature and extending
them for multimodal contexts.

References:
    - Hardt et al., "Equality of Opportunity in Supervised Learning" (NeurIPS 2016)
    - Chouldechova, "Fair Prediction with Disparate Impact" (2017)
    - Mehrabi et al., "A Survey on Bias and Fairness in Machine Learning" (2021)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix


@dataclass
class FairnessResult:
    """Container for fairness metric results."""

    metric_name: str
    value: float
    is_fair: bool
    threshold: float
    groups: Dict[str, float]
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "is_fair": self.is_fair,
            "threshold": self.threshold,
            "groups": self.groups,
            "confidence_interval": self.confidence_interval,
            "p_value": self.p_value,
            "details": self.details,
        }


class FairnessMetrics:
    """
    Comprehensive fairness metrics suite for VLM evaluation.

    This class implements multiple fairness definitions and provides methods
    to compute them across demographic groups in VLM outputs.

    Attributes:
        epsilon: Fairness threshold (default 0.1 for 80% rule)
        confidence_level: Confidence level for statistical tests

    Example:
        >>> fm = FairnessMetrics(epsilon=0.1)
        >>> predictions = {"male": [1, 1, 0, 1], "female": [0, 0, 1, 0]}
        >>> result = fm.demographic_parity(predictions)
        >>> print(f"Demographic Parity Gap: {result.value:.3f}")
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        confidence_level: float = 0.95,
    ):
        self.epsilon = epsilon
        self.confidence_level = confidence_level

    def demographic_parity(
        self,
        predictions: Dict[str, List[int]],
        reference_group: Optional[str] = None,
    ) -> FairnessResult:
        """
        Compute Demographic Parity (Statistical Parity).

        Measures whether positive prediction rates are equal across groups.
        DP holds when: P(Ŷ=1|A=a) = P(Ŷ=1|A=b) for all groups a, b

        Args:
            predictions: Dict mapping group names to binary predictions
            reference_group: Optional reference group for ratio computation

        Returns:
            FairnessResult with demographic parity metrics
        """
        group_rates = {}
        for group, preds in predictions.items():
            preds_arr = np.array(preds)
            group_rates[group] = np.mean(preds_arr)

        rates = list(group_rates.values())
        dp_gap = max(rates) - min(rates)

        # Compute disparity ratio (4/5ths rule)
        if reference_group and reference_group in group_rates:
            ref_rate = group_rates[reference_group]
        else:
            ref_rate = max(rates)

        disparity_ratios = {}
        for group, rate in group_rates.items():
            if ref_rate > 0:
                disparity_ratios[group] = rate / ref_rate
            else:
                disparity_ratios[group] = 1.0 if rate == 0 else float('inf')

        min_ratio = min(disparity_ratios.values())
        is_fair = dp_gap <= self.epsilon and min_ratio >= 0.8

        return FairnessResult(
            metric_name="demographic_parity",
            value=dp_gap,
            is_fair=is_fair,
            threshold=self.epsilon,
            groups=group_rates,
            details={
                "disparity_ratios": disparity_ratios,
                "min_ratio": min_ratio,
                "satisfies_80_percent_rule": min_ratio >= 0.8,
            },
        )

    def equalized_odds(
        self,
        predictions: Dict[str, List[int]],
        labels: Dict[str, List[int]],
    ) -> FairnessResult:
        """
        Compute Equalized Odds.

        Measures whether true positive rates and false positive rates are
        equal across groups.
        EO holds when: P(Ŷ=1|A=a,Y=y) = P(Ŷ=1|A=b,Y=y) for y ∈ {0,1}

        Args:
            predictions: Dict mapping group names to binary predictions
            labels: Dict mapping group names to ground truth labels

        Returns:
            FairnessResult with equalized odds metrics
        """
        group_metrics = {}

        for group in predictions.keys():
            preds = np.array(predictions[group])
            labs = np.array(labels[group])

            # True Positive Rate (Recall)
            pos_mask = labs == 1
            if pos_mask.sum() > 0:
                tpr = np.mean(preds[pos_mask])
            else:
                tpr = 0.0

            # False Positive Rate
            neg_mask = labs == 0
            if neg_mask.sum() > 0:
                fpr = np.mean(preds[neg_mask])
            else:
                fpr = 0.0

            group_metrics[group] = {"tpr": tpr, "fpr": fpr}

        # Compute gaps
        tprs = [m["tpr"] for m in group_metrics.values()]
        fprs = [m["fpr"] for m in group_metrics.values()]

        tpr_gap = max(tprs) - min(tprs)
        fpr_gap = max(fprs) - min(fprs)
        eo_gap = max(tpr_gap, fpr_gap)

        is_fair = eo_gap <= self.epsilon

        return FairnessResult(
            metric_name="equalized_odds",
            value=eo_gap,
            is_fair=is_fair,
            threshold=self.epsilon,
            groups={g: m["tpr"] for g, m in group_metrics.items()},
            details={
                "tpr_gap": tpr_gap,
                "fpr_gap": fpr_gap,
                "group_metrics": group_metrics,
            },
        )

    def equal_opportunity(
        self,
        predictions: Dict[str, List[int]],
        labels: Dict[str, List[int]],
    ) -> FairnessResult:
        """
        Compute Equal Opportunity (subset of Equalized Odds).

        Measures whether true positive rates are equal across groups.
        EO holds when: P(Ŷ=1|A=a,Y=1) = P(Ŷ=1|A=b,Y=1)

        Args:
            predictions: Dict mapping group names to binary predictions
            labels: Dict mapping group names to ground truth labels

        Returns:
            FairnessResult with equal opportunity metrics
        """
        group_tprs = {}

        for group in predictions.keys():
            preds = np.array(predictions[group])
            labs = np.array(labels[group])

            pos_mask = labs == 1
            if pos_mask.sum() > 0:
                tpr = np.mean(preds[pos_mask])
            else:
                tpr = 0.0

            group_tprs[group] = tpr

        tprs = list(group_tprs.values())
        tpr_gap = max(tprs) - min(tprs)
        is_fair = tpr_gap <= self.epsilon

        return FairnessResult(
            metric_name="equal_opportunity",
            value=tpr_gap,
            is_fair=is_fair,
            threshold=self.epsilon,
            groups=group_tprs,
        )

    def predictive_parity(
        self,
        predictions: Dict[str, List[int]],
        labels: Dict[str, List[int]],
    ) -> FairnessResult:
        """
        Compute Predictive Parity (Outcome Test).

        Measures whether precision (PPV) is equal across groups.
        PP holds when: P(Y=1|Ŷ=1,A=a) = P(Y=1|Ŷ=1,A=b)

        Args:
            predictions: Dict mapping group names to binary predictions
            labels: Dict mapping group names to ground truth labels

        Returns:
            FairnessResult with predictive parity metrics
        """
        group_ppvs = {}

        for group in predictions.keys():
            preds = np.array(predictions[group])
            labs = np.array(labels[group])

            pred_pos_mask = preds == 1
            if pred_pos_mask.sum() > 0:
                ppv = np.mean(labs[pred_pos_mask])
            else:
                ppv = 0.0

            group_ppvs[group] = ppv

        ppvs = list(group_ppvs.values())
        ppv_gap = max(ppvs) - min(ppvs)
        is_fair = ppv_gap <= self.epsilon

        return FairnessResult(
            metric_name="predictive_parity",
            value=ppv_gap,
            is_fair=is_fair,
            threshold=self.epsilon,
            groups=group_ppvs,
        )

    def calibration(
        self,
        probabilities: Dict[str, List[float]],
        labels: Dict[str, List[int]],
        n_bins: int = 10,
    ) -> FairnessResult:
        """
        Compute Calibration fairness across groups.

        Measures whether predicted probabilities are well-calibrated
        across demographic groups.

        Args:
            probabilities: Dict mapping group names to predicted probabilities
            labels: Dict mapping group names to ground truth labels
            n_bins: Number of bins for calibration

        Returns:
            FairnessResult with calibration metrics
        """
        group_ece = {}

        for group in probabilities.keys():
            probs = np.array(probabilities[group])
            labs = np.array(labels[group])

            # Compute Expected Calibration Error
            ece = self._compute_ece(probs, labs, n_bins)
            group_ece[group] = ece

        eces = list(group_ece.values())
        ece_gap = max(eces) - min(eces)
        max_ece = max(eces)

        is_fair = ece_gap <= self.epsilon and max_ece <= 0.1

        return FairnessResult(
            metric_name="calibration",
            value=ece_gap,
            is_fair=is_fair,
            threshold=self.epsilon,
            groups=group_ece,
            details={
                "max_ece": max_ece,
                "mean_ece": np.mean(eces),
            },
        )

    def _compute_ece(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(probabilities)

        for i in range(n_bins):
            in_bin = (probabilities > bin_boundaries[i]) & (
                probabilities <= bin_boundaries[i + 1]
            )
            prop_in_bin = in_bin.sum() / total

            if in_bin.sum() > 0:
                avg_confidence = probabilities[in_bin].mean()
                avg_accuracy = labels[in_bin].mean()
                ece += prop_in_bin * abs(avg_confidence - avg_accuracy)

        return ece

    def counterfactual_fairness(
        self,
        original_outputs: Dict[str, List[Any]],
        counterfactual_outputs: Dict[str, List[Any]],
        similarity_fn: Optional[callable] = None,
    ) -> FairnessResult:
        """
        Compute Counterfactual Fairness for VLM outputs.

        Measures whether changing protected attributes leads to
        different model outputs.

        Args:
            original_outputs: Outputs with original attributes
            counterfactual_outputs: Outputs with counterfactual attributes
            similarity_fn: Function to compute similarity between outputs

        Returns:
            FairnessResult with counterfactual fairness metrics
        """
        if similarity_fn is None:
            # Default: exact match for strings, close for numbers
            def similarity_fn(a, b):
                if isinstance(a, str) and isinstance(b, str):
                    return 1.0 if a.lower().strip() == b.lower().strip() else 0.0
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return 1.0 - min(abs(a - b) / max(abs(a), abs(b), 1), 1.0)
                else:
                    return 1.0 if a == b else 0.0

        group_consistency = {}

        for group in original_outputs.keys():
            if group not in counterfactual_outputs:
                continue

            orig = original_outputs[group]
            cf = counterfactual_outputs[group]

            similarities = [
                similarity_fn(o, c) for o, c in zip(orig, cf)
            ]
            group_consistency[group] = np.mean(similarities)

        consistencies = list(group_consistency.values())
        min_consistency = min(consistencies) if consistencies else 0.0
        mean_consistency = np.mean(consistencies) if consistencies else 0.0

        # Higher consistency = more fair (outputs don't change with attributes)
        is_fair = min_consistency >= (1 - self.epsilon)

        return FairnessResult(
            metric_name="counterfactual_fairness",
            value=1 - mean_consistency,  # Convert to "unfairness" measure
            is_fair=is_fair,
            threshold=self.epsilon,
            groups=group_consistency,
            details={
                "min_consistency": min_consistency,
                "mean_consistency": mean_consistency,
            },
        )

    def representation_disparity(
        self,
        group_counts: Dict[str, int],
        expected_distribution: Optional[Dict[str, float]] = None,
    ) -> FairnessResult:
        """
        Measure representation disparity in model outputs.

        Computes how far the distribution of groups in outputs deviates
        from expected (uniform or specified) distribution.

        Args:
            group_counts: Count of each group in outputs
            expected_distribution: Expected proportion for each group

        Returns:
            FairnessResult with representation metrics
        """
        total = sum(group_counts.values())
        if total == 0:
            return FairnessResult(
                metric_name="representation_disparity",
                value=0.0,
                is_fair=True,
                threshold=self.epsilon,
                groups={},
            )

        observed = {g: c / total for g, c in group_counts.items()}

        if expected_distribution is None:
            # Assume uniform distribution
            n_groups = len(group_counts)
            expected_distribution = {g: 1 / n_groups for g in group_counts}

        # Compute KL divergence
        kl_div = 0.0
        for group in observed:
            if group in expected_distribution and observed[group] > 0:
                kl_div += observed[group] * np.log(
                    observed[group] / expected_distribution[group]
                )

        # Compute total variation distance
        tv_distance = 0.5 * sum(
            abs(observed.get(g, 0) - expected_distribution.get(g, 0))
            for g in set(observed) | set(expected_distribution)
        )

        is_fair = tv_distance <= self.epsilon

        return FairnessResult(
            metric_name="representation_disparity",
            value=tv_distance,
            is_fair=is_fair,
            threshold=self.epsilon,
            groups=observed,
            details={
                "kl_divergence": kl_div,
                "total_variation": tv_distance,
                "expected_distribution": expected_distribution,
            },
        )

    def compute_all(
        self,
        predictions: Dict[str, List[int]],
        labels: Optional[Dict[str, List[int]]] = None,
        probabilities: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, FairnessResult]:
        """
        Compute all applicable fairness metrics.

        Args:
            predictions: Dict mapping group names to binary predictions
            labels: Optional ground truth labels
            probabilities: Optional predicted probabilities

        Returns:
            Dictionary of all computed fairness metrics
        """
        results = {}

        # Always compute demographic parity
        results["demographic_parity"] = self.demographic_parity(predictions)

        # Compute metrics requiring labels
        if labels is not None:
            results["equalized_odds"] = self.equalized_odds(predictions, labels)
            results["equal_opportunity"] = self.equal_opportunity(predictions, labels)
            results["predictive_parity"] = self.predictive_parity(predictions, labels)

        # Compute calibration if probabilities available
        if probabilities is not None and labels is not None:
            results["calibration"] = self.calibration(probabilities, labels)

        return results

    def aggregate_fairness_score(
        self,
        results: Dict[str, FairnessResult],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute aggregate fairness score from multiple metrics.

        Args:
            results: Dictionary of fairness results
            weights: Optional weights for each metric

        Returns:
            Aggregate fairness score (0 = perfectly fair, 1 = maximally unfair)
        """
        if not results:
            return 0.0

        if weights is None:
            weights = {name: 1.0 for name in results}

        total_weight = sum(weights.get(name, 0) for name in results)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            (results[name].value / results[name].threshold) * weights.get(name, 1.0)
            for name in results
        )

        # Normalize and clip to [0, 1]
        score = weighted_sum / total_weight
        return min(max(score, 0.0), 1.0)
