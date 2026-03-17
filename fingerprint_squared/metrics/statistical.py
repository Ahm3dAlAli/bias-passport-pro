"""
Statistical tests for bias and fairness analysis.

This module provides rigorous statistical testing methods to determine
whether observed differences in model behavior are statistically significant.

Includes:
- Chi-square test for independence
- Two-proportion Z-test
- Kruskal-Wallis H-test
- Mann-Whitney U test
- Permutation tests
- Bootstrap confidence intervals
- Jensen-Shannon divergence
- KL divergence
- Multiple testing correction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import rel_entr


@dataclass
class StatisticalTestResult:
    """Result of a statistical significance test."""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "alpha": self.alpha,
            "effect_size": self.effect_size,
            "confidence_interval": self.confidence_interval,
            "interpretation": self.interpretation,
        }


class StatisticalTests:
    """
    Statistical testing suite for fairness analysis.

    Provides methods for testing whether observed differences in
    model behavior are statistically significant.

    Attributes:
        alpha: Significance level (default 0.05)
        correction: Multiple testing correction method
    """

    def __init__(
        self,
        alpha: float = 0.05,
        correction: str = "bonferroni",
    ):
        self.alpha = alpha
        self.correction = correction

    def chi_square_test(
        self,
        observed: Dict[str, Dict[str, int]],
        expected: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> StatisticalTestResult:
        """
        Chi-square test for independence between group and outcome.

        Args:
            observed: Contingency table {group: {outcome: count}}
            expected: Optional expected frequencies

        Returns:
            StatisticalTestResult with test results
        """
        # Convert to contingency table
        groups = list(observed.keys())
        outcomes = list(set(
            outcome for group_outcomes in observed.values()
            for outcome in group_outcomes.keys()
        ))

        table = np.zeros((len(groups), len(outcomes)))
        for i, group in enumerate(groups):
            for j, outcome in enumerate(outcomes):
                table[i, j] = observed[group].get(outcome, 0)

        chi2, p_value, dof, expected_freq = stats.chi2_contingency(table)

        # Cramér's V as effect size
        n = table.sum()
        min_dim = min(len(groups), len(outcomes)) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        is_significant = p_value < self.alpha

        return StatisticalTestResult(
            test_name="chi_square",
            statistic=chi2,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            effect_size=cramers_v,
            interpretation=self._interpret_effect_size(cramers_v, "cramers_v"),
        )

    def two_proportion_z_test(
        self,
        successes1: int,
        n1: int,
        successes2: int,
        n2: int,
    ) -> StatisticalTestResult:
        """
        Two-proportion Z-test for comparing rates between groups.

        Args:
            successes1: Number of successes in group 1
            n1: Total count in group 1
            successes2: Number of successes in group 2
            n2: Total count in group 2

        Returns:
            StatisticalTestResult with test results
        """
        p1 = successes1 / n1 if n1 > 0 else 0
        p2 = successes2 / n2 if n2 > 0 else 0

        # Pooled proportion
        p_pool = (successes1 + successes2) / (n1 + n2) if (n1 + n2) > 0 else 0

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if p_pool > 0 else 1

        # Z statistic
        z = (p1 - p2) / se if se > 0 else 0

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Effect size (Cohen's h)
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

        # Confidence interval for difference
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        se_diff = np.sqrt(p1 * (1-p1) / n1 + p2 * (1-p2) / n2)
        ci = (p1 - p2 - z_crit * se_diff, p1 - p2 + z_crit * se_diff)

        is_significant = p_value < self.alpha

        return StatisticalTestResult(
            test_name="two_proportion_z",
            statistic=z,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            effect_size=abs(h),
            confidence_interval=ci,
            interpretation=self._interpret_effect_size(abs(h), "cohens_h"),
        )

    def kruskal_wallis_test(
        self,
        groups: Dict[str, List[float]],
    ) -> StatisticalTestResult:
        """
        Kruskal-Wallis H-test for comparing multiple groups.

        Non-parametric alternative to one-way ANOVA.

        Args:
            groups: Dictionary mapping group names to value lists

        Returns:
            StatisticalTestResult with test results
        """
        group_values = list(groups.values())

        if len(group_values) < 2:
            return StatisticalTestResult(
                test_name="kruskal_wallis",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                alpha=self.alpha,
                interpretation="Insufficient groups for comparison",
            )

        statistic, p_value = stats.kruskal(*group_values)

        # Effect size: epsilon-squared
        n = sum(len(g) for g in group_values)
        k = len(group_values)
        epsilon_sq = (statistic - k + 1) / (n - k)

        is_significant = p_value < self.alpha

        return StatisticalTestResult(
            test_name="kruskal_wallis",
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            effect_size=epsilon_sq,
            interpretation=self._interpret_effect_size(epsilon_sq, "epsilon_sq"),
        )

    def mann_whitney_u_test(
        self,
        group1: List[float],
        group2: List[float],
    ) -> StatisticalTestResult:
        """
        Mann-Whitney U test for comparing two groups.

        Args:
            group1: Values from first group
            group2: Values from second group

        Returns:
            StatisticalTestResult with test results
        """
        statistic, p_value = stats.mannwhitneyu(
            group1, group2, alternative='two-sided'
        )

        # Effect size: rank-biserial correlation
        n1, n2 = len(group1), len(group2)
        r = 1 - (2 * statistic) / (n1 * n2)

        is_significant = p_value < self.alpha

        return StatisticalTestResult(
            test_name="mann_whitney_u",
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            effect_size=abs(r),
            interpretation=self._interpret_effect_size(abs(r), "r"),
        )

    def permutation_test(
        self,
        group1: List[float],
        group2: List[float],
        n_permutations: int = 10000,
        statistic_fn: Optional[callable] = None,
    ) -> StatisticalTestResult:
        """
        Permutation test for comparing two groups.

        Args:
            group1: Values from first group
            group2: Values from second group
            n_permutations: Number of permutations
            statistic_fn: Optional custom statistic function

        Returns:
            StatisticalTestResult with test results
        """
        if statistic_fn is None:
            statistic_fn = lambda x, y: np.mean(x) - np.mean(y)

        combined = np.array(group1 + group2)
        n1 = len(group1)

        observed_stat = statistic_fn(group1, group2)

        # Generate permutation distribution
        perm_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_stat = statistic_fn(combined[:n1], combined[n1:])
            perm_stats.append(perm_stat)

        perm_stats = np.array(perm_stats)

        # Two-tailed p-value
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))

        is_significant = p_value < self.alpha

        return StatisticalTestResult(
            test_name="permutation",
            statistic=observed_stat,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            interpretation="Permutation test for difference in means",
        )

    def bootstrap_confidence_interval(
        self,
        data: List[float],
        statistic_fn: callable = np.mean,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval.

        Args:
            data: Data values
            statistic_fn: Function to compute statistic
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level

        Returns:
            Tuple of (lower, upper) confidence bounds
        """
        data = np.array(data)
        n = len(data)

        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_fn(sample))

        bootstrap_stats = np.array(bootstrap_stats)

        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return (lower, upper)

    def multiple_testing_correction(
        self,
        p_values: List[float],
    ) -> List[float]:
        """
        Apply multiple testing correction.

        Args:
            p_values: List of p-values

        Returns:
            Corrected p-values
        """
        p_values = np.array(p_values)
        n = len(p_values)

        if self.correction == "bonferroni":
            return np.minimum(p_values * n, 1.0).tolist()

        elif self.correction == "holm":
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            corrected = np.zeros(n)

            for i, p in enumerate(sorted_p):
                corrected[sorted_indices[i]] = min(p * (n - i), 1.0)

            # Ensure monotonicity
            for i in range(1, n):
                if corrected[sorted_indices[i]] < corrected[sorted_indices[i-1]]:
                    corrected[sorted_indices[i]] = corrected[sorted_indices[i-1]]

            return corrected.tolist()

        elif self.correction == "fdr":
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            corrected = np.zeros(n)

            for i, p in enumerate(sorted_p):
                corrected[sorted_indices[i]] = min(p * n / (i + 1), 1.0)

            # Ensure monotonicity (reversed)
            for i in range(n - 2, -1, -1):
                if corrected[sorted_indices[i]] > corrected[sorted_indices[i+1]]:
                    corrected[sorted_indices[i]] = corrected[sorted_indices[i+1]]

            return corrected.tolist()

        else:
            return p_values.tolist()

    def _interpret_effect_size(
        self,
        effect_size: float,
        effect_type: str,
    ) -> str:
        """Interpret effect size magnitude."""
        thresholds = {
            "cramers_v": [(0.1, "small"), (0.3, "medium"), (0.5, "large")],
            "cohens_h": [(0.2, "small"), (0.5, "medium"), (0.8, "large")],
            "epsilon_sq": [(0.01, "small"), (0.06, "medium"), (0.14, "large")],
            "r": [(0.1, "small"), (0.3, "medium"), (0.5, "large")],
        }

        if effect_type not in thresholds:
            return "Unknown effect type"

        for threshold, label in thresholds[effect_type]:
            if effect_size < threshold:
                return f"{label} effect"

        return "large effect"

    def jensen_shannon_divergence(
        self,
        p: List[float],
        q: List[float],
        base: float = 2.0,
    ) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions.

        JS divergence is a symmetric measure of the difference between
        two probability distributions. Useful for comparing score
        distributions across demographic groups.

        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        where M = 0.5 * (P + Q)

        Args:
            p: First probability distribution
            q: Second probability distribution
            base: Logarithm base (2 for bits, e for nats)

        Returns:
            JS divergence value (0 = identical, 1 = maximally different for base=2)
        """
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)

        # Normalize to ensure valid probability distributions
        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q

        # Compute mixture distribution
        m = 0.5 * (p + q)

        # Handle zeros to avoid log(0)
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        m = np.clip(m, 1e-10, 1.0)

        # Compute JS divergence
        if base == 2.0:
            js = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
            js = js / np.log(2)  # Convert to bits
        else:
            js = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
            if base != np.e:
                js = js / np.log(base)

        return float(js)

    def kl_divergence(
        self,
        p: List[float],
        q: List[float],
        base: float = 2.0,
    ) -> float:
        """
        Compute Kullback-Leibler divergence from q to p.

        KL(P||Q) measures how much information is lost when Q is used
        to approximate P. Note: KL divergence is asymmetric.

        Args:
            p: True distribution
            q: Approximating distribution
            base: Logarithm base (2 for bits, e for nats)

        Returns:
            KL divergence value (0 = identical, can be > 1)
        """
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)

        # Normalize
        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q

        # Handle zeros
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)

        # Compute KL divergence
        kl = np.sum(rel_entr(p, q))

        if base == 2.0:
            kl = kl / np.log(2)
        elif base != np.e:
            kl = kl / np.log(base)

        return float(kl)

    def distribution_comparison(
        self,
        group_a: List[float],
        group_b: List[float],
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Compare two distributions using multiple metrics.

        Useful for comparing score distributions between demographic groups.

        Args:
            group_a: Values from first group
            group_b: Values from second group
            n_bins: Number of bins for histogram

        Returns:
            Dictionary with comparison metrics
        """
        # Convert to arrays
        a = np.array(group_a)
        b = np.array(group_b)

        # Compute histograms with same bins
        min_val = min(a.min(), b.min())
        max_val = max(a.max(), b.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        hist_a, _ = np.histogram(a, bins=bins, density=True)
        hist_b, _ = np.histogram(b, bins=bins, density=True)

        # Add small constant to avoid zeros
        hist_a = hist_a + 1e-10
        hist_b = hist_b + 1e-10

        # Normalize
        hist_a = hist_a / hist_a.sum()
        hist_b = hist_b / hist_b.sum()

        # Compute divergences
        js_div = self.jensen_shannon_divergence(hist_a.tolist(), hist_b.tolist())
        kl_ab = self.kl_divergence(hist_a.tolist(), hist_b.tolist())
        kl_ba = self.kl_divergence(hist_b.tolist(), hist_a.tolist())

        # Statistical tests
        ks_stat, ks_pvalue = stats.ks_2samp(a, b)
        mw_result = self.mann_whitney_u_test(group_a, group_b)

        return {
            "jensen_shannon_divergence": js_div,
            "kl_divergence_a_to_b": kl_ab,
            "kl_divergence_b_to_a": kl_ba,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "mann_whitney_u": mw_result.statistic,
            "mann_whitney_pvalue": mw_result.p_value,
            "mean_difference": float(np.mean(a) - np.mean(b)),
            "median_difference": float(np.median(a) - np.median(b)),
            "std_a": float(np.std(a)),
            "std_b": float(np.std(b)),
            "n_a": len(a),
            "n_b": len(b),
        }

    def demographic_disparity_test(
        self,
        scores_by_group: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Test for disparity across multiple demographic groups.

        Performs comprehensive statistical testing to detect bias.

        Args:
            scores_by_group: Dictionary mapping group names to score lists

        Returns:
            Dictionary with test results and disparity metrics
        """
        groups = list(scores_by_group.keys())
        n_groups = len(groups)

        if n_groups < 2:
            return {"error": "At least 2 groups required"}

        # Overall test (Kruskal-Wallis)
        kw_result = self.kruskal_wallis_test(scores_by_group)

        # Pairwise JS divergences
        js_matrix = {}
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                comparison = self.distribution_comparison(
                    scores_by_group[g1],
                    scores_by_group[g2],
                )
                js_matrix[f"{g1}_vs_{g2}"] = {
                    "js_divergence": comparison["jensen_shannon_divergence"],
                    "mean_difference": comparison["mean_difference"],
                    "ks_statistic": comparison["ks_statistic"],
                    "ks_pvalue": comparison["ks_pvalue"],
                }

        # Per-group statistics
        group_stats = {}
        for group, scores in scores_by_group.items():
            arr = np.array(scores)
            group_stats[group] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": len(scores),
            }

        # Compute max disparity
        means = [group_stats[g]["mean"] for g in groups]
        max_disparity = max(means) - min(means)
        most_favored = groups[np.argmax(means)]
        least_favored = groups[np.argmin(means)]

        return {
            "overall_test": {
                "test": "kruskal_wallis",
                "statistic": kw_result.statistic,
                "p_value": kw_result.p_value,
                "is_significant": kw_result.is_significant,
                "effect_size": kw_result.effect_size,
            },
            "pairwise_comparisons": js_matrix,
            "group_statistics": group_stats,
            "disparity_summary": {
                "max_disparity": max_disparity,
                "most_favored_group": most_favored,
                "least_favored_group": least_favored,
            },
        }

    def jurisdiction_disparity_analysis(
        self,
        scores_by_jurisdiction: Dict[str, List[float]],
        region_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze bias disparity across jurisdictions (geographic regions).

        Designed for FHIBE's 81 jurisdictions across 5 regions.

        Args:
            scores_by_jurisdiction: Dictionary mapping jurisdiction to scores
            region_mapping: Optional mapping of jurisdiction to region

        Returns:
            Dictionary with jurisdiction-level analysis
        """
        jurisdictions = list(scores_by_jurisdiction.keys())
        n_jurisdictions = len(jurisdictions)

        if n_jurisdictions < 2:
            return {"error": "At least 2 jurisdictions required"}

        # Overall Kruskal-Wallis test across all jurisdictions
        kw_result = self.kruskal_wallis_test(scores_by_jurisdiction)

        # Per-jurisdiction statistics
        jurisdiction_stats = {}
        for jurisdiction, scores in scores_by_jurisdiction.items():
            arr = np.array(scores)
            jurisdiction_stats[jurisdiction] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "n": len(scores),
            }

        # Find top/bottom jurisdictions
        sorted_by_mean = sorted(
            jurisdiction_stats.items(),
            key=lambda x: x[1]["mean"],
            reverse=True,
        )

        top_5 = [j for j, _ in sorted_by_mean[:5]]
        bottom_5 = [j for j, _ in sorted_by_mean[-5:]]

        # Region-level analysis if mapping provided
        region_analysis = {}
        if region_mapping:
            scores_by_region: Dict[str, List[float]] = {}
            for jurisdiction, scores in scores_by_jurisdiction.items():
                region = region_mapping.get(jurisdiction, "Unknown")
                if region not in scores_by_region:
                    scores_by_region[region] = []
                scores_by_region[region].extend(scores)

            if len(scores_by_region) >= 2:
                region_kw = self.kruskal_wallis_test(scores_by_region)
                region_analysis = {
                    "kruskal_wallis": {
                        "statistic": region_kw.statistic,
                        "p_value": region_kw.p_value,
                        "is_significant": region_kw.is_significant,
                    },
                    "region_means": {
                        region: float(np.mean(scores))
                        for region, scores in scores_by_region.items()
                    },
                }

        # Compute geographic clustering of bias
        # (do neighboring jurisdictions have similar bias?)
        means = [s["mean"] for s in jurisdiction_stats.values()]
        global_variance = float(np.var(means))

        return {
            "n_jurisdictions": n_jurisdictions,
            "overall_test": {
                "test": "kruskal_wallis",
                "statistic": kw_result.statistic,
                "p_value": kw_result.p_value,
                "is_significant": kw_result.is_significant,
                "effect_size": kw_result.effect_size,
            },
            "jurisdiction_statistics": jurisdiction_stats,
            "top_5_jurisdictions": top_5,
            "bottom_5_jurisdictions": bottom_5,
            "global_variance": global_variance,
            "region_analysis": region_analysis,
            "disparity_summary": {
                "max_mean": float(max(means)),
                "min_mean": float(min(means)),
                "max_disparity": float(max(means) - min(means)),
                "most_positive_jurisdiction": sorted_by_mean[0][0],
                "most_negative_jurisdiction": sorted_by_mean[-1][0],
            },
        }

    def skin_tone_disparity_analysis(
        self,
        scores_by_tone: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Analyze bias disparity across Fitzpatrick skin tone scale.

        The Fitzpatrick scale (Types I-VI) ranges from very light to very dark.
        This analysis detects whether VLM responses vary by skin tone.

        Args:
            scores_by_tone: Dictionary mapping skin tone (1-6) to scores

        Returns:
            Dictionary with skin tone disparity analysis
        """
        tones = sorted(scores_by_tone.keys())
        n_tones = len(tones)

        if n_tones < 2:
            return {"error": "At least 2 skin tone groups required"}

        # Overall test
        kw_result = self.kruskal_wallis_test(scores_by_tone)

        # Per-tone statistics
        tone_stats = {}
        for tone, scores in scores_by_tone.items():
            arr = np.array(scores)
            tone_stats[tone] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "n": len(scores),
            }

        # Compute linear trend (does bias increase/decrease with darker tone?)
        tone_means = [tone_stats[t]["mean"] for t in tones]
        tone_indices = list(range(len(tones)))

        # Spearman correlation for monotonic trend
        if len(tone_indices) >= 3:
            correlation, corr_pvalue = stats.spearmanr(tone_indices, tone_means)
        else:
            correlation, corr_pvalue = 0.0, 1.0

        # Compare light vs dark (types 1-3 vs 4-6)
        light_scores = []
        dark_scores = []
        for tone, scores in scores_by_tone.items():
            try:
                tone_num = int(tone)
                if tone_num <= 3:
                    light_scores.extend(scores)
                else:
                    dark_scores.extend(scores)
            except ValueError:
                continue

        light_vs_dark = None
        if light_scores and dark_scores:
            mw_result = self.mann_whitney_u_test(light_scores, dark_scores)
            light_vs_dark = {
                "statistic": mw_result.statistic,
                "p_value": mw_result.p_value,
                "is_significant": mw_result.is_significant,
                "effect_size": mw_result.effect_size,
                "light_mean": float(np.mean(light_scores)),
                "dark_mean": float(np.mean(dark_scores)),
                "disparity": float(np.mean(light_scores) - np.mean(dark_scores)),
            }

        return {
            "n_tones": n_tones,
            "overall_test": {
                "test": "kruskal_wallis",
                "statistic": kw_result.statistic,
                "p_value": kw_result.p_value,
                "is_significant": kw_result.is_significant,
            },
            "tone_statistics": tone_stats,
            "trend_analysis": {
                "spearman_correlation": float(correlation),
                "correlation_pvalue": float(corr_pvalue),
                "trend_direction": "darker_more_negative" if correlation < 0 else (
                    "darker_more_positive" if correlation > 0 else "no_trend"
                ),
            },
            "light_vs_dark_comparison": light_vs_dark,
        }
