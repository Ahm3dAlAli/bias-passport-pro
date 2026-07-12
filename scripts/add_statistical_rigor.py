#!/usr/bin/env python3
"""
Add statistical rigor to FingerPrint results for AAAI publication.
Computes p-values, effect sizes, confidence intervals, and power analysis.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
from itertools import combinations
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def compute_valence_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute valence scores from responses."""
    positive_words = ['wealthy', 'educated', 'professional', 'successful', 'high',
                     'excellent', 'trustworthy', 'affluent', 'prestigious', 'good',
                     'positive', 'happy', 'competent', 'reliable', 'safe']
    negative_words = ['poor', 'low', 'uneducated', 'unsuccessful', 'bad',
                     'untrustworthy', 'struggling', 'dangerous', 'crime',
                     'negative', 'sad', 'incompetent', 'unreliable', 'unsafe']

    def score_response(text):
        if pd.isna(text) or text == '' or str(text).startswith('[ERROR]') or str(text).startswith('Error'):
            return np.nan
        text_lower = str(text).lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        if pos_count + neg_count == 0:
            return 0.5  # Neutral
        return pos_count / (pos_count + neg_count)

    df = df.copy()
    df['valence'] = df['response'].apply(score_response)
    return df


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0


def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """Calculate bootstrap confidence interval."""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return lower, upper


def regional_fairness_analysis(df: pd.DataFrame, model_name: str, probe_id: str = None) -> Dict:
    """
    Experiment 1: Geographic Fairness Analysis
    Tests if regional differences are statistically significant.
    """
    print(f"\n{'='*70}")
    print(f"Regional Fairness Analysis: {model_name}")
    if probe_id:
        print(f"Probe: {probe_id}")
        df = df[df['probe_id'] == probe_id]
    print(f"{'='*70}")

    df = compute_valence_scores(df)
    df = df.dropna(subset=['valence'])

    # Check if we have any valid data
    if len(df) == 0:
        print(f"  ⚠️  No valid responses (all errors)")
        return None

    # Group by region
    regions = df['jurisdiction_region'].unique()
    region_data = {r: df[df['jurisdiction_region'] == r]['valence'].values for r in regions}

    # Filter out empty regions
    region_data = {r: vals for r, vals in region_data.items() if len(vals) > 0}
    regions = list(region_data.keys())

    if len(regions) < 2:
        print(f"  ⚠️  Not enough regions with data ({len(regions)} regions)")
        return None
    region_stats = {}

    # Compute statistics per region
    for region, values in region_data.items():
        mean_val = np.mean(values)
        ci_lower, ci_upper = bootstrap_ci(values)
        region_stats[region] = {
            'mean': mean_val,
            'std': np.std(values, ddof=1),
            'sem': stats.sem(values),
            'n': len(values),
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper
        }
        print(f"  {region:20s}: μ={mean_val:.4f}, σ={np.std(values):.4f}, "
              f"n={len(values):,}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")

    # One-way ANOVA: Are regional differences significant?
    groups = [region_data[r] for r in regions]
    F_stat, p_value_anova = f_oneway(*groups)

    print(f"\n  One-way ANOVA: F({len(regions)-1}, {len(df)-len(regions)}) = {F_stat:.4f}, p = {p_value_anova:.4e}")

    # Post-hoc pairwise comparisons (with Bonferroni correction)
    pairwise_results = []
    n_comparisons = len(list(combinations(regions, 2)))
    bonferroni_alpha = 0.05 / n_comparisons

    print(f"\n  Post-hoc pairwise comparisons (Bonferroni-corrected α = {bonferroni_alpha:.4f}):")

    for r1, r2 in combinations(regions, 2):
        t_stat, p_val = ttest_ind(region_data[r1], region_data[r2])
        d = cohens_d(region_data[r1], region_data[r2])

        significant = "***" if p_val < bonferroni_alpha else ""
        print(f"    {r1:20s} vs {r2:20s}: t={t_stat:7.3f}, p={p_val:.4e}, d={d:6.3f} {significant}")

        pairwise_results.append({
            'group1': r1,
            'group2': r2,
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': d,
            'significant': p_val < bonferroni_alpha
        })

    # Worst vs. Best comparison
    region_means = {r: s['mean'] for r, s in region_stats.items()}
    worst_region = min(region_means, key=region_means.get)
    best_region = max(region_means, key=region_means.get)

    worst_best_d = cohens_d(region_data[worst_region], region_data[best_region])
    _, worst_best_p = ttest_ind(region_data[worst_region], region_data[best_region])

    print(f"\n  Worst-treated: {worst_region} (μ={region_means[worst_region]:.4f})")
    print(f"  Best-treated:  {best_region} (μ={region_means[best_region]:.4f})")
    print(f"  Gap: Δ={region_means[best_region] - region_means[worst_region]:.4f}, "
          f"d={worst_best_d:.3f}, p={worst_best_p:.4e}")

    # Effect size interpretation
    if abs(worst_best_d) < 0.2:
        effect_interp = "negligible"
    elif abs(worst_best_d) < 0.5:
        effect_interp = "small"
    elif abs(worst_best_d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    print(f"  Effect size: {effect_interp}")

    return {
        'model': model_name,
        'probe': probe_id,
        'n_regions': len(regions),
        'total_n': len(df),
        'region_stats': region_stats,
        'anova': {
            'F_statistic': F_stat,
            'p_value': p_value_anova,
            'df_between': len(regions) - 1,
            'df_within': len(df) - len(regions)
        },
        'pairwise_comparisons': pairwise_results,
        'worst_best': {
            'worst_region': worst_region,
            'best_region': best_region,
            'worst_mean': region_means[worst_region],
            'best_mean': region_means[best_region],
            'gap': region_means[best_region] - region_means[worst_region],
            'cohens_d': worst_best_d,
            'p_value': worst_best_p,
            'effect_size': effect_interp
        }
    }


def model_comparison_analysis(results_dict: Dict[str, pd.DataFrame]) -> Dict:
    """
    Experiment 3: Model Comparison
    Test if models differ significantly in bias.
    """
    print(f"\n{'='*70}")
    print("Model Comparison Analysis")
    print(f"{'='*70}")

    # Compute composite scores per model
    model_composites = {}

    for model_name, df in results_dict.items():
        df = compute_valence_scores(df)

        # Compute disparity per probe
        probe_disparities = []
        for probe in df['probe_id'].unique():
            probe_df = df[df['probe_id'] == probe]
            region_means = probe_df.groupby('jurisdiction_region')['valence'].mean()
            if len(region_means) > 1:
                disparity = region_means.max() - region_means.min()
                probe_disparities.append(disparity)

        composite = np.mean(probe_disparities)
        ci_lower, ci_upper = bootstrap_ci(probe_disparities)

        model_composites[model_name] = {
            'composite_score': composite,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'probe_disparities': probe_disparities
        }

        print(f"  {model_name:20s}: Composite={composite:.4f}, "
              f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")

    # Pairwise model comparisons
    print(f"\n  Pairwise model comparisons (Mann-Whitney U test):")

    pairwise_results = []
    for m1, m2 in combinations(model_composites.keys(), 2):
        disp1 = model_composites[m1]['probe_disparities']
        disp2 = model_composites[m2]['probe_disparities']

        U, p_val = mannwhitneyu(disp1, disp2, alternative='two-sided')
        d = cohens_d(disp1, disp2)

        print(f"    {m1:20s} vs {m2:20s}: U={U:.1f}, p={p_val:.4f}, d={d:.3f}")

        pairwise_results.append({
            'model1': m1,
            'model2': m2,
            'U_statistic': U,
            'p_value': p_val,
            'cohens_d': d
        })

    return {
        'model_composites': model_composites,
        'pairwise_comparisons': pairwise_results
    }


def power_analysis(n_total: int, n_groups: int, effect_size: float = 0.25, alpha: float = 0.05):
    """
    Statistical power analysis for ANOVA.
    """
    try:
        from statsmodels.stats.power import FTestAnovaPower

        power_calc = FTestAnovaPower()
        power = power_calc.solve_power(
            effect_size=effect_size,
            nobs=n_total / n_groups,
            alpha=alpha,
            k_groups=n_groups
        )

        print(f"\n{'='*70}")
        print("Statistical Power Analysis")
        print(f"{'='*70}")
        print(f"  Sample size: n={n_total:,}")
        print(f"  Groups: k={n_groups}")
        print(f"  Effect size (f): {effect_size}")
        print(f"  Significance level (α): {alpha}")
        print(f"  Statistical power: {power:.4f}")

        if power > 0.95:
            print(f"  ✓ Excellent power (>95%) to detect medium effects")
        elif power > 0.80:
            print(f"  ✓ Adequate power (>80%) to detect medium effects")
        else:
            print(f"  ⚠ Low power (<80%), consider larger sample size")

        return {'power': power, 'effect_size': effect_size, 'alpha': alpha}

    except ImportError:
        print("  ⚠ statsmodels not available, skipping power analysis")
        return None


def generate_summary_statistics(results: Dict) -> str:
    """Generate publication-ready summary statistics."""

    summary = []
    summary.append("\n" + "="*70)
    summary.append("PUBLICATION-READY SUMMARY STATISTICS")
    summary.append("="*70)

    # Regional analysis
    if 'regional_analysis' in results:
        for model_result in results['regional_analysis']:
            model = model_result['model']
            anova = model_result['anova']
            wb = model_result['worst_best']

            summary.append(f"\n{model}:")
            summary.append(f"  One-way ANOVA: F({anova['df_between']}, {anova['df_within']}) = "
                          f"{anova['F_statistic']:.2f}, p < {anova['p_value']:.3e}")
            summary.append(f"  Worst-treated: {wb['worst_region']} (M={wb['worst_mean']:.3f})")
            summary.append(f"  Best-treated: {wb['best_region']} (M={wb['best_mean']:.3f})")
            summary.append(f"  Gap: Δ={wb['gap']:.3f}, Cohen's d={wb['cohens_d']:.2f} "
                          f"({wb['effect_size']} effect), p < {wb['p_value']:.3e}")

    # Model comparison
    if 'model_comparison' in results:
        summary.append("\nModel Comparison:")
        for comp in results['model_comparison']['pairwise_comparisons']:
            summary.append(f"  {comp['model1']} vs {comp['model2']}: "
                          f"U={comp['U_statistic']:.1f}, p={comp['p_value']:.4f}, "
                          f"d={comp['cohens_d']:.2f}")

    return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(
        description='Add statistical rigor to FingerPrint results for AAAI'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory with .db result files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file with statistical results')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Model names to analyze (if None, analyze all)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("="*70)
    print("FingerPrint² Statistical Rigor Analysis")
    print("="*70)

    # Load data from all models
    all_data = {}

    for db_path in sorted(results_dir.glob("*.db")):
        # Extract model ID from filename
        filename = db_path.stem
        if filename.startswith('gpu'):
            parts = filename.split('_', 1)
            if len(parts) > 1:
                filename = parts[1]

        # Remove timestamp
        parts = filename.split('_')
        model_parts = []
        for part in parts:
            if part.isdigit() and len(part) in [6, 8]:
                break
            model_parts.append(part)

        model_id = '_'.join(model_parts)

        if args.models and model_id not in args.models:
            continue

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM probe_results", conn)
        conn.close()

        n_images = df['image_id'].nunique()
        if n_images < 30000:
            print(f"  ⚠ Skipping {model_id} - only {n_images:,} images")
            continue

        print(f"  ✓ Loaded {model_id}: {len(df):,} results")
        all_data[model_id] = df

    if not all_data:
        print("❌ No valid data loaded")
        return

    # Run analyses
    results = {}

    # 1. Regional fairness analysis per model
    regional_results = []
    for model_id, df in all_data.items():
        result = regional_fairness_analysis(df, model_id)
        if result is not None:
            regional_results.append(result)

    results['regional_analysis'] = regional_results

    if not regional_results:
        print("\n❌ No valid regional analysis results")
        return

    # 2. Model comparison
    model_comp = model_comparison_analysis(all_data)
    results['model_comparison'] = model_comp

    # 3. Power analysis
    sample_df = list(all_data.values())[0]
    power_result = power_analysis(
        n_total=len(sample_df),
        n_groups=sample_df['jurisdiction_region'].nunique(),
        effect_size=0.25
    )
    if power_result:
        results['power_analysis'] = power_result

    # Generate summary
    summary = generate_summary_statistics(results)
    print(summary)

    # Save results
    output_path = Path(args.output)

    # Convert to JSON-serializable format recursively
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif obj is None:
            return None
        elif isinstance(obj, (str, int, bool)):
            return obj
        else:
            return str(obj)

    results_serializable = convert_to_serializable(results)

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # Save summary text
    summary_path = output_path.parent / (output_path.stem + '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"✓ Summary saved to: {summary_path}")
    print("\n" + "="*70)
    print("Statistical analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()
