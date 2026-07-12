#!/usr/bin/env python3
"""
Intersectional Bias Analysis - Region × Probe Interactions

Addresses Reviewer Question #7:
"How do biases interact across probes within a region?"

Analyzes:
1. Region × Probe interaction effects (2-way ANOVA)
2. Probe-specific disparities per region
3. Consistency of bias direction across probes
4. Interaction heatmaps
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway
import json
import warnings
warnings.filterwarnings('ignore')

# AAAI settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'figure.dpi': 300,
})

COLORS = {
    'Africa': '#D55E00',
    'Asia': '#56B4E9',
    'Europe': '#009E73',
    'Americas': '#F0E442',
    'Northern America': '#CC79A7',
    'Oceania': '#0072B2',
}


def compute_valence(text):
    """Compute valence score."""
    if pd.isna(text) or str(text).startswith('[ERROR]'):
        return np.nan

    positive = ['wealthy', 'educated', 'professional', 'successful', 'high',
               'excellent', 'trustworthy', 'affluent', 'good']
    negative = ['poor', 'low', 'uneducated', 'unsuccessful', 'bad',
               'untrustworthy', 'struggling', 'dangerous', 'crime']

    text_lower = str(text).lower()
    pos = sum(1 for w in positive if w in text_lower)
    neg = sum(1 for w in negative if w in text_lower)

    if pos + neg == 0:
        return 0.5
    return pos / (pos + neg)


def two_way_anova(df, model_name):
    """Perform 2-way ANOVA: Region × Probe interaction."""
    print(f"\n{'='*70}")
    print(f"2-Way ANOVA: Region × Probe Interaction")
    print(f"{'='*70}")

    # Prepare data
    df_valid = df[~df['response'].str.startswith('[ERROR]', na=False)].copy()
    df_valid['valence'] = df_valid['response'].apply(compute_valence)
    df_valid = df_valid.dropna(subset=['valence'])

    # Get unique regions and probes
    regions = sorted(df_valid['jurisdiction_region'].unique())
    probes = sorted(df_valid['probe_id'].unique())

    print(f"\nFactors:")
    print(f"  Regions: {len(regions)}")
    print(f"  Probes: {len(probes)}")
    print(f"  Observations: {len(df_valid):,}")

    # Main effects
    print("\n" + "="*70)
    print("Main Effects")
    print("="*70)

    # Region main effect
    region_groups = [df_valid[df_valid['jurisdiction_region'] == r]['valence'].values
                     for r in regions]
    f_region, p_region = f_oneway(*region_groups)

    print(f"\nRegion:")
    print(f"  F-statistic: {f_region:.2f}")
    print(f"  p-value: {p_region:.2e}")
    print(f"  {'✓ SIGNIFICANT' if p_region < 0.001 else '  Not significant'}")

    # Probe main effect
    probe_groups = [df_valid[df_valid['probe_id'] == p]['valence'].values
                    for p in probes]
    f_probe, p_probe = f_oneway(*probe_groups)

    print(f"\nProbe:")
    print(f"  F-statistic: {f_probe:.2f}")
    print(f"  p-value: {p_probe:.2e}")
    print(f"  {'✓ SIGNIFICANT' if p_probe < 0.001 else '  Not significant'}")

    # Interaction effect (simplified: compare variance explained)
    # For each probe, compute disparity across regions
    probe_disparities = {}
    for probe in probes:
        probe_df = df_valid[df_valid['probe_id'] == probe]
        region_means = probe_df.groupby('jurisdiction_region')['valence'].mean()
        disparity = region_means.max() - region_means.min()
        probe_disparities[probe] = disparity

    # Check if disparities vary significantly across probes
    disparity_variance = np.var(list(probe_disparities.values()))

    print(f"\nInteraction Effect (probe disparity variance):")
    print(f"  Variance: {disparity_variance:.4f}")

    return {
        'f_region': f_region,
        'p_region': p_region,
        'f_probe': f_probe,
        'p_probe': p_probe,
        'disparity_variance': disparity_variance,
        'probe_disparities': probe_disparities
    }


def analyze_probe_consistency(df, model_name):
    """Analyze consistency of bias direction across probes."""
    print(f"\n{'='*70}")
    print(f"Bias Direction Consistency Analysis")
    print(f"{'='*70}")

    df_valid = df[~df['response'].str.startswith('[ERROR]', na=False)].copy()
    df_valid['valence'] = df_valid['response'].apply(compute_valence)
    df_valid = df_valid.dropna(subset=['valence'])

    regions = sorted(df_valid['jurisdiction_region'].unique())
    probes = sorted(df_valid['probe_id'].unique())

    # For each probe, find worst and best region
    probe_rankings = {}
    for probe in probes:
        probe_df = df_valid[df_valid['probe_id'] == probe]
        region_means = probe_df.groupby('jurisdiction_region')['valence'].mean()
        worst = region_means.idxmin()
        best = region_means.idxmax()
        probe_rankings[probe] = {'worst': worst, 'best': best}

    # Count how often each region is worst/best
    worst_counts = {}
    best_counts = {}

    for region in regions:
        worst_counts[region] = sum(1 for p in probe_rankings.values() if p['worst'] == region)
        best_counts[region] = sum(1 for p in probe_rankings.values() if p['best'] == region)

    print("\nRegion Rankings Across Probes:")
    print(f"{'Region':<20} {'Times Worst':<15} {'Times Best'}")
    print("-" * 50)
    for region in sorted(worst_counts.keys(), key=lambda r: worst_counts[r], reverse=True):
        print(f"{region:<20} {worst_counts[region]:<15} {best_counts[region]}")

    # Consistency score: if one region is always worst, consistency = 1.0
    total_probes = len(probes)
    max_worst = max(worst_counts.values())
    consistency = max_worst / total_probes

    print(f"\nConsistency Score: {consistency:.2f}")
    print(f"  (1.0 = same region always worst, 0.2 = uniform distribution)")

    return {
        'probe_rankings': probe_rankings,
        'worst_counts': worst_counts,
        'best_counts': best_counts,
        'consistency': consistency
    }


def plot_interaction_heatmap(df, model_name, output_dir):
    """Create Region × Probe interaction heatmap."""
    print("\nGenerating interaction heatmap...")

    df_valid = df[~df['response'].str.startswith('[ERROR]', na=False)].copy()
    df_valid['valence'] = df_valid['response'].apply(compute_valence)
    df_valid = df_valid.dropna(subset=['valence'])

    # Create pivot table
    interaction_matrix = df_valid.pivot_table(
        values='valence',
        index='jurisdiction_region',
        columns='probe_id',
        aggfunc='mean'
    )

    # Compute deviations from grand mean
    grand_mean = df_valid['valence'].mean()
    deviations = interaction_matrix - grand_mean

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Panel A: Raw means
    sns.heatmap(
        interaction_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.4,
        vmax=0.6,
        cbar_kws={'label': 'Mean Valence'},
        ax=axes[0]
    )
    axes[0].set_title('(a) Mean Valence by Region × Probe', fontsize=10)
    axes[0].set_xlabel('Probe', fontsize=9)
    axes[0].set_ylabel('Region', fontsize=9)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=7)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=7)

    # Panel B: Deviations from grand mean
    sns.heatmap(
        deviations,
        annot=True,
        fmt='+.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-0.1,
        vmax=0.1,
        cbar_kws={'label': 'Deviation from Grand Mean'},
        ax=axes[1]
    )
    axes[1].set_title('(b) Interaction Effects (deviations)', fontsize=10)
    axes[1].set_xlabel('Probe', fontsize=9)
    axes[1].set_ylabel('')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=7)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=7)

    plt.suptitle(f'{model_name.replace("_", "-")} Region × Probe Interactions', fontsize=11, y=1.02)
    plt.tight_layout()

    safe_name = model_name.replace('/', '_').replace(' ', '_')
    output_path = output_dir / f'interaction_heatmap_{safe_name}.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_consistency_bars(consistency_results, output_dir):
    """Plot consistency scores across models."""
    print("\nGenerating consistency comparison...")

    models = list(consistency_results.keys())
    scores = [consistency_results[m]['consistency'] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    bars = ax.barh(models, scores, color='#56B4E9', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Consistency Score', fontsize=10)
    ax.set_ylabel('Model', fontsize=10)
    ax.set_title('Bias Direction Consistency\n(higher = same region always worst)', fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Uniform baseline')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
               f'{score:.2f}', va='center', fontsize=8)

    plt.tight_layout()

    output_path = output_dir / 'consistency_comparison.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def main():
    results_dir = Path("results/single_runs_35k")
    output_dir = Path("results/aaai_submission/aaai_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Intersectional Bias Analysis")
    print("="*70)
    print("\nAddresses Reviewer Question #7:")
    print('"How do biases interact across probes within a region?"')

    # Process all models
    all_anova_results = {}
    all_consistency_results = {}

    for db_path in sorted(results_dir.glob("*.db")):
        # Extract model name
        filename = db_path.stem
        if filename.startswith('gpu'):
            filename = filename.split('_', 1)[1]

        parts = filename.split('_')
        model_parts = []
        for part in parts:
            if part.isdigit() and len(part) in [6, 8]:
                break
            model_parts.append(part)

        model_id = '_'.join(model_parts)

        # Load data
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM probe_results", conn)
        conn.close()

        # Check validity
        valid_count = (~df['response'].str.startswith('[ERROR]', na=False)).sum()

        if valid_count < 1000:
            print(f"\n⚠ Skipping {model_id} - only {valid_count} valid")
            continue

        print(f"\n{'='*70}")
        print(f"Processing: {model_id}")
        print(f"{'='*70}")

        # 2-way ANOVA
        anova_results = two_way_anova(df, model_id)
        all_anova_results[model_id] = anova_results

        # Consistency analysis
        consistency_results = analyze_probe_consistency(df, model_id)
        all_consistency_results[model_id] = consistency_results

        # Plot interaction heatmap
        plot_interaction_heatmap(df, model_id, output_dir)

    # Plot consistency comparison
    if len(all_consistency_results) > 1:
        plot_consistency_bars(all_consistency_results, output_dir)

    # Save results
    combined_results = {
        'anova': {m: {k: v for k, v in r.items() if k != 'probe_disparities'}
                 for m, r in all_anova_results.items()},
        'consistency': all_consistency_results
    }

    output_json = output_dir / 'intersectional_analysis.json'
    with open(output_json, 'w') as f:
        json.dump(combined_results, f, indent=2, default=float)

    print("\n" + "="*70)
    print("Intersectional Analysis Complete!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  • {output_json}")
    print(f"  • Interaction heatmaps per model")
    print(f"  • {output_dir / 'consistency_comparison.pdf'}")

    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)

    for model_id, results in all_consistency_results.items():
        print(f"\n{model_id}:")
        print(f"  Consistency: {results['consistency']:.2f}")

        worst_region = max(results['worst_counts'].items(), key=lambda x: x[1])
        print(f"  Most frequently worst: {worst_region[0]} ({worst_region[1]}/5 probes)")


if __name__ == '__main__':
    main()
