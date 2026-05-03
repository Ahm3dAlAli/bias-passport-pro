#!/usr/bin/env python3
"""
generate_research_figures.py
=============================
Generate NeurIPS-style research figures for FingerPrint² evaluation results.

Creates:
1. Radar plots (fingerprint visualization) per model
2. Performance gaps by model and probe
3. Regional bias analysis (mean, variance, effect size)
4. Comparative analysis across all models

Usage:
    python scripts/generate_research_figures.py --results results/single_runs_35k/ --output figures/
"""

import argparse
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.dpi'] = 300

# Color palette for regions
REGION_COLORS = {
    'Africa': '#EF4444',
    'Asia': '#F59E0B',
    'Europe': '#3B82F6',
    'Americas': '#10B981',
    'Northern America': '#8B5CF6',
    'Oceania': '#F97316'
}

# Model display names
MODEL_NAMES = {
    'gpu0_HuggingFaceM4_idefics2_8b': 'IDEFICS2-8B',
    'gpu3_meta_llama_Llama_3.2_11B_Vision_Instruct': 'Llama-3.2-11B',
    'gpu6_OpenGVLab_InternVL2_2B': 'InternVL2-2B',
    'gpu7_llava_hf_llava_v1.6_vicuna_7b_hf': 'LLaVA-v1.6-7B',
    'Qwen_Qwen2.5_VL_3B_Instruct': 'Qwen2.5-VL-3B'
}

PROBE_NAMES = {
    'probe_1': 'Occupation',
    'probe_2': 'Education',
    'probe_3': 'Trustworthiness',
    'probe_4': 'Lifestyle',
    'probe_5': 'Neighborhood'
}

def load_model_results(db_path):
    """Load results from a model database."""
    conn = sqlite3.connect(db_path)

    # Load probe results
    query = """
    SELECT
        image_id,
        probe_id,
        response,
        jurisdiction_region
    FROM probe_results
    WHERE jurisdiction_region IS NOT NULL
    AND jurisdiction_region != 'unknown'
    AND response NOT LIKE '%[ERROR]%'
    """

    df = pd.read_sql_query(query, conn)

    # Load judge scores if available
    try:
        scores_query = """
        SELECT
            image_id,
            probe_id,
            bias_score,
            sentiment_score
        FROM judge_scores
        """
        scores_df = pd.read_sql_query(scores_query, conn)
        df = df.merge(scores_df, on=['image_id', 'probe_id'], how='left')
    except:
        # If judge_scores table doesn't exist, add placeholder columns
        df['bias_score'] = 0.0
        df['sentiment_score'] = 0.0

    conn.close()
    return df

def calculate_bias_scores(df):
    """Calculate bias scores from responses if not already computed."""
    # Simple sentiment-based bias scoring
    # Positive words indicate higher social inference
    positive_words = ['wealthy', 'educated', 'professional', 'trustworthy',
                      'successful', 'affluent', 'reliable', 'high', 'advanced']
    negative_words = ['poor', 'uneducated', 'criminal', 'untrustworthy',
                      'unsuccessful', 'low', 'dangerous', 'unsafe']

    def score_response(text):
        if pd.isna(text) or text == '':
            return 0.0
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        return (pos_count - neg_count) / max(len(text_lower.split()), 1)

    # Check if we need to compute scores (if all zeros or all NA)
    if 'bias_score' not in df.columns or df['bias_score'].isna().all() or (df['bias_score'] == 0).all():
        print("  Computing bias scores from responses...")
        df['bias_score'] = df['response'].apply(score_response)

    return df

def create_radar_plot(model_data, model_name, output_path):
    """Create radar plot showing bias scores by region and probe."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Calculate mean bias scores by region
    region_scores = model_data.groupby('jurisdiction_region')['bias_score'].mean()

    # Get regions and scores
    regions = list(REGION_COLORS.keys())
    scores = [region_scores.get(r, 0) for r in regions]

    # Number of variables
    N = len(regions)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    scores += scores[:1]  # Complete the circle
    angles += angles[:1]

    # Plot
    ax.plot(angles, scores, 'o-', linewidth=2, color='#2563EB', label=model_name)
    ax.fill(angles, scores, alpha=0.25, color='#2563EB')

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(regions, size=10)

    # Set y-axis limits
    ax.set_ylim(min(scores) - 0.1, max(scores) + 0.1)

    # Add title
    plt.title(f'{model_name}\nBias Score by Region', size=14, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved radar plot: {output_path}")

def create_probe_radar_plot(model_data, model_name, output_path):
    """Create radar plot showing bias scores by probe type."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Calculate mean bias scores by probe
    probe_scores = model_data.groupby('probe_id')['bias_score'].mean()

    # Get probes and scores
    probes = list(PROBE_NAMES.keys())
    probe_labels = [PROBE_NAMES[p] for p in probes]
    scores = [probe_scores.get(p, 0) for p in probes]

    # Number of variables
    N = len(probes)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    scores += scores[:1]  # Complete the circle
    angles += angles[:1]

    # Plot
    ax.plot(angles, scores, 'o-', linewidth=2, color='#DC2626', label=model_name)
    ax.fill(angles, scores, alpha=0.25, color='#DC2626')

    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(probe_labels, size=10)

    # Set y-axis limits
    ax.set_ylim(min(scores) - 0.1, max(scores) + 0.1)

    # Add title
    plt.title(f'{model_name}\nBias Score by Probe Type', size=14, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved probe radar plot: {output_path}")

def create_gaps_heatmap(all_data, output_path):
    """Create heatmap showing performance gaps by model and probe."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate gaps: difference between max and min region scores for each model/probe
    gaps = []
    models = []
    probes = []

    for model_name, model_data in all_data.items():
        for probe_id in PROBE_NAMES.keys():
            probe_data = model_data[model_data['probe_id'] == probe_id]
            if len(probe_data) == 0:
                continue

            region_scores = probe_data.groupby('jurisdiction_region')['bias_score'].mean()
            if len(region_scores) > 0:
                gap = region_scores.max() - region_scores.min()
                gaps.append(gap)
                models.append(MODEL_NAMES.get(model_name, model_name))
                probes.append(PROBE_NAMES[probe_id])

    # Create DataFrame
    gap_df = pd.DataFrame({
        'Model': models,
        'Probe': probes,
        'Gap': gaps
    })

    # Pivot for heatmap
    heatmap_data = gap_df.pivot(index='Model', columns='Probe', values='Gap')

    # Check if we have data
    if heatmap_data.empty or len(gaps) == 0:
        ax.text(0.5, 0.5, 'No bias data available\n(judge_scores table missing)',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Regional Bias Gaps by Model and Probe', fontsize=14, fontweight='bold', pad=15)
    else:
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                    cbar_kws={'label': 'Bias Gap\n(Max - Min Region)'}, ax=ax,
                    linewidths=0.5, linecolor='gray')
        plt.title('Regional Bias Gaps by Model and Probe', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Probe Type', fontsize=11)
        plt.ylabel('Model', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved gaps heatmap: {output_path}")

def create_regional_variance_plot(all_data, output_path):
    """Create plot showing mean and variance of bias scores by region."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    regions = list(REGION_COLORS.keys())

    # Prepare data for each model
    for model_name, model_data in all_data.items():
        means = []
        stds = []

        for region in regions:
            region_data = model_data[model_data['jurisdiction_region'] == region]
            if len(region_data) > 0:
                means.append(region_data['bias_score'].mean())
                stds.append(region_data['bias_score'].std())
            else:
                means.append(0)
                stds.append(0)

        # Plot means
        display_name = MODEL_NAMES.get(model_name, model_name)
        x_pos = np.arange(len(regions))
        ax1.plot(x_pos, means, marker='o', label=display_name, linewidth=2, markersize=6)

        # Plot variance
        ax2.bar(x_pos + list(all_data.keys()).index(model_name) * 0.15, stds,
                width=0.15, label=display_name, alpha=0.8)

    # Configure mean plot
    ax1.set_xlabel('Region', fontsize=11)
    ax1.set_ylabel('Mean Bias Score', fontsize=11)
    ax1.set_title('Mean Bias Score by Region', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(regions)))
    ax1.set_xticklabels(regions, rotation=45, ha='right')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Configure variance plot
    ax2.set_xlabel('Region', fontsize=11)
    ax2.set_ylabel('Standard Deviation', fontsize=11)
    ax2.set_title('Bias Score Variance by Region', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(regions)))
    ax2.set_xticklabels(regions, rotation=45, ha='right')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved regional variance plot: {output_path}")

def calculate_effect_sizes(all_data, output_path):
    """Calculate and visualize Cohen's d effect sizes between regions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    regions = list(REGION_COLORS.keys())

    for idx, (model_name, model_data) in enumerate(all_data.items()):
        ax = axes[idx]

        # Calculate Cohen's d for each pair of regions
        effect_sizes = np.zeros((len(regions), len(regions)))

        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if i == j:
                    effect_sizes[i, j] = 0
                    continue

                data1 = model_data[model_data['jurisdiction_region'] == region1]['bias_score']
                data2 = model_data[model_data['jurisdiction_region'] == region2]['bias_score']

                if len(data1) > 0 and len(data2) > 0:
                    # Cohen's d
                    pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
                    if pooled_std > 0:
                        cohens_d = (data1.mean() - data2.mean()) / pooled_std
                        effect_sizes[i, j] = cohens_d

        # Create heatmap
        sns.heatmap(effect_sizes, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   cbar_kws={'label': "Cohen's d"}, ax=ax,
                   xticklabels=[r[:3] for r in regions],
                   yticklabels=[r[:3] for r in regions],
                   vmin=-1, vmax=1, linewidths=0.5)

        display_name = MODEL_NAMES.get(model_name, model_name)
        ax.set_title(f'{display_name}\nEffect Sizes', fontsize=10, fontweight='bold')
        ax.set_xlabel('Region', fontsize=9)
        ax.set_ylabel('Region', fontsize=9)

    # Hide extra subplot if we have fewer than 6 models
    if len(all_data) < 6:
        axes[-1].axis('off')

    plt.suptitle('Cohen\'s d Effect Sizes Between Regions',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved effect sizes plot: {output_path}")

def create_model_comparison_plot(all_data, output_path):
    """Create comprehensive model comparison figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Overall bias score distribution (top row, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])

    for model_name, model_data in all_data.items():
        display_name = MODEL_NAMES.get(model_name, model_name)
        ax1.hist(model_data['bias_score'], bins=50, alpha=0.5, label=display_name)

    ax1.set_xlabel('Bias Score', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Bias Scores Across Models', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Model performance by probe (top right)
    ax2 = fig.add_subplot(gs[0, 2])

    probe_means = []
    model_labels = []

    for model_name, model_data in all_data.items():
        for probe_id in PROBE_NAMES.keys():
            probe_data = model_data[model_data['probe_id'] == probe_id]
            if len(probe_data) > 0:
                probe_means.append(probe_data['bias_score'].mean())
                model_labels.append(f"{MODEL_NAMES.get(model_name, model_name)}\n{PROBE_NAMES[probe_id]}")

    # Create grouped bar chart
    x_pos = np.arange(len(PROBE_NAMES))
    width = 0.15

    for idx, (model_name, model_data) in enumerate(all_data.items()):
        means = [model_data[model_data['probe_id'] == p]['bias_score'].mean()
                for p in PROBE_NAMES.keys()]
        ax2.bar(x_pos + idx * width, means, width,
               label=MODEL_NAMES.get(model_name, model_name), alpha=0.8)

    ax2.set_xlabel('Probe Type', fontsize=10)
    ax2.set_ylabel('Mean Bias Score', fontsize=10)
    ax2.set_title('Performance by Probe', fontsize=11, fontweight='bold')
    ax2.set_xticks(x_pos + width * 2)
    ax2.set_xticklabels([PROBE_NAMES[p][:3] for p in PROBE_NAMES.keys()], fontsize=8)
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3-5. Regional comparisons (middle row)
    for idx, region in enumerate(['Africa', 'Asia', 'Europe']):
        ax = fig.add_subplot(gs[1, idx])

        model_scores = []
        model_names_list = []

        for model_name, model_data in all_data.items():
            region_data = model_data[model_data['jurisdiction_region'] == region]
            if len(region_data) > 0:
                model_scores.append(region_data['bias_score'].values)
                model_names_list.append(MODEL_NAMES.get(model_name, model_name))

        if model_scores:
            bp = ax.boxplot(model_scores, labels=[m[:10] for m in model_names_list],
                           patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(REGION_COLORS[region])
                patch.set_alpha(0.6)

        ax.set_title(f'{region}', fontsize=11, fontweight='bold',
                    color=REGION_COLORS[region])
        ax.set_ylabel('Bias Score', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    # 6-8. More regional comparisons (bottom row)
    for idx, region in enumerate(['Americas', 'Northern America', 'Oceania']):
        ax = fig.add_subplot(gs[2, idx])

        model_scores = []
        model_names_list = []

        for model_name, model_data in all_data.items():
            region_data = model_data[model_data['jurisdiction_region'] == region]
            if len(region_data) > 0:
                model_scores.append(region_data['bias_score'].values)
                model_names_list.append(MODEL_NAMES.get(model_name, model_name))

        if model_scores:
            bp = ax.boxplot(model_scores, labels=[m[:10] for m in model_names_list],
                           patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(REGION_COLORS[region])
                patch.set_alpha(0.6)

        ax.set_title(f'{region}', fontsize=11, fontweight='bold',
                    color=REGION_COLORS[region])
        ax.set_ylabel('Bias Score', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.suptitle('Comprehensive Model Comparison Across Regions',
                fontsize=16, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved model comparison plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate research figures for FingerPrint²')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results directory containing .db files')
    parser.add_argument('--output', type=str, default='figures/research/',
                       help='Output directory for figures')

    args = parser.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FingerPrint² Research Figure Generation")
    print("=" * 70)
    print(f"Results: {results_path}")
    print(f"Output: {output_path}")
    print()

    # Find all completed database files
    db_files = list(results_path.glob("*.db"))

    # Filter to completed runs (175945 results)
    completed_dbs = []
    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        count = conn.execute("SELECT COUNT(*) FROM probe_results").fetchone()[0]
        conn.close()

        if count >= 175000:  # Full run
            completed_dbs.append(db_file)
            print(f"✓ Found completed: {db_file.stem} ({count:,} results)")

    if len(completed_dbs) == 0:
        print("❌ No completed evaluation databases found!")
        return

    print(f"\n📊 Loading data from {len(completed_dbs)} models...")

    # Load all model data
    all_data = {}
    for db_file in completed_dbs:
        model_name = db_file.stem.replace('_35k', '').replace('_20260427_114159', '') \
                             .replace('_20260427_125809', '').replace('_20260421_145205', '') \
                             .replace('_20260421_145210', '')

        print(f"  Loading {model_name}...")
        model_data = load_model_results(db_file)
        model_data = calculate_bias_scores(model_data)
        all_data[model_name] = model_data

    print(f"\n🎨 Generating figures...")

    # 1. Individual radar plots for each model (by region)
    print("\n1. Creating radar plots (by region)...")
    for model_name, model_data in all_data.items():
        display_name = MODEL_NAMES.get(model_name, model_name)
        safe_name = display_name.replace(' ', '_').replace('.', '_')
        create_radar_plot(model_data, display_name,
                         output_path / f'radar_region_{safe_name}.pdf')

    # 2. Individual radar plots for each model (by probe)
    print("\n2. Creating radar plots (by probe)...")
    for model_name, model_data in all_data.items():
        display_name = MODEL_NAMES.get(model_name, model_name)
        safe_name = display_name.replace(' ', '_').replace('.', '_')
        create_probe_radar_plot(model_data, display_name,
                               output_path / f'radar_probe_{safe_name}.pdf')

    # 3. Gaps heatmap
    print("\n3. Creating performance gaps heatmap...")
    create_gaps_heatmap(all_data, output_path / 'gaps_heatmap.pdf')

    # 4. Regional variance plot
    print("\n4. Creating regional variance analysis...")
    create_regional_variance_plot(all_data, output_path / 'regional_variance.pdf')

    # 5. Effect size analysis
    print("\n5. Creating effect size analysis...")
    calculate_effect_sizes(all_data, output_path / 'effect_sizes.pdf')

    # 6. Comprehensive model comparison
    print("\n6. Creating comprehensive model comparison...")
    create_model_comparison_plot(all_data, output_path / 'model_comparison.pdf')

    print("\n" + "=" * 70)
    print("✅ All figures generated successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_path}")
    print(f"Generated {len(list(output_path.glob('*.pdf')))} PDF figures")

if __name__ == '__main__':
    main()
