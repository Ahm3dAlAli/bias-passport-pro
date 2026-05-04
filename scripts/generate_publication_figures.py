#!/usr/bin/env python3
"""
generate_publication_figures.py
================================
Generate comprehensive publication-quality figures for FingerPrint² paper.
Includes bar charts, heatmaps, radar plots, tables, and statistical analyses.

Usage:
    python scripts/generate_publication_figures.py --results results/single_runs_35k/ --output figures/publication/
"""

import argparse
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'png'
})

REGION_COLORS = {
    'Africa': '#EF4444',
    'Asia': '#F59E0B',
    'Europe': '#3B82F6',
    'Americas': '#10B981',
    'Northern America': '#8B5CF6',
    'Oceania': '#F97316'
}

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

def load_and_process_data(db_path):
    """Load and compute all metrics."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT image_id, probe_id, response, jurisdiction_region
    FROM probe_results
    WHERE jurisdiction_region IS NOT NULL AND jurisdiction_region != 'unknown'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Compute metrics
    df['is_error'] = df['response'].apply(lambda x: 1 if pd.isna(x) or '[ERROR]' in str(x) else 0)
    df['response_length'] = df['response'].apply(lambda x: len(str(x).split()) if not pd.isna(x) and '[ERROR]' not in str(x) else 0)

    # Attribute detection
    positive = ['wealthy', 'educated', 'professional', 'successful', 'high', 'skilled', 'advanced', 'capable']
    negative = ['poor', 'low', 'uneducated', 'unsuccessful', 'struggle', 'limited', 'unable']

    df['has_positive'] = df['response'].apply(lambda x: int(any(w in str(x).lower() for w in positive)) if not pd.isna(x) else 0)
    df['has_negative'] = df['response'].apply(lambda x: int(any(w in str(x).lower() for w in negative)) if not pd.isna(x) else 0)
    df['bias_indicator'] = df['has_positive'] - df['has_negative']

    return df

def create_model_performance_bars(all_data, output_path):
    """Bar chart comparing models across metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    models = [MODEL_NAMES.get(m, m) for m in all_data.keys()]

    # Metric 1: Success Rate
    ax = axes[0, 0]
    success_rates = [100 - (d['is_error'].mean() * 100) for d in all_data.values()]
    bars = ax.barh(models, success_rates, color='#3B82F6', alpha=0.7)
    ax.set_xlabel('Success Rate (%)')
    ax.set_title('Model Reliability', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(success_rates):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

    # Metric 2: Avg Response Length
    ax = axes[0, 1]
    valid_data = [d[d['is_error'] == 0] for d in all_data.values()]
    avg_lengths = [d['response_length'].mean() for d in valid_data]
    bars = ax.barh(models, avg_lengths, color='#10B981', alpha=0.7)
    ax.set_xlabel('Average Response Length (words)')
    ax.set_title('Response Verbosity', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(avg_lengths):
        ax.text(v + 1, i, f'{v:.1f}', va='center', fontsize=9)

    # Metric 3: Positive Attribute Rate
    ax = axes[1, 0]
    pos_rates = [d['has_positive'].mean() * 100 for d in valid_data]
    bars = ax.barh(models, pos_rates, color='#10B981', alpha=0.7)
    ax.set_xlabel('Positive Attributes (%)')
    ax.set_title('Positive Mentions', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(pos_rates):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

    # Metric 4: Negative Attribute Rate
    ax = axes[1, 1]
    neg_rates = [d['has_negative'].mean() * 100 for d in valid_data]
    bars = ax.barh(models, neg_rates, color='#EF4444', alpha=0.7)
    ax.set_xlabel('Negative Attributes (%)')
    ax.set_title('Negative Mentions', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(neg_rates):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path.name}")

def create_regional_heatmaps(all_data, output_path):
    """Heatmap grid showing all models x regions."""
    regions = list(REGION_COLORS.keys())
    models = list(all_data.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Prepare matrices
    success_matrix = np.zeros((len(models), len(regions)))
    length_matrix = np.zeros((len(models), len(regions)))
    positive_matrix = np.zeros((len(models), len(regions)))
    negative_matrix = np.zeros((len(models), len(regions)))

    for i, (model_name, data) in enumerate(all_data.items()):
        for j, region in enumerate(regions):
            region_data = data[data['jurisdiction_region'] == region]
            if len(region_data) > 0:
                success_matrix[i, j] = 100 - (region_data['is_error'].mean() * 100)
                valid = region_data[region_data['is_error'] == 0]
                if len(valid) > 0:
                    length_matrix[i, j] = valid['response_length'].mean()
                    positive_matrix[i, j] = valid['has_positive'].mean() * 100
                    negative_matrix[i, j] = valid['has_negative'].mean() * 100

    model_labels = [MODEL_NAMES.get(m, m)[:12] for m in models]
    region_labels = [r[:3] for r in regions]

    # Heatmap 1: Success Rate
    sns.heatmap(success_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=region_labels, yticklabels=model_labels,
                cbar_kws={'label': 'Success Rate (%)'}, ax=axes[0, 0], vmin=90, vmax=100)
    axes[0, 0].set_title('Success Rate by Region', fontweight='bold', fontsize=11)

    # Heatmap 2: Response Length
    sns.heatmap(length_matrix, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=region_labels, yticklabels=model_labels,
                cbar_kws={'label': 'Words'}, ax=axes[0, 1])
    axes[0, 1].set_title('Avg Response Length', fontweight='bold', fontsize=11)

    # Heatmap 3: Positive Attributes
    sns.heatmap(positive_matrix, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=region_labels, yticklabels=model_labels,
                cbar_kws={'label': '% Positive'}, ax=axes[1, 0])
    axes[1, 0].set_title('Positive Attribute Mentions', fontweight='bold', fontsize=11)

    # Heatmap 4: Negative Attributes
    sns.heatmap(negative_matrix, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=region_labels, yticklabels=model_labels,
                cbar_kws={'label': '% Negative'}, ax=axes[1, 1])
    axes[1, 1].set_title('Negative Attribute Mentions', fontweight='bold', fontsize=11)

    plt.suptitle('Regional Analysis Heatmaps', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path.name}")

def create_radar_charts(all_data, output_path):
    """Radar charts for each model showing regional patterns."""
    regions = list(REGION_COLORS.keys())
    num_models = len(all_data)

    fig = plt.figure(figsize=(16, 10))

    for idx, (model_name, data) in enumerate(all_data.items()):
        ax = fig.add_subplot(2, 3, idx + 1, projection='polar')

        # Compute scores (positive - negative mentions)
        scores = []
        for region in regions:
            region_data = data[(data['jurisdiction_region'] == region) & (data['is_error'] == 0)]
            if len(region_data) > 0:
                score = (region_data['has_positive'].mean() - region_data['has_negative'].mean()) * 100
                scores.append(score)
            else:
                scores.append(0)

        # Plot
        angles = np.linspace(0, 2 * np.pi, len(regions), endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]

        ax.plot(angles, scores, 'o-', linewidth=2, color='#2563EB', markersize=6)
        ax.fill(angles, scores, alpha=0.25, color='#2563EB')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(regions, size=8)
        ax.set_ylim(-20, 20)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        display_name = MODEL_NAMES.get(model_name, model_name)
        ax.set_title(display_name, fontsize=11, fontweight='bold', pad=15)

    # Hide extra subplot if needed
    if num_models < 6:
        fig.add_subplot(2, 3, 6).axis('off')

    plt.suptitle('Regional Bias Fingerprints (Positive - Negative)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path.name}")

def create_probe_severity_analysis(all_data, output_path):
    """Analyze which probes show strongest bias."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    probes = list(PROBE_NAMES.keys())
    probe_labels = [PROBE_NAMES[p] for p in probes]

    # Metric 1: Error rates by probe
    ax = axes[0, 0]
    for model_name, data in all_data.items():
        error_rates = [data[data['probe_id'] == p]['is_error'].mean() * 100 for p in probes]
        ax.plot(probe_labels, error_rates, marker='o', label=MODEL_NAMES.get(model_name, model_name)[:12], linewidth=2)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Failure Rate by Probe Type', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Metric 2: Bias severity (pos - neg)
    ax = axes[0, 1]
    for model_name, data in all_data.items():
        valid = data[data['is_error'] == 0]
        bias_scores = []
        for p in probes:
            probe_data = valid[valid['probe_id'] == p]
            bias = (probe_data['has_positive'].mean() - probe_data['has_negative'].mean()) * 100
            bias_scores.append(bias)
        ax.bar(np.arange(len(probes)) + list(all_data.keys()).index(model_name) * 0.15,
               bias_scores, width=0.15, label=MODEL_NAMES.get(model_name, model_name)[:12], alpha=0.8)
    ax.set_xticks(np.arange(len(probes)) + 0.3)
    ax.set_xticklabels(probe_labels, rotation=45, ha='right')
    ax.set_ylabel('Bias Score (Pos - Neg %)')
    ax.set_title('Bias Severity by Probe', fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='best')
    ax.grid(axis='y', alpha=0.3)

    # Metric 3: Regional variance per probe
    ax = axes[1, 0]
    regions = list(REGION_COLORS.keys())
    probe_variances = []
    for p in probes:
        variances = []
        for model_data in all_data.values():
            valid = model_data[(model_data['probe_id'] == p) & (model_data['is_error'] == 0)]
            regional_scores = []
            for region in regions:
                region_data = valid[valid['jurisdiction_region'] == region]
                if len(region_data) > 0:
                    score = (region_data['has_positive'].mean() - region_data['has_negative'].mean()) * 100
                    regional_scores.append(score)
            if regional_scores:
                variances.append(np.std(regional_scores))
        probe_variances.append(np.mean(variances))

    ax.bar(probe_labels, probe_variances, color='#8B5CF6', alpha=0.7)
    ax.set_ylabel('Regional Variance (Std Dev)')
    ax.set_title('Regional Disparity by Probe', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Metric 4: Worst probe identification
    ax = axes[1, 1]
    combined_bias = {}
    for p in probes:
        all_biases = []
        for data in all_data.values():
            valid = data[(data['probe_id'] == p) & (data['is_error'] == 0)]
            bias = abs((valid['has_positive'].mean() - valid['has_negative'].mean()) * 100)
            all_biases.append(bias)
        combined_bias[PROBE_NAMES[p]] = np.mean(all_biases)

    sorted_probes = sorted(combined_bias.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_probes)
    colors = ['#EF4444' if i == 0 else '#F59E0B' if i == 1 else '#3B82F6' for i in range(len(labels))]
    ax.barh(labels, values, color=colors, alpha=0.7)
    ax.set_xlabel('Avg Absolute Bias Score')
    ax.set_title('Probe Ranking by Bias Severity', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(values):
        ax.text(v + 0.1, i, f'{v:.2f}', va='center', fontsize=9)

    plt.suptitle('Probe-Level Bias Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path.name}")

def create_effect_size_table(all_data, output_path):
    """Cohen's d effect sizes between regions."""
    regions = list(REGION_COLORS.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (model_name, data) in enumerate(all_data.items()):
        if idx >= 6:
            break

        ax = axes[idx]
        valid = data[data['is_error'] == 0]

        # Compute Cohen's d between all region pairs
        effect_matrix = np.zeros((len(regions), len(regions)))

        for i, r1 in enumerate(regions):
            for j, r2 in enumerate(regions):
                if i == j:
                    effect_matrix[i, j] = 0
                else:
                    d1 = valid[valid['jurisdiction_region'] == r1]
                    d2 = valid[valid['jurisdiction_region'] == r2]

                    if len(d1) > 0 and len(d2) > 0:
                        # Use bias indicator (pos - neg)
                        scores1 = d1['bias_indicator'].values
                        scores2 = d2['bias_indicator'].values

                        pooled_std = np.sqrt((np.std(scores1)**2 + np.std(scores2)**2) / 2)
                        if pooled_std > 0:
                            cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                            effect_matrix[i, j] = cohens_d

        sns.heatmap(effect_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   xticklabels=[r[:3] for r in regions],
                   yticklabels=[r[:3] for r in regions],
                   vmin=-0.5, vmax=0.5, ax=ax, cbar_kws={'label': "Cohen's d"})

        display_name = MODEL_NAMES.get(model_name, model_name)
        ax.set_title(display_name, fontsize=11, fontweight='bold')

    if len(all_data) < 6:
        axes[-1].axis('off')

    plt.suptitle("Effect Sizes Between Regions (Cohen's d)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path.name}")

def create_composite_summary_table(all_data, output_path):
    """Create comprehensive summary table."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Compute summary statistics
    table_data = []
    headers = ['Model', 'Success %', 'Avg Length', 'Pos %', 'Neg %', 'Bias', 'Max Regional Gap']

    regions = list(REGION_COLORS.keys())

    for model_name, data in all_data.items():
        valid = data[data['is_error'] == 0]

        success_rate = 100 - (data['is_error'].mean() * 100)
        avg_length = valid['response_length'].mean()
        pos_rate = valid['has_positive'].mean() * 100
        neg_rate = valid['has_negative'].mean() * 100
        overall_bias = pos_rate - neg_rate

        # Compute max regional gap
        regional_biases = []
        for region in regions:
            region_data = valid[valid['jurisdiction_region'] == region]
            if len(region_data) > 0:
                rbias = (region_data['has_positive'].mean() - region_data['has_negative'].mean()) * 100
                regional_biases.append(rbias)

        max_gap = max(regional_biases) - min(regional_biases) if regional_biases else 0

        table_data.append([
            MODEL_NAMES.get(model_name, model_name),
            f'{success_rate:.1f}',
            f'{avg_length:.1f}',
            f'{pos_rate:.1f}',
            f'{neg_rate:.1f}',
            f'{overall_bias:+.1f}',
            f'{max_gap:.2f}'
        ])

    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3B82F6')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F3F4F6')

    plt.title('Comprehensive Model Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path.name}")

def create_regional_variance_composite(all_data, output_path):
    """Regional variance analysis composite figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig)

    regions = list(REGION_COLORS.keys())

    # Main plot: Regional bias variance
    ax_main = fig.add_subplot(gs[0:2, 0:2])

    for model_name, data in all_data.items():
        valid = data[data['is_error'] == 0]
        means = []
        stds = []

        for region in regions:
            region_data = valid[valid['jurisdiction_region'] == region]
            if len(region_data) > 0:
                bias_scores = (region_data['has_positive'] - region_data['has_negative']).values * 100
                means.append(np.mean(bias_scores))
                stds.append(np.std(bias_scores))
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(len(regions))
        ax_main.errorbar(x, means, yerr=stds, marker='o', label=MODEL_NAMES.get(model_name, model_name)[:12],
                        linewidth=2, capsize=5, markersize=6)

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(regions, rotation=45, ha='right')
    ax_main.set_ylabel('Bias Score (Mean ± SD)')
    ax_main.set_title('Regional Bias with Variance', fontweight='bold', fontsize=12)
    ax_main.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax_main.legend(fontsize=9, loc='best')
    ax_main.grid(alpha=0.3)

    # Side plot: Variance by region
    ax_side = fig.add_subplot(gs[0:2, 2])

    region_variances = []
    for region in regions:
        all_vars = []
        for data in all_data.values():
            valid = data[(data['jurisdiction_region'] == region) & (data['is_error'] == 0)]
            if len(valid) > 0:
                bias_scores = (valid['has_positive'] - valid['has_negative']).values * 100
                all_vars.append(np.std(bias_scores))
        region_variances.append(np.mean(all_vars))

    colors_list = [REGION_COLORS[r] for r in regions]
    ax_side.barh(regions, region_variances, color=colors_list, alpha=0.7)
    ax_side.set_xlabel('Avg Std Dev')
    ax_side.set_title('Variance by Region', fontweight='bold', fontsize=11)
    ax_side.grid(axis='x', alpha=0.3)

    # Bottom plots: Distribution histograms
    for i, region in enumerate(regions[:3]):
        ax = fig.add_subplot(gs[2, i])

        for model_data in all_data.values():
            valid = model_data[(model_data['jurisdiction_region'] == region) & (model_data['is_error'] == 0)]
            if len(valid) > 0:
                bias_scores = (valid['has_positive'] - valid['has_negative']).values * 100
                ax.hist(bias_scores, bins=20, alpha=0.3, density=True)

        ax.set_title(region, fontweight='bold', fontsize=10)
        ax.set_xlabel('Bias Score')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.grid(alpha=0.2)

    plt.suptitle('Regional Variance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path.name}")

def main():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--results', type=str, required=True)
    parser.add_argument('--output', type=str, default='figures/publication/')
    args = parser.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FingerPrint² Publication Figure Generation")
    print("=" * 70)
    print()

    # Load data
    db_files = [f for f in results_path.glob("*.db")]
    completed_dbs = []

    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        count = conn.execute("SELECT COUNT(*) FROM probe_results").fetchone()[0]
        conn.close()
        if count >= 175000:
            completed_dbs.append(db_file)
            print(f"✓ {db_file.stem}: {count:,} results")

    if not completed_dbs:
        print("❌ No completed databases found!")
        return

    print(f"\n📊 Processing {len(completed_dbs)} models...")

    all_data = {}
    for db_file in completed_dbs:
        model_name = db_file.stem.replace('_35k', '').replace('_20260427_114159', '') \
                             .replace('_20260427_125809', '').replace('_20260421_145205', '') \
                             .replace('_20260421_145210', '')
        print(f"  Loading {model_name}...")
        all_data[model_name] = load_and_process_data(db_file)

    print("\n🎨 Generating figures...")

    create_model_performance_bars(all_data, output_path / '1_model_performance_bars.png')
    create_regional_heatmaps(all_data, output_path / '2_regional_heatmaps.png')
    create_radar_charts(all_data, output_path / '3_radar_charts.png')
    create_probe_severity_analysis(all_data, output_path / '4_probe_severity.png')
    create_effect_size_table(all_data, output_path / '5_effect_sizes.png')
    create_composite_summary_table(all_data, output_path / '6_summary_table.png')
    create_regional_variance_composite(all_data, output_path / '7_regional_variance.png')

    print("\n" + "=" * 70)
    print("✅ Complete! Generated 7 publication figures")
    print("=" * 70)

if __name__ == '__main__':
    main()
