#!/usr/bin/env python3
"""
generate_interpretable_figures.py
==================================
Generate interpretable research figures from FingerPrint² raw evaluation data.
Focuses on measurable response characteristics rather than subjective bias scores.

Usage:
    python scripts/generate_interpretable_figures.py --results results/single_runs_35k/ --output figures/
"""

import argparse
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Publication style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

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

def load_model_results(db_path):
    """Load results from database."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        image_id,
        probe_id,
        response,
        jurisdiction_region
    FROM probe_results
    WHERE jurisdiction_region IS NOT NULL
    AND jurisdiction_region != 'unknown'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def compute_response_metrics(df):
    """Compute interpretable metrics from responses."""

    def get_metrics(text):
        if pd.isna(text) or '[ERROR]' in str(text):
            return {
                'length': 0,
                'word_count': 0,
                'unique_words': 0,
                'is_error': 1,
                'has_negative': 0,
                'has_positive': 0
            }

        text_lower = str(text).lower()
        words = text_lower.split()

        # Negative indicators
        negative_words = ['poor', 'low', 'unable', 'cannot', 'lack', 'difficult',
                         'struggle', 'limited', 'uneducated', 'unemployed']
        # Positive indicators
        positive_words = ['high', 'successful', 'educated', 'professional', 'skilled',
                         'affluent', 'advanced', 'capable', 'qualified', 'experienced']

        return {
            'length': len(text),
            'word_count': len(words),
            'unique_words': len(set(words)),
            'is_error': 0,
            'has_negative': int(any(w in text_lower for w in negative_words)),
            'has_positive': int(any(w in text_lower for w in positive_words))
        }

    metrics = df['response'].apply(get_metrics)
    metrics_df = pd.DataFrame(metrics.tolist())

    for col in metrics_df.columns:
        df[col] = metrics_df[col]

    return df

def create_error_rate_comparison(all_data, output_path):
    """Compare error rates across models and regions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # By model
    model_errors = []
    model_names = []
    for model_name, model_data in all_data.items():
        error_rate = model_data['is_error'].mean() * 100
        model_errors.append(error_rate)
        model_names.append(MODEL_NAMES.get(model_name, model_name))

    ax1.barh(model_names, model_errors, color='#DC2626', alpha=0.7)
    ax1.set_xlabel('Error Rate (%)', fontsize=12)
    ax1.set_title('Model Failure Rates', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # By region (averaged across models)
    regions = list(REGION_COLORS.keys())
    region_errors = []

    for region in regions:
        errors = []
        for model_data in all_data.values():
            region_data = model_data[model_data['jurisdiction_region'] == region]
            if len(region_data) > 0:
                errors.append(region_data['is_error'].mean() * 100)
        region_errors.append(np.mean(errors) if errors else 0)

    colors = [REGION_COLORS[r] for r in regions]
    ax2.bar(range(len(regions)), region_errors, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(regions)))
    ax2.set_xticklabels(regions, rotation=45, ha='right')
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_title('Regional Failure Rates (Avg)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Model Reliability Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def create_response_length_analysis(all_data, output_path):
    """Analyze response length by region and model."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    regions = list(REGION_COLORS.keys())

    for idx, (model_name, model_data) in enumerate(all_data.items()):
        if idx >= 6:
            break

        ax = axes[idx]

        # Get valid responses (non-errors)
        valid_data = model_data[model_data['is_error'] == 0]

        region_lengths = []
        region_labels = []

        for region in regions:
            region_data = valid_data[valid_data['jurisdiction_region'] == region]
            if len(region_data) > 0:
                region_lengths.append(region_data['word_count'].values)
                region_labels.append(region[:3])

        if region_lengths:
            bp = ax.boxplot(region_lengths, labels=region_labels, patch_artist=True)

            for patch, region in zip(bp['boxes'], regions[:len(region_lengths)]):
                patch.set_facecolor(REGION_COLORS[region])
                patch.set_alpha(0.6)

        display_name = MODEL_NAMES.get(model_name, model_name)
        ax.set_title(display_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Response Length (words)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)

    # Hide extra subplot
    if len(all_data) < 6:
        axes[-1].axis('off')

    plt.suptitle('Response Length Distribution by Region', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def create_attribute_mention_heatmap(all_data, output_path):
    """Show which models mention positive/negative attributes by region."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    regions = list(REGION_COLORS.keys())
    models = [MODEL_NAMES.get(m, m) for m in all_data.keys()]

    # Positive mentions
    positive_matrix = np.zeros((len(models), len(regions)))
    negative_matrix = np.zeros((len(models), len(regions)))

    for i, (model_name, model_data) in enumerate(all_data.items()):
        valid = model_data[model_data['is_error'] == 0]
        for j, region in enumerate(regions):
            region_data = valid[valid['jurisdiction_region'] == region]
            if len(region_data) > 0:
                positive_matrix[i, j] = region_data['has_positive'].mean() * 100
                negative_matrix[i, j] = region_data['has_negative'].mean() * 100

    # Positive heatmap
    sns.heatmap(positive_matrix, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=[r[:3] for r in regions],
                yticklabels=models, cbar_kws={'label': '% with positive words'},
                ax=ax1, vmin=0, vmax=50)
    ax1.set_title('Positive Attribute Mentions', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Region', fontsize=11)

    # Negative heatmap
    sns.heatmap(negative_matrix, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=[r[:3] for r in regions],
                yticklabels=models, cbar_kws={'label': '% with negative words'},
                ax=ax2, vmin=0, vmax=50)
    ax2.set_title('Negative Attribute Mentions', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Region', fontsize=11)

    plt.suptitle('Attribute Mention Rates by Model and Region', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def create_probe_comparison(all_data, output_path):
    """Compare response characteristics across probe types."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    probes = list(PROBE_NAMES.keys())

    # 1. Error rate by probe
    ax = axes[0, 0]
    for model_name, model_data in all_data.items():
        probe_errors = []
        for probe in probes:
            probe_data = model_data[model_data['probe_id'] == probe]
            probe_errors.append(probe_data['is_error'].mean() * 100)

        display_name = MODEL_NAMES.get(model_name, model_name)
        ax.plot(range(len(probes)), probe_errors, marker='o', label=display_name, linewidth=2)

    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([PROBE_NAMES[p][:4] for p in probes], rotation=45)
    ax.set_ylabel('Error Rate (%)', fontsize=11)
    ax.set_title('Error Rate by Probe Type', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # 2. Response length by probe
    ax = axes[0, 1]
    for model_name, model_data in all_data.items():
        probe_lengths = []
        valid = model_data[model_data['is_error'] == 0]
        for probe in probes:
            probe_data = valid[valid['probe_id'] == probe]
            probe_lengths.append(probe_data['word_count'].median())

        display_name = MODEL_NAMES.get(model_name, model_name)
        ax.plot(range(len(probes)), probe_lengths, marker='s', label=display_name, linewidth=2)

    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([PROBE_NAMES[p][:4] for p in probes], rotation=45)
    ax.set_ylabel('Median Response Length', fontsize=11)
    ax.set_title('Response Length by Probe Type', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # 3. Positive mentions by probe
    ax = axes[1, 0]
    for model_name, model_data in all_data.items():
        probe_pos = []
        valid = model_data[model_data['is_error'] == 0]
        for probe in probes:
            probe_data = valid[valid['probe_id'] == probe]
            probe_pos.append(probe_data['has_positive'].mean() * 100)

        display_name = MODEL_NAMES.get(model_name, model_name)
        ax.plot(range(len(probes)), probe_pos, marker='o', label=display_name, linewidth=2)

    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([PROBE_NAMES[p][:4] for p in probes], rotation=45)
    ax.set_ylabel('Positive Mentions (%)', fontsize=11)
    ax.set_title('Positive Attributes by Probe', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # 4. Negative mentions by probe
    ax = axes[1, 1]
    for model_name, model_data in all_data.items():
        probe_neg = []
        valid = model_data[model_data['is_error'] == 0]
        for probe in probes:
            probe_data = valid[valid['probe_id'] == probe]
            probe_neg.append(probe_data['has_negative'].mean() * 100)

        display_name = MODEL_NAMES.get(model_name, model_name)
        ax.plot(range(len(probes)), probe_neg, marker='s', label=display_name, linewidth=2)

    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([PROBE_NAMES[p][:4] for p in probes], rotation=45)
    ax.set_ylabel('Negative Mentions (%)', fontsize=11)
    ax.set_title('Negative Attributes by Probe', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Probe Type Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def create_regional_disparity_analysis(all_data, output_path):
    """Analyze disparities between regions for each model."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    regions = list(REGION_COLORS.keys())

    for idx, (model_name, model_data) in enumerate(all_data.items()):
        if idx >= 6:
            break

        ax = axes[idx]
        valid = model_data[model_data['is_error'] == 0]

        metrics = {
            'Resp Len': [],
            'Positive %': [],
            'Negative %': []
        }

        for region in regions:
            region_data = valid[valid['jurisdiction_region'] == region]
            if len(region_data) > 0:
                metrics['Resp Len'].append(region_data['word_count'].mean())
                metrics['Positive %'].append(region_data['has_positive'].mean() * 100)
                metrics['Negative %'].append(region_data['has_negative'].mean() * 100)
            else:
                metrics['Resp Len'].append(0)
                metrics['Positive %'].append(0)
                metrics['Negative %'].append(0)

        x = np.arange(len(regions))
        width = 0.25

        ax.bar(x - width, metrics['Resp Len'], width, label='Avg Length', alpha=0.8, color='#3B82F6')
        ax.bar(x, metrics['Positive %'], width, label='Positive %', alpha=0.8, color='#10B981')
        ax.bar(x + width, metrics['Negative %'], width, label='Negative %', alpha=0.8, color='#EF4444')

        display_name = MODEL_NAMES.get(model_name, model_name)
        ax.set_title(display_name, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([r[:3] for r in regions], rotation=45, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    if len(all_data) < 6:
        axes[-1].axis('off')

    plt.suptitle('Regional Response Characteristics', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate interpretable research figures')
    parser.add_argument('--results', type=str, required=True, help='Results directory')
    parser.add_argument('--output', type=str, default='figures/', help='Output directory')

    args = parser.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FingerPrint² Interpretable Figure Generation")
    print("=" * 70)
    print()

    # Find completed DBs
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

    print(f"\n📊 Loading {len(completed_dbs)} models...")

    all_data = {}
    for db_file in completed_dbs:
        model_name = db_file.stem.replace('_35k', '').replace('_20260427_114159', '') \
                             .replace('_20260427_125809', '').replace('_20260421_145205', '') \
                             .replace('_20260421_145210', '')
        print(f"  Processing {model_name}...")
        data = load_model_results(db_file)
        data = compute_response_metrics(data)
        all_data[model_name] = data

    print("\n🎨 Generating figures...")

    create_error_rate_comparison(all_data, output_path / 'error_rates.pdf')
    create_response_length_analysis(all_data, output_path / 'response_lengths.pdf')
    create_attribute_mention_heatmap(all_data, output_path / 'attribute_mentions.pdf')
    create_probe_comparison(all_data, output_path / 'probe_analysis.pdf')
    create_regional_disparity_analysis(all_data, output_path / 'regional_disparities.pdf')

    print("\n" + "=" * 70)
    print("✅ Complete! Generated 5 interpretable figures")
    print("=" * 70)

if __name__ == '__main__':
    main()
