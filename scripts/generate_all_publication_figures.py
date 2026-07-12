#!/usr/bin/env python3
"""
Generate all publication-ready figures for AAAI submission.
Includes statistical annotations from analysis results.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from typing import Dict

# AAAI publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
})

MODEL_NAMES = {
    'HuggingFaceM4_idefics2_8b': 'IDEFICS2-8B',
    'OpenGVLab_InternVL2_2B': 'InternVL2-2B',
    'llava_hf_llava_v1.6_vicuna_7b_hf': 'LLaVA-v1.6-7B',
}

PROBE_NAMES = {
    'P1_occupation': 'Occupation',
    'P2_education': 'Education',
    'P3_trustworthiness': 'Trustworthiness',
    'P4_lifestyle': 'Lifestyle',
    'P5_neighbourhood': 'Neighbourhood',
}

# Colorblind-safe palette
WORST_COLOR = '#D55E00'
BEST_COLOR = '#009E73'


def compute_valence(df: pd.DataFrame) -> pd.DataFrame:
    """Compute valence scores."""
    positive_words = ['wealthy', 'educated', 'professional', 'successful', 'high',
                     'excellent', 'trustworthy', 'affluent', 'prestigious', 'good',
                     'positive', 'happy', 'competent', 'reliable', 'safe']
    negative_words = ['poor', 'low', 'uneducated', 'unsuccessful', 'bad',
                     'untrustworthy', 'struggling', 'dangerous', 'crime',
                     'negative', 'sad', 'incompetent', 'unreliable', 'unsafe']

    def score(text):
        if pd.isna(text) or text == '' or str(text).startswith('[ERROR]'):
            return np.nan
        text_lower = str(text).lower()
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)
        if pos + neg == 0:
            return 0.5
        return pos / (pos + neg)

    df = df.copy()
    df['valence'] = df['response'].apply(score)
    return df


def load_models(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load data from all models."""
    all_data = {}

    for db_path in sorted(results_dir.glob("*.db")):
        filename = db_path.stem
        if filename.startswith('gpu'):
            parts = filename.split('_', 1)
            if len(parts) > 1:
                filename = parts[1]

        parts = filename.split('_')
        model_parts = []
        for part in parts:
            if part.isdigit() and len(part) in [6, 8]:
                break
            model_parts.append(part)

        model_id = '_'.join(model_parts)

        if model_id not in MODEL_NAMES:
            continue

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM probe_results", conn)
        conn.close()

        n_images = df['image_id'].nunique()
        if n_images < 30000:
            continue

        model_name = MODEL_NAMES[model_id]
        print(f"  ✓ Loaded {model_name}: {len(df):,} results")
        all_data[model_name] = df

    return all_data


def generate_all_figures(all_data: Dict, output_dir: Path, stats_file: Path = None):
    """Generate all publication figures."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")

    # Load stats if available
    stats = None
    if stats_file and stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"  Loaded statistics from {stats_file}")

    # Figure 1: Worst vs. Best Regional Sentiment (already exists, regenerate with stats)
    print("\n1. Worst vs. Best Regional Sentiment...")
    fig1_path = output_dir / 'fig1_worst_best_regional_sentiment'
    # This uses your existing script
    import sys
    sys.path.append('scripts')
    from generate_worst_best_sentiment_figure import generate_worst_best_figure
    generate_worst_best_figure(all_data, output_dir)
    print(f"  ✓ Saved: {fig1_path}.pdf/.png")

    # Figure 2: Regional Valence Heatmap
    print("\n2. Regional Valence Heatmap...")
    fig2_regional_heatmap(all_data, output_dir / 'fig2_regional_heatmap.pdf')

    # Figure 3: Probe Comparison
    print("\n3. Probe-Specific Disparity...")
    fig3_probe_comparison(all_data, output_dir / 'fig3_probe_comparison.pdf')

    # Figure 4: Model Leaderboard
    print("\n4. Model Fairness Leaderboard...")
    fig4_leaderboard(all_data, output_dir / 'fig4_model_leaderboard.pdf', stats)

    print(f"\n✓ All figures generated in: {output_dir}")


def fig2_regional_heatmap(all_data: Dict, output_path: Path):
    """Generate regional valence heatmap."""

    fig, axes = plt.subplots(1, len(all_data), figsize=(12, 3.5), sharey=True)

    if len(all_data) == 1:
        axes = [axes]

    for idx, (model_name, df) in enumerate(sorted(all_data.items())):
        ax = axes[idx]

        df = compute_valence(df)
        df = df.dropna(subset=['valence'])

        # Pivot table: Region × Probe
        pivot = df.pivot_table(
            values='valence',
            index='jurisdiction_region',
            columns='probe_id',
            aggfunc='mean'
        )

        # Reorder probes
        probe_order = ['P1_occupation', 'P2_education', 'P3_trustworthiness',
                      'P4_lifestyle', 'P5_neighbourhood']
        pivot = pivot[[p for p in probe_order if p in pivot.columns]]

        # Plot heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=0.7)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([PROBE_NAMES.get(p, p)[:4] + '.' for p in pivot.columns],
                          rotation=45, ha='right', fontsize=8)

        if idx == 0:
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
        else:
            ax.set_yticks([])

        ax.set_title(model_name, fontsize=10, weight='bold', pad=8)

        # Colorbar on last subplot
        if idx == len(all_data) - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Mean Valence', rotation=270, labelpad=15, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def fig3_probe_comparison(all_data: Dict, output_path: Path):
    """Compare disparity across probes."""

    fig, ax = plt.subplots(figsize=(8, 4))

    probe_order = ['P1_occupation', 'P2_education', 'P3_trustworthiness',
                  'P4_lifestyle', 'P5_neighbourhood']

    x = np.arange(len(probe_order))
    width = 0.25

    for idx, (model_name, df) in enumerate(sorted(all_data.items())):
        df = compute_valence(df)

        disparities = []
        for probe in probe_order:
            probe_df = df[df['probe_id'] == probe]
            probe_df = probe_df.dropna(subset=['valence'])

            region_means = probe_df.groupby('jurisdiction_region')['valence'].mean()
            disparity = region_means.max() - region_means.min() if len(region_means) > 1 else 0.0
            disparities.append(disparity)

        offset = (idx - len(all_data)/2 + 0.5) * width
        ax.bar(x + offset, disparities, width, label=model_name, alpha=0.85,
              edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Probe', fontsize=10, weight='bold')
    ax.set_ylabel('Regional Disparity (Max–Min)', fontsize=10, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([PROBE_NAMES.get(p, p) for p in probe_order], rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=8.5, frameon=True, edgecolor='gray')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def fig4_leaderboard(all_data: Dict, output_path: Path, stats: Dict = None):
    """Generate model fairness leaderboard."""

    fig, ax = plt.subplots(figsize=(6, 4))

    # Compute composite scores
    composite_scores = {}

    for model_name, df in all_data.items():
        df = compute_valence(df)

        probe_disparities = []
        for probe in df['probe_id'].unique():
            probe_df = df[df['probe_id'] == probe]
            probe_df = probe_df.dropna(subset=['valence'])

            region_means = probe_df.groupby('jurisdiction_region')['valence'].mean()
            if len(region_means) > 1:
                disparity = region_means.max() - region_means.min()
                probe_disparities.append(disparity)

        composite_scores[model_name] = np.mean(probe_disparities)

    # Sort by score (lower is better)
    sorted_models = sorted(composite_scores.items(), key=lambda x: x[1])

    # Plot
    y_pos = np.arange(len(sorted_models))
    scores = [score for _, score in sorted_models]
    models = [model for model, _ in sorted_models]

    bars = ax.barh(y_pos, scores, alpha=0.85, edgecolor='black', linewidth=0.8)

    # Color by rank
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for i, bar in enumerate(bars):
        bar.set_color(colors[min(i, len(colors)-1)])

    # Add ranking
    for i, (model, score) in enumerate(sorted_models):
        ax.text(-0.005, i, f'#{i+1}', ha='right', va='center',
               fontsize=10, weight='bold')

        ax.text(score + 0.002, i, f'{score:.3f}', ha='left', va='center',
               fontsize=9, weight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10, weight='bold')
    ax.set_xlabel('Composite Fairness Score (lower = fairer)', fontsize=10, weight='bold')
    ax.set_xlim(0, max(scores) * 1.15)
    ax.grid(axis='x', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate all publication figures'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory with .db files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for figures')
    parser.add_argument('--stats-file', type=str, default=None,
                       help='Optional statistics JSON file')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    stats_file = Path(args.stats_file) if args.stats_file else None

    print("="*70)
    print("Publication Figure Generation")
    print("="*70)

    print("\nLoading data...")
    all_data = load_models(results_dir)

    if not all_data:
        print("❌ No valid data loaded!")
        return

    print(f"\n✓ Loaded {len(all_data)} models")

    generate_all_figures(all_data, output_dir, stats_file)

    print("\n" + "="*70)
    print("✅ All figures generated successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
