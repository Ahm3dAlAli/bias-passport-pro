#!/usr/bin/env python3
"""
Generate worst- vs. best-treated group sentiment per model figure.
Shows the regional disparity in sentiment scores across models.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Publication-quality settings for AAAI
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
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
})

# Model display names
MODEL_NAMES = {
    'HuggingFaceM4_idefics2_8b': 'IDEFICS2-8B',
    'OpenGVLab_InternVL2_2B': 'InternVL2-2B',
    'llava_hf_llava_v1.6_vicuna_7b_hf': 'LLaVA-v1.6-7B',
}

# Color schemes - colorblind-friendly palette
MODEL_COLORS = {
    'IDEFICS2-8B': '#E69F00',      # Orange (colorblind-safe)
    'InternVL2-2B': '#56B4E9',     # Sky blue (colorblind-safe)
    'LLaVA-v1.6-7B': '#009E73',    # Bluish green (colorblind-safe)
}

# Colors for worst/best bars (using accessible red-green alternatives)
WORST_COLOR = '#D55E00'   # Vermillion (colorblind-safe "red")
BEST_COLOR = '#009E73'    # Bluish green (colorblind-safe "green")


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


def load_model_data(results_dir: Path) -> dict:
    """Load data from all working models."""
    all_data = {}

    for db_path in sorted(results_dir.glob("*.db")):
        filename = db_path.stem

        # Remove gpu prefix
        if filename.startswith('gpu'):
            parts = filename.split('_', 1)
            if len(parts) > 1:
                filename = parts[1]

        # Remove timestamp suffix
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
            print(f"  ⚠️  Skipping {model_id} - only {n_images:,} images")
            continue

        model_name = MODEL_NAMES[model_id]
        print(f"  ✓ Loaded {model_name}: {len(df):,} results ({n_images:,} images)")
        all_data[model_name] = df

    return all_data


def compute_worst_best_by_region(df: pd.DataFrame) -> dict:
    """Compute worst and best treated regions based on valence scores with confidence intervals."""
    df = compute_valence_scores(df)
    df = df.dropna(subset=['valence'])

    # Calculate mean valence per region with std error for error bars
    region_stats = df.groupby('jurisdiction_region')['valence'].agg(['mean', 'sem', 'count'])

    if len(region_stats) == 0:
        return {
            'worst_region': None,
            'worst_score': 0.0,
            'worst_sem': 0.0,
            'worst_n': 0,
            'best_region': None,
            'best_score': 0.0,
            'best_sem': 0.0,
            'best_n': 0,
            'gap': 0.0
        }

    worst_region = region_stats['mean'].idxmin()
    best_region = region_stats['mean'].idxmax()

    return {
        'worst_region': worst_region,
        'worst_score': region_stats.loc[worst_region, 'mean'],
        'worst_sem': region_stats.loc[worst_region, 'sem'],
        'worst_n': int(region_stats.loc[worst_region, 'count']),
        'best_region': best_region,
        'best_score': region_stats.loc[best_region, 'mean'],
        'best_sem': region_stats.loc[best_region, 'sem'],
        'best_n': int(region_stats.loc[best_region, 'count']),
        'gap': region_stats.loc[best_region, 'mean'] - region_stats.loc[worst_region, 'mean']
    }


def generate_worst_best_figure(all_data: dict, output_path: Path):
    """Generate worst vs. best treated group sentiment figure."""
    print("\nGenerating worst vs. best treated group sentiment figure...")

    # Collect data for all models
    models = []
    worst_scores = []
    best_scores = []
    worst_sems = []
    best_sems = []
    worst_regions = []
    best_regions = []
    worst_ns = []
    best_ns = []
    gaps = []

    for model_name in sorted(all_data.keys()):
        df = all_data[model_name]
        stats = compute_worst_best_by_region(df)

        models.append(model_name)
        worst_scores.append(stats['worst_score'])
        best_scores.append(stats['best_score'])
        worst_sems.append(stats['worst_sem'])
        best_sems.append(stats['best_sem'])
        worst_regions.append(stats['worst_region'])
        best_regions.append(stats['best_region'])
        worst_ns.append(stats['worst_n'])
        best_ns.append(stats['best_n'])
        gaps.append(stats['gap'])

        print(f"  {model_name}:")
        print(f"    Worst: {stats['worst_region']} ({stats['worst_score']:.3f} ± {stats['worst_sem']:.4f}, n={stats['worst_n']:,})")
        print(f"    Best:  {stats['best_region']} ({stats['best_score']:.3f} ± {stats['best_sem']:.4f}, n={stats['best_n']:,})")
        print(f"    Gap:   {stats['gap']:.3f}")

    # Create figure with AAAI-appropriate size (3.25" for single column, 6.75" for double)
    # Using 6.75" width for double-column figure
    fig, ax = plt.subplots(figsize=(6.75, 3.5))

    x = np.arange(len(models))
    width = 0.32

    # Create bars with colorblind-friendly colors and error bars
    bars1 = ax.bar(x - width/2, worst_scores, width,
                   label='Worst-treated region',
                   color=WORST_COLOR, alpha=0.9, edgecolor='black', linewidth=0.8,
                   yerr=worst_sems, capsize=3, error_kw={'linewidth': 1, 'ecolor': 'black', 'alpha': 0.7})
    bars2 = ax.bar(x + width/2, best_scores, width,
                   label='Best-treated region',
                   color=BEST_COLOR, alpha=0.9, edgecolor='black', linewidth=0.8,
                   yerr=best_sems, capsize=3, error_kw={'linewidth': 1, 'ecolor': 'black', 'alpha': 0.7})

    # Add gap indicators with connecting lines
    for i, (worst, best, gap) in enumerate(zip(worst_scores, best_scores, gaps)):
        # Draw connecting line
        ax.plot([i - width/2, i + width/2], [worst, best],
               'k--', alpha=0.3, linewidth=1.0, zorder=1)

    # Add value labels on bars with better visibility
    for i, (bar, score, region, n) in enumerate(zip(bars1, worst_scores, worst_regions, worst_ns)):
        height = bar.get_height()
        # Shorten region names for better fit
        region_short = str(region).replace('Northern ', 'N. ')
        if len(region_short) > 12:
            region_short = region_short[:12] + '.'
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.015,
               region_short,
               ha='center', va='top', fontsize=7, rotation=0,
               color='white', weight='bold')
        # Add sample size below bar
        ax.text(bar.get_x() + bar.get_width()/2., 0.41,
               f'n={n//1000}k',
               ha='center', va='bottom', fontsize=6, rotation=0,
               color='#555555', style='italic')

    for i, (bar, score, region, n) in enumerate(zip(bars2, best_scores, best_regions, best_ns)):
        height = bar.get_height()
        region_short = str(region).replace('Northern ', 'N. ')
        if len(region_short) > 12:
            region_short = region_short[:12] + '.'
        # Position above error bar
        error_top = height + best_sems[i]
        ax.text(bar.get_x() + bar.get_width()/2., error_top + 0.01,
               region_short,
               ha='center', va='bottom', fontsize=7, rotation=0,
               weight='bold', color='#333333')
        # Add sample size below bar
        ax.text(bar.get_x() + bar.get_width()/2., 0.41,
               f'n={n//1000}k',
               ha='center', va='bottom', fontsize=6, rotation=0,
               color='#555555', style='italic')

    # Formatting for AAAI standards
    ax.set_xlabel('Model', fontsize=10, weight='bold')
    ax.set_ylabel('Mean Valence Score', fontsize=10, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, weight='bold')

    # Legend with better positioning and styling
    ax.legend(loc='upper left', frameon=True, fontsize=8.5,
             edgecolor='gray', fancybox=False, framealpha=0.95,
             borderpad=0.5, handlelength=1.5)

    # Set appropriate y-axis limits with some padding
    ax.set_ylim(0.4, 0.75)

    # Grid for better readability
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5, zorder=0)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    plt.tight_layout()

    # Save figure
    pdf_path = output_path / 'worst_best_regional_sentiment.pdf'
    png_path = output_path / 'worst_best_regional_sentiment.png'

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate worst vs. best treated group sentiment figure'
    )
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results directory with .db files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for figure')
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Worst- vs. Best-Treated Regional Sentiment Figure Generation")
    print("=" * 70)
    print(f"Results: {results_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("📊 Loading model data...")
    all_data = load_model_data(results_dir)

    if len(all_data) == 0:
        print("❌ No valid data loaded!")
        return

    print(f"\n✓ Loaded {len(all_data)} working models")

    # Generate figure
    generate_worst_best_figure(all_data, output_dir)

    print("\n" + "=" * 70)
    print("✅ Figure generated successfully!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
