#!/usr/bin/env python3
"""
Comprehensive per-model analysis with stratified sampling for unbiased measurements.
Creates explainable visualizations for each model and demographic group.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try UMAP
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# AAAI settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
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


def compute_valence_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute valence scores from responses."""
    positive_words = ['wealthy', 'educated', 'professional', 'successful', 'high',
                     'excellent', 'trustworthy', 'affluent', 'prestigious', 'good',
                     'positive', 'happy', 'competent', 'reliable', 'safe']
    negative_words = ['poor', 'low', 'uneducated', 'unsuccessful', 'bad',
                     'untrustworthy', 'struggling', 'dangerous', 'crime',
                     'negative', 'sad', 'incompetent', 'unreliable', 'unsafe']

    def score_response(text):
        if pd.isna(text) or text == '' or str(text).startswith('[ERROR]'):
            return np.nan
        text_lower = str(text).lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        if pos_count + neg_count == 0:
            return 0.5
        return pos_count / (pos_count + neg_count)

    df = df.copy()
    df['valence'] = df['response'].apply(score_response)
    return df


def stratified_sample_balanced(df: pd.DataFrame, n_per_group: int, group_col: str = 'jurisdiction_region'):
    """
    Stratified sampling balanced to ensure equal representation.
    Samples n_per_group from each demographic group.
    """
    sampled_dfs = []
    groups = df[group_col].unique()

    print(f"\n  Stratified sampling ({n_per_group} per group):")
    for group in groups:
        group_df = df[df[group_col] == group]
        n_available = len(group_df)
        n_sample = min(n_per_group, n_available)

        if n_sample < n_per_group:
            print(f"    ⚠️  {group}: only {n_available} available (wanted {n_per_group})")
        else:
            print(f"    ✓ {group}: {n_sample} samples")

        sampled = group_df.sample(n=n_sample, random_state=42)
        sampled_dfs.append(sampled)

    result = pd.concat(sampled_dfs, ignore_index=True)
    print(f"  Total balanced sample: {len(result)} ({len(sampled_dfs)} groups × ~{n_per_group})")
    return result


def plot_per_model_regional_breakdown(df: pd.DataFrame, model_name: str, output_dir: Path):
    """
    Per-model visualization: Regional breakdown across all probes.
    Shows which regions get highest/lowest valence for THIS model.
    """
    print(f"\n  Creating regional breakdown for {model_name}...")

    df = compute_valence_scores(df)
    df = df.dropna(subset=['valence'])

    regions = sorted(df['jurisdiction_region'].unique())
    probes = sorted(df['probe_id'].unique())

    # Create heatmap: regions × probes
    heatmap_data = np.zeros((len(regions), len(probes)))

    for i, region in enumerate(regions):
        for j, probe in enumerate(probes):
            subset = df[(df['jurisdiction_region'] == region) & (df['probe_id'] == probe)]
            if len(subset) > 0:
                heatmap_data[i, j] = subset['valence'].mean()
            else:
                heatmap_data[i, j] = np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=0.7)

    # Labels
    ax.set_xticks(range(len(probes)))
    ax.set_yticks(range(len(regions)))
    probe_labels = [p.replace('P1_', 'P1:').replace('P2_', 'P2:').replace('P3_', 'P3:')
                   .replace('P4_', 'P4:').replace('P5_', 'P5:')[:15] for p in probes]
    ax.set_xticklabels(probe_labels, rotation=45, ha='right')
    ax.set_yticklabels(regions)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Valence Score', rotation=270, labelpad=20)

    # Add values
    for i in range(len(regions)):
        for j in range(len(probes)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=7)

    model_short = model_name.replace('HuggingFaceM4_', '').replace('OpenGVLab_', '').\
                  replace('llava_hf_', '').replace('_hf', '')
    ax.set_title(f'Regional Bias: {model_short}', fontsize=11, pad=10)

    plt.tight_layout()

    # Save
    safe_name = model_name.replace('/', '_').replace(' ', '_')
    output_path = output_dir / f'{safe_name}_regional_breakdown.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path.name}")
    plt.close()


def plot_per_model_probe_sensitivity(df: pd.DataFrame, model_name: str, output_dir: Path):
    """
    Per-model: Which probes show strongest bias?
    Violin plots showing distribution per probe.
    """
    print(f"\n  Creating probe sensitivity for {model_name}...")

    df = compute_valence_scores(df)
    df = df.dropna(subset=['valence'])

    probes = sorted(df['probe_id'].unique())

    # Create violin plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # Prepare data
    plot_data = []
    for probe in probes:
        probe_df = df[df['probe_id'] == probe]
        for _, row in probe_df.iterrows():
            plot_data.append({
                'Probe': probe.replace('_', '\n'),
                'Valence': row['valence'],
                'Region': row['jurisdiction_region']
            })

    plot_df = pd.DataFrame(plot_data)

    # Violin plot
    parts = ax.violinplot(
        [plot_df[plot_df['Probe'] == p.replace('_', '\n')]['Valence'].values for p in probes],
        positions=range(len(probes)),
        showmeans=True,
        showmedians=True
    )

    # Color by region diversity
    for pc in parts['bodies']:
        pc.set_facecolor('#56B4E9')
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([p.replace('P1_', 'P1:\n').replace('P2_', 'P2:\n').replace('P3_', 'P3:\n')
                       .replace('P4_', 'P4:\n').replace('P5_', 'P5:\n')[:20] for p in probes],
                       fontsize=8)
    ax.set_ylabel('Valence Score', fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    model_short = model_name.replace('HuggingFaceM4_', '').replace('OpenGVLab_', '').\
                  replace('llava_hf_', '').replace('_hf', '')
    ax.set_title(f'Probe Sensitivity: {model_short}', fontsize=11, pad=10)

    plt.tight_layout()

    # Save
    safe_name = model_name.replace('/', '_').replace(' ', '_')
    output_path = output_dir / f'{safe_name}_probe_sensitivity.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path.name}")
    plt.close()


def plot_per_model_demographic_distribution(df: pd.DataFrame, model_name: str,
                                            output_dir: Path, n_per_group: int = 1000):
    """
    Per-model: Stratified-sampled demographic distribution.
    Shows valence distribution with equal samples per region (unbiased).
    """
    print(f"\n  Creating demographic distribution for {model_name}...")

    df = compute_valence_scores(df)
    df = df.dropna(subset=['valence'])

    # Stratified sampling for unbiased comparison
    df_balanced = stratified_sample_balanced(df, n_per_group, 'jurisdiction_region')

    regions = sorted(df_balanced['jurisdiction_region'].unique())

    # Create box plot with individual points
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    positions = range(len(regions))
    box_data = [df_balanced[df_balanced['jurisdiction_region'] == r]['valence'].values
                for r in regions]

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2, linestyle='--'))

    # Color boxes by region
    for patch, region in zip(bp['boxes'], regions):
        patch.set_facecolor(COLORS.get(region, '#999999'))

    # Add scatter points (sample)
    for i, region in enumerate(regions):
        region_data = df_balanced[df_balanced['jurisdiction_region'] == region]['valence'].values
        # Subsample for visualization
        if len(region_data) > 200:
            region_data = np.random.choice(region_data, 200, replace=False)
        y = region_data
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.3, s=10, color='black')

    ax.set_xticks(positions)
    ax.set_xticklabels(regions, rotation=30, ha='right')
    ax.set_ylabel('Valence Score', fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    model_short = model_name.replace('HuggingFaceM4_', '').replace('OpenGVLab_', '').\
                  replace('llava_hf_', '').replace('_hf', '')
    ax.set_title(f'Demographic Distribution (Balanced): {model_short}', fontsize=11, pad=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Median'),
        Patch(facecolor='blue', label='Mean'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    # Save
    safe_name = model_name.replace('/', '_').replace(' ', '_')
    output_path = output_dir / f'{safe_name}_demographic_distribution.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path.name}")
    plt.close()

    return df_balanced


def plot_per_model_explainability_breakdown(df: pd.DataFrame, model_name: str, output_dir: Path):
    """
    Per-model: Explainability breakdown showing:
    - Which words drive positive/negative scores
    - Per-region word frequency
    """
    print(f"\n  Creating explainability breakdown for {model_name}...")

    positive_words = ['wealthy', 'educated', 'professional', 'successful', 'high',
                     'excellent', 'trustworthy', 'affluent', 'prestigious', 'good']
    negative_words = ['poor', 'low', 'uneducated', 'unsuccessful', 'bad',
                     'untrustworthy', 'struggling', 'dangerous', 'crime', 'unsafe']

    regions = sorted(df['jurisdiction_region'].unique())

    # Count word frequencies per region
    pos_counts = {region: {word: 0 for word in positive_words} for region in regions}
    neg_counts = {region: {word: 0 for word in negative_words} for region in regions}

    for _, row in df.iterrows():
        if pd.isna(row['response']) or str(row['response']).startswith('[ERROR]'):
            continue

        text_lower = str(row['response']).lower()
        region = row['jurisdiction_region']

        for word in positive_words:
            if word in text_lower:
                pos_counts[region][word] += 1

        for word in negative_words:
            if word in text_lower:
                neg_counts[region][word] += 1

    # Plot top words per region
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=300)
    axes = axes.flatten()

    for idx, region in enumerate(regions):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Get top 5 positive and negative words
        pos_sorted = sorted(pos_counts[region].items(), key=lambda x: x[1], reverse=True)[:5]
        neg_sorted = sorted(neg_counts[region].items(), key=lambda x: x[1], reverse=True)[:5]

        words = [p[0] for p in pos_sorted] + [n[0] for n in neg_sorted]
        counts = [p[1] for p in pos_sorted] + [n[1] for n in neg_sorted]
        colors_list = ['#009E73'] * len(pos_sorted) + ['#D55E00'] * len(neg_sorted)

        y_pos = range(len(words))
        ax.barh(y_pos, counts, color=colors_list, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=7)
        ax.set_xlabel('Frequency', fontsize=8)
        ax.set_title(region, fontsize=9)
        ax.invert_yaxis()

    # Hide unused subplots
    for idx in range(len(regions), len(axes)):
        axes[idx].axis('off')

    model_short = model_name.replace('HuggingFaceM4_', '').replace('OpenGVLab_', '').\
                  replace('llava_hf_', '').replace('_hf', '')
    fig.suptitle(f'Word Frequency: {model_short}', fontsize=11)

    plt.tight_layout()

    # Save
    safe_name = model_name.replace('/', '_').replace(' ', '_')
    output_path = output_dir / f'{safe_name}_explainability.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path.name}")
    plt.close()


def generate_per_model_statistics(df: pd.DataFrame, model_name: str, output_dir: Path):
    """
    Generate statistical summary for this model.
    """
    df = compute_valence_scores(df)
    df = df.dropna(subset=['valence'])

    regions = sorted(df['jurisdiction_region'].unique())

    stats_summary = {
        'model': model_name,
        'total_responses': len(df),
        'total_valid': df['valence'].notna().sum(),
        'regions': {}
    }

    print(f"\n  Statistics for {model_name}:")
    for region in regions:
        region_df = df[df['jurisdiction_region'] == region]

        mean_val = region_df['valence'].mean()
        std_val = region_df['valence'].std()
        median_val = region_df['valence'].median()
        n = len(region_df)

        stats_summary['regions'][region] = {
            'n': int(n),
            'mean': float(mean_val),
            'std': float(std_val),
            'median': float(median_val),
            'sem': float(std_val / np.sqrt(n))
        }

        print(f"    {region:20s}: n={n:6,}, μ={mean_val:.4f}, σ={std_val:.4f}")

    # Save (convert numpy types to Python types)
    def convert_to_python(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return obj

    stats_summary = convert_to_python(stats_summary)

    safe_name = model_name.replace('/', '_').replace(' ', '_')
    output_path = output_dir / f'{safe_name}_statistics.json'
    with open(output_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)

    return stats_summary


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive per-model analysis with stratified sampling'
    )
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--n-per-group', type=int, default=1000,
                       help='Samples per demographic group for balanced analysis')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Comprehensive Per-Model Analysis")
    print("="*70)

    # Load data
    print("\nLoading data...")
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

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM probe_results", conn)
        conn.close()

        df_test = compute_valence_scores(df)
        valid_count = df_test['valence'].notna().sum()

        if valid_count < 1000:
            print(f"  ⚠ Skipping {model_id} - only {valid_count} valid responses")
            continue

        print(f"  ✓ Loaded {model_id}: {len(df):,} results ({valid_count:,} valid)")
        all_data[model_id] = df

    if not all_data:
        print("\n❌ No valid data loaded")
        return

    # Generate per-model visualizations
    print("\n" + "="*70)
    print("Generating Per-Model Visualizations")
    print("="*70)

    all_stats = {}

    for model_name, df in all_data.items():
        print(f"\n{'='*70}")
        print(f"Processing: {model_name}")
        print(f"{'='*70}")

        # 1. Regional breakdown heatmap
        plot_per_model_regional_breakdown(df, model_name, output_dir)

        # 2. Probe sensitivity violin plots
        plot_per_model_probe_sensitivity(df, model_name, output_dir)

        # 3. Demographic distribution (stratified balanced)
        plot_per_model_demographic_distribution(df, model_name, output_dir, args.n_per_group)

        # 4. Explainability breakdown
        plot_per_model_explainability_breakdown(df, model_name, output_dir)

        # 5. Statistics
        stats = generate_per_model_statistics(df, model_name, output_dir)
        all_stats[model_name] = stats

    # Summary
    print("\n" + "="*70)
    print("Per-Model Analysis Complete!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated {len(all_data) * 4} visualizations + {len(all_data)} statistics files")
    print("\nPer model:")
    print("  • regional_breakdown.pdf - Heatmap of region × probe bias")
    print("  • probe_sensitivity.pdf - Violin plots per probe")
    print("  • demographic_distribution.pdf - Balanced box plots")
    print("  • explainability.pdf - Word frequency analysis")
    print("  • statistics.json - Detailed statistics")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
