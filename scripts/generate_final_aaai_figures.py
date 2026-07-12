#!/usr/bin/env python3
"""
Generate ALL AAAI figures in ONE folder for actual working models only.
Automatically detects models with valid data.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from itertools import combinations
import json
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


def compute_valence_scores(df):
    """Compute valence scores."""
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


def load_valid_models(results_dir):
    """Load only models with valid data."""
    print("="*70)
    print("Loading Models from Database")
    print("="*70)

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

        print(f"  ✓ {model_id}")
        print(f"    Total: {len(df):,} | Valid: {valid_count:,} ({valid_count/len(df)*100:.1f}%)")
        all_data[model_id] = df

    print(f"\n  Final: {len(all_data)} working models")
    return all_data


def get_model_short_name(model_id):
    """Get short display name."""
    return model_id.replace('HuggingFaceM4_', '').replace('OpenGVLab_', '').\
           replace('llava_hf_', '').replace('_hf', '').replace('_', '-')


def fig01_model_leaderboard_table(all_data, output_dir):
    """Figure/Table 1: Model Leaderboard."""
    print("\nGenerating: Table 1 - Model Leaderboard")

    leaderboard = []

    for model_id, df in all_data.items():
        df = compute_valence_scores(df)
        df_valid = df.dropna(subset=['valence'])

        valid_pct = len(df_valid) / len(df) * 100
        regions = df_valid['jurisdiction_region'].unique()
        probes = sorted(df_valid['probe_id'].unique())

        probe_gaps = []
        worst_probe_name = None
        worst_gap = 0

        for probe in probes:
            probe_df = df_valid[df_valid['probe_id'] == probe]
            region_means = {r: probe_df[probe_df['jurisdiction_region']==r]['valence'].mean()
                          for r in regions}
            region_means = {k:v for k,v in region_means.items() if not np.isnan(v)}

            if len(region_means) >= 2:
                gap = max(region_means.values()) - min(region_means.values())
                probe_gaps.append(gap)
                if gap > worst_gap:
                    worst_gap = gap
                    worst_probe_name = probe.replace('P1_','').replace('P2_','').replace('P3_','').\
                                      replace('P4_','').replace('P5_','').replace('_',' ').title()

        composite = np.mean(probe_gaps) if probe_gaps else np.nan

        leaderboard.append({
            'model': get_model_short_name(model_id),
            'composite': composite,
            'valid_pct': valid_pct,
            'worst_probe': worst_probe_name,
            'worst_gap': worst_gap
        })

    leaderboard.sort(key=lambda x: x['composite'])

    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering\\small")
    latex.append("\\begin{tabular}{lcccl}")
    latex.append("\\toprule")
    latex.append("Model & Composite $\\downarrow$ & Valid & Worst Probe & Worst Gap \\\\")
    latex.append("\\midrule")

    for item in leaderboard:
        latex.append(f"{item['model']} & {item['composite']:.3f} & "
                    f"{item['valid_pct']:.1f}\\% & {item['worst_probe']} & {item['worst_gap']:.3f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{\\textbf{Model leaderboard.} Ranked by composite disparity (lower is better).}")
    latex.append("\\label{tab:leaderboard}")
    latex.append("\\end{table}")

    output_path = output_dir / 'table1_model_leaderboard.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))

    print(f"  ✓ Saved: {output_path.name}")
    return leaderboard


def fig02_worst_best_bars(all_data, output_dir):
    """Figure 2: Worst vs Best Regional Sentiment."""
    print("\nGenerating: Fig 2 - Regional Disparity Bars")

    fig, ax = plt.subplots(figsize=(6.75, 4), dpi=300)

    models = list(all_data.keys())
    x = np.arange(len(models))
    width = 0.35

    worst_data = []
    best_data = []

    for model_id in models:
        df = compute_valence_scores(all_data[model_id]).dropna(subset=['valence'])
        regions = df['jurisdiction_region'].unique()
        region_means = {r: df[df['jurisdiction_region']==r]['valence'].mean() for r in regions}

        worst = min(region_means.values())
        best = max(region_means.values())
        worst_data.append(worst)
        best_data.append(best)

    ax.bar(x - width/2, worst_data, width, label='Worst Region', color='#D55E00', alpha=0.9)
    ax.bar(x + width/2, best_data, width, label='Best Region', color='#009E73', alpha=0.9)

    ax.set_ylabel('Mean Valence Score')
    ax.set_xticks(x)
    ax.set_xticklabels([get_model_short_name(m) for m in models], rotation=20, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'fig02_worst_best_regional_bars.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def fig03_combined_heatmap(all_data, output_dir):
    """Figure 3: Combined Regional × Probe Heatmap."""
    print("\nGenerating: Fig 3 - Regional Heatmap (All Models)")

    models = list(all_data.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(12, 4), sharey=True, dpi=300)

    if len(models) == 1:
        axes = [axes]

    for idx, (model_id, ax) in enumerate(zip(models, axes)):
        df = compute_valence_scores(all_data[model_id]).dropna(subset=['valence'])

        regions = sorted(df['jurisdiction_region'].unique())
        probes = sorted(df['probe_id'].unique())

        heatmap_data = np.zeros((len(regions), len(probes)))
        for i, region in enumerate(regions):
            for j, probe in enumerate(probes):
                subset = df[(df['jurisdiction_region']==region) & (df['probe_id']==probe)]
                heatmap_data[i,j] = subset['valence'].mean() if len(subset)>0 else np.nan

        im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0.4, vmax=0.7, aspect='auto')
        ax.set_title(get_model_short_name(model_id), fontsize=10)
        ax.set_xticks(range(len(probes)))
        ax.set_xticklabels([p.replace('_','\n')[:8] for p in probes], fontsize=7, rotation=45, ha='right')

        if idx == 0:
            ax.set_yticks(range(len(regions)))
            ax.set_yticklabels(regions, fontsize=8)

    fig.colorbar(im, ax=axes, label='Valence', fraction=0.02)
    plt.tight_layout()

    output_path = output_dir / 'fig03_regional_heatmap_combined.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def fig04_pca_embedding(all_data, output_dir):
    """Figure 4: Regional PCA Embedding."""
    print("\nGenerating: Fig 4 - Regional PCA")

    models = list(all_data.keys())
    probes = sorted(all_data[models[0]]['probe_id'].unique())
    regions = sorted(compute_valence_scores(all_data[models[0]]).dropna(subset=['valence'])['jurisdiction_region'].unique())

    embedding_matrix = np.zeros((len(regions), len(models)*len(probes)))

    for m_idx, model_id in enumerate(models):
        df = compute_valence_scores(all_data[model_id]).dropna(subset=['valence'])
        for p_idx, probe in enumerate(probes):
            col_idx = m_idx * len(probes) + p_idx
            for r_idx, region in enumerate(regions):
                subset = df[(df['jurisdiction_region']==region) & (df['probe_id']==probe)]
                embedding_matrix[r_idx, col_idx] = subset['valence'].mean() if len(subset)>0 else 0.5

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embedding_matrix)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(6.75, 5), dpi=300)

    colors = [COLORS.get(r, '#999999') for r in regions]
    ax.scatter(coords[:,0], coords[:,1], c=colors, s=150, alpha=0.7, edgecolors='black', linewidth=1)

    for i, region in enumerate(regions):
        ax.annotate(region, (coords[i,0], coords[i,1]), xytext=(5,5),
                   textcoords='offset points', fontsize=9)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'fig04_regional_pca.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()

    return pca.explained_variance_ratio_


def fig05_per_model_breakdown(all_data, output_dir):
    """Figure 5-7: Per-Model Regional Breakdown (one figure per model)."""
    print("\nGenerating: Figs 5-7 - Per-Model Breakdowns")

    fig_num = 5
    for model_id, df in all_data.items():
        df = compute_valence_scores(df).dropna(subset=['valence'])

        regions = sorted(df['jurisdiction_region'].unique())
        probes = sorted(df['probe_id'].unique())

        heatmap_data = np.zeros((len(regions), len(probes)))
        for i, region in enumerate(regions):
            for j, probe in enumerate(probes):
                subset = df[(df['jurisdiction_region']==region) & (df['probe_id']==probe)]
                heatmap_data[i,j] = subset['valence'].mean() if len(subset)>0 else np.nan

        fig, ax = plt.subplots(figsize=(6.75, 5), dpi=300)
        im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0.4, vmax=0.7, aspect='auto')

        ax.set_xticks(range(len(probes)))
        ax.set_yticks(range(len(regions)))
        ax.set_xticklabels([p.replace('_','\n')[:12] for p in probes], rotation=45, ha='right')
        ax.set_yticklabels(regions)

        # Add values
        for i in range(len(regions)):
            for j in range(len(probes)):
                if not np.isnan(heatmap_data[i,j]):
                    ax.text(j, i, f'{heatmap_data[i,j]:.2f}', ha='center', va='center', fontsize=7)

        plt.colorbar(im, ax=ax, label='Valence')
        ax.set_title(get_model_short_name(model_id))

        plt.tight_layout()

        safe_name = model_id.replace('/', '_').replace(' ', '_')
        output_path = output_dir / f'fig{fig_num:02d}_{safe_name}_regional_breakdown.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()

        fig_num += 1


def generate_statistics_summary(all_data, output_dir):
    """Generate statistics summary text file."""
    print("\nGenerating: Statistics Summary")

    lines = []
    lines.append("="*70)
    lines.append("AAAI PUBLICATION STATISTICS SUMMARY")
    lines.append("="*70)

    for model_id, df in all_data.items():
        df = compute_valence_scores(df).dropna(subset=['valence'])
        regions = sorted(df['jurisdiction_region'].unique())

        lines.append(f"\n{get_model_short_name(model_id)}:")
        lines.append("-"*70)

        region_data = []
        for region in regions:
            region_df = df[df['jurisdiction_region']==region]
            mean_val = region_df['valence'].mean()
            std_val = region_df['valence'].std()
            n = len(region_df)
            region_data.append((region, mean_val, std_val, n))

        for region, mean, std, n in region_data:
            lines.append(f"  {region:20s}: μ={mean:.4f}, σ={std:.4f}, n={n:6,}")

        # Max gap
        means = [x[1] for x in region_data]
        gap = max(means) - min(means)
        worst_region = region_data[np.argmin(means)][0]
        best_region = region_data[np.argmax(means)][0]

        lines.append(f"\n  Gap: Δ={gap:.4f} ({worst_region} → {best_region})")

    output_path = output_dir / 'statistics_summary.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ Saved: {output_path.name}")


def main():
    results_dir = Path("results/single_runs_35k")
    output_dir = Path("results/aaai_submission/aaai_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("AAAI FIGURES - FINAL GENERATION")
    print("="*70)
    print(f"\nOutput: {output_dir}")

    # Load valid models
    all_data = load_valid_models(results_dir)

    if not all_data:
        print("\n❌ No valid models found!")
        return

    print("\n" + "="*70)
    print("Generating Figures")
    print("="*70)

    # Generate all figures
    fig01_model_leaderboard_table(all_data, output_dir)
    fig02_worst_best_bars(all_data, output_dir)
    fig03_combined_heatmap(all_data, output_dir)
    variance = fig04_pca_embedding(all_data, output_dir)
    fig05_per_model_breakdown(all_data, output_dir)
    generate_statistics_summary(all_data, output_dir)

    # Summary
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)

    pdf_count = len(list(output_dir.glob("*.pdf")))
    print(f"\nGenerated: {pdf_count} PDF figures")
    print(f"Location: {output_dir}")

    print("\nFigures:")
    for pdf in sorted(output_dir.glob("*.pdf")):
        size_kb = pdf.stat().st_size / 1024
        print(f"  • {pdf.name} ({size_kb:.1f} KB)")

    print("\n" + "="*70)
    print(f"PCA Variance: PC1={variance[0]*100:.1f}%, PC2={variance[1]*100:.1f}%")
    print("="*70)


if __name__ == '__main__':
    main()
