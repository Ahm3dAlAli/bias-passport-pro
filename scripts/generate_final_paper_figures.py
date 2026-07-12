#!/usr/bin/env python3
"""
Generate publication figures for FINGERPRINT² paper.
Uses actual validated data from 3 working models on full FHIBE corpus.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import argparse

# Matplotlib settings for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
})

# Model names (3 working models only)
MODEL_NAMES = {
    'HuggingFaceM4_idefics2_8b': 'IDEFICS2-8B',
    'OpenGVLab_InternVL2_2B': 'InternVL2-2B',
    'llava_hf_llava_v1.6_vicuna_7b_hf': 'LLaVA-v1.6-7B',
}

PROBE_NAMES = {
    'P1_occupation': 'P1',
    'P2_education': 'P2',
    'P3_trustworthiness': 'P3',
    'P4_lifestyle': 'P4',
    'P5_neighbourhood': 'P5',
}

PROBE_FULL_NAMES = {
    'P1_occupation': 'Occupation',
    'P2_education': 'Education',
    'P3_trustworthiness': 'Trustworthiness',
    'P4_lifestyle': 'Lifestyle',
    'P5_neighbourhood': 'Neighbourhood',
}

MODEL_COLORS = {
    'IDEFICS2-8B': '#e67e22',      # Orange
    'Llama-3.2-11B': '#95a5a6',    # Gray (failed)
    'InternVL2-2B': '#9b59b6',     # Purple
    'LLaVA-v1.6-7B': '#2ecc71',    # Green
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
        if pd.isna(text) or text == '' or str(text).startswith('[ERROR]') or str(text).startswith('Error'):
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

def compute_disparity_metrics(df: pd.DataFrame) -> Dict[str, Dict]:
    """Compute disparity metrics for each probe."""
    metrics = {}
    
    for probe in df['probe_id'].unique():
        probe_df = df[df['probe_id'] == probe].copy()
        probe_df = compute_valence_scores(probe_df)
        probe_df = probe_df.dropna(subset=['valence'])
        
        if len(probe_df) == 0:
            continue
        
        region_means = probe_df.groupby('jurisdiction_region')['valence'].mean()
        
        if len(region_means) > 1:
            max_min_gap = region_means.max() - region_means.min()
            metrics[probe] = {
                'max_min_gap': max_min_gap,
                'max_region': region_means.idxmax(),
                'min_region': region_means.idxmin(),
                'max_val': region_means.max(),
                'min_val': region_means.min(),
            }
    
    return metrics

def load_all_data(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load data from all working models."""
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
            print(f"  ⚠️  Skipping {model_id} - only {n_images:,} images")
            continue
        
        model_name = MODEL_NAMES[model_id]
        print(f"  ✓ Loaded {model_name}: {len(df):,} results ({n_images:,} images)")
        all_data[model_name] = df
    
    return all_data

def fig1_leaderboard_and_heatmap(model_metrics: Dict, output_dir: Path):
    """Figure 1: Leaderboard + Heatmap side-by-side."""
    print("\n1. Creating leaderboard + heatmap...")
    
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.3)
    
    # LEFT: Leaderboard
    ax1 = fig.add_subplot(gs[0])
    
    composite_scores = {}
    for model, metrics in model_metrics.items():
        gaps = [m['max_min_gap'] for m in metrics.values()]
        composite_scores[model] = np.mean(gaps) if gaps else 0.0
    
    sorted_models = sorted(composite_scores.items(), key=lambda x: x[1])
    
    y_pos = np.arange(len(sorted_models))
    colors = [MODEL_COLORS.get(m, '#95a5a6') for m, _ in sorted_models]
    scores = [s for _, s in sorted_models]
    
    bars = ax1.barh(y_pos, scores, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.2, height=0.6)
    
    for i, (model, score) in enumerate(sorted_models):
        ax1.text(-0.003, i, f'#{i+1}', ha='right', va='center',
                fontsize=10, weight='bold')
        ax1.text(score + 0.002, i, f'{score:.3f}', ha='left', va='center',
                fontsize=9)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m for m, _ in sorted_models], fontsize=10, weight='bold')
    ax1.set_xlabel('Composite Disparity Score', fontsize=11, weight='bold')
    ax1.set_xlim(0, max(scores) * 1.2)
    ax1.grid(axis='x', alpha=0.25, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # RIGHT: Heatmap
    ax2 = fig.add_subplot(gs[1])
    
    probe_order = ['P1_occupation', 'P2_education', 'P3_trustworthiness',
                   'P4_lifestyle', 'P5_neighbourhood']
    probe_labels = [PROBE_FULL_NAMES[p] for p in probe_order]
    model_order = sorted(model_metrics.keys())
    
    data_matrix = []
    for model in model_order:
        metrics = model_metrics[model]
        row = [metrics.get(p, {}).get('max_min_gap', 0.0) for p in probe_order]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    im = ax2.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0.0, vmax=0.25)
    
    ax2.set_xticks(np.arange(len(probe_labels)))
    ax2.set_yticks(np.arange(len(model_order)))
    ax2.set_xticklabels(probe_labels, fontsize=10, rotation=0)
    ax2.set_yticklabels(model_order, fontsize=10, weight='bold')
    
    for edge, spine in ax2.spines.items():
        spine.set_visible(False)
    ax2.set_xticks(np.arange(data_matrix.shape[1] + 1) - 0.5, minor=True)
    ax2.set_yticks(np.arange(data_matrix.shape[0] + 1) - 0.5, minor=True)
    ax2.grid(which='minor', color='white', linestyle='-', linewidth=2.5)
    ax2.tick_params(which='minor', size=0)
    
    for i in range(len(model_order)):
        for j in range(len(probe_labels)):
            value = data_matrix[i, j]
            text_color = 'white' if value > 0.15 else 'black'
            ax2.text(j, i, f'{value:.3f}', ha='center', va='center',
                   color=text_color, fontsize=9, weight='bold')
    
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Max-Min Valence Gap', rotation=270, labelpad=18, fontsize=10, weight='bold')
    
    plt.savefig(output_dir / 'fig1_leaderboard_heatmap.pdf')
    plt.savefig(output_dir / 'fig1_leaderboard_heatmap.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: fig1_leaderboard_heatmap.pdf + .png")

def fig2_radar_fingerprints(model_metrics: Dict, output_dir: Path):
    """Figure 2: Radar fingerprints for each model."""
    print("\n2. Creating radar fingerprints...")
    
    probe_order = ['P1_occupation', 'P2_education', 'P3_trustworthiness',
                   'P4_lifestyle', 'P5_neighbourhood']
    probe_labels = [PROBE_NAMES[p] for p in probe_order]
    
    models = sorted(model_metrics.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(4.5*n_models, 4),
                            subplot_kw=dict(projection='polar'))
    
    if n_models == 1:
        axes = [axes]
    
    num_vars = len(probe_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, (model_name, ax) in enumerate(zip(models, axes)):
        metrics = model_metrics[model_name]
        values = [metrics.get(p, {}).get('max_min_gap', 0.0) for p in probe_order]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5,
                color=MODEL_COLORS.get(model_name, '#95a5a6'),
                markersize=7)
        ax.fill(angles, values, alpha=0.35,
                color=MODEL_COLORS.get(model_name, '#95a5a6'))
        
        ax.set_ylim(0, 0.25)
        ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25])
        ax.set_yticklabels(['0.05', '0.10', '0.15', '0.20', '0.25'], fontsize=8, color='gray')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(probe_labels, fontsize=10, weight='bold')
        
        ax.set_title(model_name, fontsize=12, weight='bold', pad=15)
        
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')
        ax.spines['polar'].set_linewidth(1.5)
        ax.spines['polar'].set_color('black')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_radar_fingerprints.pdf')
    plt.savefig(output_dir / 'fig2_radar_fingerprints.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: fig2_radar_fingerprints.pdf + .png")

def fig3_probe_comparison(model_metrics: Dict, output_dir: Path):
    """Figure 3: Per-probe disparity comparison."""
    print("\n3. Creating per-probe comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    probe_order = ['P1_occupation', 'P2_education', 'P3_trustworthiness',
                   'P4_lifestyle', 'P5_neighbourhood']
    probe_short = [PROBE_NAMES[p] + '_' + PROBE_FULL_NAMES[p][:4]
                   for p in probe_order]
    
    x = np.arange(len(probe_order))
    width = 0.25
    models = sorted(model_metrics.keys())
    
    for idx, model_name in enumerate(models):
        metrics = model_metrics[model_name]
        values = [metrics.get(p, {}).get('max_min_gap', 0.0) for p in probe_order]
        offset = (idx - len(models)/2 + 0.5) * width
        
        ax.bar(x + offset, values, width, label=model_name,
               color=MODEL_COLORS.get(model_name, '#95a5a6'),
               alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Probe', fontsize=11, weight='bold')
    ax.set_ylabel('Max-Min Valence Gap', fontsize=11, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_short, fontsize=9.5)
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_probe_comparison.pdf')
    plt.savefig(output_dir / 'fig3_probe_comparison.png', dpi=300)
    plt.close()
    print(f"  ✓ Saved: fig3_probe_comparison.pdf + .png")

def generate_metadata(all_data: Dict, model_metrics: Dict, output_dir: Path):
    """Generate dataset metadata and statistics."""
    print("\n📝 Generating metadata...")
    
    # Get one model's data for corpus stats
    sample_df = list(all_data.values())[0]
    
    metadata = []
    metadata.append("=" * 70)
    metadata.append("FINGERPRINT² DATASET METADATA")
    metadata.append("=" * 70)
    metadata.append("")
    
    n_images = sample_df['image_id'].nunique()
    metadata.append(f"Total Images: {n_images:,}")
    metadata.append(f"Models Evaluated: {len(all_data)}")
    metadata.append(f"Probes: {sample_df['probe_id'].nunique()}")
    metadata.append("")
    
    region_counts = sample_df.groupby('jurisdiction_region').size()
    metadata.append("Regional Distribution:")
    for region, count in region_counts.items():
        pct = (count / len(sample_df)) * 100
        metadata.append(f"  {region}: {count:,} ({pct:.1f}%)")
    metadata.append("")
    
    metadata.append("=" * 70)
    metadata.append("MODEL DISPARITY SCORES")
    metadata.append("=" * 70)
    metadata.append("")
    
    composite_scores = {}
    for model, metrics in model_metrics.items():
        gaps = [m['max_min_gap'] for m in metrics.values()]
        composite_scores[model] = np.mean(gaps) if gaps else 0.0
    
    sorted_models = sorted(composite_scores.items(), key=lambda x: x[1])
    
    for rank, (model, score) in enumerate(sorted_models, 1):
        metadata.append(f"#{rank} {model}: {score:.3f}")
        for probe, m in sorted(model_metrics[model].items()):
            gap = m['max_min_gap']
            metadata.append(f"    {probe}: {gap:.3f} (worst: {m['min_region']}, best: {m['max_region']})")
        metadata.append("")
    
    metadata_text = '\n'.join(metadata)
    
    with open(output_dir / 'dataset_metadata.txt', 'w') as f:
        f.write(metadata_text)
    
    print(f"  ✓ Saved: dataset_metadata.txt")
    print()
    print(metadata_text)

def main():
    parser = argparse.ArgumentParser(description='Generate FINGERPRINT² paper figures')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results directory with .db files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for figures')
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FINGERPRINT² Paper Figure Generation")
    print("=" * 70)
    print(f"Results: {results_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    print("📊 Loading model data...")
    all_data = load_all_data(results_dir)
    
    if len(all_data) == 0:
        print("❌ No valid data loaded!")
        return
    
    print(f"\n✓ Loaded {len(all_data)} working models")
    print()
    
    # Compute metrics
    print("🔢 Computing disparity metrics...")
    model_metrics = {}
    for model_name, df in all_data.items():
        print(f"  {model_name}...")
        model_metrics[model_name] = compute_disparity_metrics(df)
    
    print()
    
    # Generate figures
    print("🎨 Generating figures...")
    fig1_leaderboard_and_heatmap(model_metrics, output_dir)
    fig2_radar_fingerprints(model_metrics, output_dir)
    fig3_probe_comparison(model_metrics, output_dir)
    
    # Generate metadata
    generate_metadata(all_data, model_metrics, output_dir)
    
    print("=" * 70)
    print("✅ All figures generated successfully!")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print()

if __name__ == '__main__':
    main()
