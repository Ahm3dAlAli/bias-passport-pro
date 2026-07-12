#!/usr/bin/env python3
"""
Enhanced visualizations for VLM bias analysis with UMAP and deeper insights.
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
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  UMAP not installed. Run: pip install umap-learn")

# AAAI publication settings
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

# Colorblind-safe palette (Wong)
COLORS = {
    'Africa': '#D55E00',           # Vermillion
    'Asia': '#56B4E9',             # Sky blue
    'Europe': '#009E73',           # Bluish green
    'Americas': '#F0E442',         # Yellow
    'Northern America': '#CC79A7', # Reddish purple
    'Oceania': '#0072B2',          # Blue
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


def plot_umap_per_image(all_data: dict, output_dir: Path, n_samples: int = 5000):
    """
    UMAP visualization of individual images (sampled) colored by region.
    Shows how individual images from different regions cluster.
    Note: Uses stratified sampling to ensure representation from all regions.
    """
    if not HAS_UMAP:
        print("  Skipping UMAP (not installed)")
        return None

    print("\nCreating UMAP visualization (per-image)...")

    # Collect image-level data
    image_data = []
    models = list(all_data.keys())
    probes = sorted(all_data[models[0]]['probe_id'].unique())

    # Stratified sampling: equal samples per region
    first_model_df = all_data[models[0]]
    regions = first_model_df['jurisdiction_region'].unique()

    samples_per_region = n_samples // len(regions)
    sampled_images = []

    for region in regions:
        region_images = first_model_df[first_model_df['jurisdiction_region'] == region]['image_id'].drop_duplicates()
        n_region = min(samples_per_region, len(region_images))
        sampled = region_images.sample(n=n_region, random_state=42)
        sampled_images.extend(sampled.tolist())

    print(f"  Using stratified sample: {len(sampled_images)} images ({samples_per_region} per region)")
    print(f"  Total available: {first_model_df['image_id'].nunique()} images")

    for image_id in sampled_images:
        # Build feature vector for this image across all models and probes
        features = []
        region = None

        for model_name in models:
            df = all_data[model_name]
            df = compute_valence_scores(df)
            image_df = df[df['image_id'] == image_id]

            if len(image_df) == 0:
                features.extend([0.5] * len(probes))
                continue

            if region is None:
                region = image_df['jurisdiction_region'].iloc[0]

            for probe in probes:
                probe_df = image_df[image_df['probe_id'] == probe]
                if len(probe_df) > 0 and pd.notna(probe_df['valence'].iloc[0]):
                    features.append(probe_df['valence'].iloc[0])
                else:
                    features.append(0.5)

        if region is not None:
            image_data.append({
                'image_id': image_id,
                'region': region,
                'features': features
            })

    # Convert to arrays
    X = np.array([d['features'] for d in image_data])
    regions = [d['region'] for d in image_data]

    print(f"  Feature matrix: {X.shape}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # UMAP
    print("  Running UMAP (this may take a minute)...")
    umap = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    X_umap = umap.fit_transform(X_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)

    # Scatter by region
    for region in sorted(set(regions)):
        mask = np.array(regions) == region
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                  c=COLORS.get(region, '#999999'),
                  label=region, s=20, alpha=0.6, edgecolors='none')

    ax.set_xlabel('UMAP 1', fontsize=10)
    ax.set_ylabel('UMAP 2', fontsize=10)
    ax.set_title('Individual Images in Bias Space (UMAP)', fontsize=11, pad=10)
    ax.legend(loc='best', framealpha=0.9, markerscale=1.5)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'fig_umap_per_image.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return X_umap, regions


def plot_bias_trajectory(all_data: dict, output_dir: Path):
    """
    Show how each region's valence changes across models.
    Visualizes model-specific bias patterns.
    """
    print("\nCreating bias trajectory visualization...")

    models = list(all_data.keys())
    regions = sorted(all_data[models[0]]['jurisdiction_region'].unique())

    # Compute mean valence per region per model
    trajectory_data = {region: [] for region in regions}

    for model_name in models:
        df = all_data[model_name]
        df = compute_valence_scores(df)
        df = df.dropna(subset=['valence'])

        for region in regions:
            region_df = df[df['jurisdiction_region'] == region]
            mean_valence = region_df['valence'].mean() if len(region_df) > 0 else 0.5
            trajectory_data[region].append(mean_valence)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # Shorten model names for x-axis
    model_labels = [m.replace('HuggingFaceM4_', '').replace('OpenGVLab_', '')
                   .replace('llava_hf_', '').replace('_hf', '')[:15] for m in models]

    for region in regions:
        ax.plot(range(len(models)), trajectory_data[region],
               marker='o', linewidth=2, markersize=8,
               color=COLORS.get(region, '#999999'),
               label=region, alpha=0.8)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, rotation=30, ha='right')
    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('Mean Valence Score', fontsize=10)
    ax.set_title('Regional Bias Across Models', fontsize=11, pad=10)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([0.45, 0.70])

    plt.tight_layout()

    # Save
    output_path = output_dir / 'fig_bias_trajectory.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return trajectory_data


def plot_probe_specific_bias(all_data: dict, output_dir: Path):
    """
    Show bias patterns for each probe separately.
    Reveals which probes show strongest bias.
    """
    print("\nCreating probe-specific bias visualization...")

    models = list(all_data.keys())
    probes = sorted(all_data[models[0]]['probe_id'].unique())
    regions = sorted(all_data[models[0]]['jurisdiction_region'].unique())

    # Compute mean valence per probe per region (averaged across models)
    probe_data = {probe: {region: [] for region in regions} for probe in probes}

    for model_name in models:
        df = all_data[model_name]
        df = compute_valence_scores(df)
        df = df.dropna(subset=['valence'])

        for probe in probes:
            probe_df = df[df['probe_id'] == probe]
            for region in regions:
                region_probe_df = probe_df[probe_df['jurisdiction_region'] == region]
                if len(region_probe_df) > 0:
                    probe_data[probe][region].append(region_probe_df['valence'].mean())

    # Average across models
    for probe in probes:
        for region in regions:
            if len(probe_data[probe][region]) > 0:
                probe_data[probe][region] = np.mean(probe_data[probe][region])
            else:
                probe_data[probe][region] = 0.5

    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=300)
    axes = axes.flatten()

    probe_labels = {
        'P1_occupation': 'P1: Occupation',
        'P2_education': 'P2: Education',
        'P3_trustworthiness': 'P3: Trust',
        'P4_lifestyle': 'P4: Lifestyle',
        'P5_neighbourhood': 'P5: Neighborhood'
    }

    for idx, probe in enumerate(probes):
        ax = axes[idx]

        # Get data for this probe
        values = [probe_data[probe][r] for r in regions]
        colors_list = [COLORS.get(r, '#999999') for r in regions]

        # Bar plot
        bars = ax.bar(range(len(regions)), values, color=colors_list, alpha=0.8,
                     edgecolor='black', linewidth=0.5)

        ax.set_ylim([0.45, 0.70])
        ax.set_ylabel('Valence', fontsize=9)
        ax.set_title(probe_labels.get(probe, probe), fontsize=10)
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels([r[:3] for r in regions], fontsize=7, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Highlight min/max
        min_idx = np.argmin(values)
        max_idx = np.argmax(values)
        bars[min_idx].set_edgecolor('red')
        bars[min_idx].set_linewidth(2)
        bars[max_idx].set_edgecolor('green')
        bars[max_idx].set_linewidth(2)

    # Hide extra subplot
    axes[-1].axis('off')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'fig_probe_specific_bias.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return probe_data


def plot_model_comparison_radar(all_data: dict, output_dir: Path):
    """
    Radar plot comparing models on fairness across regions.
    """
    print("\nCreating model comparison radar plot...")

    models = list(all_data.keys())
    regions = sorted(all_data[models[0]]['jurisdiction_region'].unique())

    # Compute mean valence per model per region
    model_data = {model: {} for model in models}

    for model_name in models:
        df = all_data[model_name]
        df = compute_valence_scores(df)
        df = df.dropna(subset=['valence'])

        for region in regions:
            region_df = df[df['jurisdiction_region'] == region]
            model_data[model_name][region] = region_df['valence'].mean() if len(region_df) > 0 else 0.5

    # Radar plot
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300, subplot_kw=dict(projection='polar'))

    # Angles for each region
    angles = np.linspace(0, 2 * np.pi, len(regions), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Short model names
    model_names_short = [m.replace('HuggingFaceM4_', '').replace('OpenGVLab_', '')
                        .replace('llava_hf_', '').replace('_hf', '')[:20] for m in models]

    # Plot each model
    for idx, model_name in enumerate(models):
        values = [model_data[model_name][r] for r in regions]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=model_names_short[idx], alpha=0.7)
        ax.fill(angles, values, alpha=0.1)

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(regions, fontsize=9)
    ax.set_ylim([0.45, 0.70])
    ax.set_yticks([0.50, 0.55, 0.60, 0.65])
    ax.set_yticklabels(['0.50', '0.55', '0.60', '0.65'], fontsize=8)
    ax.set_title('Model Comparison: Regional Fairness', fontsize=11, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'fig_model_comparison_radar.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_pca_with_loadings(all_data: dict, output_dir: Path):
    """
    Enhanced PCA plot with component loadings to interpret axes.
    """
    print("\nCreating enhanced PCA with loadings...")

    # Build regional embeddings
    models = list(all_data.keys())
    probes = sorted(all_data[models[0]]['probe_id'].unique())
    regions = sorted(all_data[models[0]]['jurisdiction_region'].unique())

    embedding_matrix = np.zeros((len(regions), len(models) * len(probes)))
    feature_names = []

    for model_idx, model_name in enumerate(models):
        df = all_data[model_name]
        df = compute_valence_scores(df)
        df = df.dropna(subset=['valence'])

        for probe_idx, probe in enumerate(probes):
            col_idx = model_idx * len(probes) + probe_idx
            feature_names.append(f"{model_name[:15]}_{probe}")

            probe_df = df[df['probe_id'] == probe]

            for region_idx, region in enumerate(regions):
                region_probe_df = probe_df[probe_df['jurisdiction_region'] == region]
                if len(region_probe_df) > 0:
                    embedding_matrix[region_idx, col_idx] = region_probe_df['valence'].mean()
                else:
                    embedding_matrix[region_idx, col_idx] = 0.5

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embedding_matrix)

    # PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)

    # Get loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Scatter regions
    colors = [COLORS.get(r, '#999999') for r in regions]
    ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=colors, s=150,
              alpha=0.7, edgecolors='black', linewidth=1)

    # Annotate regions
    for i, region in enumerate(regions):
        ax.annotate(region, (pca_coords[i, 0], pca_coords[i, 1]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, fontweight='bold')

    # Plot top loadings as arrows (top 5 features)
    loading_magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_features = np.argsort(loading_magnitudes)[-5:]

    for idx in top_features:
        ax.arrow(0, 0, loadings[idx, 0]*2, loadings[idx, 1]*2,
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
        ax.text(loadings[idx, 0]*2.2, loadings[idx, 1]*2.2,
               feature_names[idx].replace('_', '\n'), fontsize=6,
               ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=10)
    ax.set_title('Regional Bias PCA with Feature Loadings', fontsize=11, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'fig_pca_with_loadings.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return pca, loadings, feature_names


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced visualizations for VLM bias analysis'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory with .db result files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for enhanced visualizations')
    parser.add_argument('--n-samples', type=int, default=5000,
                       help='Number of images to sample for UMAP (default: 5000)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Enhanced Visualizations for VLM Bias")
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

        # Check for valid data
        df_test = compute_valence_scores(df)
        valid_count = df_test['valence'].notna().sum()

        if valid_count < 1000:
            print(f"  ⚠ Skipping {model_id} - only {valid_count} valid responses")
            continue

        print(f"  ✓ Loaded {model_id}: {len(df):,} results ({valid_count:,} valid)")
        all_data[model_id] = df

    if len(all_data) < 2:
        print("\n❌ Need at least 2 models with valid data")
        return

    # Generate enhanced visualizations
    print("\n" + "="*70)
    print("Generating Enhanced Visualizations")
    print("="*70)

    # 1. UMAP per-image
    if HAS_UMAP:
        umap_coords, regions = plot_umap_per_image(all_data, output_dir, args.n_samples)
    else:
        print("\n⚠️  Install UMAP: pip install umap-learn")

    # 2. Bias trajectory
    trajectory_data = plot_bias_trajectory(all_data, output_dir)

    # 3. Probe-specific bias
    probe_data = plot_probe_specific_bias(all_data, output_dir)

    # 4. Model comparison radar
    plot_model_comparison_radar(all_data, output_dir)

    # 5. Enhanced PCA with loadings
    pca, loadings, feature_names = plot_pca_with_loadings(all_data, output_dir)

    print("\n" + "="*70)
    print("Enhanced Visualizations Complete!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  • fig_umap_per_image.pdf - UMAP of individual images")
    print("  • fig_bias_trajectory.pdf - Regional bias across models")
    print("  • fig_probe_specific_bias.pdf - Bias per probe dimension")
    print("  • fig_model_comparison_radar.pdf - Radar plot comparison")
    print("  • fig_pca_with_loadings.pdf - PCA with interpretable axes")

    # Data usage summary
    print("\n" + "="*70)
    print("Data Usage Summary")
    print("="*70)
    total_images = list(all_data.values())[0]['image_id'].nunique()
    total_responses = sum(len(df) for df in all_data.values())
    print(f"\nTotal images in dataset: {total_images:,}")
    print(f"Total responses analyzed: {total_responses:,}")
    print(f"Models used: {len(all_data)}")
    print(f"\nData usage per visualization:")
    print(f"  • UMAP: {args.n_samples:,} images (stratified sample)")
    print(f"  • Trajectory: {total_responses:,} responses (ALL DATA)")
    print(f"  • Probe-specific: {total_responses:,} responses (ALL DATA)")
    print(f"  • Radar: {total_responses:,} responses (ALL DATA)")
    print(f"  • PCA: {total_responses:,} responses (ALL DATA)")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
