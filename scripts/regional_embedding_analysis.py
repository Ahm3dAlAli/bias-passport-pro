#!/usr/bin/env python3
"""
Regional Embedding Analysis for VLM Bias
Creates embeddings from regional valence patterns and visualizes bias space.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import argparse
import json

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
})


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


def create_regional_embeddings(all_data: dict, output_dir: Path):
    """
    Create regional embeddings from valence patterns across models and probes.

    Each region is represented as a vector:
    [model1_probe1_valence, model1_probe2_valence, ..., modelN_probeM_valence]
    """
    print("\n" + "="*70)
    print("Creating Regional Embeddings")
    print("="*70)

    # Build embedding matrix: regions × (models × probes)
    models = list(all_data.keys())

    # Get all probes from first model
    first_model_df = all_data[models[0]]
    probes = sorted(first_model_df['probe_id'].unique())
    regions = sorted(first_model_df['jurisdiction_region'].unique())

    print(f"\nDimensions:")
    print(f"  Regions: {len(regions)}")
    print(f"  Models: {len(models)}")
    print(f"  Probes: {len(probes)}")
    print(f"  Embedding size: {len(models) * len(probes)}")

    # Create embedding matrix
    embedding_matrix = np.zeros((len(regions), len(models) * len(probes)))
    feature_names = []

    for model_idx, model_name in enumerate(models):
        df = all_data[model_name]
        df = compute_valence_scores(df)
        df = df.dropna(subset=['valence'])

        for probe_idx, probe in enumerate(probes):
            col_idx = model_idx * len(probes) + probe_idx
            feature_names.append(f"{model_name[:20]}_{probe}")

            probe_df = df[df['probe_id'] == probe]

            for region_idx, region in enumerate(regions):
                region_probe_df = probe_df[probe_df['jurisdiction_region'] == region]
                if len(region_probe_df) > 0:
                    embedding_matrix[region_idx, col_idx] = region_probe_df['valence'].mean()
                else:
                    embedding_matrix[region_idx, col_idx] = 0.5  # Neutral

    print(f"\nEmbedding matrix shape: {embedding_matrix.shape}")
    print(f"Non-zero entries: {np.count_nonzero(embedding_matrix)}/{embedding_matrix.size}")

    return embedding_matrix, regions, feature_names


def visualize_pca(embedding_matrix, regions, output_dir: Path):
    """PCA visualization of regional bias space."""
    print("\nPerforming PCA analysis...")

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(embedding_matrix)

    explained_var = pca.explained_variance_ratio_
    print(f"  PC1 explains {explained_var[0]*100:.1f}% of variance")
    print(f"  PC2 explains {explained_var[1]*100:.1f}% of variance")
    print(f"  Total: {sum(explained_var)*100:.1f}%")

    # Create figure
    fig, ax = plt.subplots(figsize=(6.75, 5), dpi=300)

    # Color by continent (simplified grouping)
    continent_colors = {
        'Africa': '#D55E00',           # Vermillion
        'Asia': '#56B4E9',              # Sky blue
        'Europe': '#009E73',            # Bluish green
        'Americas': '#F0E442',          # Yellow
        'Northern America': '#CC79A7',  # Reddish purple
        'Oceania': '#0072B2',           # Blue
    }

    colors = [continent_colors.get(r, '#999999') for r in regions]

    # Scatter plot
    scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                        c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Annotate points
    for i, region in enumerate(regions):
        ax.annotate(region, (pca_coords[i, 0], pca_coords[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=10)
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=10)
    ax.set_title('Regional Bias Space (PCA)', fontsize=11, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=region)
                      for region, color in continent_colors.items()]
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'fig_regional_embedding_pca.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return pca, pca_coords, explained_var


def visualize_tsne(embedding_matrix, regions, output_dir: Path):
    """t-SNE visualization of regional bias space."""
    print("\nPerforming t-SNE analysis...")

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(regions)-1))
    tsne_coords = tsne.fit_transform(embedding_matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=(6.75, 5), dpi=300)

    # Color by continent
    continent_colors = {
        'Africa': '#D55E00',
        'Asia': '#56B4E9',
        'Europe': '#009E73',
        'Americas': '#F0E442',
        'Northern America': '#CC79A7',
        'Oceania': '#0072B2',
    }

    colors = [continent_colors.get(r, '#999999') for r in regions]

    # Scatter plot
    scatter = ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                        c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Annotate points
    for i, region in enumerate(regions):
        ax.annotate(region, (tsne_coords[i, 0], tsne_coords[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax.set_title('Regional Bias Space (t-SNE)', fontsize=11, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=region)
                      for region, color in continent_colors.items()]
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'fig_regional_embedding_tsne.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return tsne_coords


def compute_regional_similarity(embedding_matrix, regions, output_dir: Path):
    """Compute and visualize pairwise regional similarity."""
    print("\nComputing regional similarity...")

    # Compute pairwise distances (cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embedding_matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=(6.75, 6.75), dpi=300)

    # Heatmap
    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0.8, vmax=1.0, aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)

    # Axis labels
    ax.set_xticks(range(len(regions)))
    ax.set_yticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(regions, fontsize=8)

    # Add values in cells
    for i in range(len(regions)):
        for j in range(len(regions)):
            if i != j:  # Skip diagonal
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=6)

    ax.set_title('Regional Treatment Similarity', fontsize=11, pad=10)
    plt.tight_layout()

    # Save
    output_path = output_dir / 'fig_regional_similarity.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    # Find most/least similar pairs
    print("\n  Most similar regional pairs:")
    upper_triangle = np.triu_indices_from(similarity_matrix, k=1)
    similarities = []
    for i, j in zip(*upper_triangle):
        similarities.append((regions[i], regions[j], similarity_matrix[i, j]))

    similarities.sort(key=lambda x: x[2], reverse=True)
    for r1, r2, sim in similarities[:5]:
        print(f"    {r1:20s} <-> {r2:20s}: {sim:.4f}")

    print("\n  Most different regional pairs:")
    for r1, r2, sim in similarities[-5:]:
        print(f"    {r1:20s} <-> {r2:20s}: {sim:.4f}")

    return similarity_matrix, similarities


def hierarchical_clustering(embedding_matrix, regions, output_dir: Path):
    """Hierarchical clustering of regions."""
    print("\nPerforming hierarchical clustering...")

    # Compute linkage
    linkage_matrix = linkage(embedding_matrix, method='ward')

    # Create figure
    fig, ax = plt.subplots(figsize=(6.75, 5), dpi=300)

    dendrogram(linkage_matrix, labels=regions, ax=ax, leaf_font_size=9)

    ax.set_title('Regional Treatment Hierarchy', fontsize=11, pad=10)
    ax.set_xlabel('Region', fontsize=10)
    ax.set_ylabel('Ward Distance', fontsize=10)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'fig_regional_clustering.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Regional embedding analysis for VLM bias'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory with .db result files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for figures and analysis')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Regional Embedding Analysis")
    print("="*70)

    # Load data from all models
    print("\nLoading data...")
    all_data = {}

    for db_path in sorted(results_dir.glob("*.db")):
        # Extract model ID
        filename = db_path.stem
        if filename.startswith('gpu'):
            parts = filename.split('_', 1)
            if len(parts) > 1:
                filename = parts[1]

        # Remove timestamp
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

    # Create regional embeddings
    embedding_matrix, regions, feature_names = create_regional_embeddings(all_data, output_dir)

    # Save embedding matrix
    embedding_df = pd.DataFrame(embedding_matrix, index=regions, columns=feature_names)
    embedding_path = output_dir / 'regional_embeddings.csv'
    embedding_df.to_csv(embedding_path)
    print(f"\n✓ Saved embedding matrix: {embedding_path}")

    # Visualizations
    pca, pca_coords, explained_var = visualize_pca(embedding_matrix, regions, output_dir)
    tsne_coords = visualize_tsne(embedding_matrix, regions, output_dir)
    similarity_matrix, similarities = compute_regional_similarity(embedding_matrix, regions, output_dir)
    hierarchical_clustering(embedding_matrix, regions, output_dir)

    # Save analysis results
    results = {
        'regions': regions,
        'embedding_dimensions': embedding_matrix.shape[1],
        'pca_explained_variance': {
            'PC1': float(explained_var[0]),
            'PC2': float(explained_var[1]),
            'total_2d': float(sum(explained_var))
        },
        'most_similar_pairs': [
            {'region1': r1, 'region2': r2, 'similarity': float(sim)}
            for r1, r2, sim in similarities[:10]
        ],
        'most_different_pairs': [
            {'region1': r1, 'region2': r2, 'similarity': float(sim)}
            for r1, r2, sim in similarities[-10:]
        ]
    }

    results_path = output_dir / 'regional_embedding_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("Regional Embedding Analysis Complete!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  • regional_embeddings.csv - Embedding matrix")
    print("  • fig_regional_embedding_pca.pdf - PCA visualization")
    print("  • fig_regional_embedding_tsne.pdf - t-SNE visualization")
    print("  • fig_regional_similarity.pdf - Similarity heatmap")
    print("  • fig_regional_clustering.pdf - Hierarchical clustering")
    print("  • regional_embedding_analysis.json - Analysis results")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
