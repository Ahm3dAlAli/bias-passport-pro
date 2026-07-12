#!/usr/bin/env python3
"""
Generate sample-level embedding visualizations showing how VLMs treat individual images.
Creates UMAP/t-SNE embeddings of actual image responses colored by region.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  UMAP not available. Install: pip install umap-learn")

# AAAI settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
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


def load_model_data(db_path):
    """Load data from database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM probe_results", conn)
    conn.close()
    return df


def create_response_embeddings(responses, method='tfidf'):
    """
    Create embeddings from text responses.

    Args:
        responses: List of response texts
        method: 'tfidf' or 'bow' (bag of words)

    Returns:
        Embedding matrix (n_samples, n_features)
    """
    print(f"  Creating {method.upper()} embeddings...")

    # Clean responses
    clean_responses = []
    for r in responses:
        if pd.isna(r) or str(r).startswith('[ERROR]'):
            clean_responses.append('')
        else:
            clean_responses.append(str(r).lower())

    # TF-IDF vectorization
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            min_df=5,
            max_df=0.8
        )
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(
            max_features=100,
            stop_words='english',
            min_df=5,
            max_df=0.8
        )

    try:
        embeddings = vectorizer.fit_transform(clean_responses).toarray()
        print(f"    Shape: {embeddings.shape}")
        return embeddings, vectorizer
    except:
        print("    ⚠️  Failed to create embeddings")
        return None, None


def stratified_sample_images(df, n_per_region=500):
    """Sample images stratified by region."""
    print(f"  Stratified sampling ({n_per_region} per region)...")

    sampled = []
    regions = df['jurisdiction_region'].unique()

    for region in regions:
        region_df = df[df['jurisdiction_region'] == region]
        n_available = len(region_df)
        n_sample = min(n_per_region, n_available)

        sample = region_df.sample(n=n_sample, random_state=42)
        sampled.append(sample)
        print(f"    {region:20s}: {n_sample:,} samples")

    result = pd.concat(sampled, ignore_index=True)
    print(f"  Total: {len(result):,} samples")
    return result


def plot_umap_embeddings(embeddings, labels, regions, title, output_path):
    """Create UMAP visualization."""
    if not HAS_UMAP:
        print("  ⚠️  Skipping UMAP (not installed)")
        return

    print("  Running UMAP...")
    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = umap.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot by region
    unique_regions = sorted(set(regions))
    for region in unique_regions:
        mask = np.array(regions) == region
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=COLORS.get(region, '#999999'),
                  label=region, s=15, alpha=0.5, edgecolors='none')

    ax.set_xlabel('UMAP 1', fontsize=10)
    ax.set_ylabel('UMAP 2', fontsize=10)
    ax.legend(loc='best', framealpha=0.9, markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_tsne_embeddings(embeddings, labels, regions, title, output_path):
    """Create t-SNE visualization."""
    print("  Running t-SNE...")

    # Use PCA first to reduce dimensions (faster)
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)
        print(f"    PCA reduction: {embeddings.shape}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot by region
    unique_regions = sorted(set(regions))
    for region in unique_regions:
        mask = np.array(regions) == region
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=COLORS.get(region, '#999999'),
                  label=region, s=15, alpha=0.5, edgecolors='none')

    ax.set_xlabel('t-SNE 1', fontsize=10)
    ax.set_ylabel('t-SNE 2', fontsize=10)
    ax.legend(loc='best', framealpha=0.9, markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_pca_embeddings(embeddings, labels, regions, title, output_path):
    """Create PCA visualization with variance explained."""
    print("  Running PCA...")

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    var1, var2 = pca.explained_variance_ratio_
    print(f"    PC1: {var1*100:.1f}%, PC2: {var2*100:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot by region
    unique_regions = sorted(set(regions))
    for region in unique_regions:
        mask = np.array(regions) == region
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=COLORS.get(region, '#999999'),
                  label=region, s=15, alpha=0.5, edgecolors='none')

    ax.set_xlabel(f'PC1 ({var1*100:.1f}% variance)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var2*100:.1f}% variance)', fontsize=10)
    ax.legend(loc='best', framealpha=0.9, markerscale=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_sample_responses_grid(df_sample, output_path):
    """Show sample responses in a grid by region and probe."""
    print("  Creating sample responses grid...")

    regions = sorted(df_sample['jurisdiction_region'].unique())
    probes = sorted(df_sample['probe_id'].unique())

    # Select one example per region × probe
    examples = []
    for region in regions:
        for probe in probes:
            subset = df_sample[
                (df_sample['jurisdiction_region'] == region) &
                (df_sample['probe_id'] == probe)
            ]
            if len(subset) > 0:
                example = subset.iloc[0]
                examples.append({
                    'region': region,
                    'probe': probe,
                    'response': str(example['response'])[:100] + '...' if len(str(example['response'])) > 100 else str(example['response'])
                })

    # Create text visualization
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.axis('off')

    # Create grid
    n_regions = len(regions)
    n_probes = len(probes)

    cell_width = 1.0 / n_probes
    cell_height = 1.0 / n_regions

    # Headers
    for j, probe in enumerate(probes):
        ax.text((j + 0.5) * cell_width, 0.98, probe.replace('_', '\n'),
               ha='center', va='top', fontsize=7, fontweight='bold')

    for i, region in enumerate(regions):
        ax.text(0.02, 1 - (i + 0.5) * cell_height, region,
               ha='left', va='center', fontsize=7, fontweight='bold',
               color=COLORS.get(region, '#999999'))

    # Fill cells with responses
    for example in examples:
        i = regions.index(example['region'])
        j = probes.index(example['probe'])

        x = (j + 0.5) * cell_width
        y = 1 - (i + 0.5) * cell_height

        ax.text(x, y, example['response'][:50],
               ha='center', va='center', fontsize=5,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=COLORS.get(example['region'], '#999999')))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def main():
    results_dir = Path("results/single_runs_35k")
    output_dir = Path("results/aaai_submission/aaai_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Sample-Level VLM Embedding Visualizations")
    print("="*70)

    # Load models
    models = []
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

        # Check for valid data
        df = load_model_data(db_path)
        valid_count = df[~df['response'].str.startswith('[ERROR]', na=False)].shape[0]

        if valid_count > 1000:
            print(f"\n✓ {model_id}: {valid_count:,} valid responses")
            models.append((model_id, db_path))

    if not models:
        print("\n❌ No valid models found")
        return

    # Process each model
    for model_id, db_path in models:
        print(f"\n{'='*70}")
        print(f"Processing: {model_id}")
        print(f"{'='*70}")

        # Load data
        df = load_model_data(db_path)

        # Filter valid responses
        df_valid = df[~df['response'].str.startswith('[ERROR]', na=False)].copy()

        # Stratified sample
        df_sample = stratified_sample_images(df_valid, n_per_region=500)

        # Create response embeddings
        embeddings, vectorizer = create_response_embeddings(df_sample['response'].tolist())

        if embeddings is None:
            continue

        regions = df_sample['jurisdiction_region'].tolist()
        labels = df_sample['image_id'].tolist()

        model_short = model_id.replace('HuggingFaceM4_', '').replace('OpenGVLab_', '').\
                     replace('llava_hf_', '').replace('_hf', '').replace('_', '-')

        # Generate visualizations
        safe_name = model_id.replace('/', '_').replace(' ', '_')

        # PCA
        plot_pca_embeddings(
            embeddings, labels, regions,
            f'Response Embedding PCA: {model_short}',
            output_dir / f'fig_sample_pca_{safe_name}.pdf'
        )

        # t-SNE
        plot_tsne_embeddings(
            embeddings, labels, regions,
            f'Response Embedding t-SNE: {model_short}',
            output_dir / f'fig_sample_tsne_{safe_name}.pdf'
        )

        # UMAP (if available)
        if HAS_UMAP:
            plot_umap_embeddings(
                embeddings, labels, regions,
                f'Response Embedding UMAP: {model_short}',
                output_dir / f'fig_sample_umap_{safe_name}.pdf'
            )

    print("\n" + "="*70)
    print("Sample Embedding Visualizations Complete!")
    print("="*70)
    print(f"\nOutput: {output_dir}")

    # Count files
    sample_figs = list(output_dir.glob("fig_sample_*.pdf"))
    print(f"\nGenerated {len(sample_figs)} sample embedding figures")

    for fig in sorted(sample_figs):
        size_kb = fig.stat().st_size / 1024
        print(f"  • {fig.name} ({size_kb:.1f} KB)")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
