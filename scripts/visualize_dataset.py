#!/usr/bin/env python3
"""
visualize_dataset.py
====================
Generate visualization images showing FHIBE dataset diversity across regions,
gender, and age groups.

Usage:
    python scripts/visualize_dataset.py --dataset /path/to/fhibe --output figures/
"""

import argparse
import random
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np

def load_image_metadata(dataset_path):
    """Load image metadata from FHIBE dataset."""
    dataset_path = Path(dataset_path)

    # Try to find metadata CSV
    csv_path = dataset_path.parent / "fhibe.20250716.u.gT5_rFTA_downsampled_public" / "data" / "aggregated_results" / "aggregated_scores" / "fhibe_scores.csv"

    if not csv_path.exists():
        print(f"Warning: Metadata CSV not found at {csv_path}")
        return None

    import pandas as pd
    df = pd.read_csv(csv_path)
    return df

def find_sample_images(dataset_path, metadata_df, samples_per_region=3):
    """Find representative sample images for each region."""
    dataset_path = Path(dataset_path)

    # Get all image directories
    image_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    samples = {
        'Africa': [],
        'Asia': [],
        'Europe': [],
        'Americas': [],
        'Northern America': [],
        'Oceania': []
    }

    # For each region, find sample images
    for region in samples.keys():
        # Find image IDs for this region from metadata
        if metadata_df is not None:
            region_images = metadata_df[metadata_df['jurisdiction_region'] == region]
            if len(region_images) > 0:
                sample_ids = region_images['image_id'].sample(min(samples_per_region, len(region_images))).tolist()
            else:
                sample_ids = []
        else:
            # Fallback: random sample from all images
            all_images = list(dataset_path.glob("*/main_*.jpg"))
            sample_ids = random.sample([img.stem for img in all_images], min(samples_per_region, len(all_images)))

        # Find actual image files
        for image_id in sample_ids:
            # Search for images with this ID
            found = False
            for img_dir in image_dirs:
                main_img = img_dir / f"main_{image_id}.jpg"
                if main_img.exists():
                    samples[region].append(main_img)
                    found = True
                    break

            if not found:
                # Try alternative patterns
                matches = list(dataset_path.glob(f"*/*{image_id}*.jpg"))
                if matches:
                    samples[region].append(matches[0])

    return samples

def create_region_grid(samples, output_path):
    """Create a grid visualization showing samples from each region."""
    regions = ['Africa', 'Asia', 'Europe', 'Americas', 'Northern America', 'Oceania']
    colors = ['#EF4444', '#F59E0B', '#3B82F6', '#10B981', '#8B5CF6', '#F97316']

    fig = plt.figure(figsize=(18, 12))

    max_samples = max(len(samples[r]) for r in regions)

    for i, region in enumerate(regions):
        region_samples = samples[region]

        for j, img_path in enumerate(region_samples):
            ax = plt.subplot(len(regions), max_samples, i * max_samples + j + 1)

            try:
                img = Image.open(img_path)
                # Resize to square
                img = img.resize((300, 300), Image.Resampling.LANCZOS)
                ax.imshow(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                ax.text(0.5, 0.5, 'Image\nNot\nAvailable',
                       ha='center', va='center', fontsize=12)

            ax.axis('off')

            # Add region label on first image of each row
            if j == 0:
                ax.text(-0.1, 0.5, region, transform=ax.transAxes,
                       fontsize=14, fontweight='bold', va='center', ha='right',
                       color=colors[i], rotation=0)

            # Add border color
            for spine in ax.spines.values():
                spine.set_edgecolor(colors[i])
                spine.set_linewidth(3)

    plt.suptitle('FHIBE Dataset: Regional Diversity Samples',
                fontsize=20, fontweight='bold', y=0.98)

    # Add legend
    patches = [mpatches.Patch(color=colors[i], label=f'{regions[i]}') for i in range(len(regions))]
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=6, frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved regional diversity grid to {output_path}")
    plt.close()

def create_demographics_summary(dataset_path, output_path):
    """Create demographic distribution visualization."""
    # Load metadata
    csv_path = Path(dataset_path).parent / "fhibe.20250716.u.gT5_rFTA_downsampled_public" / "data" / "aggregated_results" / "aggregated_scores" / "fhibe_scores.csv"

    if not csv_path.exists():
        print(f"Warning: Cannot create demographics summary - CSV not found")
        return

    import pandas as pd
    df = pd.read_csv(csv_path)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Regional distribution
    region_counts = df['jurisdiction_region'].value_counts()
    colors = ['#EF4444', '#F59E0B', '#3B82F6', '#10B981', '#8B5CF6', '#F97316']
    axes[0, 0].barh(region_counts.index, region_counts.values, color=colors[:len(region_counts)])
    axes[0, 0].set_xlabel('Number of Images', fontsize=12)
    axes[0, 0].set_title('Distribution by Region', fontsize=14, fontweight='bold')

    # Gender distribution (if available)
    if 'pronoun' in df.columns:
        gender_counts = df['pronoun'].value_counts()
        axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                      colors=['#3B82F6', '#EF4444', '#10B981', '#9CA3AF'])
        axes[0, 1].set_title('Distribution by Gender', fontsize=14, fontweight='bold')

    # Age distribution (if available)
    if 'age_group' in df.columns:
        age_counts = df['age_group'].value_counts()
        axes[1, 0].bar(range(len(age_counts)), age_counts.values,
                      color='#8B5CF6', alpha=0.7)
        axes[1, 0].set_xticks(range(len(age_counts)))
        axes[1, 0].set_xticklabels(age_counts.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Distribution by Age Group', fontsize=14, fontweight='bold')

    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    FHIBE Dataset Summary
    ━━━━━━━━━━━━━━━━━━━━━

    Total Images: {len(df):,}
    Unique Subjects: {df['image_id'].nunique():,}

    Regions: {df['jurisdiction_region'].nunique()}
    Countries: {df['jurisdiction'].nunique()}

    Consent: ✓ Explicit informed consent
    Demographics: ✓ Self-reported
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=14,
                   verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('FHIBE Dataset Demographics Overview',
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved demographics summary to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize FHIBE dataset diversity')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to FHIBE fullres dataset directory')
    parser.add_argument('--output', type=str, default='figures/',
                       help='Output directory for figures')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of sample images per region')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FHIBE Dataset Visualization")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Samples per region: {args.samples}")
    print("")

    # Load metadata
    print("Loading metadata...")
    metadata_df = load_image_metadata(args.dataset)

    # Find sample images
    print("Finding sample images...")
    samples = find_sample_images(args.dataset, metadata_df, args.samples)

    for region, imgs in samples.items():
        print(f"  {region}: {len(imgs)} samples")

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. Regional diversity grid
    create_region_grid(samples, output_dir / 'dataset_regional_diversity.png')

    # 2. Demographics summary
    create_demographics_summary(args.dataset, output_dir / 'dataset_demographics_summary.png')

    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
