#!/usr/bin/env python3
"""
visualize_dataset_simple.py
============================
Generate simple visualization of FHIBE dataset showing sample images.
Does not require metadata CSV.

Usage:
    python scripts/visualize_dataset_simple.py --dataset /path/to/fhibe --output figures/
"""

import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def find_sample_images(dataset_path, num_samples=24):
    """Find random sample images from dataset."""
    dataset_path = Path(dataset_path)

    # FHIBE fullres structure: subject_id/image_id/main_*.png
    # Find all main_*.png images (not masks, bbox, keypoints)
    all_images = []

    # Pattern: */*/main_[uuid].png (but not main_masks, main_bbox, main_keypoints)
    for subject_dir in dataset_path.iterdir():
        if subject_dir.is_dir() and not subject_dir.name.startswith('.'):
            for image_dir in subject_dir.iterdir():
                if image_dir.is_dir():
                    # Find main image (not derivatives)
                    for img_file in image_dir.glob("main_*.png"):
                        # Skip masks, bbox, keypoints
                        if not any(x in img_file.name for x in ['masks', 'bbox', 'keypoints']):
                            all_images.append(img_file)

    # Also try faces_crop_and_align as backup
    if len(all_images) < num_samples:
        for subject_dir in dataset_path.iterdir():
            if subject_dir.is_dir() and not subject_dir.name.startswith('.'):
                for image_dir in subject_dir.iterdir():
                    if image_dir.is_dir():
                        for img_file in image_dir.glob("faces_crop_and_align_*.png"):
                            all_images.append(img_file)

    print(f"Found {len(all_images)} total images")

    # Random sample
    if len(all_images) > num_samples:
        samples = random.sample(all_images, num_samples)
    else:
        samples = all_images[:num_samples]

    return samples

def create_image_grid(image_paths, output_path, title="FHIBE Dataset Sample Images"):
    """Create a grid of sample images."""
    n_images = len(image_paths)

    # Calculate grid size
    cols = 6
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.5))

    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        try:
            img = Image.open(img_path)
            # Resize to consistent size
            img = img.resize((400, 400), Image.Resampling.LANCZOS)
            ax.imshow(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            ax.text(0.5, 0.5, 'Error\nLoading\nImage',
                   ha='center', va='center', fontsize=12)

        ax.axis('off')

        # Add image ID as subtitle
        img_id = img_path.stem.replace('main_', '')[:12]
        ax.set_title(f"...{img_id}", fontsize=8, color='gray')

    # Hide extra subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()

def create_dataset_summary(dataset_path, output_path):
    """Create a simple text summary of the dataset."""
    dataset_path = Path(dataset_path)

    # Count images (FHIBE fullres structure)
    all_images = []
    for subject_dir in dataset_path.iterdir():
        if subject_dir.is_dir() and not subject_dir.name.startswith('.'):
            for image_dir in subject_dir.iterdir():
                if image_dir.is_dir():
                    for img_file in image_dir.glob("main_*.png"):
                        if not any(x in img_file.name for x in ['masks', 'bbox', 'keypoints']):
                            all_images.append(img_file)

    # Count directories (subjects)
    subject_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    summary_text = f"""
    FHIBE Dataset Summary
    {'=' * 50}

    Total Images Found: {len(all_images):,}
    Subject Directories: {len(subject_dirs):,}

    Dataset Path:
    {dataset_path}

    Properties:
    • Consented images from diverse jurisdictions
    • Self-reported demographics
    • Multiple images per subject
    • High-resolution portrait images

    Note: Full metadata analysis requires CSV file
    """

    ax.text(0.1, 0.5, summary_text, fontsize=13,
           verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('FHIBE Dataset Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize FHIBE dataset samples')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to FHIBE fullres dataset directory')
    parser.add_argument('--output', type=str, default='figures/',
                       help='Output directory for figures')
    parser.add_argument('--samples', type=int, default=24,
                       help='Number of sample images to show')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FHIBE Dataset Visualization (Simple)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Sample size: {args.samples}")
    print("")

    # Find sample images
    print("Finding sample images...")
    samples = find_sample_images(args.dataset, args.samples)
    print(f"Selected {len(samples)} images")

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. Image grid
    create_image_grid(samples, output_dir / 'dataset_samples.png')

    # 2. Dataset summary
    create_dataset_summary(args.dataset, output_dir / 'dataset_summary.png')

    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
