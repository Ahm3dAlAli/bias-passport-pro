#!/usr/bin/env python3
"""
FHIBE Dataset Setup Script

This script helps you:
1. Download the FHIBE dataset from Sony
2. Extract and organize the files
3. Verify the installation

Prerequisites:
- Request access at: https://fairnessbenchmark.ai.sony/download
- You'll receive a download link after approval

Usage:
    python scripts/setup_fhibe.py --download-path /path/to/downloaded.tar
    python scripts/setup_fhibe.py --extract-only /path/to/fhibe_directory
"""

import argparse
import os
import sys
import tarfile
import json
from pathlib import Path
from typing import Optional


def extract_fhibe_dataset(tar_path: str, extract_to: str = "./fhibe_data") -> str:
    """Extract the FHIBE dataset from the downloaded tar file."""
    print(f"Extracting {tar_path} to {extract_to}...")

    os.makedirs(extract_to, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=extract_to)

    # Find the extracted directory
    extracted_dirs = [d for d in os.listdir(extract_to) if d.startswith("fhibe.")]
    if extracted_dirs:
        return os.path.join(extract_to, extracted_dirs[0])
    return extract_to


def verify_fhibe_structure(fhibe_root: str) -> dict:
    """Verify the FHIBE dataset structure and return statistics."""
    stats = {
        "valid": False,
        "images_found": 0,
        "metadata_files": [],
        "issues": [],
    }

    # Expected paths
    data_dir = Path(fhibe_root) / "data"
    raw_dir = data_dir / "raw" / "fhibe_downsampled"
    processed_dir = data_dir / "processed"

    if not data_dir.exists():
        stats["issues"].append(f"Missing data directory: {data_dir}")
        return stats

    # Count images
    if raw_dir.exists():
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        for root, dirs, files in os.walk(raw_dir):
            for f in files:
                if Path(f).suffix.lower() in image_extensions:
                    stats["images_found"] += 1
    else:
        stats["issues"].append(f"Missing raw images directory: {raw_dir}")

    # Check metadata
    if processed_dir.exists():
        for f in processed_dir.glob("*.csv"):
            stats["metadata_files"].append(f.name)
        for f in processed_dir.glob("*.json"):
            stats["metadata_files"].append(f.name)
    else:
        stats["issues"].append(f"Missing processed directory: {processed_dir}")

    # Determine validity
    if stats["images_found"] > 0 and not stats["issues"]:
        stats["valid"] = True

    return stats


def create_fingerprint_config(fhibe_root: str, output_path: str = "./fhibe_config.json"):
    """Create a configuration file for Fingerprint² to use FHIBE."""
    config = {
        "dataset_name": "fhibe",
        "root_path": os.path.abspath(fhibe_root),
        "images_path": os.path.join(fhibe_root, "data", "raw", "fhibe_downsampled"),
        "metadata_path": os.path.join(fhibe_root, "data", "processed"),
        "face_crops_path": os.path.join(fhibe_root, "data", "raw", "fhibe_face_crop_align"),
        "supported_tasks": [
            "face_detection",
            "face_verification",
            "person_localization",
            "face_parsing",
            "keypoint_detection",
            "face_attribute",
            "object_detection",
            "instance_segmentation",
            "semantic_segmentation",
        ],
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {output_path}")
    return config


def print_download_instructions():
    """Print instructions for downloading the FHIBE dataset."""
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     FHIBE Dataset Download Instructions                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. Visit: https://fairnessbenchmark.ai.sony/download                        ║
║                                                                              ║
║  2. Fill out the access request form                                         ║
║     - Provide your institutional email                                       ║
║     - Describe your research use case                                        ║
║                                                                              ║
║  3. Wait for approval email (typically 1-3 business days)                    ║
║                                                                              ║
║  4. Download the file: *downsampled_public.tar                               ║
║     (approximately 50-100 GB)                                                ║
║                                                                              ║
║  5. Run this script with the downloaded file:                                ║
║                                                                              ║
║     python scripts/setup_fhibe.py --download-path /path/to/downloaded.tar    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          Dataset Information                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  • ~10,318 high-resolution images                                            ║
║  • 1,981 unique subjects                                                     ║
║  • 81 jurisdictions (countries/regions)                                      ║
║  • Attributes: age, ancestry, skin tone (Fitzpatrick), gender                ║
║  • Annotations: bounding boxes, keypoints, segmentation masks                ║
║                                                                              ║
║  IMPORTANT: The dataset is GDPR compliant. Subjects can revoke consent       ║
║  at any time, so always use the latest version.                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(instructions)


def main():
    parser = argparse.ArgumentParser(
        description="FHIBE Dataset Setup for Fingerprint²",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--download-path",
        type=str,
        help="Path to the downloaded FHIBE tar file",
    )
    parser.add_argument(
        "--extract-to",
        type=str,
        default="./fhibe_data",
        help="Directory to extract the dataset to",
    )
    parser.add_argument(
        "--fhibe-root",
        type=str,
        help="Path to already extracted FHIBE dataset (skip extraction)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing installation",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print download instructions and exit",
    )

    args = parser.parse_args()

    if args.instructions:
        print_download_instructions()
        return

    if not args.download_path and not args.fhibe_root:
        print_download_instructions()
        print("\nNo dataset path provided. See instructions above.")
        print("\nUsage:")
        print("  python scripts/setup_fhibe.py --download-path /path/to/fhibe.tar")
        print("  python scripts/setup_fhibe.py --fhibe-root /path/to/extracted/fhibe")
        return

    # Determine FHIBE root
    if args.fhibe_root:
        fhibe_root = args.fhibe_root
    elif args.download_path:
        if not os.path.exists(args.download_path):
            print(f"Error: Download file not found: {args.download_path}")
            sys.exit(1)
        fhibe_root = extract_fhibe_dataset(args.download_path, args.extract_to)

    # Verify structure
    print(f"\nVerifying FHIBE dataset at: {fhibe_root}")
    stats = verify_fhibe_structure(fhibe_root)

    print(f"\n{'='*60}")
    print("FHIBE Dataset Verification Results")
    print(f"{'='*60}")
    print(f"  Valid:          {stats['valid']}")
    print(f"  Images found:   {stats['images_found']}")
    print(f"  Metadata files: {len(stats['metadata_files'])}")

    if stats["metadata_files"]:
        for f in stats["metadata_files"][:5]:
            print(f"    - {f}")
        if len(stats["metadata_files"]) > 5:
            print(f"    ... and {len(stats['metadata_files']) - 5} more")

    if stats["issues"]:
        print("\n  Issues:")
        for issue in stats["issues"]:
            print(f"    ⚠ {issue}")

    if stats["valid"]:
        # Create config for Fingerprint²
        config_path = "./fhibe_config.json"
        create_fingerprint_config(fhibe_root, config_path)

        print(f"\n{'='*60}")
        print("Setup Complete!")
        print(f"{'='*60}")
        print(f"\nFHIBE dataset is ready at: {fhibe_root}")
        print(f"Configuration saved to: {config_path}")
        print("\nNext steps:")
        print("  1. Run benchmark:")
        print(f"     python scripts/run_benchmark.py --dataset {fhibe_root}/data/raw/fhibe_downsampled --n-images 100")
        print("\n  2. Or start the dashboard:")
        print("     python -m uvicorn fingerprint_squared.api.server:app --reload")
    else:
        print("\nDataset verification failed. Please check the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
