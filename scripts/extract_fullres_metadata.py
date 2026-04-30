#!/usr/bin/env python3
"""
extract_fullres_metadata.py - Extract demographic metadata from FHIBE fullres JSON files

The fullres FHIBE dataset stores demographic information in individual JSON annotation
files rather than a consolidated CSV. This script extracts the demographics and creates
a metadata CSV compatible with run_fhibe_benchmark.py.

Usage:
    python scripts/extract_fullres_metadata.py \
        --dataset /local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres \
        --output /local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres/metadata.csv
"""

import argparse
import json
import csv
from pathlib import Path
from tqdm import tqdm
import sys


def extract_metadata_from_json(json_path: Path) -> dict:
    """Extract demographic metadata from a FHIBE JSON annotation file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract demographics - handle different possible structures
        metadata = {
            'uid': json_path.stem.replace('main_annos_', ''),
            'json_path': str(json_path),
        }

        # Try different keys where demographics might be stored
        # Based on FHIBE Data Card: ancestry, nationality, pronouns are self-reported

        # Direct keys
        for key in ['ancestry', 'nationality', 'pronouns', 'pronoun', 'age', 'gender']:
            if key in data:
                metadata[key] = data[key]

        # Nested under 'demographics' or 'attributes'
        for container in ['demographics', 'attributes', 'self_reported', 'metadata']:
            if container in data and isinstance(data[container], dict):
                for key in ['ancestry', 'nationality', 'pronouns', 'pronoun', 'age', 'gender']:
                    if key in data[container]:
                        metadata[key] = data[container][key]

        # Nested under 'person' or 'subject'
        for container in ['person', 'subject']:
            if container in data and isinstance(data[container], dict):
                for key in ['ancestry', 'nationality', 'pronouns', 'pronoun', 'age', 'gender']:
                    if key in data[container]:
                        metadata[key] = data[container][key]

        return metadata

    except Exception as e:
        print(f"Error reading {json_path}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract FHIBE fullres metadata from JSON files')
    parser.add_argument('--dataset', type=str, required=True, help='Path to fullres dataset')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of files (0=all)')
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    # First, check if filepaths.csv exists and read JSON paths from it
    filepaths_csv = dataset_path / 'filepaths.csv'

    if filepaths_csv.exists():
        print(f"Reading JSON paths from {filepaths_csv}")
        json_paths = []
        img_paths = {}  # uid -> image path mapping

        with open(filepaths_csv, 'r') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            print(f"CSV columns: {columns}")

            for row in reader:
                uid = row.get('uid', '')

                # Find JSON path column
                json_col = None
                for col in ['json', 'json_path', 'annotation', 'annos']:
                    if col in row and row[col]:
                        json_col = col
                        break

                if json_col:
                    json_path = dataset_path / row[json_col]
                    if json_path.exists():
                        json_paths.append(json_path)

                        # Also store image path
                        for img_col in ['img', 'image', 'img_path', 'img_face_crop_and_align']:
                            if img_col in row and row[img_col]:
                                img_paths[uid] = row[img_col]
                                break

                if args.limit and len(json_paths) >= args.limit:
                    break

        print(f"Found {len(json_paths)} JSON files referenced in CSV")
    else:
        # Fall back to glob search
        print(f"filepaths.csv not found, searching for JSON files...")
        json_paths = list(dataset_path.rglob('main_annos_*.json'))
        img_paths = {}

        if args.limit:
            json_paths = json_paths[:args.limit]

        print(f"Found {len(json_paths)} JSON files via glob")

    if not json_paths:
        print("No JSON files found!")

        # Show what's in the first sample JSON to understand structure
        sample_jsons = list(dataset_path.rglob('*.json'))[:5]
        if sample_jsons:
            print("\nSample JSON files found:")
            for jp in sample_jsons:
                print(f"  {jp}")
                try:
                    with open(jp) as f:
                        data = json.load(f)
                    print(f"    Keys: {list(data.keys())[:10]}")
                except Exception as e:
                    print(f"    Error: {e}")
        return

    # Extract metadata from each JSON
    print(f"\nExtracting metadata from {len(json_paths)} JSON files...")
    all_metadata = []
    all_keys = set()

    for json_path in tqdm(json_paths, desc="Processing"):
        meta = extract_metadata_from_json(json_path)
        if meta:
            all_metadata.append(meta)
            all_keys.update(meta.keys())

    print(f"\nExtracted metadata from {len(all_metadata)} files")
    print(f"Available fields: {sorted(all_keys)}")

    # Check if we got demographics
    has_ancestry = sum(1 for m in all_metadata if 'ancestry' in m)
    has_nationality = sum(1 for m in all_metadata if 'nationality' in m)
    has_pronoun = sum(1 for m in all_metadata if 'pronoun' in m or 'pronouns' in m)

    print(f"\nDemographic coverage:")
    print(f"  ancestry: {has_ancestry}/{len(all_metadata)}")
    print(f"  nationality: {has_nationality}/{len(all_metadata)}")
    print(f"  pronoun: {has_pronoun}/{len(all_metadata)}")

    if has_ancestry == 0 and has_nationality == 0 and has_pronoun == 0:
        print("\nWARNING: No demographic fields found in JSON files!")
        print("Let's examine a sample JSON structure:")

        if json_paths:
            sample_path = json_paths[0]
            print(f"\nSample file: {sample_path}")
            with open(sample_path) as f:
                data = json.load(f)
            print(f"Full structure:")
            print(json.dumps(data, indent=2, default=str)[:2000])
        return

    # Write output CSV
    print(f"\nWriting metadata to {args.output}")

    # Define columns - ensure required ones are first
    columns = ['uid', 'img_path']
    for key in ['pronoun', 'pronouns', 'ancestry', 'nationality', 'age', 'gender']:
        if key in all_keys and key not in columns:
            columns.append(key)
    # Add remaining keys
    for key in sorted(all_keys):
        if key not in columns:
            columns.append(key)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        for meta in all_metadata:
            uid = meta.get('uid', '')
            meta['img_path'] = img_paths.get(uid, '')
            writer.writerow(meta)

    print(f"Done! Wrote {len(all_metadata)} rows to {args.output}")
    print(f"\nYou can now run the benchmark with:")
    print(f"  python scripts/run_fhibe_benchmark.py --dataset {args.dataset} ...")


if __name__ == '__main__':
    main()
