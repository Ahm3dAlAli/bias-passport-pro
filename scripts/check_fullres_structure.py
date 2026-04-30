#!/usr/bin/env python3
"""Quick script to check fullres FHIBE dataset structure."""

import csv
import json
from pathlib import Path
import sys

def main():
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres")

    print(f"Checking: {dataset_path}")
    print("=" * 60)

    # Check filepaths.csv
    filepaths_csv = dataset_path / "filepaths.csv"
    if filepaths_csv.exists():
        print(f"\n1. Found: {filepaths_csv}")
        with open(filepaths_csv, 'r') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            print(f"   Columns ({len(cols)}): {cols}")

            # Read first row
            first_row = next(reader, None)
            if first_row:
                print(f"\n   Sample row:")
                for k, v in first_row.items():
                    val_preview = str(v)[:80] + "..." if len(str(v)) > 80 else str(v)
                    print(f"     {k}: {val_preview}")

                # Check if JSON file exists and has demographics
                json_path = first_row.get("json", "")
                if json_path:
                    full_json_path = dataset_path / json_path
                    print(f"\n2. Checking JSON: {full_json_path}")
                    if full_json_path.exists():
                        with open(full_json_path, 'r') as jf:
                            json_data = json.load(jf)
                        print(f"   JSON keys: {list(json_data.keys())}")

                        # Print full JSON structure
                        print(f"\n   Full JSON content:")
                        print(json.dumps(json_data, indent=2, default=str)[:3000])
                    else:
                        print(f"   JSON file not found!")

                # Check image paths
                for col in ["img", "img_face_crop_and_align"]:
                    if col in first_row and first_row[col]:
                        img_path = dataset_path / first_row[col]
                        print(f"\n3. Image ({col}): {img_path}")
                        print(f"   Exists: {img_path.exists()}")

    else:
        print(f"filepaths.csv not found at {filepaths_csv}")

        # Try to find any CSV
        csvs = list(dataset_path.glob("*.csv"))
        print(f"\nFound {len(csvs)} CSV files:")
        for c in csvs[:5]:
            print(f"  - {c.name}")

if __name__ == "__main__":
    main()
