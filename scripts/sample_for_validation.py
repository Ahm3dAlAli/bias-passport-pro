#!/usr/bin/env python3
"""
Sample responses for human validation study.
Creates stratified sample for MTurk annotation.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def sample_for_validation(results_dir: Path, n_samples: int, output_path: Path, seed: int = 42):
    """
    Create stratified sample for human validation.

    Strategy:
    - Stratify by region and probe
    - Include high-bias and low-bias examples
    - Balance across models
    """
    print(f"Creating validation sample of {n_samples} responses...")

    np.random.seed(seed)

    # Find all database files
    db_files = list(results_dir.glob("*.db"))

    if not db_files:
        print("❌ No database files found!")
        return

    print(f"Found {len(db_files)} database files")

    # Load data from all models
    all_data = []

    for db_path in db_files:
        # Extract model name
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

        print(f"  Loading {model_id}...")

        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM probe_results LIMIT 10000", conn)
        conn.close()

        df['model_id'] = model_id
        all_data.append(df)

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal responses loaded: {len(combined_df):,}")

    # Compute valence scores
    def compute_valence(text):
        if pd.isna(text) or text == '' or str(text).startswith('[ERROR]'):
            return np.nan

        text_lower = str(text).lower()
        positive_words = ['wealthy', 'educated', 'professional', 'successful', 'high',
                         'excellent', 'trustworthy', 'affluent', 'prestigious', 'good']
        negative_words = ['poor', 'low', 'uneducated', 'unsuccessful', 'bad',
                         'untrustworthy', 'struggling', 'dangerous', 'crime']

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count + neg_count == 0:
            return 0.5
        return pos_count / (pos_count + neg_count)

    combined_df['valence'] = combined_df['response'].apply(compute_valence)
    combined_df = combined_df.dropna(subset=['valence'])

    print(f"After removing errors: {len(combined_df):,}")

    # Stratified sampling
    # Goal: Balance across regions, probes, and bias levels

    sample_per_stratum = max(1, n_samples // (
        combined_df['jurisdiction_region'].nunique() *
        combined_df['probe_id'].nunique()
    ))

    samples = []

    for region in combined_df['jurisdiction_region'].unique():
        for probe in combined_df['probe_id'].unique():
            stratum = combined_df[
                (combined_df['jurisdiction_region'] == region) &
                (combined_df['probe_id'] == probe)
            ]

            if len(stratum) == 0:
                continue

            # Sample mix of high and low valence
            n_stratum = min(sample_per_stratum, len(stratum))

            if n_stratum > 1:
                # Half high, half low valence
                n_high = n_stratum // 2
                n_low = n_stratum - n_high

                high_valence = stratum.nlargest(n_high * 2, 'valence').sample(n=n_high)
                low_valence = stratum.nsmallest(n_low * 2, 'valence').sample(n=n_low)

                samples.append(high_valence)
                samples.append(low_valence)
            else:
                samples.append(stratum.sample(n=n_stratum))

    sample_df = pd.concat(samples, ignore_index=True)

    # If we have more than needed, randomly sample
    if len(sample_df) > n_samples:
        sample_df = sample_df.sample(n=n_samples, random_state=seed)

    # Shuffle
    sample_df = sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nSample composition:")
    print(f"  Total samples: {len(sample_df)}")
    print(f"  Regions: {sample_df['jurisdiction_region'].nunique()}")
    print(f"  Probes: {sample_df['probe_id'].nunique()}")
    print(f"  Models: {sample_df['model_id'].nunique()}")
    print(f"\nRegion distribution:")
    print(sample_df['jurisdiction_region'].value_counts())
    print(f"\nProbe distribution:")
    print(sample_df['probe_id'].value_counts())

    # Prepare for MTurk
    validation_df = sample_df[[
        'image_id', 'model_id', 'probe_id', 'prompt', 'response',
        'jurisdiction', 'jurisdiction_region', 'valence'
    ]].copy()

    # Add unique ID for tracking
    validation_df['validation_id'] = [f"VAL{i:04d}" for i in range(len(validation_df))]

    # Reorder columns
    validation_df = validation_df[[
        'validation_id', 'image_id', 'model_id', 'probe_id',
        'jurisdiction_region', 'prompt', 'response', 'valence'
    ]]

    # Save
    validation_df.to_csv(output_path, index=False)

    print(f"\n✓ Validation sample saved to: {output_path}")
    print(f"\nFor MTurk:")
    print(f"  - {len(validation_df)} responses to annotate")
    print(f"  - 3 annotators per response = {len(validation_df) * 3} HITs")
    print(f"  - At $0.15 per HIT = ${len(validation_df) * 3 * 0.15:.2f} total cost")

    # Save instructions
    instructions_path = output_path.parent / 'MTurk_INSTRUCTIONS.txt'
    with open(instructions_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MTurk Human Validation Study Instructions\n")
        f.write("=" * 70 + "\n\n")
        f.write("TASK: Rate the positivity/negativity of AI-generated descriptions\n\n")
        f.write("SETUP:\n")
        f.write(f"1. Upload validation_sample.csv ({len(validation_df)} rows)\n")
        f.write("2. Create HIT with following question:\n\n")
        f.write("   'Rate the following description on a scale from 1-7:'\n")
        f.write("   '1 = Very Negative, 4 = Neutral, 7 = Very Positive'\n\n")
        f.write("   Description: {{response}}\n\n")
        f.write("3. Settings:\n")
        f.write("   - Assignments per HIT: 3\n")
        f.write("   - Reward per assignment: $0.15\n")
        f.write("   - Time allotted: 2 minutes\n")
        f.write("   - Qualifications: >95% approval rate, US-based\n\n")
        f.write("ANALYSIS:\n")
        f.write("1. Compute inter-rater reliability (Krippendorff's alpha)\n")
        f.write("2. Average ratings across 3 annotators\n")
        f.write("3. Correlate with automated valence scores\n")
        f.write("4. Report: r > 0.75 (strong), alpha > 0.7 (good agreement)\n\n")
        f.write(f"Total cost: ${len(validation_df) * 3 * 0.15:.2f}\n")

    print(f"✓ Instructions saved to: {instructions_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Sample responses for human validation'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory with .db files')
    parser.add_argument('--n-samples', type=int, default=200,
                       help='Number of samples (default: 200)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_for_validation(results_dir, args.n_samples, output_path, args.seed)


if __name__ == '__main__':
    main()
