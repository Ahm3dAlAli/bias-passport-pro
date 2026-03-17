#!/usr/bin/env python
"""
Script to compare multiple VLMs using Fingerprint².

Usage:
    python scripts/compare_models.py --models gpt-4o claude-3-opus gemini-1.5-pro
    python scripts/compare_models.py --results-dir ./fp2_results
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fingerprint_squared.core.fingerprint import FingerprintComparator, ModelFingerprint
from fingerprint_squared.visualization.plots import BiasRadarChart, ComparisonPlot
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple VLMs using Fingerprint² results"
    )

    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="./fp2_results",
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Specific models to compare"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./comparison_results",
        help="Output directory for comparison"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "html"],
        default="text",
        help="Output format"
    )

    return parser.parse_args()


def load_fingerprints(results_dir: Path, models: list = None) -> dict:
    """Load fingerprints from results directory."""
    fingerprints = {}
    fp_dir = results_dir / "fingerprints"

    if not fp_dir.exists():
        print(f"No fingerprints found in {fp_dir}")
        return fingerprints

    for fp_file in fp_dir.glob("*.json"):
        with open(fp_file, "r") as f:
            data = json.load(f)

        fp = ModelFingerprint.from_dict(data)

        if models is None or fp.model_name in models:
            # Keep latest fingerprint per model
            if fp.model_name not in fingerprints or fp.timestamp > fingerprints[fp.model_name].timestamp:
                fingerprints[fp.model_name] = fp

    return fingerprints


def print_comparison(fingerprints: dict, comparator: FingerprintComparator):
    """Print comparison results to console."""
    models = list(fingerprints.keys())

    print("\n" + "="*70)
    print("FINGERPRINT² MODEL COMPARISON")
    print("="*70 + "\n")

    # Summary table
    print("MODEL SUMMARY")
    print("-" * 70)
    print(f"{'Model':<25} {'Bias Score':<12} {'Fairness':<12} {'Level':<15}")
    print("-" * 70)

    for model, fp in sorted(fingerprints.items(), key=lambda x: x[1].bias_scores.get('overall', 0)):
        bias = fp.bias_scores.get('overall', 0)
        fairness = fp.fairness_scores.get('overall', 0)
        print(f"{model:<25} {bias:<12.3f} {fairness:<12.3f} {fp.bias_level:<15}")

    print("-" * 70)

    # Rankings
    print("\n\nRANKINGS (Lower Bias = Better)")
    print("-" * 40)

    rankings = comparator.rank_models("overall_bias", ascending=True)
    for i, (model, score) in enumerate(rankings, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} #{i}: {model} ({score:.3f})")

    # Pairwise comparisons
    if len(models) >= 2:
        print("\n\nPAIRWISE COMPARISONS")
        print("-" * 40)

        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                fp1 = fingerprints[m1]
                fp2 = fingerprints[m2]
                comparison = comparator.compare(fp1, fp2)

                print(f"\n{m1} vs {m2}:")
                print(f"  Similarity: {comparison['similarity']:.3f}")
                print(f"  Bias: {comparison['bias_comparison']}")
                print(f"  Fairness: {comparison['fairness_comparison']}")

    # Risk areas
    print("\n\nRISK AREAS BY MODEL")
    print("-" * 40)

    for model, fp in fingerprints.items():
        if fp.risk_areas:
            print(f"\n{model}:")
            for risk in fp.risk_areas[:5]:
                print(f"  ⚠️  {risk}")
        else:
            print(f"\n{model}: No significant risk areas")

    print("\n" + "="*70 + "\n")


def save_json_comparison(fingerprints: dict, comparator: FingerprintComparator, output_path: Path):
    """Save comparison results as JSON."""
    results = {
        "models": {},
        "rankings": {},
        "pairwise_comparisons": {},
    }

    for model, fp in fingerprints.items():
        results["models"][model] = fp.to_dict()

    results["rankings"]["by_bias"] = comparator.rank_models("overall_bias", ascending=True)
    results["rankings"]["by_fairness"] = comparator.rank_models("overall_fairness", ascending=False)

    models = list(fingerprints.keys())
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            comparison = comparator.compare(fingerprints[m1], fingerprints[m2])
            results["pairwise_comparisons"][f"{m1}_vs_{m2}"] = comparison

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"JSON comparison saved to: {output_path}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fingerprints
    fingerprints = load_fingerprints(results_dir, args.models)

    if not fingerprints:
        print("No fingerprints found. Run evaluations first.")
        sys.exit(1)

    print(f"Loaded {len(fingerprints)} fingerprints")

    # Initialize comparator
    comparator = FingerprintComparator()
    for fp in fingerprints.values():
        comparator.add_fingerprint(fp)

    # Output results
    if args.format == "text":
        print_comparison(fingerprints, comparator)

    elif args.format == "json":
        save_json_comparison(fingerprints, comparator, output_dir / "comparison.json")

    elif args.format == "html":
        # Generate visualizations
        try:
            # Radar chart
            scores = {
                model: fp.dimension_scores
                for model, fp in fingerprints.items()
            }
            radar = BiasRadarChart()
            radar.plot(scores, output_path=output_dir / "radar_comparison.png")
            print(f"Radar chart saved to: {output_dir / 'radar_comparison.png'}")

            # Ranking chart
            rankings = comparator.rank_models("overall_bias", ascending=True)
            comp_plot = ComparisonPlot()
            comp_plot.plot_ranking(
                rankings,
                title="Model Rankings by Bias Score",
                output_path=output_dir / "rankings.png"
            )
            print(f"Rankings chart saved to: {output_dir / 'rankings.png'}")

        except ImportError as e:
            print(f"Could not generate visualizations: {e}")

        # Also save JSON
        save_json_comparison(fingerprints, comparator, output_dir / "comparison.json")


if __name__ == "__main__":
    main()
