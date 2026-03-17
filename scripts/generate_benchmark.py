#!/usr/bin/env python
"""
Script to generate benchmark datasets for Fingerprint² evaluation.

Usage:
    python scripts/generate_benchmark.py --benchmark fp2-core --output ./data/benchmarks
    python scripts/generate_benchmark.py --all --output ./data/benchmarks
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fingerprint_squared.benchmarks.loader import BenchmarkLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Fingerprint² benchmark datasets"
    )

    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        help="Benchmark to generate (fp2-core, fp2-visual, fp2-occupation, fp2-intersectional)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate all benchmarks"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/benchmarks",
        help="Output directory"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        help="Maximum samples per benchmark"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = BenchmarkLoader()
    benchmarks = loader.list_benchmarks()

    if args.all:
        to_generate = list(benchmarks.keys())
    elif args.benchmark:
        to_generate = [args.benchmark]
    else:
        print("Please specify --benchmark or --all")
        print("\nAvailable benchmarks:")
        for name, desc in benchmarks.items():
            print(f"  {name}: {desc}")
        sys.exit(1)

    print(f"\nGenerating {len(to_generate)} benchmark(s)...\n")

    for benchmark_name in to_generate:
        if benchmark_name not in benchmarks:
            print(f"Unknown benchmark: {benchmark_name}")
            continue

        print(f"Generating: {benchmark_name}")

        dataset = loader.load(benchmark_name, max_samples=args.max_samples)

        # Save to JSON
        output_path = output_dir / f"{benchmark_name}.json"
        loader.save_dataset(dataset, output_path)

        print(f"  Samples: {len(dataset)}")
        print(f"  Saved to: {output_path}")

        # Print sample statistics
        bias_types = {}
        for sample in dataset:
            bt = sample.bias_type or "unknown"
            bias_types[bt] = bias_types.get(bt, 0) + 1

        print(f"  Bias types: {bias_types}")
        print()

    print(f"\nAll benchmarks saved to: {output_dir}")


if __name__ == "__main__":
    main()
