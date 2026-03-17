#!/usr/bin/env python
"""
Script to run Fingerprint² evaluation on VLMs.

Usage:
    python scripts/run_evaluation.py --model gpt-4o --api-key $OPENAI_API_KEY
    python scripts/run_evaluation.py --config configs/comprehensive.yaml
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fingerprint_squared import FingerprintSquared
from fingerprint_squared.core.evaluator import EvaluationConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Fingerprint² evaluation on Vision-Language Models"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model to evaluate (e.g., gpt-4o, claude-3-opus)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="API key for the model"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./fp2_results",
        help="Output directory"
    )
    parser.add_argument(
        "--n-probes", "-n",
        type=int,
        default=50,
        help="Number of probes per type"
    )
    parser.add_argument(
        "--dimensions", "-d",
        nargs="+",
        default=["gender", "race", "age"],
        help="Demographic dimensions to analyze"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Load configuration
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config_data = yaml.safe_load(f)

        config = EvaluationConfig(
            probe_types=config_data.get("probe_types", ["stereotype_association", "counterfactual"]),
            demographic_dimensions=config_data.get("demographic_dimensions", ["gender", "race", "age"]),
            n_probes_per_type=config_data.get("n_probes_per_type", 50),
            max_tokens=config_data.get("max_tokens", 512),
            temperature=config_data.get("temperature", 0.0),
            fairness_threshold=config_data.get("fairness_threshold", 0.1),
            bias_threshold=config_data.get("bias_threshold", 0.5),
        )
        models = config_data.get("models", [])
        output_dir = config_data.get("output_dir", args.output)
    else:
        config = EvaluationConfig(
            probe_types=["stereotype_association", "counterfactual", "representation"],
            demographic_dimensions=args.dimensions,
            n_probes_per_type=args.n_probes,
        )
        models = [args.model] if args.model else []
        output_dir = args.output

    if not models:
        print("Error: No model specified. Use --model or --config")
        sys.exit(1)

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    # Initialize framework
    log_level = "DEBUG" if args.verbose else "INFO"
    fp2 = FingerprintSquared(config=config, output_dir=output_dir, log_level=log_level)

    print(f"\n{'='*60}")
    print("Fingerprint² - Ethical AI Assessment Framework")
    print(f"{'='*60}\n")

    # Evaluate each model
    for model in models:
        print(f"Evaluating: {model}")
        print("-" * 40)

        try:
            result = await fp2.evaluate(
                model,
                api_key=api_key,
                generate_report=not args.no_report,
            )

            print(f"\nResults for {model}:")
            print(f"  Overall Bias Score: {result.overall_bias_score:.3f}")
            print(f"  Overall Fairness Score: {result.overall_fairness_score:.3f}")
            print(f"  Total Probes: {result.total_probes}")
            print(f"  Valid Responses: {result.total_responses}")

            # Get fingerprint
            fp_key = f"{result.model_name}_{result.timestamp}"
            if fp_key in fp2._fingerprints:
                fp = fp2._fingerprints[fp_key]
                print(f"\n  Fingerprint Hash: {fp.fingerprint_hash}")
                print(f"  Bias Level: {fp.bias_level}")
                print(f"  Fairness Level: {fp.fairness_level}")
                if fp.risk_areas:
                    print(f"  Risk Areas: {', '.join(fp.risk_areas[:3])}")

        except Exception as e:
            print(f"Error evaluating {model}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

        print()

    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
