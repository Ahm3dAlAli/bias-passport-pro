#!/usr/bin/env python3
"""
Quick-start script for running bias fingerprinting.

This script demonstrates the full pipeline:
1. Load a dataset (FHIBE, UTKFace, or synthetic)
2. Run probes against one or more VLMs
3. Score responses with LLM-as-judge
4. Generate fingerprints and reports

Usage:
    # With real dataset
    python run_fingerprint.py --dataset ./images --model openai:gpt-4o

    # With synthetic data (for testing)
    python run_fingerprint.py --synthetic --model openai:gpt-4o

    # Compare multiple models
    python run_fingerprint.py --dataset ./images --model openai:gpt-4o anthropic:claude-3-sonnet
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from fingerprint_squared import (
    FingerprintPipeline,
    MultiModelPipeline,
    PipelineConfig,
    FHIBELoader,
    BiasPassportPDF,
    load_fhibe,
)


async def run_single_model(args):
    """Run fingerprinting on a single model."""
    print(f"\n{'='*60}")
    print("Fingerprint Squared - Bias Fingerprinting Pipeline")
    print(f"{'='*60}\n")

    # Load dataset
    print("[1/4] Loading dataset...")
    loader = FHIBELoader()

    if args.synthetic:
        dataset = loader.create_synthetic_dataset(n_per_intersection=args.n_images)
        print(f"  Created synthetic dataset with {len(dataset)} images")
    else:
        dataset = loader.load_from_directory(args.dataset, format=args.format)
        print(f"  Loaded {len(dataset)} images from {args.dataset}")

    # Initialize VLM
    print("\n[2/4] Initializing VLM...")
    provider, model_name = args.model[0].split(":", 1) if ":" in args.model[0] else ("openai", args.model[0])

    vlm = create_vlm(provider, model_name)
    if vlm is None:
        print(f"  ERROR: Could not initialize {provider}:{model_name}")
        return

    print(f"  Initialized {provider}:{model_name}")

    # Run pipeline
    print("\n[3/4] Running evaluation...")
    config = PipelineConfig(
        n_images_per_group=args.n_images,
        output_dir=args.output,
        use_llm_judge=not args.no_judge,
        verbose=args.verbose,
    )

    pipeline = FingerprintPipeline(
        config=config,
        progress_callback=lambda stage, progress: print(f"  {stage}: {progress:.0%}"),
    )

    results = await pipeline.run(
        vlm=vlm,
        dataset=dataset,
        model_id=f"{provider}_{model_name}",
        model_name=model_name,
    )

    # Save and display results
    print("\n[4/4] Saving results...")
    paths = results.save(args.output)

    for name, path in paths.items():
        print(f"  Saved {name}: {path}")

    # Generate PDF passport
    if args.passport:
        pdf_path = Path(args.output) / f"{results.fingerprint.model_id}_passport.pdf"
        generator = BiasPassportPDF()
        generator.generate(results.fingerprint, str(pdf_path))
        print(f"  Generated passport: {pdf_path}")

    # Display summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    display_fingerprint(results.fingerprint)

    return results


async def run_comparison(args):
    """Run fingerprinting comparison across multiple models."""
    print(f"\n{'='*60}")
    print("Fingerprint Squared - Model Comparison")
    print(f"{'='*60}\n")

    # Load dataset
    print("[1/3] Loading dataset...")
    loader = FHIBELoader()

    if args.synthetic:
        dataset = loader.create_synthetic_dataset(n_per_intersection=args.n_images)
    else:
        dataset = loader.load_from_directory(args.dataset, format=args.format)

    print(f"  Loaded {len(dataset)} images")

    # Initialize VLMs
    print("\n[2/3] Initializing models...")
    vlms = []
    model_ids = []
    model_names = []

    for model_spec in args.model:
        provider, model_name = model_spec.split(":", 1) if ":" in model_spec else ("openai", model_spec)

        vlm = create_vlm(provider, model_name)
        if vlm:
            vlms.append(vlm)
            model_ids.append(f"{provider}_{model_name}")
            model_names.append(model_name)
            print(f"  Initialized {provider}:{model_name}")
        else:
            print(f"  WARNING: Could not initialize {model_spec}")

    if not vlms:
        print("  ERROR: No valid models")
        return

    # Run comparison
    print("\n[3/3] Running comparison...")
    config = PipelineConfig(
        n_images_per_group=args.n_images,
        output_dir=args.output,
        use_llm_judge=not args.no_judge,
    )

    pipeline = MultiModelPipeline(config=config)
    results = await pipeline.run_comparison(
        models=vlms,
        model_ids=model_ids,
        model_names=model_names,
        dataset=dataset,
    )

    # Display comparison
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    sorted_fps = sorted(
        results["fingerprints"].values(),
        key=lambda x: x.overall_bias_score
    )

    print(f"\n{'Rank':<6} {'Model':<25} {'Overall':<10} {'Valence':<10} {'Stereotype':<10}")
    print("-" * 60)

    for i, fp in enumerate(sorted_fps):
        print(
            f"#{i+1:<5} {fp.model_name:<25} "
            f"{fp.overall_bias_score*100:>6.1f}%   "
            f"{fp.valence_bias*100:>6.1f}%   "
            f"{fp.stereotype_bias*100:>6.1f}%"
        )

    return results


def create_vlm(provider: str, model_name: str):
    """Create VLM instance."""
    try:
        if provider == "openai":
            from fingerprint_squared.models.openai_vlm import OpenAIVLM
            return OpenAIVLM(model=model_name)
        elif provider == "anthropic":
            from fingerprint_squared.models.anthropic_vlm import AnthropicVLM
            return AnthropicVLM(model=model_name)
        elif provider == "google":
            from fingerprint_squared.models.google_vlm import GoogleVLM
            return GoogleVLM(model=model_name)
        elif provider == "huggingface":
            from fingerprint_squared.models.huggingface_vlm import HuggingFaceVLM
            return HuggingFaceVLM(model_id=model_name)
        else:
            print(f"Unknown provider: {provider}")
            return None
    except Exception as e:
        print(f"Error creating VLM: {e}")
        return None


def display_fingerprint(fp):
    """Display fingerprint summary."""
    # Grade
    if fp.overall_bias_score < 0.2:
        grade = "A (Excellent)"
    elif fp.overall_bias_score < 0.35:
        grade = "B (Good)"
    elif fp.overall_bias_score < 0.5:
        grade = "C (Fair)"
    elif fp.overall_bias_score < 0.65:
        grade = "D (Poor)"
    else:
        grade = "F (Failing)"

    print(f"\nModel: {fp.model_name}")
    print(f"Grade: {grade}")
    print(f"\nScores:")
    print(f"  Overall Bias:    {fp.overall_bias_score*100:>6.1f}%")
    print(f"  Valence Bias:    {fp.valence_bias*100:>6.1f}%")
    print(f"  Stereotype Bias: {fp.stereotype_bias*100:>6.1f}%")
    print(f"  Confidence Bias: {fp.confidence_bias*100:>6.1f}%")
    print(f"  Refusal Rate:    {fp.refusal_rate*100:>6.1f}%")

    if fp.radar_dimensions:
        print(f"\nProbe Breakdown:")
        for probe, score in sorted(fp.radar_dimensions.items(), key=lambda x: -x[1]):
            bar = "=" * int(score * 20)
            print(f"  {probe:20} [{bar:20}] {score*100:>5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run bias fingerprinting on Vision-Language Models"
    )

    parser.add_argument(
        "--model", "-m",
        nargs="+",
        required=True,
        help="Model(s) to evaluate (format: provider:model_name)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset (no real images needed)"
    )
    parser.add_argument(
        "--format", "-f",
        default="auto",
        choices=["auto", "fhibe", "utkface", "fairface", "custom"],
        help="Dataset format"
    )
    parser.add_argument(
        "--output", "-o",
        default="./fingerprint_output",
        help="Output directory"
    )
    parser.add_argument(
        "--n-images", "-n",
        type=int,
        default=10,
        help="Images per demographic group"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring"
    )
    parser.add_argument(
        "--passport",
        action="store_true",
        help="Generate PDF passport"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Validate args
    if not args.synthetic and not args.dataset:
        parser.error("Either --dataset or --synthetic is required")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run appropriate pipeline
    if len(args.model) > 1:
        asyncio.run(run_comparison(args))
    else:
        asyncio.run(run_single_model(args))


if __name__ == "__main__":
    main()
