#!/usr/bin/env python3
"""
Fingerprint² Benchmark Runner

Run comprehensive bias evaluation across multiple VLMs.
Results are saved to SQLite and can be viewed on the dashboard.

Usage:
    python scripts/run_benchmark.py --models gpt-4o,claude-3.5-sonnet --n-images 100
    python scripts/run_benchmark.py --all-models --full  # Full benchmark
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.live import Live

console = Console()


# Available models for benchmarking
AVAILABLE_MODELS = {
    # Proprietary APIs via OpenRouter
    "gpt-4o": "openrouter:openai/gpt-4o",
    "gpt-4o-mini": "openrouter:openai/gpt-4o-mini",
    "claude-3.5-sonnet": "openrouter:anthropic/claude-3.5-sonnet",
    "claude-3-opus": "openrouter:anthropic/claude-3-opus",
    "gemini-pro": "openrouter:google/gemini-pro-vision",
    "gemini-1.5-flash": "openrouter:google/gemini-flash-1.5",

    # Open source models (local or via API)
    "llava-1.6": "openrouter:liuhaotian/llava-v1.6-34b",
    "qwen-vl": "local:qwen:Qwen2.5-VL-7B-Instruct",
    "internvl": "local:internvl:InternVL3-8B",
    "llama-vision": "local:llama:Llama-3.2-11B-Vision-Instruct",
}

# Default models for quick benchmark
DEFAULT_MODELS = ["gpt-4o", "claude-3.5-sonnet", "llava-1.6"]


async def run_single_model_evaluation(
    model_key: str,
    model_id: str,
    dataset,
    n_images: int,
    storage,
    progress,
    task_id,
):
    """Run evaluation for a single model."""
    from fingerprint_squared import (
        SocialInferenceBattery,
        LLMJudge,
        FingerprintAggregator,
        ProbeType,
    )
    from fingerprint_squared.models.openrouter_vlm import MultiProviderVLM
    from PIL import Image

    try:
        # Create VLM
        vlm = MultiProviderVLM.create(model_id)

        # Initialize components
        battery = SocialInferenceBattery()
        judge = LLMJudge()
        aggregator = FingerprintAggregator()

        # Get balanced sample
        sample = dataset.get_balanced_sample(n_per_group=max(1, n_images // 24), seed=42)
        images_to_process = list(sample.images)[:n_images]

        # Create experiment
        experiment_id = storage.create_experiment(
            model_id=model_key,
            model_name=model_key,
            dataset_name="fhibe",
            config={
                "n_images": len(images_to_process),
                "n_probes": 6,
                "timestamp": datetime.now().isoformat(),
            },
        )

        progress.update(task_id, description=f"[cyan]{model_key}[/] - Loading images...")

        # Load images and run probes
        all_responses = []
        demographics_map = {}

        total_probes = len(images_to_process) * len(ProbeType)
        completed = 0

        for img_data in images_to_process:
            try:
                image = Image.open(img_data.image_path).convert("RGB")
                demographics_map[img_data.image_id] = img_data.demographics

                for probe_type in ProbeType:
                    progress.update(
                        task_id,
                        description=f"[cyan]{model_key}[/] - {probe_type.value} ({completed}/{total_probes})",
                        completed=completed,
                        total=total_probes,
                    )

                    response = await battery.run_single_probe(
                        vlm, image, img_data.image_id, probe_type
                    )
                    all_responses.append(response)
                    completed += 1

            except Exception as e:
                console.print(f"[yellow]Warning: Error processing {img_data.image_id}: {e}[/]")
                continue

        progress.update(task_id, description=f"[cyan]{model_key}[/] - Scoring responses...")

        # Score responses
        probe_questions = {pt: battery.get_probe_prompt(pt) for pt in ProbeType}
        scored_responses = await judge.score_batch(
            all_responses,
            demographics_map,
            probe_questions,
        )

        # Add demographic info
        for response in scored_responses:
            response.demographic_info = demographics_map.get(response.image_id, {})

        # Save responses
        storage.save_responses(experiment_id, scored_responses, demographics_map)

        progress.update(task_id, description=f"[cyan]{model_key}[/] - Computing fingerprint...")

        # Generate fingerprint
        fingerprint = aggregator.aggregate(
            model_id=model_key,
            model_name=model_key,
            responses=scored_responses,
        )

        # Save fingerprint
        storage.save_fingerprint(experiment_id, fingerprint)

        # Update experiment
        storage.update_experiment(
            experiment_id,
            status="completed",
            n_images=len(images_to_process),
            n_probes=len(all_responses),
        )

        # Cleanup
        if hasattr(vlm, 'close'):
            await vlm.close()

        progress.update(task_id, description=f"[green]✓ {model_key}[/] - Complete!", completed=total_probes, total=total_probes)

        return {
            "model": model_key,
            "experiment_id": experiment_id,
            "fingerprint": fingerprint,
            "n_images": len(images_to_process),
            "n_responses": len(scored_responses),
        }

    except Exception as e:
        progress.update(task_id, description=f"[red]✗ {model_key}[/] - Error: {str(e)[:50]}")
        return {
            "model": model_key,
            "error": str(e),
        }


async def run_benchmark(
    models: List[str],
    dataset_path: Optional[str],
    n_images: int,
    output_dir: str,
):
    """Run the full benchmark."""
    from fingerprint_squared import FHIBELoader, SQLiteStorage

    console.print(Panel.fit(
        "[bold cyan]Fingerprint²[/] VLM Bias Benchmark\n"
        f"Models: {', '.join(models)}\n"
        f"Images per model: {n_images}",
        title="🔬 Starting Benchmark",
        border_style="cyan",
    ))

    # Initialize storage
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "fingerprints.db")
    storage = SQLiteStorage(db_path)

    # Load dataset
    console.print("\n[bold]Loading dataset...[/]")
    loader = FHIBELoader()

    if dataset_path and Path(dataset_path).exists():
        dataset = loader.load_from_directory(dataset_path)
        console.print(f"[green]✓[/] Loaded {len(dataset)} images from {dataset_path}")
    else:
        console.print("[yellow]Using synthetic dataset for testing[/]")
        dataset = loader.create_synthetic_dataset(n_per_intersection=5)
        console.print(f"[green]✓[/] Created {len(dataset)} synthetic images")

    # Run evaluations
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        tasks = {}
        for model in models:
            model_id = AVAILABLE_MODELS.get(model, model)
            task_id = progress.add_task(f"[cyan]{model}[/] - Initializing...", total=n_images * 6)
            tasks[model] = task_id

        # Run models sequentially (to avoid rate limits)
        for model in models:
            model_id = AVAILABLE_MODELS.get(model, model)
            result = await run_single_model_evaluation(
                model_key=model,
                model_id=model_id,
                dataset=dataset,
                n_images=n_images,
                storage=storage,
                progress=progress,
                task_id=tasks[model],
            )
            results.append(result)

    # Display results
    console.print("\n")
    display_results(results, storage)

    # Save results JSON
    results_file = os.path.join(output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "models": models,
                "n_images": n_images,
                "results": [
                    {
                        "model": r["model"],
                        "experiment_id": r.get("experiment_id"),
                        "n_images": r.get("n_images"),
                        "n_responses": r.get("n_responses"),
                        "error": r.get("error"),
                        "fingerprint": r["fingerprint"].to_dict() if r.get("fingerprint") else None,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
            default=str,
        )

    console.print(f"\n[green]✓[/] Results saved to: {results_file}")
    console.print(f"[green]✓[/] Database: {db_path}")
    console.print(f"\n[bold]View dashboard at:[/] http://localhost:8000")

    return results


def display_results(results, storage):
    """Display benchmark results in a nice table."""

    # Create leaderboard table
    table = Table(
        title="🏆 Bias Leaderboard",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Rank", style="dim", width=6)
    table.add_column("Model", style="bold")
    table.add_column("P1 Occ.", justify="right")
    table.add_column("P2 Edu.", justify="right")
    table.add_column("P3 Auth.", justify="right")
    table.add_column("P4 Trust", justify="right")
    table.add_column("P5 Life.", justify="right")
    table.add_column("P6 Geo.", justify="right")
    table.add_column("Overall", justify="right", style="bold")
    table.add_column("Severity")

    # Sort by overall bias (lower is better)
    valid_results = [r for r in results if r.get("fingerprint")]
    sorted_results = sorted(
        valid_results,
        key=lambda r: r["fingerprint"].overall_bias_score,
    )

    for rank, result in enumerate(sorted_results, 1):
        fp = result["fingerprint"]
        radar = fp.radar_dimensions

        # Color based on score
        def score_color(score):
            if score < 0.4:
                return f"[green]{score:.2f}[/]"
            elif score < 0.6:
                return f"[yellow]{score:.2f}[/]"
            else:
                return f"[red]{score:.2f}[/]"

        # Severity badge
        overall = fp.overall_bias_score
        if overall < 0.4:
            severity = "[green]LOW[/]"
        elif overall < 0.6:
            severity = "[yellow]MED[/]"
        else:
            severity = "[red]HIGH[/]"

        table.add_row(
            f"#{rank}",
            result["model"],
            score_color(radar.get("occupation", 0)),
            score_color(radar.get("education", 0)),
            score_color(radar.get("leadership", 0)),
            score_color(radar.get("trustworthiness", 0)),
            score_color(radar.get("lifestyle", 0)),
            score_color(radar.get("neighborhood", 0)),
            score_color(overall),
            severity,
        )

    console.print(table)

    # Show errors if any
    errors = [r for r in results if r.get("error")]
    if errors:
        console.print("\n[red]Errors:[/]")
        for r in errors:
            console.print(f"  • {r['model']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Fingerprint² VLM Bias Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with default models
  python scripts/run_benchmark.py --n-images 50

  # Specific models
  python scripts/run_benchmark.py --models gpt-4o,claude-3.5-sonnet --n-images 100

  # Full benchmark with all models
  python scripts/run_benchmark.py --all-models --n-images 200

  # With custom dataset
  python scripts/run_benchmark.py --dataset ./fhibe_images --n-images 100

Available models:
  Proprietary: gpt-4o, gpt-4o-mini, claude-3.5-sonnet, claude-3-opus, gemini-pro
  Open Source: llava-1.6, qwen-vl, internvl, llama-vision
        """,
    )

    parser.add_argument(
        "--models", "-m",
        type=str,
        help="Comma-separated list of models to evaluate",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Evaluate all available models",
    )
    parser.add_argument(
        "--n-images", "-n",
        type=int,
        default=50,
        help="Number of images per model (default: 50)",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Path to FHIBE dataset directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        console.print("\n[bold]Available Models:[/]\n")
        for key, model_id in AVAILABLE_MODELS.items():
            console.print(f"  [cyan]{key:20}[/] → {model_id}")
        console.print()
        return

    # Determine models to run
    if args.all_models:
        models = list(AVAILABLE_MODELS.keys())
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = DEFAULT_MODELS

    # Validate models
    invalid = [m for m in models if m not in AVAILABLE_MODELS and not m.startswith(("openrouter:", "local:"))]
    if invalid:
        console.print(f"[red]Error:[/] Unknown models: {', '.join(invalid)}")
        console.print(f"Use --list-models to see available models")
        sys.exit(1)

    # Run benchmark
    asyncio.run(run_benchmark(
        models=models,
        dataset_path=args.dataset,
        n_images=args.n_images,
        output_dir=args.output,
    ))


if __name__ == "__main__":
    main()
