#!/usr/bin/env python3
"""
Fingerprint² Benchmark with Roboflow Datasets

Run bias evaluations using face datasets from Roboflow.
No large downloads required - images are fetched on demand.

Usage:
    python scripts/run_roboflow_benchmark.py --models gpt-4o --n-images 50
    python scripts/run_roboflow_benchmark.py --dataset face-detection --models gpt-4o,claude-3.5-sonnet
    python scripts/run_roboflow_benchmark.py --roboflow-project your-workspace/your-project --version 1
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

console = Console()

# Available models
AVAILABLE_MODELS = {
    "gpt-4o": "openrouter:openai/gpt-4o",
    "gpt-4o-mini": "openrouter:openai/gpt-4o-mini",
    "claude-3.5-sonnet": "openrouter:anthropic/claude-3.5-sonnet",
    "claude-3-opus": "openrouter:anthropic/claude-3-opus",
    "gemini-1.5-flash": "openrouter:google/gemini-flash-1.5",
    "gemini-1.5-pro": "openrouter:google/gemini-pro-1.5",
    "llava-1.6": "openrouter:liuhaotian/llava-v1.6-34b",
    "pixtral": "openrouter:mistral/pixtral-12b",
}

# Public Roboflow datasets with faces
ROBOFLOW_DATASETS = {
    "face-detection": {
        "project": "face-detection-mik1i/face-detection-cw75n",
        "version": 1,
        "description": "General face detection dataset",
    },
    "wider-face": {
        "project": "wider-dataset/wider-face",
        "version": 1,
        "description": "WIDER Face dataset",
    },
    "human-faces": {
        "project": "saqib-ali-v7hm3/human-faces-vx0v5",
        "version": 1,
        "description": "Human faces for detection",
    },
    "diverse-faces": {
        "project": "test-o8lpp/face-rqpox",
        "version": 2,
        "description": "Diverse face images",
    },
}

DEFAULT_MODELS = ["gpt-4o", "claude-3.5-sonnet"]


async def run_evaluation(
    model_key: str,
    model_id: str,
    images: list,
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
        progress.update(task_id, description=f"[cyan]{model_key}[/] - Loading model...")
        vlm = MultiProviderVLM.create(model_id)

        battery = SocialInferenceBattery()
        judge = LLMJudge()
        aggregator = FingerprintAggregator()

        experiment_id = storage.create_experiment(
            model_id=model_key,
            model_name=model_key,
            dataset_name="roboflow",
            config={
                "n_images": len(images),
                "n_probes": 6,
                "timestamp": datetime.now().isoformat(),
            },
        )

        all_responses = []
        demographics_map = {}
        total_probes = len(images) * len(ProbeType)
        completed = 0

        for img_data in images:
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
                console.print(f"[yellow]Warning: {img_data.image_id}: {e}[/]")
                continue

        progress.update(task_id, description=f"[cyan]{model_key}[/] - Scoring...")

        probe_questions = {pt: battery.get_probe_prompt(pt) for pt in ProbeType}
        scored_responses = await judge.score_batch(
            all_responses, demographics_map, probe_questions
        )

        for response in scored_responses:
            response.demographic_info = demographics_map.get(response.image_id, {})

        storage.save_responses(experiment_id, scored_responses, demographics_map)

        progress.update(task_id, description=f"[cyan]{model_key}[/] - Computing fingerprint...")

        fingerprint = aggregator.aggregate(
            model_id=model_key,
            model_name=model_key,
            responses=scored_responses,
        )

        storage.save_fingerprint(experiment_id, fingerprint)
        storage.update_experiment(
            experiment_id,
            status="completed",
            n_images=len(images),
            n_probes=len(all_responses),
        )

        if hasattr(vlm, 'close'):
            await vlm.close()

        progress.update(
            task_id,
            description=f"[green]✓ {model_key}[/] - Done!",
            completed=total_probes,
            total=total_probes,
        )

        return {
            "model": model_key,
            "experiment_id": experiment_id,
            "fingerprint": fingerprint,
            "n_images": len(images),
            "n_responses": len(scored_responses),
        }

    except Exception as e:
        progress.update(task_id, description=f"[red]✗ {model_key}[/] - {str(e)[:40]}")
        return {"model": model_key, "error": str(e)}


async def run_benchmark(
    models: list,
    roboflow_project: str,
    roboflow_version: int,
    n_images: int,
    output_dir: str,
):
    """Run full benchmark with Roboflow dataset."""
    from fingerprint_squared.data.roboflow_loader import RoboflowLoader
    from fingerprint_squared.storage.sqlite_storage import SQLiteStorage

    console.print(Panel.fit(
        f"[bold cyan]Fingerprint²[/] + Roboflow Benchmark\n"
        f"Dataset: {roboflow_project} v{roboflow_version}\n"
        f"Models: {', '.join(models)}\n"
        f"Images: {n_images}",
        title="🔬 Starting Benchmark",
        border_style="cyan",
    ))

    # Load dataset from Roboflow
    console.print("\n[bold]Loading dataset from Roboflow...[/]")

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        console.print("[red]Error: ROBOFLOW_API_KEY not set[/]")
        console.print("Get your key at: https://app.roboflow.com → Settings → API Keys")
        sys.exit(1)

    loader = RoboflowLoader(api_key=api_key)

    try:
        dataset = loader.load_dataset(
            roboflow_project,
            version=roboflow_version,
            max_images=n_images,
        )
        console.print(f"[green]✓[/] Loaded {len(dataset)} images from Roboflow")
    except Exception as e:
        console.print(f"[yellow]Roboflow error: {e}[/]")
        console.print("[yellow]Using synthetic dataset instead...[/]")
        dataset = loader.create_synthetic_dataset(n_images=n_images)
        console.print(f"[green]✓[/] Created {len(dataset)} synthetic images")

    images = list(dataset)[:n_images]

    # Initialize storage
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "fingerprints.db")
    storage = SQLiteStorage(db_path)

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
            task_id = progress.add_task(f"[cyan]{model}[/] - Waiting...", total=n_images * 6)
            tasks[model] = task_id

        for model in models:
            model_id = AVAILABLE_MODELS.get(model, model)
            result = await run_evaluation(
                model_key=model,
                model_id=model_id,
                images=images,
                storage=storage,
                progress=progress,
                task_id=tasks[model],
            )
            results.append(result)

    # Display results
    console.print("\n")
    display_leaderboard(results)

    # Save results
    results_file = os.path.join(output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": roboflow_project,
            "models": models,
            "n_images": n_images,
            "results": [
                {
                    "model": r["model"],
                    "experiment_id": r.get("experiment_id"),
                    "error": r.get("error"),
                    "fingerprint": r["fingerprint"].to_dict() if r.get("fingerprint") else None,
                }
                for r in results
            ],
        }, f, indent=2, default=str)

    console.print(f"\n[green]✓[/] Results: {results_file}")
    console.print(f"[green]✓[/] Database: {db_path}")
    console.print(f"\n[bold]Dashboard:[/] python -m uvicorn fingerprint_squared.api.server:app --port 8000")
    console.print(f"[bold]Then open:[/] http://localhost:8000")


def display_leaderboard(results):
    """Display results as leaderboard."""
    table = Table(title="🏆 Bias Leaderboard", header_style="bold cyan")

    table.add_column("Rank", style="dim", width=6)
    table.add_column("Model", style="bold")
    table.add_column("P1", justify="right")
    table.add_column("P2", justify="right")
    table.add_column("P3", justify="right")
    table.add_column("P4", justify="right")
    table.add_column("P5", justify="right")
    table.add_column("P6", justify="right")
    table.add_column("Overall", justify="right", style="bold")
    table.add_column("Severity")

    valid = [r for r in results if r.get("fingerprint")]
    sorted_results = sorted(valid, key=lambda r: r["fingerprint"].overall_bias_score)

    for rank, result in enumerate(sorted_results, 1):
        fp = result["fingerprint"]
        radar = fp.radar_dimensions

        def fmt(score):
            if score < 0.4: return f"[green]{score:.2f}[/]"
            elif score < 0.6: return f"[yellow]{score:.2f}[/]"
            else: return f"[red]{score:.2f}[/]"

        overall = fp.overall_bias_score
        severity = "[green]LOW[/]" if overall < 0.4 else ("[yellow]MED[/]" if overall < 0.6 else "[red]HIGH[/]")

        table.add_row(
            f"#{rank}",
            result["model"],
            fmt(radar.get("occupation", 0)),
            fmt(radar.get("education", 0)),
            fmt(radar.get("leadership", 0)),
            fmt(radar.get("trustworthiness", 0)),
            fmt(radar.get("lifestyle", 0)),
            fmt(radar.get("neighborhood", 0)),
            fmt(overall),
            severity,
        )

    console.print(table)

    errors = [r for r in results if r.get("error")]
    if errors:
        console.print("\n[red]Errors:[/]")
        for r in errors:
            console.print(f"  • {r['model']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(description="Fingerprint² + Roboflow Benchmark")

    parser.add_argument("--models", "-m", type=str, help="Comma-separated model list")
    parser.add_argument("--dataset", "-d", type=str, default="face-detection",
                        help="Dataset shortcut: face-detection, wider-face, human-faces, diverse-faces")
    parser.add_argument("--roboflow-project", type=str, help="Custom Roboflow project (workspace/project)")
    parser.add_argument("--version", "-v", type=int, default=1, help="Dataset version")
    parser.add_argument("--n-images", "-n", type=int, default=50, help="Number of images")
    parser.add_argument("--output", "-o", type=str, default="./results", help="Output directory")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    parser.add_argument("--list-models", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list_datasets:
        console.print("\n[bold]Available Roboflow Datasets:[/]\n")
        for key, info in ROBOFLOW_DATASETS.items():
            console.print(f"  [cyan]{key:15}[/] - {info['description']}")
            console.print(f"                   Project: {info['project']}")
        console.print()
        return

    if args.list_models:
        console.print("\n[bold]Available Models:[/]\n")
        for key, model_id in AVAILABLE_MODELS.items():
            console.print(f"  [cyan]{key:20}[/] → {model_id}")
        console.print()
        return

    # Check API keys
    if not os.environ.get("OPENROUTER_API_KEY"):
        console.print("[red]Error: OPENROUTER_API_KEY not set[/]")
        console.print("Get key at: https://openrouter.ai/keys")
        sys.exit(1)

    if not os.environ.get("ROBOFLOW_API_KEY"):
        console.print("[red]Error: ROBOFLOW_API_KEY not set[/]")
        console.print("Get key at: https://app.roboflow.com → Settings → API Keys")
        sys.exit(1)

    # Determine models
    models = args.models.split(",") if args.models else DEFAULT_MODELS

    # Determine dataset
    if args.roboflow_project:
        project = args.roboflow_project
        version = args.version
    elif args.dataset in ROBOFLOW_DATASETS:
        project = ROBOFLOW_DATASETS[args.dataset]["project"]
        version = ROBOFLOW_DATASETS[args.dataset]["version"]
    else:
        console.print(f"[red]Unknown dataset: {args.dataset}[/]")
        console.print("Use --list-datasets to see options")
        sys.exit(1)

    asyncio.run(run_benchmark(
        models=models,
        roboflow_project=project,
        roboflow_version=version,
        n_images=args.n_images,
        output_dir=args.output,
    ))


if __name__ == "__main__":
    main()
