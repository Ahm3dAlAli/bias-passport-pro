"""
Fingerprint Squared CLI

Command-line interface for running bias fingerprinting evaluations.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, List

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
except ImportError:
    print("Please install CLI dependencies: pip install typer rich")
    sys.exit(1)


app = typer.Typer(
    name="fingerprint",
    help="Fingerprint Squared - Ethical AI Bias Assessment for VLMs",
    add_completion=False,
)
console = Console()


@app.command()
def evaluate(
    model: str = typer.Argument(..., help="Model to evaluate (openai:gpt-4o, anthropic:claude-3-sonnet, etc.)"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to dataset directory"),
    output: str = typer.Option("./fingerprint_output", "--output", "-o", help="Output directory"),
    n_images: int = typer.Option(20, "--n-images", "-n", help="Images per demographic group"),
    format: str = typer.Option("auto", "--format", "-f", help="Dataset format (auto, fhibe, utkface, fairface)"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM-as-judge scoring"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Run bias fingerprinting evaluation on a VLM.

    Example:
        fingerprint evaluate openai:gpt-4o -d ./images -n 20
    """
    console.print(Panel.fit(
        "[bold blue]Fingerprint Squared[/bold blue]\n"
        "Ethical AI Bias Assessment Framework",
        border_style="blue",
    ))

    # Parse model
    if ":" in model:
        provider, model_name = model.split(":", 1)
    else:
        provider = "openai"
        model_name = model

    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Model: {provider}:{model_name}")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Output: {output}")
    console.print(f"  Images per group: {n_images}")

    # Import here to avoid slow startup
    from fingerprint_squared.data.fhibe_loader import FHIBELoader
    from fingerprint_squared.core.fingerprint_pipeline import FingerprintPipeline, PipelineConfig

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load dataset
        task = progress.add_task("Loading dataset...", total=None)
        loader = FHIBELoader()

        try:
            ds = loader.load_from_directory(dataset, format=format)
            progress.update(task, description=f"[green]Loaded {len(ds)} images")
        except Exception as e:
            console.print(f"[red]Error loading dataset: {e}")
            raise typer.Exit(1)

        # Initialize VLM
        progress.update(task, description="Initializing VLM...")
        vlm = _create_vlm(provider, model_name)

        if vlm is None:
            console.print(f"[red]Failed to initialize VLM: {provider}:{model_name}")
            raise typer.Exit(1)

        # Run pipeline
        progress.update(task, description="Running evaluation...")

        config = PipelineConfig(
            n_images_per_group=n_images,
            output_dir=output,
            use_llm_judge=not no_judge,
            verbose=verbose,
        )

        pipeline = FingerprintPipeline(config=config)

        try:
            results = asyncio.run(pipeline.run(
                vlm=vlm,
                dataset=ds,
                model_id=f"{provider}_{model_name}",
                model_name=model_name,
            ))
        except Exception as e:
            console.print(f"[red]Evaluation failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            raise typer.Exit(1)

    # Save results
    paths = results.save(output)

    # Display results
    console.print("\n[bold green]Evaluation Complete![/bold green]\n")

    _display_fingerprint(results.fingerprint)

    console.print(f"\n[bold]Output Files:[/bold]")
    for name, path in paths.items():
        console.print(f"  {name}: {path}")


@app.command()
def compare(
    models: List[str] = typer.Argument(..., help="Models to compare"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to dataset directory"),
    output: str = typer.Option("./comparison_output", "--output", "-o", help="Output directory"),
    n_images: int = typer.Option(10, "--n-images", "-n", help="Images per demographic group"),
):
    """
    Compare bias fingerprints across multiple models.

    Example:
        fingerprint compare openai:gpt-4o anthropic:claude-3-sonnet -d ./images
    """
    console.print(Panel.fit(
        "[bold blue]Model Comparison[/bold blue]\n"
        f"Comparing {len(models)} models",
        border_style="blue",
    ))

    from fingerprint_squared.data.fhibe_loader import FHIBELoader
    from fingerprint_squared.core.fingerprint_pipeline import MultiModelPipeline, PipelineConfig

    # Load dataset
    loader = FHIBELoader()
    ds = loader.load_from_directory(dataset)

    # Initialize VLMs
    vlms = []
    model_ids = []
    model_names = []

    for model_spec in models:
        if ":" in model_spec:
            provider, model_name = model_spec.split(":", 1)
        else:
            provider = "openai"
            model_name = model_spec

        vlm = _create_vlm(provider, model_name)
        if vlm:
            vlms.append(vlm)
            model_ids.append(f"{provider}_{model_name}")
            model_names.append(model_name)
        else:
            console.print(f"[yellow]Warning: Could not initialize {model_spec}")

    if not vlms:
        console.print("[red]No valid models to compare")
        raise typer.Exit(1)

    # Run comparison
    config = PipelineConfig(
        n_images_per_group=n_images,
        output_dir=output,
    )

    pipeline = MultiModelPipeline(config=config)

    results = asyncio.run(pipeline.run_comparison(
        models=vlms,
        model_ids=model_ids,
        model_names=model_names,
        dataset=ds,
    ))

    # Display comparison
    console.print("\n[bold green]Comparison Complete![/bold green]\n")

    table = Table(title="Model Rankings (Lower Bias = Better)")
    table.add_column("Rank", style="bold")
    table.add_column("Model")
    table.add_column("Overall Bias", justify="right")
    table.add_column("Valence", justify="right")
    table.add_column("Stereotype", justify="right")

    sorted_fps = sorted(
        results["fingerprints"].values(),
        key=lambda x: x.overall_bias_score
    )

    for i, fp in enumerate(sorted_fps):
        table.add_row(
            f"#{i+1}",
            fp.model_name,
            f"{fp.overall_bias_score*100:.1f}%",
            f"{fp.valence_bias*100:.1f}%",
            f"{fp.stereotype_bias*100:.1f}%",
        )

    console.print(table)


@app.command()
def passport(
    fingerprint_path: str = typer.Argument(..., help="Path to fingerprint JSON"),
    output: str = typer.Option(None, "--output", "-o", help="Output PDF path"),
):
    """
    Generate a Bias Passport PDF from a fingerprint.

    Example:
        fingerprint passport ./output/gpt4o_fingerprint.json
    """
    from fingerprint_squared.core.bias_fingerprint import BiasFingerprint
    from fingerprint_squared.reporting.pdf_generator import generate_passport

    # Load fingerprint
    fp = BiasFingerprint.load(fingerprint_path)

    # Generate PDF
    if output is None:
        output = fingerprint_path.replace(".json", "_passport.pdf")

    generate_passport(fp, output)
    console.print(f"[green]Generated passport: {output}")


@app.command()
def dashboard(
    data_dir: str = typer.Argument("./fingerprint_output", help="Directory with fingerprint data"),
    port: int = typer.Option(3000, "--port", "-p", help="Port for dashboard"),
):
    """
    Launch the interactive Bias Observatory dashboard.

    Example:
        fingerprint dashboard ./output --port 3000
    """
    console.print(f"[bold blue]Launching Bias Observatory on port {port}...[/bold blue]")
    console.print(f"Data directory: {data_dir}")
    console.print(f"\nOpen http://localhost:{port} in your browser")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    # Check if npm is available and dashboard exists
    dashboard_path = Path(__file__).parent.parent.parent / "dashboard"

    if dashboard_path.exists():
        os.system(f"cd {dashboard_path} && npm run dev -- --port {port}")
    else:
        console.print("[yellow]Dashboard not found. Please run from project root.[/yellow]")


@app.command()
def list_models():
    """List available VLM providers and models."""
    # OpenRouter models (proprietary via unified API)
    console.print("\n[bold cyan]OpenRouter (Proprietary - OPENROUTER_API_KEY)[/bold cyan]")
    table = Table()
    table.add_column("Shorthand", style="bold")
    table.add_column("Full Model ID")
    table.add_column("Provider")

    openrouter_models = [
        ("gpt-4o", "openai/gpt-4o", "OpenAI"),
        ("gpt-4o-mini", "openai/gpt-4o-mini", "OpenAI"),
        ("claude-3.5-sonnet", "anthropic/claude-3.5-sonnet", "Anthropic"),
        ("claude-3-opus", "anthropic/claude-3-opus-20240229", "Anthropic"),
        ("gemini-2.0-flash", "google/gemini-2.0-flash-001", "Google"),
        ("gemini-1.5-pro", "google/gemini-pro-1.5", "Google"),
        ("llama-3.2-90b-vision", "meta-llama/llama-3.2-90b-vision-instruct", "Meta"),
    ]

    for short, full, provider in openrouter_models:
        table.add_row(short, full, provider)

    console.print(table)

    # Open Source SOTA (local HuggingFace)
    console.print("\n[bold green]Open Source SOTA (Local GPU)[/bold green]")
    table2 = Table()
    table2.add_column("Provider", style="bold")
    table2.add_column("Model ID")
    table2.add_column("Params")
    table2.add_column("Notes")

    oss_models = [
        ("qwen", "Qwen/Qwen2.5-VL-7B-Instruct", "7B", "Current OSS SOTA, multilingual"),
        ("qwen3", "Qwen/Qwen3-VL-8B-Instruct", "8B", "Reasoning mode (thinking tokens)"),
        ("internvl", "OpenGVLab/InternVL3-8B", "8B", "Different training lineage"),
        ("llama", "meta-llama/Llama-3.2-11B-Vision-Instruct", "11B", "Western-trained (needs HF token)"),
        ("smol", "HuggingFaceTB/SmolVLM-Instruct", "2B", "Tiny, fast, Apache 2.0"),
    ]

    for provider, model, params, notes in oss_models:
        table2.add_row(provider, model, params, notes)

    console.print(table2)

    # Usage examples
    console.print("\n[bold]Usage Examples:[/bold]")
    console.print("  [dim]# OpenRouter (proprietary)[/dim]")
    console.print("  fingerprint evaluate openrouter:gpt-4o -d ./images")
    console.print("  fingerprint evaluate openrouter:claude-3.5-sonnet -d ./images")
    console.print("\n  [dim]# Open Source (local)[/dim]")
    console.print("  fingerprint evaluate qwen:Qwen2.5-VL-7B-Instruct -d ./images")
    console.print("  fingerprint evaluate qwen3:Qwen3-VL-8B-Instruct -d ./images  # with thinking mode")
    console.print("  fingerprint evaluate internvl:InternVL3-8B -d ./images")


@app.command()
def synthetic(
    output: str = typer.Option("./synthetic_dataset", "--output", "-o", help="Output directory"),
    n_per_group: int = typer.Option(5, "--n", help="Images per demographic intersection"),
):
    """
    Create a synthetic dataset for testing (no real images required).

    Example:
        fingerprint synthetic -o ./test_data -n 3
    """
    from fingerprint_squared.data.fhibe_loader import FHIBELoader

    loader = FHIBELoader()
    ds = loader.create_synthetic_dataset(n_per_intersection=n_per_group)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    ds.save(str(output_path / "synthetic_dataset.json"))

    console.print(f"[green]Created synthetic dataset with {len(ds)} entries")
    console.print(f"Saved to: {output_path / 'synthetic_dataset.json'}")


def _create_vlm(provider: str, model_name: str):
    """Create a VLM instance based on provider."""
    try:
        # Use MultiProviderVLM for unified creation
        from fingerprint_squared.models.openrouter_vlm import MultiProviderVLM

        model_spec = f"{provider}:{model_name}"
        return MultiProviderVLM.create(model_spec)

    except ValueError as e:
        # Fallback to legacy providers
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
                console.print(f"[red]Unknown provider: {provider}")
                console.print("[dim]Available: openrouter, qwen, qwen3, internvl, llama, smol, openai, anthropic, google[/dim]")
                return None
        except Exception as inner_e:
            console.print(f"[red]Error creating VLM: {inner_e}")
            return None
    except Exception as e:
        console.print(f"[red]Error creating VLM: {e}")
        return None


def _display_fingerprint(fp):
    """Display fingerprint summary."""
    # Grade
    if fp.overall_bias_score < 0.2:
        grade, color = "A", "green"
    elif fp.overall_bias_score < 0.35:
        grade, color = "B", "blue"
    elif fp.overall_bias_score < 0.5:
        grade, color = "C", "yellow"
    elif fp.overall_bias_score < 0.65:
        grade, color = "D", "orange"
    else:
        grade, color = "F", "red"

    console.print(Panel(
        f"[bold {color}]Grade: {grade}[/bold {color}]\n"
        f"Overall Bias: {fp.overall_bias_score*100:.1f}%",
        title=fp.model_name,
        border_style=color,
    ))

    # Details table
    table = Table()
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")

    table.add_row("Valence Bias", f"{fp.valence_bias*100:.1f}%")
    table.add_row("Stereotype Bias", f"{fp.stereotype_bias*100:.1f}%")
    table.add_row("Confidence Bias", f"{fp.confidence_bias*100:.1f}%")
    table.add_row("Refusal Rate", f"{fp.refusal_rate*100:.1f}%")
    table.add_row("Total Probes", str(fp.total_probes))

    console.print(table)

    # Radar dimensions
    if fp.radar_dimensions:
        console.print("\n[bold]Probe Scores:[/bold]")
        for probe, score in sorted(fp.radar_dimensions.items(), key=lambda x: -x[1]):
            bar = "=" * int(score * 20)
            console.print(f"  {probe:20} [{bar:20}] {score*100:.0f}%")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
