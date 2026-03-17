"""
Command-line interface for Fingerprint².

Provides CLI commands for evaluating VLMs, comparing models,
and generating reports.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from fingerprint_squared import FingerprintSquared
from fingerprint_squared.core.evaluator import EvaluationConfig
from fingerprint_squared.models.registry import list_models

app = typer.Typer(
    name="fingerprint-squared",
    help="Fingerprint²: Ethical AI Assessment Framework for Vision-Language Models",
    add_completion=False,
)

console = Console()


@app.command()
def evaluate(
    model: str = typer.Argument(..., help="Model to evaluate (e.g., gpt-4o, claude-3-opus)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key for the model"),
    output_dir: Path = typer.Option("./fp2_results", "--output", "-o", help="Output directory"),
    probe_types: Optional[List[str]] = typer.Option(
        None, "--probe", "-p", help="Specific probe types to run"
    ),
    dimensions: Optional[List[str]] = typer.Option(
        None, "--dimension", "-d", help="Demographic dimensions to analyze"
    ),
    n_probes: int = typer.Option(50, "--n-probes", "-n", help="Number of probes per type"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip report generation"),
    format: str = typer.Option("html", "--format", "-f", help="Report format (html/markdown)"),
):
    """
    Evaluate a Vision-Language Model for bias and fairness.

    Example:
        fp2 evaluate gpt-4o --api-key $OPENAI_API_KEY
        fp2 evaluate claude-3-opus -d gender -d race -n 100
    """
    console.print(Panel.fit(
        f"[bold blue]Fingerprint²[/bold blue]\n"
        f"Evaluating: [green]{model}[/green]",
        title="🔍 Evaluation Starting",
    ))

    # Build config
    config = EvaluationConfig(
        probe_types=probe_types or ["stereotype_association", "counterfactual", "representation"],
        demographic_dimensions=dimensions or ["gender", "race", "age"],
        n_probes_per_type=n_probes,
    )

    # Initialize framework
    fp2 = FingerprintSquared(config=config, output_dir=output_dir)

    # Run evaluation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running evaluation...", total=None)

        try:
            result = fp2.evaluate_sync(
                model,
                api_key=api_key,
                generate_report=not no_report,
            )
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    # Display results
    _display_results(result, fp2._fingerprints.get(f"{result.model_name}_{result.timestamp}"))

    console.print(f"\n[green]✓[/green] Results saved to: {output_dir}")


@app.command()
def compare(
    models: List[str] = typer.Argument(..., help="Models to compare"),
    output_dir: Path = typer.Option("./fp2_results", "--output", "-o", help="Output directory"),
):
    """
    Compare multiple VLMs.

    Example:
        fp2 compare gpt-4o claude-3-opus gemini-1.5-pro
    """
    console.print(Panel.fit(
        f"[bold blue]Fingerprint²[/bold blue]\n"
        f"Comparing: [green]{', '.join(models)}[/green]",
        title="⚖️ Model Comparison",
    ))

    fp2 = FingerprintSquared(output_dir=output_dir)

    # Load existing results or evaluate
    for model in models:
        existing = fp2.load_results(model)
        if existing is None:
            console.print(f"[yellow]Warning: No existing results for {model}. Run evaluation first.[/yellow]")

    # Display comparison
    try:
        rankings = fp2.rank_models("overall_bias")
        _display_rankings(rankings, "Bias Score (lower is better)")
    except Exception as e:
        console.print(f"[yellow]Could not generate rankings: {e}[/yellow]")


@app.command("list-models")
def list_available_models():
    """List all supported VLM models."""
    models = list_models()

    table = Table(title="Supported Models")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="green")

    providers = {
        "gpt": "OpenAI",
        "claude": "Anthropic",
        "gemini": "Google",
        "llava": "HuggingFace",
        "blip": "HuggingFace",
    }

    for model in sorted(models):
        provider = "Unknown"
        for key, prov in providers.items():
            if key in model.lower():
                provider = prov
                break
        table.add_row(model, provider)

    console.print(table)


@app.command()
def report(
    model: str = typer.Argument(..., help="Model name"),
    timestamp: Optional[str] = typer.Option(None, "--timestamp", "-t", help="Specific timestamp"),
    output_dir: Path = typer.Option("./fp2_results", "--output", "-o", help="Output directory"),
    format: str = typer.Option("html", "--format", "-f", help="Report format"),
):
    """
    Generate a report from existing evaluation results.

    Example:
        fp2 report gpt-4o --format html
    """
    fp2 = FingerprintSquared(output_dir=output_dir)

    result = fp2.load_results(model, timestamp)
    if result is None:
        console.print(f"[red]No results found for {model}[/red]")
        raise typer.Exit(1)

    # For now, just indicate that results were found
    console.print(f"[green]Results found for {model}[/green]")
    console.print(f"Report generation requires full EvaluationResult object.")


@app.command()
def visualize(
    model: str = typer.Argument(..., help="Model to visualize"),
    output_dir: Path = typer.Option("./fp2_results", "--output", "-o", help="Output directory"),
    output_file: Optional[Path] = typer.Option(None, "--file", "-f", help="Output file path"),
):
    """
    Generate visualizations for a model's evaluation.

    Example:
        fp2 visualize gpt-4o -f ./charts/gpt4o.png
    """
    fp2 = FingerprintSquared(output_dir=output_dir)

    try:
        fig = fp2.visualize_fingerprint(model, output_path=output_file)
        output = output_file or (output_dir / f"{model}_fingerprint.png")
        console.print(f"[green]✓[/green] Visualization saved to: {output}")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Run evaluation first: fp2 evaluate {model}")
        raise typer.Exit(1)


@app.command()
def summary(
    output_dir: Path = typer.Option("./fp2_results", "--output", "-o", help="Output directory"),
):
    """
    Display summary of all evaluations.

    Example:
        fp2 summary
    """
    fp2 = FingerprintSquared(output_dir=output_dir)

    # Load any existing fingerprints
    fp_dir = output_dir / "fingerprints"
    if fp_dir.exists():
        from fingerprint_squared.utils.io import load_json
        from fingerprint_squared.core.fingerprint import ModelFingerprint

        for fp_file in fp_dir.glob("*.json"):
            try:
                data = load_json(fp_file)
                fp = ModelFingerprint.from_dict(data)
                fp2.comparator.add_fingerprint(fp)
                fp2._fingerprints[f"{fp.model_name}_{fp.timestamp}"] = fp
            except Exception:
                pass

    summary_data = fp2.get_summary()

    console.print(Panel.fit(
        f"[bold blue]Fingerprint² Summary[/bold blue]\n\n"
        f"Total Evaluations: [green]{summary_data['total_evaluations']}[/green]\n"
        f"Models Evaluated: [cyan]{', '.join(summary_data['models_evaluated']) or 'None'}[/cyan]",
        title="📊 Summary",
    ))

    if summary_data['rankings']['by_bias']:
        _display_rankings(summary_data['rankings']['by_bias'], "Top Models by Bias (lower is better)")


def _display_results(result, fingerprint):
    """Display evaluation results in a nice format."""
    # Summary table
    table = Table(title=f"Evaluation Results: {result.model_name}")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")

    # Overall scores
    bias_status = "🟢 Low" if result.overall_bias_score < 0.3 else "🟡 Medium" if result.overall_bias_score < 0.6 else "🔴 High"
    fairness_status = "🟢 Good" if result.overall_fairness_score > 0.7 else "🟡 Fair" if result.overall_fairness_score > 0.4 else "🔴 Poor"

    table.add_row("Overall Bias Score", f"{result.overall_bias_score:.3f}", bias_status)
    table.add_row("Overall Fairness Score", f"{result.overall_fairness_score:.3f}", fairness_status)
    table.add_row("Total Probes", str(result.total_probes), "-")
    table.add_row("Valid Responses", str(result.total_responses), "-")

    console.print(table)

    # Fingerprint info
    if fingerprint:
        console.print(f"\n[bold]Fingerprint Hash:[/bold] [cyan]{fingerprint.fingerprint_hash}[/cyan]")
        console.print(f"[bold]Bias Level:[/bold] {fingerprint.bias_level}")
        console.print(f"[bold]Fairness Level:[/bold] {fingerprint.fairness_level}")

        if fingerprint.risk_areas:
            console.print(f"\n[bold red]Risk Areas:[/bold red]")
            for risk in fingerprint.risk_areas[:3]:
                console.print(f"  ⚠️  {risk}")

        if fingerprint.strengths:
            console.print(f"\n[bold green]Strengths:[/bold green]")
            for strength in fingerprint.strengths[:3]:
                console.print(f"  ✓  {strength}")


def _display_rankings(rankings, title):
    """Display model rankings."""
    table = Table(title=title)

    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Model", style="green")
    table.add_column("Score", style="yellow")

    for i, (model, score) in enumerate(rankings, 1):
        table.add_row(f"#{i}", model, f"{score:.3f}")

    console.print(table)


@app.callback()
def main():
    """
    Fingerprint²: Ethical AI Assessment Framework for Vision-Language Models

    A comprehensive framework for evaluating bias and fairness in VLMs.
    """
    pass


if __name__ == "__main__":
    app()
