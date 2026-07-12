#!/usr/bin/env python3
"""
Master script to generate ALL AAAI publication figures.
Creates comprehensive visualizations for paper submission.
"""

import subprocess
import sys
from pathlib import Path
import shutil


def run_script(script_name, args, description):
    """Run a script and report status."""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)

    cmd = [sys.executable, script_name] + args
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"✓ {description} - COMPLETE")
        return True
    else:
        print(f"✗ {description} - FAILED")
        return False


def organize_figures(base_output_dir: Path):
    """Organize all figures into paper-ready structure."""
    print("\n" + "="*70)
    print("Organizing Figures for AAAI Paper")
    print("="*70)

    # Create organized structure
    paper_figs = base_output_dir / "paper_figures"
    paper_figs.mkdir(exist_ok=True)

    # Create subdirectories
    (paper_figs / "main").mkdir(exist_ok=True)
    (paper_figs / "supplementary").mkdir(exist_ok=True)
    (paper_figs / "tables").mkdir(exist_ok=True)

    return paper_figs


def main():
    print("="*70)
    print("AAAI Publication Figures - Master Generator")
    print("="*70)

    # Paths
    scripts_dir = Path(__file__).parent
    results_dir = Path("results/single_runs_35k")
    base_output = Path("results/aaai_submission")

    success_count = 0
    total_count = 0

    # 1. Model Leaderboard (Table 2)
    total_count += 1
    if run_script(
        str(scripts_dir / "generate_model_leaderboard.py"),
        ["--results-dir", str(results_dir), "--output-dir", str(base_output / "leaderboard")],
        "Model Leaderboard (Table 2)"
    ):
        success_count += 1

    # 2. Statistical Analysis
    total_count += 1
    if run_script(
        str(scripts_dir / "add_statistical_rigor.py"),
        ["--results-dir", str(results_dir), "--output", str(base_output / "statistical_analysis.json")],
        "Statistical Analysis (ANOVA, Effect Sizes)"
    ):
        success_count += 1

    # 3. Regional Embedding Analysis
    total_count += 1
    if run_script(
        str(scripts_dir / "regional_embedding_analysis.py"),
        ["--results-dir", str(results_dir), "--output-dir", str(base_output / "regional_embeddings")],
        "Regional Embedding Analysis (PCA, t-SNE, Similarity)"
    ):
        success_count += 1

    # 4. Enhanced Visualizations (UMAP, Trajectory, etc.)
    total_count += 1
    if run_script(
        str(scripts_dir / "enhanced_visualizations.py"),
        ["--results-dir", str(results_dir), "--output-dir", str(base_output / "enhanced_viz"), "--n-samples", "6000"],
        "Enhanced Visualizations (UMAP, Trajectory, Radar)"
    ):
        success_count += 1

    # 5. Per-Model Analysis (Stratified Balanced)
    total_count += 1
    if run_script(
        str(scripts_dir / "comprehensive_per_model_analysis.py"),
        ["--results-dir", str(results_dir), "--output-dir", str(base_output / "per_model_analysis"), "--n-per-group", "1000"],
        "Per-Model Analysis (Stratified Balanced Sampling)"
    ):
        success_count += 1

    # 6. Validation Sampling
    total_count += 1
    if run_script(
        str(scripts_dir / "sample_for_validation.py"),
        ["--results-dir", str(results_dir), "--output", str(base_output / "validation_sample.csv"), "--n-samples", "486"],
        "MTurk Validation Sampling"
    ):
        success_count += 1

    # 7. Qualitative Examples
    total_count += 1
    if run_script(
        str(scripts_dir / "extract_qualitative_examples.py"),
        ["--results-dir", str(results_dir), "--output-dir", str(base_output)],
        "Qualitative Examples Extraction"
    ):
        success_count += 1

    # 8. Publication-Style Figures (Original)
    total_count += 1
    if run_script(
        str(scripts_dir / "generate_all_publication_figures.py"),
        ["--results-dir", str(results_dir), "--output-dir", str(base_output / "figures")],
        "Publication Figures (Original 4 Figures)"
    ):
        success_count += 1

    # Organize figures
    organize_figures(base_output)

    # Summary
    print("\n" + "="*70)
    print("AAAI Figure Generation - COMPLETE")
    print("="*70)
    print(f"\nSuccess: {success_count}/{total_count} scripts")
    print(f"\nAll outputs in: {base_output}")

    # Create index
    create_figure_index(base_output)

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Review all figures:")
    print(f"   open {base_output}")
    print("\n2. Check the figure index:")
    print(f"   cat {base_output}/FIGURE_INDEX.md")
    print("\n3. Copy LaTeX table:")
    print(f"   cat {base_output}/leaderboard/model_leaderboard_table.tex")
    print("\n4. Add to paper following the guide in AAAI_FIGURE_USAGE_GUIDE.md")
    print("="*70)


def create_figure_index(output_dir: Path):
    """Create index of all generated figures."""

    index = []
    index.append("# AAAI Publication Figures - Complete Index")
    index.append(f"\nGenerated: {Path.cwd()}")
    index.append("\n" + "="*70)

    # Main Figures
    index.append("\n## Main Paper Figures")
    index.append("\n### Original 4 Figures")
    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        for fig in sorted(figures_dir.glob("*.pdf")):
            size_kb = fig.stat().st_size / 1024
            index.append(f"- [{fig.name}]({fig.relative_to(output_dir)}) ({size_kb:.1f} KB)")

    # Enhanced Visualizations
    index.append("\n### Enhanced Visualizations (5 figures)")
    enhanced_dir = output_dir / "enhanced_viz"
    if enhanced_dir.exists():
        for fig in sorted(enhanced_dir.glob("*.pdf")):
            size_kb = fig.stat().st_size / 1024
            index.append(f"- [{fig.name}]({fig.relative_to(output_dir)}) ({size_kb:.1f} KB)")

    # Regional Embeddings
    index.append("\n### Regional Embeddings (4 figures)")
    embed_dir = output_dir / "regional_embeddings"
    if embed_dir.exists():
        for fig in sorted(embed_dir.glob("*.pdf")):
            size_kb = fig.stat().st_size / 1024
            index.append(f"- [{fig.name}]({fig.relative_to(output_dir)}) ({size_kb:.1f} KB)")

    # Per-Model Analysis
    index.append("\n### Per-Model Analysis (12 figures)")
    per_model_dir = output_dir / "per_model_analysis"
    if per_model_dir.exists():
        index.append("\n#### IDEFICS2-8B (4 figures)")
        for fig in sorted(per_model_dir.glob("HuggingFaceM4_idefics2_8b*.pdf")):
            size_kb = fig.stat().st_size / 1024
            index.append(f"- [{fig.name}]({fig.relative_to(output_dir)}) ({size_kb:.1f} KB)")

        index.append("\n#### InternVL2-2B (4 figures)")
        for fig in sorted(per_model_dir.glob("OpenGVLab_InternVL2_2B*.pdf")):
            size_kb = fig.stat().st_size / 1024
            index.append(f"- [{fig.name}]({fig.relative_to(output_dir)}) ({size_kb:.1f} KB)")

        index.append("\n#### LLaVA-1.6-7B (4 figures)")
        for fig in sorted(per_model_dir.glob("llava*.pdf")):
            size_kb = fig.stat().st_size / 1024
            index.append(f"- [{fig.name}]({fig.relative_to(output_dir)}) ({size_kb:.1f} KB)")

    # Tables
    index.append("\n## Tables")
    index.append("\n### Model Leaderboard (Table 2)")
    leaderboard_dir = output_dir / "leaderboard"
    if leaderboard_dir.exists():
        for file in sorted(leaderboard_dir.glob("*.tex")):
            size_kb = file.stat().st_size / 1024
            index.append(f"- [{file.name}]({file.relative_to(output_dir)}) ({size_kb:.1f} KB)")

    # Statistics
    index.append("\n## Statistics Files")
    stat_files = [
        "statistical_analysis.json",
        "statistical_analysis_summary.txt",
        "validation_sample.csv",
        "qualitative_examples.json"
    ]
    for stat_file in stat_files:
        path = output_dir / stat_file
        if path.exists():
            size_kb = path.stat().st_size / 1024
            index.append(f"- [{stat_file}]({stat_file}) ({size_kb:.1f} KB)")

    # Summary
    index.append("\n" + "="*70)
    index.append("\n## Summary")

    # Count PDFs
    total_pdfs = len(list(output_dir.rglob("*.pdf")))
    index.append(f"\nTotal PDF figures: {total_pdfs}")

    # Total size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*.pdf")) / (1024 * 1024)
    index.append(f"Total size (PDFs): {total_size:.1f} MB")

    index.append("\n" + "="*70)

    # Write index
    index_path = output_dir / "FIGURE_INDEX.md"
    with open(index_path, 'w') as f:
        f.write("\n".join(index))

    print(f"\n✓ Created figure index: {index_path}")


if __name__ == '__main__':
    main()
