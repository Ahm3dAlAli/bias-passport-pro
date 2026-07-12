#!/usr/bin/env python3
"""
Master script to run all AAAI-required analyses for FingerPrint.
This script runs everything needed for publication in one go.

Usage:
    python3 scripts/run_all_aaai_analyses.py --results-dir results/single_runs_35k --output-dir results/aaai_submission

What it does:
    1. Statistical rigor analysis (p-values, effect sizes, CIs)
    2. Prompt sensitivity analysis
    3. Sample data for human validation
    4. Generate all publication figures
    5. Create summary report

Time: ~15-30 minutes
Output: Complete analysis package ready for paper
"""

import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime
import json


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"▶ {description}...")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Error running {description}")
        print(f"STDERR: {result.stderr}")
        return False

    print(result.stdout)
    print(f"✓ {description} complete\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run all AAAI-required analyses for FingerPrint'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory with .db result files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for all analyses')
    parser.add_argument('--skip-sensitivity', action='store_true',
                       help='Skip prompt sensitivity analysis (saves time)')
    parser.add_argument('--validation-samples', type=int, default=200,
                       help='Number of samples for human validation (default: 200)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    log_file = output_dir / f'analysis_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

    print_section("FingerPrint AAAI Submission Analysis Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_file}\n")

    results = {
        'timestamp': datetime.now().isoformat(),
        'results_dir': str(results_dir),
        'output_dir': str(output_dir),
        'steps_completed': []
    }

    # Step 1: Statistical Rigor Analysis
    print_section("STEP 1: Statistical Rigor Analysis")
    print("Computing p-values, effect sizes, confidence intervals, power analysis...")

    stats_cmd = [
        sys.executable,
        'scripts/add_statistical_rigor.py',
        '--results-dir', str(results_dir),
        '--output', str(output_dir / 'statistical_analysis.json')
    ]

    if run_command(stats_cmd, "Statistical rigor analysis"):
        results['steps_completed'].append('statistical_analysis')
        print("📊 Statistical analysis complete!")
        print(f"   Results: {output_dir / 'statistical_analysis.json'}")
        print(f"   Summary: {output_dir / 'statistical_analysis_summary.txt'}")

    # Step 2: Prompt Sensitivity Analysis
    if not args.skip_sensitivity:
        print_section("STEP 2: Prompt Sensitivity Analysis")
        print("Testing robustness across prompt formulations...")

        # Find first database file
        db_files = list(results_dir.glob("*.db"))
        if db_files:
            sensitivity_cmd = [
                sys.executable,
                'scripts/prompt_sensitivity_analysis.py',
                '--results-db', str(db_files[0]),
                '--sample-size', '1000',
                '--output-dir', str(output_dir / 'sensitivity')
            ]

            if run_command(sensitivity_cmd, "Prompt sensitivity analysis"):
                results['steps_completed'].append('sensitivity_analysis')
                print("🔍 Sensitivity analysis complete!")
                print(f"   Results: {output_dir / 'sensitivity' / 'prompt_sensitivity_analysis.json'}")
    else:
        print_section("STEP 2: Prompt Sensitivity Analysis [SKIPPED]")

    # Step 3: Sample for Human Validation
    print_section("STEP 3: Human Validation Sample")
    print(f"Sampling {args.validation_samples} responses for MTurk validation...")

    sample_cmd = [
        sys.executable,
        'scripts/sample_for_validation.py',
        '--results-dir', str(results_dir),
        '--n-samples', str(args.validation_samples),
        '--output', str(output_dir / 'validation_sample.csv')
    ]

    if run_command(sample_cmd, "Human validation sampling"):
        results['steps_completed'].append('validation_sampling')
        print("📝 Validation sample created!")
        print(f"   File: {output_dir / 'validation_sample.csv'}")
        print(f"   Next: Upload to MTurk (~${args.validation_samples * 3 * 0.15:.2f})")

    # Step 4: Generate Publication Figures
    print_section("STEP 4: Generate Publication Figures")
    print("Creating all AAAI-quality figures...")

    figures_cmd = [
        sys.executable,
        'scripts/generate_all_publication_figures.py',
        '--results-dir', str(results_dir),
        '--output-dir', str(output_dir / 'figures'),
        '--stats-file', str(output_dir / 'statistical_analysis.json')
    ]

    if run_command(figures_cmd, "Publication figure generation"):
        results['steps_completed'].append('figure_generation')
        print("📈 All figures generated!")
        print(f"   Directory: {output_dir / 'figures'}")

    # Step 5: Extract Qualitative Examples
    print_section("STEP 5: Extract Qualitative Examples")
    print("Finding high/low bias examples for paper...")

    examples_cmd = [
        sys.executable,
        'scripts/extract_qualitative_examples.py',
        '--results-dir', str(results_dir),
        '--n-examples', '10',
        '--output', str(output_dir / 'qualitative_examples.json')
    ]

    if run_command(examples_cmd, "Qualitative example extraction"):
        results['steps_completed'].append('qualitative_examples')
        print("📖 Examples extracted!")
        print(f"   File: {output_dir / 'qualitative_examples.json'}")

    # Save results summary
    print_section("ANALYSIS COMPLETE")

    results_file = output_dir / 'analysis_summary.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✓ All analyses complete!")
    print(f"✓ {len(results['steps_completed'])} steps completed")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - statistical_analysis.json (p-values, effect sizes, CIs)")
    print(f"  - statistical_analysis_summary.txt (publication-ready text)")
    print(f"  - sensitivity/ (prompt robustness analysis)")
    print(f"  - validation_sample.csv (for MTurk, {args.validation_samples} samples)")
    print(f"  - figures/ (all publication figures)")
    print(f"  - qualitative_examples.json (example responses)")
    print(f"  - analysis_summary.json (this summary)")

    print(f"\n{'='*70}")
    print("NEXT STEPS FOR PAPER:")
    print(f"{'='*70}")
    print("1. Review statistical_analysis_summary.txt for paper text")
    print("2. Upload validation_sample.csv to MTurk (~$90)")
    print("3. Use figures/ for paper (already AAAI-formatted)")
    print("4. Use qualitative_examples.json for Table of example responses")
    print("5. Start writing Methods section (use AAAI_RESEARCH_METHODOLOGY.md)")
    print("\nEstimated time to submission: 3-4 weeks")
    print("See READY_FOR_AAAI_CHECKLIST.md for complete timeline\n")


if __name__ == '__main__':
    main()
