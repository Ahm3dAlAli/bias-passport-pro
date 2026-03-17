#!/usr/bin/env python3
"""
generate_figures.py
===================
Generate publication-quality figures for the Fingerprint² NeurIPS paper.
Uses actual benchmark results from patched_results.json.

Usage:
    python paper/generate_figures.py

Output:
    paper/figures/*.pdf (vector graphics for LaTeX)
    paper/figures/*.png (raster for preview)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette (colorblind-friendly)
COLORS = {
    'paligemma-3b-mix-448': '#2ecc71',      # Green (best)
    'SmolVLM2-2.2B-Instruct': '#3498db',    # Blue
    'Qwen2.5-VL-3B-Instruct': '#9b59b6',    # Purple
    'InternVL2-2B': '#e67e22',              # Orange
    'moondream2': '#e74c3c',                # Red (worst)
}

MODEL_SHORT = {
    'paligemma-3b-mix-448': 'PaLiGemma-3B',
    'SmolVLM2-2.2B-Instruct': 'SmolVLM2-2.2B',
    'Qwen2.5-VL-3B-Instruct': 'Qwen2.5-VL-3B',
    'InternVL2-2B': 'InternVL2-2B',
    'moondream2': 'Moondream2',
}

PROBE_LABELS = {
    'P1_occupation': 'P1: Occupation',
    'P2_education': 'P2: Education',
    'P3_trustworthiness': 'P3: Trust',
    'P4_lifestyle': 'P4: Lifestyle',
    'P5_neighbourhood': 'P5: Neighbourhood',
}

PROBE_SHORT = {
    'P1_occupation': 'Occupation',
    'P2_education': 'Education',
    'P3_trustworthiness': 'Trust',
    'P4_lifestyle': 'Lifestyle',
    'P5_neighbourhood': 'Neighbourhood',
}

REGION_ORDER = ['Africa', 'Asia', 'Europe', 'Americas', 'Northern America', 'Oceania']

# ============================================================================
# Load Data
# ============================================================================

def load_results():
    """Load benchmark results from JSON."""
    results_path = Path(__file__).parent.parent / 'patched_results.json'
    if not results_path.exists():
        results_path = Path('/Users/ahmeda./Desktop/patched_results.json')

    with open(results_path) as f:
        return json.load(f)

# ============================================================================
# Figure 1: Radar Chart - Bias Fingerprints
# ============================================================================

def fig1_radar_fingerprints(results, output_dir):
    """Create radar chart showing bias fingerprints for all models."""

    probes = list(PROBE_LABELS.keys())
    n_probes = len(probes)
    angles = np.linspace(0, 2 * np.pi, n_probes, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for model_name, data in results.items():
        if model_name not in COLORS:
            continue

        values = [data['dimensions'].get(p, {}).get('disparity', 0) for p in probes]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2,
                label=MODEL_SHORT[model_name], color=COLORS[model_name],
                markersize=5)
        ax.fill(angles, values, alpha=0.15, color=COLORS[model_name])

    # Customize radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([PROBE_SHORT[p] for p in probes], size=10)
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5'], size=8, color='gray')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True,
              fancybox=True, shadow=False, framealpha=0.95)

    plt.title('Bias Fingerprints: Disparity by Probe', pad=20, fontweight='bold')

    # Save
    fig.savefig(output_dir / 'fig1_radar_fingerprints.pdf', format='pdf')
    fig.savefig(output_dir / 'fig1_radar_fingerprints.png', format='png', dpi=300)
    plt.close(fig)
    print("✓ Figure 1: Radar fingerprints")

# ============================================================================
# Figure 2: Bar Chart - Composite Scores Leaderboard
# ============================================================================

def fig2_composite_leaderboard(results, output_dir):
    """Create horizontal bar chart showing composite bias scores."""

    # Sort models by composite score
    models = [(name, data['composite_score']) for name, data in results.items()
              if name in COLORS]
    models.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(7, 3.5))

    y_pos = np.arange(len(models))
    scores = [m[1] for m in models]
    names = [MODEL_SHORT[m[0]] for m in models]
    colors = [COLORS[m[0]] for m in models]

    bars = ax.barh(y_pos, scores, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', ha='left', fontsize=9, fontweight='bold')

    # Severity thresholds
    ax.axvline(x=0.40, color='#f39c12', linestyle='--', linewidth=1.5, alpha=0.7, label='Medium threshold')
    ax.axvline(x=0.60, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, label='High threshold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Composite Disparity Score (lower is better)')
    ax.set_xlim(0, 0.45)
    ax.set_title('Model Bias Leaderboard', fontweight='bold', pad=10)

    # Add severity zone labels
    ax.text(0.20, -0.8, 'LOW', ha='center', fontsize=9, color='#27ae60', fontweight='bold')
    ax.text(0.50, -0.8, 'MED', ha='center', fontsize=9, color='#f39c12', fontweight='bold')

    ax.legend(loc='lower right', framealpha=0.95)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_composite_leaderboard.pdf', format='pdf')
    fig.savefig(output_dir / 'fig2_composite_leaderboard.png', format='png', dpi=300)
    plt.close(fig)
    print("✓ Figure 2: Composite leaderboard")

# ============================================================================
# Figure 3: Heatmap - Disparity by Model × Probe
# ============================================================================

def fig3_disparity_heatmap(results, output_dir):
    """Create heatmap showing disparity scores for all model-probe combinations."""

    probes = list(PROBE_LABELS.keys())
    models = [m for m in COLORS.keys() if m in results]

    # Build matrix
    matrix = np.zeros((len(models), len(probes)))
    for i, model in enumerate(models):
        for j, probe in enumerate(probes):
            matrix[i, j] = results[model]['dimensions'].get(probe, {}).get('disparity', 0)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.6)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Disparity Score', rotation=270, labelpad=15)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(probes)):
            val = matrix[i, j]
            color = 'white' if val > 0.35 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')

    # Labels
    ax.set_xticks(np.arange(len(probes)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([PROBE_SHORT[p] for p in probes], rotation=45, ha='right')
    ax.set_yticklabels([MODEL_SHORT[m] for m in models])

    ax.set_title('Disparity Scores: Model × Probe', fontweight='bold', pad=10)

    # Add grid
    ax.set_xticks(np.arange(len(probes)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(models)+1)-.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_disparity_heatmap.pdf', format='pdf')
    fig.savefig(output_dir / 'fig3_disparity_heatmap.png', format='png', dpi=300)
    plt.close(fig)
    print("✓ Figure 3: Disparity heatmap")

# ============================================================================
# Figure 4: Regional Bias - Group Means Comparison
# ============================================================================

def fig4_regional_bias(results, output_dir):
    """Create grouped bar chart showing valence by region for each model."""

    # Focus on P4 Lifestyle (shows clearest pattern)
    probe = 'P4_lifestyle'

    models = [m for m in COLORS.keys() if m in results]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(REGION_ORDER))
    width = 0.15
    offsets = np.linspace(-2*width, 2*width, len(models))

    for i, model in enumerate(models):
        group_means = results[model]['dimensions'].get(probe, {}).get('group_means', {})
        values = [group_means.get(r, 0) for r in REGION_ORDER]

        bars = ax.bar(x + offsets[i], values, width, label=MODEL_SHORT[model],
                      color=COLORS[model], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER, rotation=30, ha='right')
    ax.set_ylabel('Mean Valence Score')
    ax.set_xlabel('Demographic Region')
    ax.set_title(f'Regional Valence Disparity: {PROBE_SHORT[probe]} Probe', fontweight='bold', pad=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.95)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_ylim(-0.1, 0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_regional_bias.pdf', format='pdf')
    fig.savefig(output_dir / 'fig4_regional_bias.png', format='png', dpi=300)
    plt.close(fig)
    print("✓ Figure 4: Regional bias (Lifestyle probe)")

# ============================================================================
# Figure 5: Effect Size Comparison
# ============================================================================

def fig5_effect_sizes(results, output_dir):
    """Create bar chart showing effect sizes (Cohen's d) for significant findings."""

    # Collect effect sizes
    effect_data = []
    for model_name, data in results.items():
        if model_name not in COLORS:
            continue
        for probe_id, probe_data in data['dimensions'].items():
            effect_data.append({
                'model': MODEL_SHORT[model_name],
                'probe': PROBE_SHORT[probe_id],
                'effect_size': probe_data.get('effect_size', 0),
                'significant': probe_data.get('significant', False),
                'color': COLORS[model_name]
            })

    # Sort by effect size
    effect_data.sort(key=lambda x: x['effect_size'], reverse=True)
    top_12 = effect_data[:12]  # Top 12 largest effects

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(top_12))
    values = [d['effect_size'] for d in top_12]
    labels = [f"{d['model']}\n{d['probe']}" for d in top_12]
    colors = [d['color'] for d in top_12]

    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    # Effect size thresholds
    ax.axvline(x=0.2, color='#3498db', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0.5, color='#f39c12', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0.8, color='#e74c3c', linestyle=':', linewidth=1.5, alpha=0.7)

    # Add threshold labels
    ax.text(0.2, len(top_12) + 0.3, 'Small', ha='center', fontsize=8, color='#3498db')
    ax.text(0.5, len(top_12) + 0.3, 'Medium', ha='center', fontsize=8, color='#f39c12')
    ax.text(0.8, len(top_12) + 0.3, 'Large', ha='center', fontsize=8, color='#e74c3c')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', ha='left', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cohen's d Effect Size")
    ax.set_title('Largest Effect Sizes Across Model-Probe Combinations', fontweight='bold', pad=15)
    ax.set_xlim(0, 1.6)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_effect_sizes.pdf', format='pdf')
    fig.savefig(output_dir / 'fig5_effect_sizes.png', format='png', dpi=300)
    plt.close(fig)
    print("✓ Figure 5: Effect sizes")

# ============================================================================
# Figure 6: Worst/Best Group Analysis
# ============================================================================

def fig6_worst_best_groups(results, output_dir):
    """Create visualization showing which regions are worst/best across models."""

    probes = list(PROBE_LABELS.keys())
    models = [m for m in COLORS.keys() if m in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Count worst groups
    worst_counts = {r: 0 for r in REGION_ORDER}
    best_counts = {r: 0 for r in REGION_ORDER}

    for model in models:
        for probe in probes:
            probe_data = results[model]['dimensions'].get(probe, {})
            worst = probe_data.get('worst_group', '')
            best = probe_data.get('best_group', '')
            if worst in worst_counts:
                worst_counts[worst] += 1
            if best in best_counts:
                best_counts[best] += 1

    # Worst groups chart
    regions = list(worst_counts.keys())
    worst_vals = [worst_counts[r] for r in regions]
    best_vals = [best_counts[r] for r in regions]

    colors_worst = ['#e74c3c' if v == max(worst_vals) else '#f5b7b1' for v in worst_vals]
    colors_best = ['#27ae60' if v == max(best_vals) else '#abebc6' for v in best_vals]

    ax1.barh(regions, worst_vals, color=colors_worst, edgecolor='white')
    ax1.set_xlabel('Count (across 5 models × 5 probes)')
    ax1.set_title('Worst Group Frequency\n(Lowest Valence)', fontweight='bold')
    ax1.set_xlim(0, 15)
    for i, v in enumerate(worst_vals):
        ax1.text(v + 0.2, i, str(v), va='center', fontsize=10, fontweight='bold')

    ax2.barh(regions, best_vals, color=colors_best, edgecolor='white')
    ax2.set_xlabel('Count (across 5 models × 5 probes)')
    ax2.set_title('Best Group Frequency\n(Highest Valence)', fontweight='bold')
    ax2.set_xlim(0, 15)
    for i, v in enumerate(best_vals):
        ax2.text(v + 0.2, i, str(v), va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_worst_best_groups.pdf', format='pdf')
    fig.savefig(output_dir / 'fig6_worst_best_groups.png', format='png', dpi=300)
    plt.close(fig)
    print("✓ Figure 6: Worst/best groups")

# ============================================================================
# Figure 7: Probe-wise Disparity Comparison
# ============================================================================

def fig7_probe_comparison(results, output_dir):
    """Create grouped bar chart comparing all models across probes."""

    probes = list(PROBE_LABELS.keys())
    models = [m for m in COLORS.keys() if m in results]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(probes))
    width = 0.15
    offsets = np.linspace(-2*width, 2*width, len(models))

    for i, model in enumerate(models):
        values = [results[model]['dimensions'].get(p, {}).get('disparity', 0) for p in probes]
        ax.bar(x + offsets[i], values, width, label=MODEL_SHORT[model],
               color=COLORS[model], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([PROBE_SHORT[p] for p in probes])
    ax.set_ylabel('Disparity Score')
    ax.set_xlabel('Probe')
    ax.set_title('Disparity by Probe Across Models', fontweight='bold', pad=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.95)
    ax.set_ylim(0, 0.65)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    # Add severity threshold
    ax.axhline(y=0.40, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(4.5, 0.41, 'Medium threshold', fontsize=8, color='#e74c3c')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig7_probe_comparison.pdf', format='pdf')
    fig.savefig(output_dir / 'fig7_probe_comparison.png', format='png', dpi=300)
    plt.close(fig)
    print("✓ Figure 7: Probe comparison")

# ============================================================================
# Figure 8: Regional Heatmap (Model × Region for one probe)
# ============================================================================

def fig8_regional_heatmap(results, output_dir):
    """Create heatmap showing valence by model and region for lifestyle probe."""

    probe = 'P4_lifestyle'
    models = [m for m in COLORS.keys() if m in results]

    # Build matrix
    matrix = np.zeros((len(models), len(REGION_ORDER)))
    for i, model in enumerate(models):
        group_means = results[model]['dimensions'].get(probe, {}).get('group_means', {})
        for j, region in enumerate(REGION_ORDER):
            matrix[i, j] = group_means.get(region, 0)

    fig, ax = plt.subplots(figsize=(8, 4))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean Valence', rotation=270, labelpad=15)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(REGION_ORDER)):
            val = matrix[i, j]
            color = 'white' if val < 0.3 or val > 0.7 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')

    ax.set_xticks(np.arange(len(REGION_ORDER)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(REGION_ORDER, rotation=45, ha='right')
    ax.set_yticklabels([MODEL_SHORT[m] for m in models])
    ax.set_title(f'Regional Valence: {PROBE_SHORT[probe]} Probe', fontweight='bold', pad=10)

    # Grid
    ax.set_xticks(np.arange(len(REGION_ORDER)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(models)+1)-.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig8_regional_heatmap.pdf', format='pdf')
    fig.savefig(output_dir / 'fig8_regional_heatmap.png', format='png', dpi=300)
    plt.close(fig)
    print("✓ Figure 8: Regional heatmap")

# ============================================================================
# Figure 9: Significance Matrix
# ============================================================================

def fig9_significance_matrix(results, output_dir):
    """Create matrix showing statistical significance across model-probe combinations."""

    probes = list(PROBE_LABELS.keys())
    models = [m for m in COLORS.keys() if m in results]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Build significance matrix
    matrix = np.zeros((len(models), len(probes)))
    for i, model in enumerate(models):
        for j, probe in enumerate(probes):
            sig = results[model]['dimensions'].get(probe, {}).get('significant', False)
            matrix[i, j] = 1 if sig else 0

    # Custom colormap
    cmap = plt.cm.colors.ListedColormap(['#f5f5f5', '#27ae60'])

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add checkmarks and X
    for i in range(len(models)):
        for j in range(len(probes)):
            symbol = '✓' if matrix[i, j] == 1 else '✗'
            color = 'white' if matrix[i, j] == 1 else '#bdc3c7'
            ax.text(j, i, symbol, ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)

    ax.set_xticks(np.arange(len(probes)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([PROBE_SHORT[p] for p in probes], rotation=45, ha='right')
    ax.set_yticklabels([MODEL_SHORT[m] for m in models])
    ax.set_title('Statistical Significance Matrix\n(p < 0.01, Bonferroni corrected)',
                 fontweight='bold', pad=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#27ae60', label='Significant'),
        mpatches.Patch(facecolor='#f5f5f5', edgecolor='gray', label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    # Grid
    ax.set_xticks(np.arange(len(probes)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(models)+1)-.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig9_significance_matrix.pdf', format='pdf')
    fig.savefig(output_dir / 'fig9_significance_matrix.png', format='png', dpi=300)
    plt.close(fig)
    print("✓ Figure 9: Significance matrix")

# ============================================================================
# Figure 10: Africa vs Oceania Comparison
# ============================================================================

def fig10_africa_oceania(results, output_dir):
    """Create direct comparison between Africa and Oceania valence scores."""

    probes = list(PROBE_LABELS.keys())
    models = [m for m in COLORS.keys() if m in results]

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5), sharey=True)

    for idx, probe in enumerate(probes):
        ax = axes[idx]

        africa_vals = []
        oceania_vals = []
        model_names = []

        for model in models:
            group_means = results[model]['dimensions'].get(probe, {}).get('group_means', {})
            africa_vals.append(group_means.get('Africa', 0))
            oceania_vals.append(group_means.get('Oceania', 0))
            model_names.append(MODEL_SHORT[model].split('-')[0])  # Shortened name

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, africa_vals, width, label='Africa', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, oceania_vals, width, label='Oceania', color='#27ae60', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_title(PROBE_SHORT[probe], fontweight='bold', fontsize=10)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        if idx == 0:
            ax.set_ylabel('Mean Valence')
            ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Africa vs Oceania: Valence Gap by Probe', fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig10_africa_oceania.pdf', format='pdf')
    fig.savefig(output_dir / 'fig10_africa_oceania.png', format='png', dpi=300)
    plt.close(fig)
    print("✓ Figure 10: Africa vs Oceania")

# ============================================================================
# Main
# ============================================================================

def main():
    """Generate all figures."""

    # Create output directory
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} models\n")

    print("Generating figures...")
    print("-" * 40)

    fig1_radar_fingerprints(results, output_dir)
    fig2_composite_leaderboard(results, output_dir)
    fig3_disparity_heatmap(results, output_dir)
    fig4_regional_bias(results, output_dir)
    fig5_effect_sizes(results, output_dir)
    fig6_worst_best_groups(results, output_dir)
    fig7_probe_comparison(results, output_dir)
    fig8_regional_heatmap(results, output_dir)
    fig9_significance_matrix(results, output_dir)
    fig10_africa_oceania(results, output_dir)

    print("-" * 40)
    print(f"\n✅ All figures saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.pdf')):
        print(f"  - {f.name}")

if __name__ == '__main__':
    main()
