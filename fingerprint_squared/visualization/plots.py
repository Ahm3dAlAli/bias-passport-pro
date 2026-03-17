"""
Visualization plots for Fingerprint² evaluation results.

Provides rich visualizations including radar charts, heatmaps,
and comparative plots for bias and fairness analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import seaborn as sns


class BiasRadarChart:
    """
    Radar chart visualization for multi-dimensional bias profiles.

    Creates spider/radar charts showing bias scores across
    different dimensions for one or more models.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 10),
        color_palette: str = "Set2",
    ):
        self.figsize = figsize
        self.color_palette = color_palette

    def plot(
        self,
        scores: Dict[str, Dict[str, float]],
        title: str = "Bias Profile Comparison",
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create a radar chart comparing multiple models.

        Args:
            scores: Dict mapping model names to dimension scores
            title: Chart title
            output_path: Optional path to save figure

        Returns:
            Matplotlib Figure
        """
        # Get dimensions
        first_model = list(scores.keys())[0]
        dimensions = list(scores[first_model].keys())
        n_dims = len(dimensions)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(polar=True))

        # Compute angles
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]  # Close the plot

        # Colors
        colors = sns.color_palette(self.color_palette, len(scores))

        # Plot each model
        for (model_name, model_scores), color in zip(scores.items(), colors):
            values = [model_scores.get(dim, 0) for dim in dimensions]
            values += values[:1]  # Close the plot

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, size=10)

        # Set y-axis
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)

        # Add threshold line
        threshold_values = [0.5] * (n_dims + 1)
        ax.plot(angles, threshold_values, '--', color='red', alpha=0.5, linewidth=1, label='Threshold')

        # Legend and title
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title(title, size=14, fontweight='bold', y=1.08)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig


class FairnessHeatmap:
    """
    Heatmap visualization for fairness metrics across groups.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = "RdYlGn_r",
    ):
        self.figsize = figsize
        self.cmap = cmap

    def plot(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Fairness Metrics by Group",
        output_path: Optional[Path] = None,
        annotate: bool = True,
    ) -> Figure:
        """
        Create a heatmap of fairness metrics.

        Args:
            data: Dict mapping groups to metrics
            title: Chart title
            output_path: Optional path to save figure
            annotate: Whether to show values on cells

        Returns:
            Matplotlib Figure
        """
        # Convert to matrix
        groups = list(data.keys())
        metrics = list(data[groups[0]].keys())

        matrix = np.zeros((len(groups), len(metrics)))
        for i, group in enumerate(groups):
            for j, metric in enumerate(metrics):
                matrix[i, j] = data[group].get(metric, 0)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create heatmap
        sns.heatmap(
            matrix,
            annot=annotate,
            fmt='.3f',
            cmap=self.cmap,
            xticklabels=metrics,
            yticklabels=groups,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={'label': 'Score (lower = more fair)'}
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Groups', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_intersectional(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Intersectional Bias Matrix",
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create a heatmap for intersectional analysis.

        Args:
            data: Intersectional group scores
            title: Chart title
            output_path: Optional path to save figure

        Returns:
            Matplotlib Figure
        """
        return self.plot(data, title, output_path)


class FingerprintVisualizer:
    """
    Visualize model fingerprints.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 6),
    ):
        self.figsize = figsize

    def plot_fingerprint(
        self,
        fingerprint: Any,  # ModelFingerprint
        title: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Visualize a single model fingerprint.

        Args:
            fingerprint: ModelFingerprint instance
            title: Optional title
            output_path: Optional path to save figure

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        # 1. Bias scores bar chart
        ax1 = axes[0]
        bias_dims = list(fingerprint.dimension_scores.keys())
        bias_values = list(fingerprint.dimension_scores.values())

        colors = ['green' if v < 0.3 else 'orange' if v < 0.6 else 'red' for v in bias_values]
        bars = ax1.barh(bias_dims, bias_values, color=colors)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Bias Score')
        ax1.set_title('Bias by Dimension')
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')

        # 2. Overall scores gauge
        ax2 = axes[1]
        self._plot_gauge(ax2, fingerprint.bias_scores.get('overall', 0), 'Bias Level')

        # 3. Risk areas and strengths
        ax3 = axes[2]
        ax3.axis('off')

        risk_text = "Risk Areas:\n" + "\n".join([f"• {r}" for r in fingerprint.risk_areas[:5]])
        strength_text = "\nStrengths:\n" + "\n".join([f"• {s}" for s in fingerprint.strengths[:5]])

        ax3.text(0.1, 0.9, risk_text, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', color='red')
        ax3.text(0.1, 0.4, strength_text, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', color='green')
        ax3.set_title('Risk Assessment')

        # Main title
        main_title = title or f"{fingerprint.model_name} Fingerprint"
        fig.suptitle(main_title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig

    def _plot_gauge(self, ax, value: float, label: str):
        """Plot a gauge/speedometer chart."""
        # Create semi-circle
        theta = np.linspace(0, np.pi, 100)
        r = 1

        # Background segments
        segments = [
            (0, np.pi/3, 'green', 'Low'),
            (np.pi/3, 2*np.pi/3, 'orange', 'Medium'),
            (2*np.pi/3, np.pi, 'red', 'High'),
        ]

        for start, end, color, label_text in segments:
            theta_seg = np.linspace(start, end, 30)
            ax.fill_between(theta_seg, 0.8, 1.0, color=color, alpha=0.3)

        # Needle
        needle_angle = np.pi * (1 - value)
        ax.arrow(0, 0, 0.7 * np.cos(needle_angle), 0.7 * np.sin(needle_angle),
                 head_width=0.05, head_length=0.05, fc='black', ec='black')

        # Center dot
        ax.plot(0, 0, 'ko', markersize=10)

        # Value text
        ax.text(0, -0.3, f'{value:.2f}', ha='center', fontsize=14, fontweight='bold')
        ax.text(0, -0.5, label, ha='center', fontsize=12)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.axis('off')
        ax.set_aspect('equal')


class ComparisonPlot:
    """
    Comparative plots for multiple models.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
    ):
        self.figsize = figsize

    def plot_comparison_bars(
        self,
        scores: Dict[str, Dict[str, float]],
        title: str = "Model Comparison",
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create grouped bar chart comparing models.

        Args:
            scores: Dict mapping model names to metric scores
            title: Chart title
            output_path: Optional path to save figure

        Returns:
            Matplotlib Figure
        """
        models = list(scores.keys())
        metrics = list(scores[models[0]].keys())

        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        fig, ax = plt.subplots(figsize=self.figsize)
        colors = sns.color_palette("Set2", len(models))

        for i, (model, model_scores) in enumerate(scores.items()):
            values = [model_scores.get(m, 0) for m in metrics]
            ax.bar(x + i * width, values, width, label=model, color=colors[i])

        ax.set_ylabel('Score')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)

        # Add threshold line
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_ranking(
        self,
        rankings: List[Tuple[str, float]],
        title: str = "Model Rankings",
        metric_name: str = "Bias Score",
        output_path: Optional[Path] = None,
        lower_is_better: bool = True,
    ) -> Figure:
        """
        Create a ranking visualization.

        Args:
            rankings: List of (model_name, score) tuples
            title: Chart title
            metric_name: Name of the metric being ranked
            output_path: Optional path to save figure
            lower_is_better: Whether lower scores are better

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, max(6, len(rankings) * 0.5)))

        models = [r[0] for r in rankings]
        scores = [r[1] for r in rankings]

        # Color based on score
        if lower_is_better:
            colors = ['green' if s < 0.3 else 'orange' if s < 0.6 else 'red' for s in scores]
        else:
            colors = ['red' if s < 0.3 else 'orange' if s < 0.6 else 'green' for s in scores]

        y_pos = np.arange(len(models))
        bars = ax.barh(y_pos, scores, color=colors)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel(metric_name)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)

        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=9)

        # Add rank numbers
        for i, (bar, model) in enumerate(zip(bars, models)):
            ax.text(-0.05, bar.get_y() + bar.get_height()/2,
                    f'#{i+1}', va='center', ha='right', fontsize=10, fontweight='bold')

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_similarity_matrix(
        self,
        similarities: Dict[str, Dict[str, float]],
        title: str = "Model Similarity Matrix",
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create a similarity matrix heatmap.

        Args:
            similarities: Pairwise similarity scores
            title: Chart title
            output_path: Optional path to save figure

        Returns:
            Matplotlib Figure
        """
        models = list(similarities.keys())
        n = len(models)

        matrix = np.zeros((n, n))
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if m1 == m2:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = similarities.get(m1, {}).get(m2, 0)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            xticklabels=models,
            yticklabels=models,
            vmin=0,
            vmax=1,
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig
