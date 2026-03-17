"""
Main Fingerprint² pipeline orchestration.

This module provides the high-level API for running evaluations,
generating fingerprints, and producing reports.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fingerprint_squared.core.evaluator import VLMEvaluator, EvaluationConfig, EvaluationResult
from fingerprint_squared.core.fingerprint import (
    FingerprintGenerator,
    FingerprintComparator,
    ModelFingerprint,
)
from fingerprint_squared.models.base import BaseVLM
from fingerprint_squared.models.registry import get_model, list_models
from fingerprint_squared.visualization.reports import ReportGenerator
from fingerprint_squared.visualization.plots import BiasRadarChart, FingerprintVisualizer
from fingerprint_squared.utils.logging import setup_logging, get_logger
from fingerprint_squared.utils.io import save_json, load_json


class FingerprintSquared:
    """
    Main interface for the Fingerprint² framework.

    Provides a unified API for evaluating VLMs, generating fingerprints,
    comparing models, and producing reports.

    Example:
        >>> fp2 = FingerprintSquared()
        >>> result = fp2.evaluate("gpt-4o")
        >>> fp2.generate_report(result)
        >>> fingerprint = fp2.generate_fingerprint(result)
    """

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        output_dir: Union[str, Path] = "./fp2_results",
        log_level: str = "INFO",
    ):
        """
        Initialize Fingerprint² framework.

        Args:
            config: Evaluation configuration
            output_dir: Directory for outputs
            log_level: Logging level
        """
        self.config = config or EvaluationConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(level=log_level)
        self.logger = get_logger("fingerprint_squared")

        # Initialize components
        self.evaluator = VLMEvaluator(config=self.config)
        self.fingerprint_gen = FingerprintGenerator()
        self.comparator = FingerprintComparator()
        self.report_gen = ReportGenerator()

        # Cache for results
        self._results: Dict[str, EvaluationResult] = {}
        self._fingerprints: Dict[str, ModelFingerprint] = {}

    async def evaluate(
        self,
        model: Union[str, BaseVLM],
        api_key: Optional[str] = None,
        generate_fingerprint: bool = True,
        generate_report: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a VLM for bias and fairness.

        Args:
            model: Model name or BaseVLM instance
            api_key: Optional API key
            generate_fingerprint: Whether to generate fingerprint
            generate_report: Whether to generate report

        Returns:
            EvaluationResult with all metrics
        """
        self.logger.info(f"Starting evaluation for: {model}")

        # Run evaluation
        result = await self.evaluator.evaluate(model, api_key)

        # Cache result
        cache_key = f"{result.model_name}_{result.timestamp}"
        self._results[cache_key] = result

        # Generate fingerprint
        if generate_fingerprint:
            fingerprint = self.fingerprint_gen.generate(result)
            self._fingerprints[cache_key] = fingerprint
            self.comparator.add_fingerprint(fingerprint)
            self.logger.info(f"Generated fingerprint: {fingerprint.fingerprint_hash}")

        # Generate report
        if generate_report:
            report_path = self.generate_report(
                result,
                fingerprint if generate_fingerprint else None,
            )
            self.logger.info(f"Report saved to: {report_path}")

        # Save results
        self._save_results(result, fingerprint if generate_fingerprint else None)

        return result

    def evaluate_sync(
        self,
        model: Union[str, BaseVLM],
        api_key: Optional[str] = None,
        generate_fingerprint: bool = True,
        generate_report: bool = True,
    ) -> EvaluationResult:
        """Synchronous wrapper for evaluate."""
        return asyncio.run(self.evaluate(
            model, api_key, generate_fingerprint, generate_report
        ))

    async def evaluate_multiple(
        self,
        models: List[Union[str, BaseVLM]],
        api_keys: Optional[Dict[str, str]] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple models.

        Args:
            models: List of model names or instances
            api_keys: Optional dict mapping model names to API keys

        Returns:
            List of EvaluationResults
        """
        results = []
        api_keys = api_keys or {}

        for model in models:
            model_name = model if isinstance(model, str) else model.model_name
            api_key = api_keys.get(model_name)

            result = await self.evaluate(model, api_key)
            results.append(result)

        return results

    def generate_fingerprint(
        self,
        result: EvaluationResult,
    ) -> ModelFingerprint:
        """
        Generate a fingerprint from evaluation results.

        Args:
            result: EvaluationResult instance

        Returns:
            ModelFingerprint
        """
        fingerprint = self.fingerprint_gen.generate(result)
        cache_key = f"{result.model_name}_{result.timestamp}"
        self._fingerprints[cache_key] = fingerprint
        self.comparator.add_fingerprint(fingerprint)
        return fingerprint

    def generate_report(
        self,
        result: EvaluationResult,
        fingerprint: Optional[ModelFingerprint] = None,
        format: str = "html",
    ) -> Path:
        """
        Generate evaluation report.

        Args:
            result: EvaluationResult instance
            fingerprint: Optional ModelFingerprint
            format: Report format ("html" or "markdown")

        Returns:
            Path to generated report
        """
        report_dir = self.output_dir / "reports"

        if format == "html":
            return self.report_gen.generate_html_report(
                result, fingerprint, report_dir
            )
        else:
            return self.report_gen.generate_markdown_report(
                result, fingerprint, report_dir
            )

    def compare_models(
        self,
        model1: str,
        model2: str,
    ) -> Dict[str, Any]:
        """
        Compare two evaluated models.

        Args:
            model1: First model name
            model2: Second model name

        Returns:
            Comparison results
        """
        fp1 = self._get_latest_fingerprint(model1)
        fp2 = self._get_latest_fingerprint(model2)

        if fp1 is None or fp2 is None:
            raise ValueError(f"Fingerprints not found for {model1} or {model2}")

        return self.comparator.compare(fp1, fp2)

    def rank_models(
        self,
        by: str = "overall_bias",
        ascending: bool = True,
    ) -> List[tuple]:
        """
        Rank all evaluated models.

        Args:
            by: Metric to rank by
            ascending: Sort order

        Returns:
            List of (model_name, score) tuples
        """
        return self.comparator.rank_models(by=by, ascending=ascending)

    def visualize_fingerprint(
        self,
        model: str,
        output_path: Optional[Path] = None,
    ) -> Any:
        """
        Visualize a model's fingerprint.

        Args:
            model: Model name
            output_path: Optional output path

        Returns:
            Matplotlib Figure
        """
        fingerprint = self._get_latest_fingerprint(model)
        if fingerprint is None:
            raise ValueError(f"Fingerprint not found for {model}")

        visualizer = FingerprintVisualizer()
        return visualizer.plot_fingerprint(
            fingerprint,
            output_path=output_path or self.output_dir / f"{model}_fingerprint.png",
        )

    def visualize_comparison(
        self,
        models: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
    ) -> Any:
        """
        Create radar chart comparison of models.

        Args:
            models: List of model names (default: all evaluated)
            output_path: Optional output path

        Returns:
            Matplotlib Figure
        """
        if models is None:
            models = list(set(fp.model_name for fp in self._fingerprints.values()))

        scores = {}
        for model in models:
            fp = self._get_latest_fingerprint(model)
            if fp:
                scores[model] = fp.dimension_scores

        radar = BiasRadarChart()
        return radar.plot(
            scores,
            output_path=output_path or self.output_dir / "model_comparison.png",
        )

    def _get_latest_fingerprint(self, model_name: str) -> Optional[ModelFingerprint]:
        """Get the most recent fingerprint for a model."""
        matching = [
            (key, fp) for key, fp in self._fingerprints.items()
            if fp.model_name == model_name
        ]

        if not matching:
            return None

        # Sort by timestamp and return most recent
        matching.sort(key=lambda x: x[0], reverse=True)
        return matching[0][1]

    def _save_results(
        self,
        result: EvaluationResult,
        fingerprint: Optional[ModelFingerprint],
    ) -> None:
        """Save results to disk."""
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)

        # Save evaluation result
        result_path = results_dir / f"{result.model_name}_{result.timestamp}.json"
        save_json(result.to_dict(), result_path)

        # Save fingerprint
        if fingerprint:
            fp_dir = self.output_dir / "fingerprints"
            fp_dir.mkdir(exist_ok=True)
            fp_path = fp_dir / f"{fingerprint.model_name}_{fingerprint.timestamp}.json"
            save_json(fingerprint.to_dict(), fp_path)

    def load_results(
        self,
        model_name: str,
        timestamp: Optional[str] = None,
    ) -> Optional[EvaluationResult]:
        """
        Load previously saved results.

        Args:
            model_name: Model name
            timestamp: Optional specific timestamp

        Returns:
            EvaluationResult if found
        """
        results_dir = self.output_dir / "results"

        if timestamp:
            result_path = results_dir / f"{model_name}_{timestamp}.json"
            if result_path.exists():
                data = load_json(result_path)
                # Convert back to EvaluationResult
                return data
        else:
            # Find most recent
            matching = list(results_dir.glob(f"{model_name}_*.json"))
            if matching:
                matching.sort(reverse=True)
                data = load_json(matching[0])
                return data

        return None

    def list_available_models(self) -> List[str]:
        """List all supported models."""
        return list_models()

    def list_evaluated_models(self) -> List[str]:
        """List all models that have been evaluated."""
        return list(set(fp.model_name for fp in self._fingerprints.values()))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        return {
            "total_evaluations": len(self._results),
            "models_evaluated": self.list_evaluated_models(),
            "rankings": {
                "by_bias": self.rank_models("overall_bias", ascending=True)[:5],
                "by_fairness": self.rank_models("overall_fairness", ascending=False)[:5],
            },
        }


# Convenience functions
def evaluate(
    model: str,
    api_key: Optional[str] = None,
    output_dir: str = "./fp2_results",
) -> EvaluationResult:
    """
    Quick evaluation of a single model.

    Args:
        model: Model name
        api_key: Optional API key
        output_dir: Output directory

    Returns:
        EvaluationResult
    """
    fp2 = FingerprintSquared(output_dir=output_dir)
    return fp2.evaluate_sync(model, api_key)


def compare(
    models: List[str],
    api_keys: Optional[Dict[str, str]] = None,
    output_dir: str = "./fp2_results",
) -> Dict[str, Any]:
    """
    Quick comparison of multiple models.

    Args:
        models: List of model names
        api_keys: Optional dict of API keys
        output_dir: Output directory

    Returns:
        Comparison results
    """
    fp2 = FingerprintSquared(output_dir=output_dir)

    # Evaluate all models
    asyncio.run(fp2.evaluate_multiple(models, api_keys))

    # Return rankings
    return fp2.get_summary()
