"""
Experiment runner for reproducible Fingerprint² evaluations.

Provides a structured way to run experiments with configurable
parameters, logging, and result tracking.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ExperimentConfig:
    """Configuration for a Fingerprint² experiment."""

    # Experiment metadata
    name: str
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)

    # Models to evaluate
    models: List[str] = field(default_factory=list)

    # Evaluation settings
    benchmark: str = "fp2-core"
    probe_types: List[str] = field(default_factory=lambda: [
        "stereotype_association", "counterfactual", "representation"
    ])
    demographic_dimensions: List[str] = field(default_factory=lambda: [
        "gender", "race", "age"
    ])
    n_probes_per_type: int = 50
    max_samples: Optional[int] = None

    # Model settings
    max_tokens: int = 512
    temperature: float = 0.0
    max_concurrent: int = 5

    # Thresholds
    fairness_threshold: float = 0.1
    bias_threshold: float = 0.5

    # Output settings
    output_dir: str = "./experiments"
    save_raw_responses: bool = True
    generate_reports: bool = True
    report_format: str = "html"

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        return cls(**data)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @property
    def experiment_hash(self) -> str:
        """Generate a hash for this experiment configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


@dataclass
class ExperimentRun:
    """Record of a single experiment run."""

    run_id: str
    config: ExperimentConfig
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"  # running, completed, failed
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "results": self.results,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class ExperimentRunner:
    """
    Runner for Fingerprint² experiments.

    Manages experiment execution, logging, and result tracking
    for reproducible research.

    Example:
        >>> config = ExperimentConfig(
        ...     name="vlm_bias_study",
        ...     models=["gpt-4o", "claude-3-opus"],
        ... )
        >>> runner = ExperimentRunner(config)
        >>> results = runner.run()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        api_keys: Optional[Dict[str, str]] = None,
    ):
        self.config = config
        self.api_keys = api_keys or {}
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up experiment logging."""
        import logging
        from fingerprint_squared.utils.logging import setup_logging

        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"{self.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(level="INFO", log_file=log_file)

        self.logger = logging.getLogger("fingerprint_squared.experiment")

    def run(self) -> ExperimentRun:
        """
        Run the experiment.

        Returns:
            ExperimentRun with results
        """
        import asyncio
        from fingerprint_squared import FingerprintSquared
        from fingerprint_squared.core.evaluator import EvaluationConfig

        # Create run record
        run_id = f"{self.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.config.experiment_hash}"
        run = ExperimentRun(
            run_id=run_id,
            config=self.config,
            start_time=datetime.now().isoformat(),
        )

        self.logger.info(f"Starting experiment: {run_id}")
        self.logger.info(f"Configuration: {self.config.name}")
        self.logger.info(f"Models: {self.config.models}")

        # Save config
        config_path = self.output_dir / f"{run_id}_config.yaml"
        self.config.save_yaml(config_path)

        try:
            # Build evaluation config
            eval_config = EvaluationConfig(
                probe_types=self.config.probe_types,
                demographic_dimensions=self.config.demographic_dimensions,
                n_probes_per_type=self.config.n_probes_per_type,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                max_concurrent=self.config.max_concurrent,
                fairness_threshold=self.config.fairness_threshold,
                bias_threshold=self.config.bias_threshold,
                save_raw_responses=self.config.save_raw_responses,
            )

            # Initialize framework
            fp2 = FingerprintSquared(
                config=eval_config,
                output_dir=self.output_dir / run_id,
            )

            # Evaluate each model
            model_results = {}
            for model in self.config.models:
                self.logger.info(f"Evaluating model: {model}")

                try:
                    api_key = self.api_keys.get(model)
                    result = asyncio.run(fp2.evaluate(
                        model,
                        api_key=api_key,
                        generate_fingerprint=True,
                        generate_report=self.config.generate_reports,
                    ))

                    model_results[model] = {
                        "overall_bias_score": result.overall_bias_score,
                        "overall_fairness_score": result.overall_fairness_score,
                        "total_probes": result.total_probes,
                        "fingerprint": fp2._fingerprints.get(
                            f"{result.model_name}_{result.timestamp}"
                        ).to_dict() if f"{result.model_name}_{result.timestamp}" in fp2._fingerprints else None,
                    }

                    self.logger.info(
                        f"  Bias Score: {result.overall_bias_score:.3f}, "
                        f"Fairness Score: {result.overall_fairness_score:.3f}"
                    )

                except Exception as e:
                    self.logger.error(f"Error evaluating {model}: {e}")
                    run.errors.append(f"{model}: {str(e)}")
                    model_results[model] = {"error": str(e)}

            # Compile results
            run.results = {
                "model_results": model_results,
                "summary": fp2.get_summary(),
                "rankings": {
                    "by_bias": fp2.rank_models("overall_bias", ascending=True),
                    "by_fairness": fp2.rank_models("overall_fairness", ascending=False),
                },
            }

            # Generate comparison visualization
            if len(self.config.models) > 1:
                try:
                    fp2.visualize_comparison(
                        output_path=self.output_dir / run_id / "comparison.png"
                    )
                except Exception as e:
                    self.logger.warning(f"Could not generate comparison: {e}")

            run.status = "completed"

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            run.status = "failed"
            run.errors.append(str(e))
            raise

        finally:
            run.end_time = datetime.now().isoformat()

            # Save run record
            run_path = self.output_dir / f"{run_id}_run.json"
            with open(run_path, "w") as f:
                json.dump(run.to_dict(), f, indent=2)

            self.logger.info(f"Experiment completed: {run.status}")
            self.logger.info(f"Results saved to: {self.output_dir / run_id}")

        return run

    def run_ablation(
        self,
        parameter: str,
        values: List[Any],
    ) -> Dict[str, ExperimentRun]:
        """
        Run ablation study varying a single parameter.

        Args:
            parameter: Parameter name to vary
            values: List of values to test

        Returns:
            Dict mapping parameter values to runs
        """
        results = {}

        for value in values:
            self.logger.info(f"Ablation: {parameter}={value}")

            # Create modified config
            config_dict = self.config.to_dict()
            config_dict[parameter] = value
            config_dict["name"] = f"{self.config.name}_ablation_{parameter}_{value}"

            modified_config = ExperimentConfig.from_dict(config_dict)
            runner = ExperimentRunner(modified_config, self.api_keys)

            try:
                run = runner.run()
                results[str(value)] = run
            except Exception as e:
                self.logger.error(f"Ablation failed for {parameter}={value}: {e}")
                results[str(value)] = None

        return results


def run_experiment(
    config_path: Union[str, Path],
    api_keys: Optional[Dict[str, str]] = None,
) -> ExperimentRun:
    """
    Run an experiment from a configuration file.

    Args:
        config_path: Path to YAML configuration
        api_keys: Optional API keys

    Returns:
        ExperimentRun with results
    """
    config = ExperimentConfig.from_yaml(config_path)
    runner = ExperimentRunner(config, api_keys)
    return runner.run()


def create_experiment_config(
    name: str,
    models: List[str],
    output_dir: str = "./experiments",
    **kwargs,
) -> ExperimentConfig:
    """
    Create an experiment configuration.

    Args:
        name: Experiment name
        models: Models to evaluate
        output_dir: Output directory
        **kwargs: Additional configuration

    Returns:
        ExperimentConfig instance
    """
    return ExperimentConfig(
        name=name,
        models=models,
        output_dir=output_dir,
        **kwargs,
    )
