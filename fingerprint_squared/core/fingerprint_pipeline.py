"""
Fingerprint Pipeline

Orchestrates the complete bias fingerprinting workflow:
1. Load dataset (FHIBE or compatible)
2. Run probes against VLM
3. Score responses with LLM judge
4. Aggregate into fingerprint
5. Generate outputs (JSON, PDF, dashboard data)

This is the main entry point for running evaluations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from tqdm.asyncio import tqdm

from fingerprint_squared.data.fhibe_loader import FHIBEDataset, FHIBEImage, FHIBELoader
from fingerprint_squared.probes.social_inference_battery import (
    SocialInferenceBattery,
    ProbeResponse,
    ProbeType,
)
from fingerprint_squared.scoring.llm_judge import LLMJudge
from fingerprint_squared.core.bias_fingerprint import (
    BiasFingerprint,
    FingerprintAggregator,
    FingerprintComparator,
)


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the fingerprinting pipeline."""

    # Dataset settings
    n_images_per_group: int = 20  # Images per demographic intersection
    balanced_sampling: bool = True
    random_seed: int = 42

    # Probe settings
    probes_to_run: List[ProbeType] = field(
        default_factory=lambda: list(ProbeType)
    )
    max_concurrent_probes: int = 10

    # Scoring settings
    use_llm_judge: bool = True
    max_concurrent_scoring: int = 10

    # Output settings
    output_dir: str = "./fingerprint_outputs"
    save_raw_responses: bool = True
    generate_report: bool = True

    # Logging
    verbose: bool = True
    log_file: Optional[str] = None


@dataclass
class PipelineResults:
    """Results from a fingerprinting pipeline run."""

    fingerprint: BiasFingerprint
    raw_responses: List[ProbeResponse]
    config: PipelineConfig
    runtime_seconds: float
    errors: List[str] = field(default_factory=list)

    def save(self, output_dir: str) -> Dict[str, str]:
        """Save all outputs to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save fingerprint
        fp_path = output_dir / f"{self.fingerprint.model_id}_fingerprint.json"
        self.fingerprint.save(str(fp_path))
        paths["fingerprint"] = str(fp_path)

        # Save raw responses
        responses_path = output_dir / f"{self.fingerprint.model_id}_responses.json"
        responses_data = [
            {
                "probe_type": r.probe_type.value,
                "image_id": r.image_id,
                "raw_response": r.raw_response,
                "valence_score": r.valence_score,
                "stereotype_alignment": r.stereotype_alignment,
                "confidence_score": r.confidence_score,
                "refusal": r.refusal,
                "error": r.error,
                "demographic_info": r.demographic_info,
                "latency_ms": r.latency_ms,
            }
            for r in self.raw_responses
        ]
        with open(responses_path, "w") as f:
            json.dump(responses_data, f, indent=2)
        paths["responses"] = str(responses_path)

        # Save config
        config_path = output_dir / f"{self.fingerprint.model_id}_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "n_images_per_group": self.config.n_images_per_group,
                "probes_to_run": [p.value for p in self.config.probes_to_run],
                "random_seed": self.config.random_seed,
                "runtime_seconds": self.runtime_seconds,
            }, f, indent=2)
        paths["config"] = str(config_path)

        return paths


class FingerprintPipeline:
    """
    Main pipeline for bias fingerprinting VLMs.

    Example:
        >>> from fingerprint_squared.models.openai_vlm import OpenAIVLM
        >>>
        >>> pipeline = FingerprintPipeline(config=PipelineConfig())
        >>> vlm = OpenAIVLM(model="gpt-4o")
        >>> dataset = load_fhibe("path/to/images")
        >>>
        >>> results = await pipeline.run(
        ...     vlm=vlm,
        ...     dataset=dataset,
        ...     model_id="gpt-4o",
        ...     model_name="GPT-4 Vision",
        ... )
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        judge: Optional[LLMJudge] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        self.config = config or PipelineConfig()
        self.judge = judge or LLMJudge()
        self.progress_callback = progress_callback

        self.probe_battery = SocialInferenceBattery()
        self.aggregator = FingerprintAggregator()

        # Setup output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Setup logging
        if self.config.log_file:
            logging.basicConfig(
                filename=self.config.log_file,
                level=logging.DEBUG if self.config.verbose else logging.INFO,
            )

    def _report_progress(self, stage: str, progress: float):
        """Report progress to callback if set."""
        if self.progress_callback:
            self.progress_callback(stage, progress)
        if self.config.verbose:
            logger.info(f"{stage}: {progress:.1%}")

    async def run(
        self,
        vlm,  # BaseVLM instance
        dataset: FHIBEDataset,
        model_id: str,
        model_name: str,
    ) -> PipelineResults:
        """
        Run the complete fingerprinting pipeline.

        Args:
            vlm: VLM instance to evaluate
            dataset: FHIBE dataset
            model_id: Unique model identifier
            model_name: Human-readable model name

        Returns:
            PipelineResults with fingerprint and raw data
        """
        start_time = datetime.now()
        errors = []

        self._report_progress("Starting pipeline", 0.0)

        # Step 1: Sample dataset
        if self.config.balanced_sampling:
            sampled = dataset.get_balanced_sample(
                n_per_group=self.config.n_images_per_group,
                seed=self.config.random_seed,
            )
        else:
            sampled = dataset

        self._report_progress("Dataset sampled", 0.1)
        logger.info(f"Using {len(sampled)} images from {len(dataset)} total")

        # Step 2: Run probes
        self._report_progress("Running probes", 0.2)
        all_responses = []

        try:
            all_responses = await self._run_probes(vlm, sampled)
        except Exception as e:
            errors.append(f"Probe error: {str(e)}")
            logger.error(f"Probe execution failed: {e}")

        self._report_progress("Probes complete", 0.5)

        # Step 3: Score with LLM judge
        if self.config.use_llm_judge and all_responses:
            self._report_progress("Scoring responses", 0.6)

            try:
                all_responses = await self._score_responses(all_responses, sampled)
            except Exception as e:
                errors.append(f"Scoring error: {str(e)}")
                logger.error(f"Scoring failed: {e}")

        self._report_progress("Scoring complete", 0.8)

        # Step 4: Aggregate into fingerprint
        self._report_progress("Aggregating fingerprint", 0.9)

        fingerprint = self.aggregator.aggregate(
            model_id=model_id,
            model_name=model_name,
            responses=all_responses,
        )

        runtime = (datetime.now() - start_time).total_seconds()

        self._report_progress("Pipeline complete", 1.0)

        return PipelineResults(
            fingerprint=fingerprint,
            raw_responses=all_responses,
            config=self.config,
            runtime_seconds=runtime,
            errors=errors,
        )

    async def _run_probes(
        self,
        vlm,
        dataset: FHIBEDataset,
    ) -> List[ProbeResponse]:
        """Run all probes against all images."""
        responses = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_probes)

        async def run_single(image: FHIBEImage, probe_type: ProbeType):
            async with semaphore:
                try:
                    response = await self.probe_battery.run_single_probe(
                        vlm=vlm,
                        image_path=image.image_path,
                        image_id=image.image_id,
                        probe_type=probe_type,
                        demographics=image.demographics,
                    )
                    return response
                except Exception as e:
                    logger.error(f"Probe failed for {image.image_id}/{probe_type.value}: {e}")
                    return ProbeResponse(
                        probe_type=probe_type,
                        image_id=image.image_id,
                        raw_response="",
                        error=str(e),
                        demographic_info=image.demographics,
                    )

        # Create all tasks
        tasks = []
        for image in dataset:
            for probe_type in self.config.probes_to_run:
                tasks.append(run_single(image, probe_type))

        # Run with progress bar
        if self.config.verbose:
            results = []
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Running probes",
            ):
                result = await coro
                results.append(result)
        else:
            results = await asyncio.gather(*tasks)

        return list(results)

    async def _score_responses(
        self,
        responses: List[ProbeResponse],
        dataset: FHIBEDataset,
    ) -> List[ProbeResponse]:
        """Score all responses with LLM judge."""
        # Build demographics map
        demographics_map = {
            img.image_id: img.demographics
            for img in dataset
        }

        # Build probe questions map
        probe_questions = {
            probe_type: self.probe_battery.get_probe_question(probe_type)
            for probe_type in ProbeType
        }

        # Score batch
        scored = await self.judge.score_batch(
            responses=responses,
            demographics_map=demographics_map,
            probe_questions=probe_questions,
            max_concurrent=self.config.max_concurrent_scoring,
        )

        return scored


class MultiModelPipeline:
    """
    Run fingerprinting across multiple models for comparison.

    Example:
        >>> pipeline = MultiModelPipeline()
        >>> results = await pipeline.run_comparison(
        ...     models=[gpt4o, claude_sonnet, llava],
        ...     model_names=["GPT-4o", "Claude Sonnet", "LLaVA"],
        ...     dataset=dataset,
        ... )
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):
        self.config = config or PipelineConfig()
        self.comparator = FingerprintComparator()

    async def run_comparison(
        self,
        models: List[Any],  # List of VLM instances
        model_ids: List[str],
        model_names: List[str],
        dataset: FHIBEDataset,
    ) -> Dict[str, Any]:
        """
        Run fingerprinting on multiple models and compare.

        Args:
            models: List of VLM instances
            model_ids: List of model identifiers
            model_names: List of human-readable names
            dataset: Shared dataset for evaluation

        Returns:
            Comparison results
        """
        if len(models) != len(model_ids) or len(models) != len(model_names):
            raise ValueError("models, model_ids, and model_names must have same length")

        fingerprints = []
        all_results = {}

        for vlm, model_id, model_name in zip(models, model_ids, model_names):
            logger.info(f"Evaluating {model_name}...")

            pipeline = FingerprintPipeline(config=self.config)
            results = await pipeline.run(
                vlm=vlm,
                dataset=dataset,
                model_id=model_id,
                model_name=model_name,
            )

            fingerprints.append(results.fingerprint)
            all_results[model_id] = results

            # Save individual results
            results.save(self.config.output_dir)

        # Compare fingerprints
        comparison = self.comparator.compare(fingerprints)

        # Save comparison
        comparison_path = Path(self.config.output_dir) / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)

        return {
            "comparison": comparison,
            "fingerprints": {fp.model_id: fp for fp in fingerprints},
            "results": all_results,
        }


# Convenience function for quick single-model evaluation
async def fingerprint_model(
    vlm,
    dataset: FHIBEDataset,
    model_id: str,
    model_name: str,
    output_dir: str = "./fingerprint_outputs",
    n_images_per_group: int = 20,
) -> BiasFingerprint:
    """
    Quick fingerprinting function for a single model.

    Example:
        >>> fingerprint = await fingerprint_model(
        ...     vlm=OpenAIVLM(model="gpt-4o"),
        ...     dataset=load_fhibe("path/to/images"),
        ...     model_id="gpt-4o",
        ...     model_name="GPT-4 Vision",
        ... )
    """
    config = PipelineConfig(
        n_images_per_group=n_images_per_group,
        output_dir=output_dir,
    )

    pipeline = FingerprintPipeline(config=config)
    results = await pipeline.run(
        vlm=vlm,
        dataset=dataset,
        model_id=model_id,
        model_name=model_name,
    )

    results.save(output_dir)
    return results.fingerprint
