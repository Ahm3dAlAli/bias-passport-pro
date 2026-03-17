"""
Background Worker for Fingerprint² Evaluations

This worker processes evaluation jobs from a queue (in production, would use Redis/RabbitMQ).
For now, it monitors the database for pending evaluations and processes them.

Usage:
    python -m fingerprint_squared.worker
"""

import asyncio
import os
import sys
import signal
from datetime import datetime
from pathlib import Path

# Graceful shutdown handling
shutdown_event = asyncio.Event()


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    print("\n[Worker] Shutdown signal received, finishing current task...")
    shutdown_event.set()


async def process_pending_evaluations():
    """Process any pending evaluations in the database."""
    from fingerprint_squared.storage.sqlite_storage import SQLiteStorage

    db_path = os.environ.get("DATABASE_PATH", "./data/fingerprints.db")
    storage = SQLiteStorage(db_path)

    print(f"[Worker] Connected to database: {db_path}")
    print("[Worker] Monitoring for pending evaluations...")

    while not shutdown_event.is_set():
        try:
            # Check for pending experiments
            experiments = storage.list_experiments(status="pending", limit=1)

            if experiments:
                exp = experiments[0]
                print(f"[Worker] Processing experiment {exp.experiment_id} for model {exp.model_id}")

                # Update status to running
                storage.update_experiment(exp.experiment_id, status="running")

                try:
                    # Run the evaluation
                    await run_evaluation(storage, exp)
                    print(f"[Worker] Completed experiment {exp.experiment_id}")
                except Exception as e:
                    print(f"[Worker] Error processing {exp.experiment_id}: {e}")
                    storage.update_experiment(
                        exp.experiment_id,
                        status="failed",
                        error=str(e),
                    )
            else:
                # No pending work, wait a bit
                await asyncio.sleep(5)

        except Exception as e:
            print(f"[Worker] Error in main loop: {e}")
            await asyncio.sleep(10)

    print("[Worker] Shutdown complete")


async def run_evaluation(storage, experiment):
    """Run an evaluation for an experiment."""
    from PIL import Image
    from fingerprint_squared.models.openrouter_vlm import MultiProviderVLM
    from fingerprint_squared.data.fhibe_loader import FHIBELoader, load_fhibe
    from fingerprint_squared.probes.social_inference_battery import (
        SocialInferenceBattery,
        ProbeType,
    )
    from fingerprint_squared.scoring.llm_judge import LLMJudge
    from fingerprint_squared.core.bias_fingerprint import FingerprintAggregator

    config = experiment.config or {}
    model_id = experiment.model_id
    model_name = experiment.model_name
    dataset_path = config.get("dataset_path") or os.environ.get("FHIBE_PATH", "./fhibe_images")
    n_images = config.get("n_images", 20)
    n_per_group = config.get("n_per_group", 5)
    seed = config.get("seed", 42)

    # Create VLM
    vlm = MultiProviderVLM.create(model_id)

    # Load or create dataset
    if dataset_path and Path(dataset_path).exists():
        dataset = load_fhibe(dataset_path)
    else:
        loader = FHIBELoader()
        dataset = loader.create_synthetic_dataset(n_per_intersection=n_per_group)

    # Get balanced sample
    sample = dataset.get_balanced_sample(n_per_group=n_per_group, seed=seed)

    # Initialize components
    battery = SocialInferenceBattery()
    judge = LLMJudge()
    aggregator = FingerprintAggregator()

    # Prepare images
    images = []
    demographics_map = {}
    for img in sample:
        try:
            pil_image = Image.open(img.image_path).convert("RGB")
            images.append((img.image_id, pil_image))
            demographics_map[img.image_id] = img.demographics
        except Exception:
            continue

    if not images:
        raise ValueError("No valid images found")

    images = images[:n_images]

    print(f"[Worker] Processing {len(images)} images with {len(ProbeType)} probes each")

    # Run probes
    all_responses = []
    for i, (image_id, image) in enumerate(images):
        for probe_type in ProbeType:
            response = await battery.run_single_probe(vlm, image, image_id, probe_type)
            all_responses.append(response)

        if (i + 1) % 10 == 0:
            print(f"[Worker] Progress: {i + 1}/{len(images)} images")

    # Score responses
    probe_questions = {pt: battery.get_probe_prompt(pt) for pt in ProbeType}
    scored_responses = await judge.score_batch(
        all_responses,
        demographics_map,
        probe_questions,
    )

    # Add demographic info
    for response in scored_responses:
        response.demographic_info = demographics_map.get(response.image_id, {})

    # Save responses
    storage.save_responses(experiment.experiment_id, scored_responses, demographics_map)

    # Generate fingerprint
    fingerprint = aggregator.aggregate(
        model_id=model_id,
        model_name=model_name or model_id,
        responses=scored_responses,
    )

    # Save fingerprint
    storage.save_fingerprint(experiment.experiment_id, fingerprint)

    # Update experiment status
    storage.update_experiment(
        experiment.experiment_id,
        status="completed",
        n_images=len(images),
        n_probes=len(all_responses),
    )

    # Cleanup
    if hasattr(vlm, 'close'):
        await vlm.close()

    print(f"[Worker] Fingerprint generated: overall_bias={fingerprint.overall_bias_score:.3f}")


def main():
    """Main entry point for the worker."""
    print("=" * 60)
    print("Fingerprint² Background Worker")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Database: {os.environ.get('DATABASE_PATH', './data/fingerprints.db')}")
    print("=" * 60)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the worker
    asyncio.run(process_pending_evaluations())


if __name__ == "__main__":
    main()
