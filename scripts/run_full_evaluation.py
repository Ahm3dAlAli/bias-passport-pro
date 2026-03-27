#!/usr/bin/env python3
"""
run_full_evaluation.py
======================
Run Fingerprint² evaluation on the FULL FHIBE dataset (10,318 images).

This script provides more control than run_fhibe_benchmark.py:
- Run models one at a time or in batches
- Resume from checkpoints
- Aggregate results from multiple runs

Usage:
    # Run all models on full dataset
    python scripts/run_full_evaluation.py --dataset /path/to/fhibe --all-models

    # Run specific models
    python scripts/run_full_evaluation.py --dataset /path/to/fhibe \
        --models moondream2,paligemma,qwen

    # Resume from checkpoint
    python scripts/run_full_evaluation.py --dataset /path/to/fhibe --resume
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Model configurations - Updated with latest SOTA VLMs (March 2025)
MODELS = {
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 1: Original Fingerprint² models (verified working)
    # ═══════════════════════════════════════════════════════════════════════════
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "smolvlm": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "paligemma": "google/paligemma-3b-mix-448",
    # "moondream2": "vikhyat/moondream2",  # Requires HF auth
    "internvl2-2b": "OpenGVLab/InternVL2-2B",

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 2: Latest Qwen Vision Models (2024-2025)
    # ═══════════════════════════════════════════════════════════════════════════
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",  # Requires multi-GPU
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B",

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 3: Meta Llama Vision Models
    # ═══════════════════════════════════════════════════════════════════════════
    "llama-3.2-11b-vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "llama-3.2-90b-vision": "meta-llama/Llama-3.2-90B-Vision-Instruct",  # Requires multi-GPU

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 4: InternVL Series (Latest)
    # ═══════════════════════════════════════════════════════════════════════════
    "internvl2-8b": "OpenGVLab/InternVL2-8B",
    "internvl2.5-8b": "OpenGVLab/InternVL2_5-8B",
    "internvl3-2b": "OpenGVLab/InternVL3-2B",
    "internvl3-8b": "OpenGVLab/InternVL3-8B",

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 5: Mistral Pixtral Vision Models
    # ═══════════════════════════════════════════════════════════════════════════
    "pixtral-12b": "mistralai/Pixtral-12B-2409",

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 6: DeepSeek Vision Models
    # ═══════════════════════════════════════════════════════════════════════════
    "deepseek-vl2-tiny": "deepseek-ai/deepseek-vl2-tiny",
    "deepseek-vl2-small": "deepseek-ai/deepseek-vl2-small",

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 7: Google Gemma 3 Multimodal (March 2025)
    # ═══════════════════════════════════════════════════════════════════════════
    "gemma3-4b-it": "google/gemma-3-4b-it",
    "gemma3-12b-it": "google/gemma-3-12b-it",
    "gemma3-27b-it": "google/gemma-3-27b-it",

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 8: Classic Foundation Models (CLIP, FLAVA, Flamingo)
    # ═══════════════════════════════════════════════════════════════════════════
    "flava": "facebook/flava-full",
    "openflamingo-4b": "openflamingo/OpenFlamingo-4B-vitl-rpj3b",
    "openflamingo-9b": "openflamingo/OpenFlamingo-9B-vitl-mpt7b",

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 9: IDEFICS Models (Open reproduction of Flamingo)
    # ═══════════════════════════════════════════════════════════════════════════
    "idefics2-8b": "HuggingFaceM4/idefics2-8b",
    "idefics3-8b-llama3": "HuggingFaceM4/Idefics3-8B-Llama3",

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 10: Other Notable VLMs
    # ═══════════════════════════════════════════════════════════════════════════
    "phi-3.5-vision": "microsoft/Phi-3.5-vision-instruct",
    "llava-1.6-vicuna-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava-1.6-mistral-7b": "llava-hf/llava-v1.6-mistral-7b-hf",

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 11: Known Issues (100% refusal or captioning-only)
    # ═══════════════════════════════════════════════════════════════════════════
    "minicpm-v2": "openbmb/MiniCPM-V-2",  # Known high refusal
    "florence-2": "microsoft/Florence-2-large",  # Captioning-only, not Q&A
}

# Default models to run - representative subset across model families
DEFAULT_MODELS = [
    # Original verified models
    "qwen2.5-vl-3b", "smolvlm", "paligemma", "internvl2-2b",
    # Latest SOTA (single GPU capable)
    "qwen3-vl-2b", "llama-3.2-11b-vision", "internvl3-2b",
    "pixtral-12b", "deepseek-vl2-tiny", "gemma3-4b-it",
    # Foundation models
    "flava", "idefics2-8b",
]

# Lightweight models for quick testing
LIGHTWEIGHT_MODELS = [
    "qwen2.5-vl-3b", "smolvlm", "internvl2-2b",
    "qwen3-vl-2b", "internvl3-2b", "deepseek-vl2-tiny",
]

# Large models requiring multi-GPU or high VRAM (>48GB)
LARGE_MODELS = [
    "qwen2.5-vl-72b", "llama-3.2-90b-vision", "gemma3-27b-it",
]


def run_model(model_key: str, model_id: str, dataset_path: str,
              output_dir: Path, gpu: int = 0, use_4bit: bool = True) -> dict:
    """Run evaluation for a single model."""

    output_file = output_dir / f"{model_key}_results.json"
    log_file = output_dir / f"{model_key}.log"

    # Check if already completed
    if output_file.exists():
        print(f"[SKIP] {model_key}: Already completed ({output_file})")
        with open(output_file) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"Running: {model_key} ({model_id})")
    print(f"Output:  {output_file}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "scripts/run_fhibe_benchmark.py",
        "--dataset", dataset_path,
        "--models", model_id,
        "--output", str(output_file),
        "--html", str(output_dir / f"{model_key}_dashboard.html"),
        "--gpu", str(gpu),
    ]

    if use_4bit:
        cmd.append("--4bit")

    # NOTE: No --sample flag = full dataset (all 10,318 images)

    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent.parent,
        )

    if result.returncode != 0:
        print(f"[ERROR] {model_key} failed. See {log_file}")
        return {"model": model_key, "error": f"Exit code {result.returncode}"}

    print(f"[DONE] {model_key} completed successfully")

    if output_file.exists():
        with open(output_file) as f:
            return json.load(f)
    return {"model": model_key, "status": "completed"}


def aggregate_results(output_dir: Path) -> dict:
    """Aggregate results from all model runs."""

    aggregated = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "FHIBE (full)",
        "n_images": 10318,
        "models": {},
    }

    for result_file in output_dir.glob("*_results.json"):
        model_key = result_file.stem.replace("_results", "")
        try:
            with open(result_file) as f:
                data = json.load(f)

            # Extract fingerprint data
            if "fingerprints" in data:
                for fp in data["fingerprints"]:
                    aggregated["models"][fp.get("model_name", model_key)] = fp
            elif isinstance(data, dict) and "composite_score" in data:
                aggregated["models"][model_key] = data
            else:
                aggregated["models"][model_key] = data

        except Exception as e:
            print(f"[WARN] Could not parse {result_file}: {e}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Run Fingerprint² on full FHIBE dataset (10,318 images)"
    )
    parser.add_argument(
        "--dataset", "-d", required=True,
        help="Path to FHIBE dataset directory"
    )
    parser.add_argument(
        "--models", "-m", type=str, default=None,
        help="Comma-separated model keys (e.g., 'moondream2,paligemma,qwen')"
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Run all available models (including those with known issues)"
    )
    parser.add_argument(
        "--lightweight", action="store_true",
        help="Run only lightweight models (suitable for single GPU with <24GB VRAM)"
    )
    parser.add_argument(
        "--large", action="store_true",
        help="Include large models (requires multi-GPU or >48GB VRAM)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/full_evaluation",
        help="Output directory"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index to use"
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint (skip completed models)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("\n" + "="*80)
        print("FINGERPRINT² SUPPORTED VLMS - Full Model Catalog")
        print("="*80)
        print(f"\n{'Model Key':<25} {'HuggingFace ID':<45} {'Tags'}")
        print("-"*80)
        for key, model_id in MODELS.items():
            tags = []
            if key in DEFAULT_MODELS:
                tags.append("default")
            if key in LIGHTWEIGHT_MODELS:
                tags.append("lightweight")
            if key in LARGE_MODELS:
                tags.append("LARGE")
            tag_str = f"[{', '.join(tags)}]" if tags else ""
            print(f"  {key:<23} {model_id:<45} {tag_str}")
        print("-"*80)
        print(f"\nTotal models: {len(MODELS)}")
        print(f"Default models ({len(DEFAULT_MODELS)}): {', '.join(DEFAULT_MODELS)}")
        print(f"Lightweight models ({len(LIGHTWEIGHT_MODELS)}): {', '.join(LIGHTWEIGHT_MODELS)}")
        print(f"Large models ({len(LARGE_MODELS)}): {', '.join(LARGE_MODELS)}")
        print()
        return

    # Determine which models to run
    if args.all_models:
        models_to_run = list(MODELS.keys())
    elif args.lightweight:
        models_to_run = LIGHTWEIGHT_MODELS.copy()
    elif args.models:
        models_to_run = [m.strip() for m in args.models.split(",")]
        # Validate
        invalid = [m for m in models_to_run if m not in MODELS]
        if invalid:
            print(f"[ERROR] Unknown models: {invalid}")
            print(f"Available: {list(MODELS.keys())}")
            sys.exit(1)
    else:
        models_to_run = DEFAULT_MODELS.copy()

    # Add large models if requested
    if args.large:
        for m in LARGE_MODELS:
            if m not in models_to_run:
                models_to_run.append(m)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Fingerprint² Full FHIBE Evaluation")
    print(f"{'='*60}")
    print(f"Dataset:    {args.dataset}")
    print(f"Output:     {output_dir}")
    print(f"GPU:        {args.gpu}")
    print(f"Models:     {', '.join(models_to_run)}")
    print(f"4-bit:      {not args.no_4bit}")
    print(f"Resume:     {args.resume}")
    print()
    print("WARNING: Running on FULL dataset (10,318 images)")
    print("Estimated time: 4-8 hours per model")
    print(f"{'='*60}\n")

    # Run each model
    results = {}
    for model_key in models_to_run:
        model_id = MODELS[model_key]
        result = run_model(
            model_key=model_key,
            model_id=model_id,
            dataset_path=args.dataset,
            output_dir=output_dir,
            gpu=args.gpu,
            use_4bit=not args.no_4bit,
        )
        results[model_key] = result

    # Aggregate all results
    print(f"\n{'='*60}")
    print("Aggregating results...")
    print(f"{'='*60}\n")

    aggregated = aggregate_results(output_dir)
    aggregated_file = output_dir / "full_benchmark_aggregated.json"

    with open(aggregated_file, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    print(f"[DONE] Aggregated results: {aggregated_file}")
    print(f"\nSummary:")
    print("-" * 40)
    for model, data in aggregated.get("models", {}).items():
        if isinstance(data, dict):
            score = data.get("composite_score", "N/A")
            print(f"  {model:30} → {score}")
    print()


if __name__ == "__main__":
    main()
