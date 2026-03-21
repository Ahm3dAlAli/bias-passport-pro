#!/bin/bash
# =============================================================================
# run_full_benchmark.sh
# =============================================================================
# Run Fingerprint² evaluation on the FULL FHIBE dataset (10,318 images)
# across all VLM models.
#
# This will take significant time and GPU memory. Estimated:
#   - 10,318 images × 5 probes × 8 models = ~413,000 inference calls
#   - ~24-48 hours depending on GPU and model sizes
#
# Usage:
#   ./scripts/run_full_benchmark.sh /path/to/fhibe_dataset
#
# =============================================================================

set -e

# Configuration
DATASET_PATH="${1:-/shares/fhibe/fhibe_face_crop_align}"
OUTPUT_DIR="results/full_benchmark_$(date +%Y%m%d_%H%M%S)"
GPU="${2:-0}"

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS TO EVALUATE - Updated with latest SOTA VLMs (March 2025)
# ═══════════════════════════════════════════════════════════════════════════════

# Default set: Representative models across all families (single GPU capable)
MODELS=(
    # Original Fingerprint² models (verified working)
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    "google/paligemma-3b-mix-448"
    "vikhyat/moondream2"
    "OpenGVLab/InternVL2-2B"

    # Latest Qwen Vision (2024-2025)
    "Qwen/Qwen3-VL-2B"

    # Meta Llama Vision
    "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # InternVL Latest
    "OpenGVLab/InternVL3-2B"

    # Mistral Pixtral
    "mistralai/Pixtral-12B-2409"

    # DeepSeek Vision
    "deepseek-ai/deepseek-vl2-tiny"

    # Google Gemma 3 (March 2025)
    "google/gemma-3-4b-it"

    # IDEFICS (Flamingo-style)
    "HuggingFaceM4/idefics2-8b"

    # Microsoft Phi Vision
    "microsoft/Phi-3.5-vision-instruct"

    # LLaVA
    "llava-hf/llava-v1.6-vicuna-7b-hf"
)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL ADDITIONAL MODELS
# ═══════════════════════════════════════════════════════════════════════════════

# Larger models (require multi-GPU or >48GB VRAM):
# "Qwen/Qwen2.5-VL-72B-Instruct"
# "meta-llama/Llama-3.2-90B-Vision-Instruct"
# "google/gemma-3-27b-it"

# Foundation/Embedding models (limited text generation):
# "facebook/flava-full"
# "openflamingo/OpenFlamingo-4B-vitl-rpj3b"

# Models with known issues (100% refusal in initial tests):
# "openbmb/MiniCPM-V-2"
# "microsoft/Florence-2-large"  # Captioning-only, not Q&A

echo "=============================================="
echo "Fingerprint² Full FHIBE Benchmark"
echo "=============================================="
echo "Dataset:    $DATASET_PATH"
echo "Output:     $OUTPUT_DIR"
echo "GPU:        $GPU"
echo "Models:     ${#MODELS[@]}"
echo ""
echo "WARNING: This will process ALL 10,318 images!"
echo "Estimated time: 24-48 hours"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Join models with comma
MODELS_STR=$(IFS=,; echo "${MODELS[*]}")

echo ""
echo "Starting benchmark at $(date)"
echo "=============================================="

# Run the benchmark (NO --sample flag = full dataset)
python scripts/run_fhibe_benchmark.py \
    --dataset "$DATASET_PATH" \
    --models "$MODELS_STR" \
    --output "$OUTPUT_DIR/full_results.json" \
    --html "$OUTPUT_DIR/dashboard.html" \
    --gpu "$GPU" \
    --4bit \
    2>&1 | tee "$OUTPUT_DIR/benchmark.log"

echo ""
echo "=============================================="
echo "Benchmark completed at $(date)"
echo "Results: $OUTPUT_DIR/full_results.json"
echo "Dashboard: $OUTPUT_DIR/dashboard.html"
echo "=============================================="
