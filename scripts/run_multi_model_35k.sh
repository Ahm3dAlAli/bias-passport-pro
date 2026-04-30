#!/bin/bash
# =============================================================================
# run_multi_model_35k.sh
# =============================================================================
# Run Fingerprint² evaluation on the FULL 35K FHIBE dataset across multiple
# VLM models. This extends the SmolVLM evaluation you already completed.
#
# Usage:
#   ./scripts/run_multi_model_35k.sh [gpu_id]
#
# Example:
#   ./scripts/run_multi_model_35k.sh 5    # Use GPU 5 (default)
#
# =============================================================================

set -e

# Configuration
DATASET_PATH="${1:-/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres}"
GPU="${2:-5}"
OUTPUT_BASE="results/multi_model_35k_$(date +%Y%m%d_%H%M%S)"

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS TO EVALUATE
# Selected models that should work well with 4-bit quantization on a single GPU
# SmolVLM2-2.2B-Instruct already completed (175,945 results in gpu5_20260325.db)
# ═══════════════════════════════════════════════════════════════════════════════

MODELS=(
    # Small efficient models (2-4GB VRAM in 4-bit)
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "OpenGVLab/InternVL2-2B"
    "OpenGVLab/InternVL3-2B"
    "google/paligemma-3b-mix-448"
    "Qwen/Qwen3-VL-2B"

    # Medium models (6-10GB VRAM in 4-bit)
    "llava-hf/llava-v1.6-vicuna-7b-hf"
    "HuggingFaceM4/idefics2-8b"
    "microsoft/Phi-3.5-vision-instruct"

    # Larger models (12-20GB VRAM in 4-bit) - comment out if GPU memory limited
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
    "mistralai/Pixtral-12B-2409"
)

# Models with known compatibility issues (skip for now)
# "deepseek-ai/deepseek-vl2-tiny"  # May require special handling
# "google/gemma-3-4b-it"            # Newer model, verify compatibility first

echo "=============================================="
echo "Fingerprint² Multi-Model 35K Evaluation"
echo "=============================================="
echo "Dataset:    $DATASET_PATH"
echo "Output:     $OUTPUT_BASE"
echo "GPU:        $GPU"
echo "Models:     ${#MODELS[@]}"
echo ""
echo "Models to evaluate:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""
echo "NOTE: SmolVLM2-2.2B-Instruct already completed"
echo "      (175,945 results in gpu5_20260325.db)"
echo ""
echo "Estimated time: 24-48 hours per model"
echo "Total estimated: $(( ${#MODELS[@]} * 36 )) hours (~$((${#MODELS[@]} * 36 / 24)) days)"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Log file for the full run
MASTER_LOG="$OUTPUT_BASE/master_log.txt"

echo "Starting multi-model evaluation at $(date)" | tee "$MASTER_LOG"
echo "============================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# ═══════════════════════════════════════════════════════════════════════════════
# Run each model sequentially
# ═══════════════════════════════════════════════════════════════════════════════

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    model_num=$((i + 1))
    total_models="${#MODELS[@]}"

    # Create clean model name for output files
    model_name=$(echo "$model" | sed 's/\//_/g' | sed 's/-/_/g')
    output_db="$OUTPUT_BASE/${model_name}_35k.db"
    output_json="$OUTPUT_BASE/${model_name}_35k.json"
    output_html="$OUTPUT_BASE/${model_name}_dashboard.html"
    model_log="$OUTPUT_BASE/${model_name}.log"

    echo "" | tee -a "$MASTER_LOG"
    echo "═══════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
    echo "Model $model_num/$total_models: $model" | tee -a "$MASTER_LOG"
    echo "═══════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
    echo "Started: $(date)" | tee -a "$MASTER_LOG"
    echo "Output: $output_db" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # Run the benchmark for this model
    python scripts/run_fhibe_benchmark.py \
        --dataset "$DATASET_PATH" \
        --models "$model" \
        --output "$output_json" \
        --html "$output_html" \
        --gpu "$GPU" \
        --4bit \
        2>&1 | tee "$model_log"

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "" | tee -a "$MASTER_LOG"
        echo "✓ Model $model_num/$total_models completed successfully" | tee -a "$MASTER_LOG"
        echo "  Finished: $(date)" | tee -a "$MASTER_LOG"

        # Count results
        if [ -f "$output_db" ]; then
            result_count=$(sqlite3 "$output_db" "SELECT COUNT(*) FROM probe_results;" 2>/dev/null || echo "N/A")
            echo "  Results: $result_count probe responses" | tee -a "$MASTER_LOG"
        fi
    else
        echo "" | tee -a "$MASTER_LOG"
        echo "✗ Model $model_num/$total_models FAILED (exit code: $exit_code)" | tee -a "$MASTER_LOG"
        echo "  Check log: $model_log" | tee -a "$MASTER_LOG"
    fi

    echo "" | tee -a "$MASTER_LOG"

    # Optional: Add a delay between models to let GPU cool down
    if [ $model_num -lt $total_models ]; then
        echo "Waiting 60 seconds before next model..." | tee -a "$MASTER_LOG"
        sleep 60
    fi
done

echo "" | tee -a "$MASTER_LOG"
echo "============================================" | tee -a "$MASTER_LOG"
echo "All models completed at $(date)" | tee -a "$MASTER_LOG"
echo "============================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Results summary:" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Generate summary table
for db in "$OUTPUT_BASE"/*.db; do
    if [ -f "$db" ]; then
        db_name=$(basename "$db")
        model_name=$(echo "$db_name" | sed 's/_35k\.db$//')
        result_count=$(sqlite3 "$db" "SELECT COUNT(*) FROM probe_results;" 2>/dev/null || echo "0")
        image_count=$(sqlite3 "$db" "SELECT COUNT(DISTINCT image_id) FROM probe_results;" 2>/dev/null || echo "0")

        printf "  %-50s: %7s results (%s images)\n" "$model_name" "$result_count" "$image_count" | tee -a "$MASTER_LOG"
    fi
done

echo "" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "Output directory: $OUTPUT_BASE" | tee -a "$MASTER_LOG"
echo "============================================" | tee -a "$MASTER_LOG"
