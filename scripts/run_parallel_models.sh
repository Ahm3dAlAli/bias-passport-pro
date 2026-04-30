#!/bin/bash
# =============================================================================
# run_parallel_models.sh
# =============================================================================
# Run multiple models in parallel on different GPUs using screen sessions
#
# Usage:
#   ./scripts/run_parallel_models.sh
#
# This will start one model per available GPU, each in its own screen session
# =============================================================================

set -e

DATASET_PATH="/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres"
OUTPUT_BASE="results/parallel_35k_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "$OUTPUT_BASE"

# ═══════════════════════════════════════════════════════════════════════════════
# GPU to MODEL mapping
# Assign one model per GPU
# ═══════════════════════════════════════════════════════════════════════════════

# Format: "GPU_ID:MODEL_NAME"
declare -a GPU_MODELS=(
    # GPU 0
    "0:Qwen/Qwen2.5-VL-3B-Instruct"

    # GPU 1
    "1:OpenGVLab/InternVL2-2B"

    # GPU 2
    "2:OpenGVLab/InternVL3-2B"

    # GPU 3
    "3:google/paligemma-3b-mix-448"

    # GPU 4
    "4:Qwen/Qwen3-VL-2B"

    # GPU 5 - Already running Qwen2.5-VL-3B-Instruct, so use different model
    "5:llava-hf/llava-v1.6-vicuna-7b-hf"

    # GPU 6
    "6:HuggingFaceM4/idefics2-8b"

    # GPU 7
    "7:microsoft/Phi-3.5-vision-instruct"

    # Add more GPUs if available
    # "8:meta-llama/Llama-3.2-11B-Vision-Instruct"
    # "9:mistralai/Pixtral-12B-2409"
)

echo "=============================================="
echo "Fingerprint² Parallel Multi-GPU Evaluation"
echo "=============================================="
echo "Dataset:    $DATASET_PATH"
echo "Output:     $OUTPUT_BASE"
echo "Models:     ${#GPU_MODELS[@]}"
echo ""
echo "GPU assignments:"
for assignment in "${GPU_MODELS[@]}"; do
    gpu=$(echo "$assignment" | cut -d: -f1)
    model=$(echo "$assignment" | cut -d: -f2-)
    echo "  GPU $gpu: $model"
done
echo ""
read -p "Start all models in parallel? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Master log file
MASTER_LOG="$OUTPUT_BASE/master_parallel_log.txt"
echo "Starting parallel evaluation at $(date)" | tee "$MASTER_LOG"
echo "============================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# ═══════════════════════════════════════════════════════════════════════════════
# Launch each model in its own screen session
# ═══════════════════════════════════════════════════════════════════════════════

for assignment in "${GPU_MODELS[@]}"; do
    gpu=$(echo "$assignment" | cut -d: -f1)
    model=$(echo "$assignment" | cut -d: -f2-)

    # Create clean model name for screen session and files
    model_name=$(echo "$model" | sed 's/\//_/g' | sed 's/-/_/g')
    screen_name="fp_gpu${gpu}_${model_name}"

    # Output files
    output_db="$OUTPUT_BASE/gpu${gpu}_${model_name}.db"
    output_json="$OUTPUT_BASE/gpu${gpu}_${model_name}.json"
    output_html="$OUTPUT_BASE/gpu${gpu}_${model_name}_dashboard.html"
    model_log="$OUTPUT_BASE/gpu${gpu}_${model_name}.log"

    echo "Starting GPU $gpu: $model" | tee -a "$MASTER_LOG"
    echo "  Screen: $screen_name" | tee -a "$MASTER_LOG"
    echo "  Output: $output_db" | tee -a "$MASTER_LOG"

    # Kill existing screen with same name if it exists
    screen -S "$screen_name" -X quit 2>/dev/null || true

    # Start screen session for this model
    screen -dmS "$screen_name" bash -c "
        cd /local/scratch/alali/FingerPrint
        echo 'Starting evaluation on GPU $gpu at \$(date)' | tee '$model_log'
        echo 'Model: $model' | tee -a '$model_log'
        echo '========================================' | tee -a '$model_log'
        echo '' | tee -a '$model_log'

        python scripts/run_fhibe_benchmark.py \\
            --dataset '$DATASET_PATH' \\
            --models '$model' \\
            --output '$output_json' \\
            --html '$output_html' \\
            --gpu $gpu \\
            --4bit \\
            2>&1 | tee -a '$model_log'

        exit_code=\$?
        echo '' | tee -a '$model_log'
        echo '========================================' | tee -a '$model_log'
        echo \"Evaluation finished at \$(date)\" | tee -a '$model_log'
        echo \"Exit code: \$exit_code\" | tee -a '$model_log'

        if [ -f '$output_db' ]; then
            result_count=\$(sqlite3 '$output_db' 'SELECT COUNT(*) FROM probe_results;' 2>/dev/null || echo 'N/A')
            echo \"Total results: \$result_count\" | tee -a '$model_log'
        fi

        echo '' | tee -a '$model_log'
        echo 'Press Enter to close this screen session...'
        read
    "

    echo "  ✓ Screen session '$screen_name' started" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # Small delay between launches to avoid resource contention
    sleep 5
done

echo "" | tee -a "$MASTER_LOG"
echo "============================================" | tee -a "$MASTER_LOG"
echo "All models launched!" | tee -a "$MASTER_LOG"
echo "============================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "To monitor progress:" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "List all screen sessions:" | tee -a "$MASTER_LOG"
echo "  screen -ls" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Attach to a specific GPU:" | tee -a "$MASTER_LOG"
for assignment in "${GPU_MODELS[@]}"; do
    gpu=$(echo "$assignment" | cut -d: -f1)
    model=$(echo "$assignment" | cut -d: -f2-)
    model_name=$(echo "$model" | sed 's/\//_/g' | sed 's/-/_/g')
    screen_name="fp_gpu${gpu}_${model_name}"
    echo "  screen -r $screen_name   # GPU $gpu" | tee -a "$MASTER_LOG"
done
echo "" | tee -a "$MASTER_LOG"
echo "Check GPU usage:" | tee -a "$MASTER_LOG"
echo "  watch -n 1 nvidia-smi" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "View logs:" | tee -a "$MASTER_LOG"
echo "  tail -f $OUTPUT_BASE/gpu*.log" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "============================================" | tee -a "$MASTER_LOG"
