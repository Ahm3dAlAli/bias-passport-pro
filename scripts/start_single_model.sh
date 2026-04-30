#!/bin/bash
# =============================================================================
# start_single_model.sh
# =============================================================================
# Start a single model evaluation on a specific GPU in a screen session
#
# Usage:
#   ./scripts/start_single_model.sh <gpu_id> <model_name>
#
# Example:
#   ./scripts/start_single_model.sh 3 "OpenGVLab/InternVL2-2B"
#   ./scripts/start_single_model.sh 4 "Qwen/Qwen3-VL-2B"
#
# =============================================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <gpu_id> <model_name>"
    echo ""
    echo "Example:"
    echo "  $0 3 \"OpenGVLab/InternVL2-2B\""
    echo "  $0 4 \"Qwen/Qwen3-VL-2B\""
    echo ""
    echo "Available models:"
    echo "  Qwen/Qwen2.5-VL-3B-Instruct"
    echo "  OpenGVLab/InternVL2-2B"
    echo "  OpenGVLab/InternVL3-2B"
    echo "  google/paligemma-3b-mix-448"
    echo "  Qwen/Qwen3-VL-2B"
    echo "  llava-hf/llava-v1.6-vicuna-7b-hf"
    echo "  HuggingFaceM4/idefics2-8b"
    echo "  microsoft/Phi-3.5-vision-instruct"
    echo "  meta-llama/Llama-3.2-11B-Vision-Instruct"
    echo "  mistralai/Pixtral-12B-2409"
    exit 1
fi

GPU="$1"
MODEL="$2"

DATASET_PATH="/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres"
OUTPUT_BASE="results/single_runs_35k"

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Create clean model name
model_name=$(echo "$MODEL" | sed 's/\//_/g' | sed 's/-/_/g')
screen_name="fp_gpu${GPU}_${model_name}"

# Output files
timestamp=$(date +%Y%m%d_%H%M%S)
output_db="$OUTPUT_BASE/gpu${GPU}_${model_name}_${timestamp}.db"
output_json="$OUTPUT_BASE/gpu${GPU}_${model_name}_${timestamp}.json"
output_html="$OUTPUT_BASE/gpu${GPU}_${model_name}_${timestamp}.html"
model_log="$OUTPUT_BASE/gpu${GPU}_${model_name}_${timestamp}.log"

echo "=============================================="
echo "Starting Single Model Evaluation"
echo "=============================================="
echo "GPU:       $GPU"
echo "Model:     $MODEL"
echo "Dataset:   $DATASET_PATH"
echo "Screen:    $screen_name"
echo "Output DB: $output_db"
echo "Log:       $model_log"
echo "=============================================="
echo ""

# Kill existing screen with same name if it exists
screen -S "$screen_name" -X quit 2>/dev/null || true

# Start screen session
screen -dmS "$screen_name" bash -c "
    cd /local/scratch/alali/FingerPrint
    echo 'Starting evaluation on GPU $GPU at \$(date)' | tee '$model_log'
    echo 'Model: $MODEL' | tee -a '$model_log'
    echo '========================================' | tee -a '$model_log'
    echo '' | tee -a '$model_log'

    python scripts/run_fhibe_benchmark.py \\
        --dataset '$DATASET_PATH' \\
        --models '$MODEL' \\
        --output '$output_json' \\
        --html '$output_html' \\
        --gpu $GPU \\
        --4bit \\
        2>&1 | tee -a '$model_log'

    exit_code=\$?
    echo '' | tee -a '$model_log'
    echo '========================================' | tee -a '$model_log'
    echo \"Evaluation finished at \$(date)\" | tee -a '$model_log'
    echo \"Exit code: \$exit_code\" | tee -a '$model_log'

    if [ -f '$output_db' ]; then
        result_count=\$(sqlite3 '$output_db' 'SELECT COUNT(*) FROM probe_results;' 2>/dev/null || echo 'N/A')
        image_count=\$(sqlite3 '$output_db' 'SELECT COUNT(DISTINCT image_id) FROM probe_results;' 2>/dev/null || echo 'N/A')
        echo \"Total results: \$result_count\" | tee -a '$model_log'
        echo \"Unique images: \$image_count\" | tee -a '$model_log'
    fi

    echo '' | tee -a '$model_log'
    echo 'Press Enter to close this screen session...'
    read
"

echo "✓ Screen session '$screen_name' started"
echo ""
echo "To attach to this session:"
echo "  screen -r $screen_name"
echo ""
echo "To detach: Ctrl+A, then D"
echo ""
echo "To view log:"
echo "  tail -f $model_log"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"
echo "=============================================="
