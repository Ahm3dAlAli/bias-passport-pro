#!/bin/bash
# =============================================================================
# sync_and_run_rolf.sh
# =============================================================================
# Sync Fingerprint² scripts to ROLF and run the full benchmark
#
# Usage:
#   ./sync_and_run_rolf.sh                    # Sync only
#   ./sync_and_run_rolf.sh --setup            # Sync and setup environment (first time)
#   ./sync_and_run_rolf.sh --run              # Sync and run benchmark
#   ./sync_and_run_rolf.sh --run --lightweight # Run lightweight models only
#   ./sync_and_run_rolf.sh --run --gpu 7      # Run on specific GPU
# =============================================================================

set -e

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Update these for your setup
# ═══════════════════════════════════════════════════════════════════════════════

ROLF_HOST="rolf"                              # SSH host alias (from ~/.ssh/config)
ROLF_USER="alali"                            # Your ROLF username (for display only)
REMOTE_DIR="/local/scratch/alali/FingerPrint" # Remote project directory
DATASET_PATH="/local/scratch/alali/fhibe_data"  # FHIBE dataset on ROLF
LOCAL_DIR="/Users/ahmeda./Desktop/FingerPrint"

# Parse arguments
RUN_BENCHMARK=false
SETUP_ENV=false
LIGHTWEIGHT=false
GPU=0
MODELS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --setup)
            SETUP_ENV=true
            shift
            ;;
        --run)
            RUN_BENCHMARK=true
            shift
            ;;
        --lightweight)
            LIGHTWEIGHT=true
            shift
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Fingerprint² ROLF Sync & Run"
echo "=============================================="
echo "Local:      $LOCAL_DIR"
echo "Remote:     ${ROLF_HOST}:${REMOTE_DIR}"
echo "Dataset:    $DATASET_PATH"
echo "Run:        $RUN_BENCHMARK"
echo "Lightweight: $LIGHTWEIGHT"
echo "GPU:        $GPU"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Sync files to ROLF
# ═══════════════════════════════════════════════════════════════════════════════

echo "[1/3] Syncing files to ROLF..."

# Create remote directory if it doesn't exist
ssh ${ROLF_HOST} "mkdir -p ${REMOTE_DIR}/scripts ${REMOTE_DIR}/results"

# Sync scripts
rsync -avz --progress \
    ${LOCAL_DIR}/scripts/run_fhibe_benchmark.py \
    ${LOCAL_DIR}/scripts/run_full_evaluation.py \
    ${LOCAL_DIR}/scripts/run_full_benchmark.sh \
    ${LOCAL_DIR}/scripts/setup_rolf.sh \
    ${LOCAL_DIR}/scripts/extract_fullres_metadata.py \
    ${ROLF_HOST}:${REMOTE_DIR}/scripts/

# Sync fingerprint_squared module if it exists
if [ -d "${LOCAL_DIR}/fingerprint_squared" ]; then
    rsync -avz --progress \
        ${LOCAL_DIR}/fingerprint_squared/ \
        ${ROLF_HOST}:${REMOTE_DIR}/fingerprint_squared/
fi

# Sync requirements if exists
if [ -f "${LOCAL_DIR}/requirements.txt" ]; then
    rsync -avz ${LOCAL_DIR}/requirements.txt \
        ${ROLF_HOST}:${REMOTE_DIR}/
fi

echo "[OK] Files synced successfully"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Setup environment (if requested) or check it
# ═══════════════════════════════════════════════════════════════════════════════

if [ "$SETUP_ENV" = true ]; then
    echo "[2/3] Setting up environment on ROLF (this may take a while)..."
    ssh -t ${ROLF_HOST} "cd ${REMOTE_DIR} && bash scripts/setup_rolf.sh"
    echo "[OK] Environment setup complete"
else
    echo "[2/3] Checking remote environment..."

    ssh ${ROLF_HOST} << REMOTE_CHECK
echo "Python version:"
python3 --version

echo ""
echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

echo ""
echo "Checking venv and packages..."
if [ -d "${REMOTE_DIR}/venv" ]; then
    source ${REMOTE_DIR}/venv/bin/activate
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
else
    echo "Virtual environment not found. Run with --setup to create it."
fi
REMOTE_CHECK

    echo "[OK] Environment check complete"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Run benchmark (if requested)
# ═══════════════════════════════════════════════════════════════════════════════

if [ "$RUN_BENCHMARK" = true ]; then
    echo "[3/3] Starting benchmark on ROLF..."

    # Build the command - activate venv first
    CMD="cd ${REMOTE_DIR} && source venv/bin/activate && CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/run_full_evaluation.py"
    CMD="${CMD} --dataset ${DATASET_PATH}"
    CMD="${CMD} --output results/full_evaluation_$(date +%Y%m%d_%H%M%S)"

    if [ "$LIGHTWEIGHT" = true ]; then
        CMD="${CMD} --lightweight"
    fi

    if [ -n "$MODELS" ]; then
        CMD="${CMD} --models ${MODELS}"
    fi

    echo "Running command:"
    echo "  $CMD"
    echo ""

    # Run in screen/tmux for persistence
    ssh ${ROLF_HOST} << REMOTE_RUN
# Create a screen session for the benchmark
screen -dmS fingerprint_benchmark bash -c '
    cd ${REMOTE_DIR}
    echo "Starting Fingerprint² benchmark at \$(date)"
    echo "Command: $CMD"
    echo ""
    $CMD 2>&1 | tee results/benchmark_\$(date +%Y%m%d_%H%M%S).log
    echo ""
    echo "Benchmark completed at \$(date)"
'
echo "Benchmark started in screen session 'fingerprint_benchmark'"
echo ""
echo "To monitor progress:"
echo "  ssh ${ROLF_HOST}"
echo "  screen -r fingerprint_benchmark"
echo ""
echo "To detach from screen: Ctrl+A, then D"
REMOTE_RUN

    echo "[OK] Benchmark started on ROLF"
else
    echo "[3/3] Skipping benchmark run (use --run to start)"
fi

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
echo ""
echo "Useful commands:"
echo "  # Check benchmark status"
echo "  ssh ${ROLF_HOST} 'screen -r fingerprint_benchmark'"
echo ""
echo "  # List running screens"
echo "  ssh ${ROLF_HOST} 'screen -ls'"
echo ""
echo "  # Download results when complete"
echo "  rsync -avz ${ROLF_HOST}:${REMOTE_DIR}/results/ ${LOCAL_DIR}/results/"
echo ""
