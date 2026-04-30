#!/bin/bash
# =============================================================================
# run_multi_model_rolf.sh
# =============================================================================
# Sync FingerPrint code to rolf and run multi-model 35K evaluation in screen
#
# Usage:
#   ./run_multi_model_rolf.sh           # Just sync
#   ./run_multi_model_rolf.sh --run     # Sync and run in screen
#
# =============================================================================

set -e

REMOTE_USER="alali"
REMOTE_HOST="rolf.ifi.uzh.ch"
REMOTE_DIR="/local/scratch/alali/FingerPrint"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse arguments
RUN_EVALUATION=false
if [[ "$1" == "--run" ]]; then
    RUN_EVALUATION=true
fi

echo "=============================================="
echo "Syncing FingerPrint to rolf..."
echo "=============================================="

# Sync code to rolf (excluding large files and results)
rsync -avz --delete \
    --exclude 'results/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.venv/' \
    --exclude 'venv/' \
    --exclude '*.db' \
    --exclude '*.zip' \
    --exclude 'fingerprint2_overleaf.zip' \
    "$LOCAL_DIR/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo ""
echo "✓ Sync completed"
echo ""

if [ "$RUN_EVALUATION" = true ]; then
    echo "=============================================="
    echo "Starting multi-model evaluation in screen..."
    echo "=============================================="
    echo ""

    # Create screen session and run evaluation
    ssh "${REMOTE_USER}@${REMOTE_HOST}" << 'ENDSSH'
        cd /local/scratch/alali/FingerPrint

        # Kill existing screen session if it exists
        screen -S fingerprint_multimodel -X quit 2>/dev/null || true

        # Create new screen session and run evaluation
        screen -dmS fingerprint_multimodel bash -c '
            cd /local/scratch/alali/FingerPrint
            echo "Starting multi-model evaluation at $(date)"
            echo "=========================================="
            echo ""

            # Run the evaluation script
            ./scripts/run_multi_model_35k.sh /shares/fhibe/fullres 5

            echo ""
            echo "=========================================="
            echo "Multi-model evaluation completed at $(date)"
            echo ""
            echo "Press Enter to close this screen session..."
            read
        '

        echo "✓ Screen session 'fingerprint_multimodel' started"
        echo ""
        echo "To attach to the screen session:"
        echo "  ssh rolf"
        echo "  screen -r fingerprint_multimodel"
        echo ""
        echo "To detach: Ctrl+A, then D"
        echo ""
        echo "To list all screens:"
        echo "  screen -ls"
ENDSSH

    echo ""
    echo "=============================================="
    echo "Evaluation started on rolf in screen session"
    echo "=============================================="
else
    echo "=============================================="
    echo "Sync complete. To run evaluation:"
    echo "  ./run_multi_model_rolf.sh --run"
    echo ""
    echo "Or manually on rolf:"
    echo "  ssh rolf"
    echo "  cd /local/scratch/alali/FingerPrint"
    echo "  screen -S fingerprint_multimodel"
    echo "  ./scripts/run_multi_model_35k.sh /shares/fhibe/fullres 5"
    echo "=============================================="
fi
