#!/bin/bash
# Run Fingerprint² Benchmark on rolf
# Usage: ./run_on_rolf.sh [n_images] [models]
#
# Example:
#   ./run_on_rolf.sh 100 "gpt-4o,claude-3.5-sonnet"
#   ./run_on_rolf.sh 200 "gpt-4o,claude-3.5-sonnet,gemini-1.5-flash"

set -e

# Configuration
USERNAME=$(whoami)
PROJECT_DIR="/local/scratch/${USERNAME}/fingerprint_squared/FingerPrint"
CONDA_ENV="fingerprint2"

# Arguments
N_IMAGES=${1:-50}
MODELS=${2:-"gpt-4o,claude-3.5-sonnet"}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Fingerprint² Benchmark on rolf                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Images: ${N_IMAGES}                                         "
echo "║  Models: ${MODELS}                                           "
echo "╚══════════════════════════════════════════════════════════════╝"

# Check API keys
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "[ERROR] OPENROUTER_API_KEY not set!"
    echo "Run: export OPENROUTER_API_KEY='your-key'"
    exit 1
fi

if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo "[ERROR] ROBOFLOW_API_KEY not set!"
    echo "Run: export ROBOFLOW_API_KEY='your-key'"
    exit 1
fi

# Check GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# Change to project directory
cd "${PROJECT_DIR}"

# Run with nice (low priority to be a good citizen on shared server)
echo "Starting benchmark with nice -n 20 (low priority)..."
echo "Press Ctrl+A d to detach from screen session"
echo ""

nice -n 20 python scripts/run_roboflow_benchmark.py \
    --models "${MODELS}" \
    --n-images "${N_IMAGES}" \
    --dataset face-detection \
    --output "./results"

echo ""
echo "Benchmark complete! Results saved to ./results/"
echo ""
echo "To view dashboard, set up SSH tunnel from your local machine:"
echo "  ssh -L 8000:localhost:8000 ${USERNAME}@rolf.ifi.uzh.ch"
echo "Then run on rolf:"
echo "  python -m uvicorn fingerprint_squared.api.server:app --port 8000"
echo "And open http://localhost:8000 on your local browser"
