#!/bin/bash
# =============================================================================
# sync_dpe_to_rolf.sh
# =============================================================================
# Sync the Demographic Positional Encoding (DPE) code to rolf.
# Uploads only what the DPE experiment needs — it does NOT touch the existing
# baseline result DBs already on rolf.
#
# Usage:
#   ./sync_dpe_to_rolf.sh
# =============================================================================

set -e

REMOTE_USER="alali"
REMOTE_HOST="rolf.ifi.uzh.ch"
REMOTE_DIR="/local/scratch/alali/FingerPrint"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "Syncing DPE code to rolf"
echo "=============================================="
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
echo ""

# 1. The debiasing module (new package)
rsync -avz \
    "$LOCAL_DIR/fingerprint_squared/debiasing/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/fingerprint_squared/debiasing/"

# 1b. The VLM loader (patched for transformers 5.x: AutoModelForImageTextToText,
#     BitsAndBytesConfig quantization, dtype rename)
rsync -avz \
    "$LOCAL_DIR/fingerprint_squared/models/huggingface_vlm.py" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/fingerprint_squared/models/huggingface_vlm.py"

# 2. The DPE scripts + the run orchestrator.
#    run_fhibe_benchmark.py is included because DPE reuses its client classes and
#    we patched the Idefics2 processor call for transformers 5.x.
rsync -avz \
    "$LOCAL_DIR/scripts/run_dpe_benchmark.py" \
    "$LOCAL_DIR/scripts/compare_dpe_baseline.py" \
    "$LOCAL_DIR/scripts/generate_dpe_paper_tables.py" \
    "$LOCAL_DIR/scripts/run_fhibe_benchmark.py" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/scripts/"

rsync -avz \
    "$LOCAL_DIR/run_dpe_on_rolf.sh" \
    "$LOCAL_DIR/run_dpe_parallel_rolf.sh" \
    "$LOCAL_DIR/run_dpe_alpha_sweep_rolf.sh" \
    "$LOCAL_DIR/run_dpe_ablation_rolf.sh" \
    "$LOCAL_DIR/run_dpe_ablation_one.sh" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo ""
echo "✓ DPE code synced"
echo ""
echo "=============================================="
echo "Next: run the experiment on rolf"
echo "=============================================="
echo ""
echo "  ssh rolf"
echo "  cd ${REMOTE_DIR}"
echo "  chmod +x run_dpe_on_rolf.sh"
echo "  ./run_dpe_on_rolf.sh            # runs in a screen session"
echo ""
echo "Or one-shot from here:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'cd ${REMOTE_DIR} && chmod +x run_dpe_on_rolf.sh && ./run_dpe_on_rolf.sh'"
echo ""
