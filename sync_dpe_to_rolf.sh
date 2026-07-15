#!/bin/bash
# =============================================================================
# sync_dpe_to_rolf.sh
# =============================================================================
# Sync the Demographic Positional Encoding (DPE) code to rolf.
# Uploads only what the DPE experiment needs — it does NOT touch the existing
# baseline result DBs already on rolf.
#
# Uses ONE multiplexed SSH connection (ControlMaster), so you enter your OTP
# only ONCE for the whole sync instead of per-transfer.
#
# Usage:
#   ./sync_dpe_to_rolf.sh
# =============================================================================

set -e

REMOTE_USER="alali"
REMOTE_HOST="rolf.ifi.uzh.ch"
REMOTE_DIR="/local/scratch/alali/FingerPrint"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

# --- SSH connection multiplexing: authenticate ONCE, reuse for all rsync ------
CTRL_PATH="/tmp/cm_dpe_%C"          # %C = short hash of host/port/user
SSH_OPTS="-o ControlMaster=auto -o ControlPath=${CTRL_PATH} -o ControlPersist=300"

cleanup_master() {
    ssh ${SSH_OPTS} -O exit "${REMOTE}" 2>/dev/null || true
}
trap cleanup_master EXIT

echo "=============================================="
echo "Syncing DPE code to rolf"
echo "=============================================="
echo "Remote: ${REMOTE}:${REMOTE_DIR}"
echo ""
echo ">>> Opening one SSH connection — enter your OTP once. <<<"
# Establish the master connection (this is the only auth prompt).
ssh ${SSH_OPTS} "${REMOTE}" true
echo "✓ Connection established (reused for all transfers)"
echo ""

# All rsync calls reuse the master connection via -e "ssh $SSH_OPTS".
RSYNC_SSH="ssh ${SSH_OPTS}"

# 1. The debiasing module (new package)
rsync -avz -e "$RSYNC_SSH" \
    "$LOCAL_DIR/fingerprint_squared/debiasing/" \
    "${REMOTE}:${REMOTE_DIR}/fingerprint_squared/debiasing/"

# 1b. The VLM loader (patched for transformers 5.x)
rsync -avz -e "$RSYNC_SSH" \
    "$LOCAL_DIR/fingerprint_squared/models/huggingface_vlm.py" \
    "${REMOTE}:${REMOTE_DIR}/fingerprint_squared/models/huggingface_vlm.py"

# 2. The DPE scripts (+ run_fhibe_benchmark.py — DPE reuses its client classes
#    and we patched the Idefics2 processor call + InternVL loader for tf 5.x).
rsync -avz -e "$RSYNC_SSH" \
    "$LOCAL_DIR/scripts/run_dpe_benchmark.py" \
    "$LOCAL_DIR/scripts/compare_dpe_baseline.py" \
    "$LOCAL_DIR/scripts/generate_dpe_paper_tables.py" \
    "$LOCAL_DIR/scripts/run_fhibe_benchmark.py" \
    "${REMOTE}:${REMOTE_DIR}/scripts/"

# 3. The orchestration scripts
rsync -avz -e "$RSYNC_SSH" \
    "$LOCAL_DIR/run_dpe_on_rolf.sh" \
    "$LOCAL_DIR/run_dpe_parallel_rolf.sh" \
    "$LOCAL_DIR/run_dpe_alpha_sweep_rolf.sh" \
    "$LOCAL_DIR/run_dpe_ablation_rolf.sh" \
    "$LOCAL_DIR/run_dpe_ablation_one.sh" \
    "${REMOTE}:${REMOTE_DIR}/"

echo ""
echo "✓ DPE code synced (single connection)"
echo ""
echo "=============================================="
echo "Next: run the ablation on rolf (on FREE GPUs)"
echo "=============================================="
echo "  ssh rolf && cd ${REMOTE_DIR}"
echo "  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader   # pick free GPUs"
echo "  GPUS=\"0 1\" ./run_dpe_ablation_rolf.sh"
echo ""
