#!/bin/bash
# =============================================================================
# sync_dpe_from_rolf.sh
# =============================================================================
# Download DPE result directories (DBs, figures, summary JSONs) from rolf
# back to your local machine.
#
# Usage:
#   ./sync_dpe_from_rolf.sh
# =============================================================================

set -e

REMOTE_USER="alali"
REMOTE_HOST="rolf.ifi.uzh.ch"
REMOTE_DIR="/local/scratch/alali/FingerPrint"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # repo root (script now in shell/)

# --- SSH: authenticate ONCE, and avoid "Too many authentication failures" -----
# IdentitiesOnly=yes stops ssh from offering every key in the agent (which
# exhausts the server's auth-attempt limit before you reach the OTP prompt).
# ControlMaster multiplexes one connection so a single OTP covers every rsync.
CTRL_PATH="/tmp/cm_dpe_%C"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=${CTRL_PATH} -o ControlPersist=300 \
          -o IdentitiesOnly=yes -o PreferredAuthentications=keyboard-interactive,password \
          -o NumberOfPasswordPrompts=3"

echo "=============================================="
echo "Downloading DPE results from rolf"
echo "=============================================="
echo ">>> Opening one SSH connection — enter your OTP once. <<<"
echo ""

# open (or reuse) the master connection; this is where you type the OTP
ssh ${SSH_OPTS} "${REMOTE}" true

# Pull every results/dpe_* directory produced by the DPE runs
rsync -avz --progress -e "ssh ${SSH_OPTS}" \
    --include='dpe_*/' --include='dpe_*/**' \
    --include='dpe/' --include='dpe/**' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_DIR}/results/" \
    "$LOCAL_DIR/results/"

# close the shared connection
ssh ${SSH_OPTS} -O exit "${REMOTE}" 2>/dev/null || true

echo ""
echo "✓ DPE results downloaded to results/"
echo ""
echo "Comparison figures + summaries are in each results/dpe_*/compare_<model>/ folder:"
find "$LOCAL_DIR/results" -maxdepth 2 -name "dpe_comparison_summary.json" 2>/dev/null | sed 's/^/  /'
echo ""
