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
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "Downloading DPE results from rolf"
echo "=============================================="
echo ""

# Pull every results/dpe_* directory produced by run_dpe_on_rolf.sh
rsync -avz --progress \
    --include='dpe_*/' --include='dpe_*/**' \
    --include='dpe/' --include='dpe/**' \
    --exclude='*' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/results/" \
    "$LOCAL_DIR/results/"

echo ""
echo "✓ DPE results downloaded to results/"
echo ""
echo "Comparison figures + summaries are in each results/dpe_*/compare_<model>/ folder:"
find "$LOCAL_DIR/results" -maxdepth 2 -name "dpe_comparison_summary.json" 2>/dev/null | sed 's/^/  /'
echo ""
