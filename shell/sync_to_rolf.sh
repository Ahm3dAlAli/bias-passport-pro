#!/bin/bash
# Sync FingerPrint to Rolf and run AAAI analyses

set -e

echo "========================================================================"
echo "  Syncing FingerPrint to Rolf"
echo "========================================================================"
echo ""

# Configuration
ROLF_USER="alali"
ROLF_HOST="rolf.ifi.uzh.ch"
ROLF_DIR="/local/scratch/alali/FingerPrint"
LOCAL_DIR="/Users/ahmeda./Desktop/FingerPrint"

echo "Local:  $LOCAL_DIR"
echo "Remote: ${ROLF_USER}@${ROLF_HOST}:${ROLF_DIR}"
echo ""

# Check if we can connect to Rolf
echo "Testing connection to Rolf..."
if ! ssh -o ConnectTimeout=5 ${ROLF_USER}@${ROLF_HOST} "echo 'Connected successfully'"; then
    echo "❌ Cannot connect to Rolf"
    echo "Please check:"
    echo "  1. VPN connection (if required)"
    echo "  2. SSH key is set up"
    echo "  3. Username is correct"
    exit 1
fi

echo "✓ Connection successful"
echo ""

# Create directory on Rolf if it doesn't exist
echo "Creating directory on Rolf..."
ssh ${ROLF_USER}@${ROLF_HOST} "mkdir -p ${ROLF_DIR}"

echo ""
echo "========================================================================"
echo "  Syncing files to Rolf..."
echo "========================================================================"
echo ""

# Sync everything except large files
rsync -avz --progress \
    --exclude 'results/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'venv/' \
    --exclude 'env/' \
    ${LOCAL_DIR}/ ${ROLF_USER}@${ROLF_HOST}:${ROLF_DIR}/

echo ""
echo "========================================================================"
echo "  Syncing database files (this may take a while)..."
echo "========================================================================"
echo ""

# Sync results directory (large files)
rsync -avz --progress \
    ${LOCAL_DIR}/results/single_runs_35k/*.db \
    ${ROLF_USER}@${ROLF_HOST}:${ROLF_DIR}/results/single_runs_35k/

echo ""
echo "✓ Sync complete!"
echo ""
echo "========================================================================"
echo "  Files on Rolf"
echo "========================================================================"
echo ""

# List what's on Rolf
ssh ${ROLF_USER}@${ROLF_HOST} "
cd ${ROLF_DIR}
echo 'Directory: $(pwd)'
echo ''
echo 'Analysis scripts:'
ls -lh scripts/*.py | grep -E '(add_statistical|prompt_sensitivity|sample_for|generate_all|extract_qualitative)'
echo ''
echo 'Documentation:'
ls -lh *.md | head -5
echo ''
echo 'Database files:'
ls -lh results/single_runs_35k/*.db 2>/dev/null | wc -l | xargs echo 'Found' && echo 'database files'
echo ''
"

echo "========================================================================"
echo "  NEXT STEPS"
echo "========================================================================"
echo ""
echo "1. SSH to Rolf:"
echo "   ssh ${ROLF_USER}@${ROLF_HOST}"
echo ""
echo "2. Navigate to project:"
echo "   cd ${ROLF_DIR}"
echo ""
echo "3. Run analyses:"
echo "   ./RUN_AAAI_ANALYSIS.sh"
echo ""
echo "OR run in one command:"
echo "   ssh ${ROLF_USER}@${ROLF_HOST} 'cd ${ROLF_DIR} && ./RUN_AAAI_ANALYSIS.sh'"
echo ""
echo "4. Sync results back when done:"
echo "   ./sync_from_rolf.sh"
echo ""
echo "========================================================================"
