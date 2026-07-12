#!/bin/bash
# Sync results back from Rolf after analysis

set -e

echo "========================================================================"
echo "  Syncing Results from Rolf"
echo "========================================================================"
echo ""

# Configuration
ROLF_USER="alali"
ROLF_HOST="rolf.ifi.uzh.ch"
ROLF_DIR="/local/scratch/alali/FingerPrint"
LOCAL_DIR="/Users/ahmeda./Desktop/FingerPrint"

echo "Remote: ${ROLF_USER}@${ROLF_HOST}:${ROLF_DIR}"
echo "Local:  $LOCAL_DIR"
echo ""

# Check connection
echo "Testing connection to Rolf..."
if ! ssh -o ConnectTimeout=5 ${ROLF_USER}@${ROLF_HOST} "echo 'Connected'"; then
    echo "❌ Cannot connect to Rolf"
    exit 1
fi
echo "✓ Connected"
echo ""

# Check if results exist on Rolf
echo "Checking for results on Rolf..."
if ! ssh ${ROLF_USER}@${ROLF_HOST} "test -d ${ROLF_DIR}/results/aaai_submission"; then
    echo "❌ No results found at ${ROLF_DIR}/results/aaai_submission"
    echo ""
    echo "Did you run the analysis on Rolf?"
    echo "Run: ssh ${ROLF_USER}@${ROLF_HOST} 'cd ${ROLF_DIR} && ./RUN_AAAI_ANALYSIS.sh'"
    exit 1
fi

echo "✓ Results found"
echo ""

# Show what will be synced
echo "Results to sync:"
ssh ${ROLF_USER}@${ROLF_HOST} "ls -lh ${ROLF_DIR}/results/aaai_submission/"
echo ""

echo "Press ENTER to continue, or Ctrl+C to cancel..."
read

echo ""
echo "========================================================================"
echo "  Syncing results..."
echo "========================================================================"
echo ""

# Create local directory
mkdir -p ${LOCAL_DIR}/results/aaai_submission

# Sync results
rsync -avz --progress \
    ${ROLF_USER}@${ROLF_HOST}:${ROLF_DIR}/results/aaai_submission/ \
    ${LOCAL_DIR}/results/aaai_submission/

echo ""
echo "✓ Sync complete!"
echo ""
echo "========================================================================"
echo "  Results synced to local machine"
echo "========================================================================"
echo ""
echo "Output directory: ${LOCAL_DIR}/results/aaai_submission/"
echo ""

# List what was synced
echo "Files synced:"
ls -lh ${LOCAL_DIR}/results/aaai_submission/
echo ""

# Check for key files
echo "Key files:"
if [ -f "${LOCAL_DIR}/results/aaai_submission/statistical_analysis_summary.txt" ]; then
    echo "  ✓ statistical_analysis_summary.txt"
fi
if [ -f "${LOCAL_DIR}/results/aaai_submission/validation_sample.csv" ]; then
    echo "  ✓ validation_sample.csv"
fi
if [ -d "${LOCAL_DIR}/results/aaai_submission/figures" ]; then
    FIG_COUNT=$(ls ${LOCAL_DIR}/results/aaai_submission/figures/*.pdf 2>/dev/null | wc -l)
    echo "  ✓ $FIG_COUNT figures"
fi
if [ -f "${LOCAL_DIR}/results/aaai_submission/qualitative_examples.json" ]; then
    echo "  ✓ qualitative_examples.json"
fi

echo ""
echo "========================================================================"
echo "  NEXT STEPS"
echo "========================================================================"
echo ""
echo "1. Review results:"
echo "   cat results/aaai_submission/statistical_analysis_summary.txt"
echo ""
echo "2. Check figures:"
echo "   open results/aaai_submission/figures/"
echo ""
echo "3. Upload to MTurk:"
echo "   results/aaai_submission/validation_sample.csv"
echo ""
echo "4. Start writing paper!"
echo ""
echo "========================================================================"
