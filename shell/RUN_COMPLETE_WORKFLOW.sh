#!/bin/bash
# Complete workflow: Sync to Rolf, Run, and Sync back

set -e

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                                                                   ║"
echo "║          COMPLETE AAAI ANALYSIS WORKFLOW                          ║"
echo "║                                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
ROLF_USER="alali"
ROLF_HOST="rolf.ifi.uzh.ch"
ROLF_DIR="/local/scratch/alali/FingerPrint"
LOCAL_DIR="/Users/ahmeda./Desktop/FingerPrint"

echo "This will:"
echo "  1. Sync files to Rolf"
echo "  2. Run analysis on Rolf"
echo "  3. Sync results back"
echo ""
echo "Press ENTER to continue, or Ctrl+C to cancel..."
read

# Step 1: Sync to Rolf
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 1/3: Syncing to Rolf"
echo "════════════════════════════════════════════════════════════════════"
echo ""

./sync_to_rolf.sh

if [ $? -ne 0 ]; then
    echo "❌ Sync to Rolf failed!"
    exit 1
fi

# Step 2: Run on Rolf
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 2/3: Running Analysis on Rolf"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "This will take 15-30 minutes..."
echo ""

ssh ${ROLF_USER}@${ROLF_HOST} "cd ${ROLF_DIR} && ./RUN_AAAI_ANALYSIS.sh"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Analysis failed on Rolf!"
    echo ""
    echo "You can:"
    echo "  1. SSH to Rolf manually and check: ssh ${ROLF_USER}@${ROLF_HOST}"
    echo "  2. Run locally instead: ./RUN_AAAI_ANALYSIS.sh"
    exit 1
fi

# Step 3: Sync results back
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 3/3: Syncing Results Back"
echo "════════════════════════════════════════════════════════════════════"
echo ""

./sync_from_rolf.sh

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                                                                   ║"
echo "║          ✅ COMPLETE! All analyses finished                       ║"
echo "║                                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results are in: ${LOCAL_DIR}/results/aaai_submission/"
echo ""
echo "Next steps:"
echo "  1. Review: cat results/aaai_submission/statistical_analysis_summary.txt"
echo "  2. View figures: open results/aaai_submission/figures/"
echo "  3. MTurk: results/aaai_submission/validation_sample.csv"
echo ""
