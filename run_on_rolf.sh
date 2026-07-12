#!/bin/bash
# Run AAAI analyses on Rolf (optimized for cluster)

set -e

echo "========================================================================"
echo "  Running AAAI Analysis on Rolf"
echo "========================================================================"
echo ""

# Check if we're on Rolf
if [[ ! $(hostname) == *"rolf"* ]]; then
    echo "⚠️  This script should be run ON Rolf, not locally"
    echo ""
    echo "Do you want to:"
    echo "  1. Sync to Rolf and run there"
    echo "  2. Run SSH command to execute on Rolf"
    echo "  3. Exit"
    echo ""
    read -p "Choice (1/2/3): " choice

    case $choice in
        1)
            echo "Running sync_to_rolf.sh first..."
            ./sync_to_rolf.sh
            echo ""
            echo "Now SSH to Rolf and run:"
            echo "  cd /local/scratch/alali/FingerPrint"
            echo "  ./run_on_rolf.sh"
            exit 0
            ;;
        2)
            ROLF_USER="alali"
            ROLF_HOST="rolf.ifi.uzh.ch"
            ROLF_DIR="/local/scratch/alali/FingerPrint"

            echo "Executing on Rolf via SSH..."
            ssh ${ROLF_USER}@${ROLF_HOST} "cd ${ROLF_DIR} && ./run_on_rolf.sh"
            exit 0
            ;;
        3)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
fi

# We're on Rolf - proceed with analysis
echo "✓ Running on Rolf"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Check for conda/python
echo "Checking Python environment..."
if command -v conda &> /dev/null; then
    echo "✓ Conda found"

    # Check if fingerprint env exists
    if conda env list | grep -q "fingerprint"; then
        echo "✓ Activating fingerprint environment"
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate fingerprint
    else
        echo "⚠️  fingerprint environment not found"
        echo "Using base Python: $(which python3)"
    fi
else
    echo "Using system Python: $(which python3)"
fi

python3 --version
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import scipy, pandas, numpy, matplotlib, seaborn" 2>/dev/null && echo "✓ All packages available" || {
    echo "⚠️  Missing packages. Installing..."
    pip install --user scipy pandas numpy matplotlib seaborn statsmodels
}
echo ""

# Check for database files
echo "Checking for database files..."
DB_COUNT=$(find results/single_runs_35k -name "*.db" 2>/dev/null | wc -l)
echo "Found $DB_COUNT database files"

if [ "$DB_COUNT" -eq 0 ]; then
    echo "❌ No database files found!"
    echo "Please check: results/single_runs_35k/"
    exit 1
fi
echo ""

# Set output directory
OUTPUT_DIR="results/aaai_submission"
mkdir -p $OUTPUT_DIR

echo "========================================================================"
echo "  Starting Analysis"
echo "========================================================================"
echo ""
echo "Output will be saved to: $OUTPUT_DIR"
echo ""

# Run the master analysis script
python3 scripts/run_all_aaai_analyses.py \
    --results-dir results/single_runs_35k \
    --output-dir $OUTPUT_DIR \
    --validation-samples 200

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ ANALYSIS COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh $OUTPUT_DIR/
    echo ""

    if [ -d "$OUTPUT_DIR/figures" ]; then
        echo "Figures:"
        ls -lh $OUTPUT_DIR/figures/
        echo ""
    fi

    echo "========================================================================"
    echo "  NEXT STEPS"
    echo "========================================================================"
    echo ""
    echo "1. Review results on Rolf:"
    echo "   cat $OUTPUT_DIR/statistical_analysis_summary.txt"
    echo ""
    echo "2. Sync results back to local machine:"
    echo "   exit  # (leave Rolf)"
    echo "   ./sync_from_rolf.sh  # (on local machine)"
    echo ""
    echo "3. Or download specific files:"
    echo "   scp -r ahmeda@rolf.cs.washington.edu:$(pwd)/$OUTPUT_DIR ."
    echo ""
    echo "========================================================================"
else
    echo ""
    echo "❌ Analysis failed!"
    echo "Check error messages above"
    exit 1
fi
