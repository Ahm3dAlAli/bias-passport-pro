#!/bin/bash
# Master script to run all AAAI analyses
# This generates everything needed for your paper submission

set -e  # Exit on error

echo "========================================================================"
echo "  FingerPrint AAAI Submission Analysis Pipeline"
echo "========================================================================"
echo ""
echo "This will generate:"
echo "  ✓ Statistical analysis (p-values, effect sizes, CIs)"
echo "  ✓ Prompt sensitivity analysis"
echo "  ✓ Human validation sample (for MTurk)"
echo "  ✓ All publication figures (AAAI-formatted)"
echo "  ✓ Qualitative examples"
echo ""
echo "Estimated time: 15-30 minutes"
echo "========================================================================"
echo ""

# Configuration
RESULTS_DIR="results/single_runs_35k"
OUTPUT_DIR="results/aaai_submission"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Error: Results directory not found: $RESULTS_DIR"
    echo "Please update RESULTS_DIR in this script to point to your .db files"
    exit 1
fi

# Count database files
DB_COUNT=$(find "$RESULTS_DIR" -name "*.db" | wc -l | tr -d ' ')
echo "Found $DB_COUNT database files in $RESULTS_DIR"

if [ "$DB_COUNT" -eq 0 ]; then
    echo "❌ Error: No .db files found in $RESULTS_DIR"
    exit 1
fi

echo ""
echo "Press ENTER to start, or Ctrl+C to cancel..."
read

# Run the master analysis script
python3 scripts/run_all_aaai_analyses.py \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --validation-samples 200

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ ANALYSIS COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - statistical_analysis.json       (all statistics)"
    echo "  - statistical_analysis_summary.txt (copy-paste for paper)"
    echo "  - sensitivity/                     (prompt robustness)"
    echo "  - validation_sample.csv            (upload to MTurk, ~\$90)"
    echo "  - figures/                         (all publication figures)"
    echo "  - qualitative_examples.json        (example responses)"
    echo ""
    echo "NEXT STEPS:"
    echo "  1. Review statistical_analysis_summary.txt"
    echo "  2. Upload validation_sample.csv to MTurk"
    echo "  3. Use figures/ in your paper"
    echo "  4. Start writing (see READY_FOR_AAAI_CHECKLIST.md)"
    echo ""
    echo "Timeline to submission: 3-4 weeks"
    echo "========================================================================"
else
    echo ""
    echo "❌ Analysis failed. Check error messages above."
    exit 1
fi
