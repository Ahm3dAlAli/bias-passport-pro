#!/bin/bash
# Generate NeurIPS-style research figures from FingerPrint² results

RESULTS_DIR="results/single_runs_35k"
OUTPUT_DIR="figures/research"

echo "=============================================="
echo "FingerPrint² Research Figure Generation"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Install required packages if needed
echo "Checking dependencies..."
python3 -c "import seaborn; import scipy" 2>/dev/null || {
    echo "Installing required packages..."
    pip install seaborn scipy --quiet
}

echo ""
echo "Generating research figures..."
echo ""

# Run the visualization script
python3 scripts/generate_research_figures.py \
    --results "$RESULTS_DIR" \
    --output "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "✓ Figures saved to $OUTPUT_DIR/"
echo "=============================================="
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.pdf 2>/dev/null || echo "No PDF files found"

echo ""
echo "To sync from rolf:"
echo "  rsync -avz rolf:/local/scratch/alali/FingerPrint/$OUTPUT_DIR/ ./$OUTPUT_DIR/"
