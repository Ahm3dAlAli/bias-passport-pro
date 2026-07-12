#!/bin/bash
# Generate paper-style figures matching existing publication figures

RESULTS_DIR="results/single_runs_35k"
OUTPUT_DIR="figures/paper_style"

echo "======================================================================"
echo "FingerPrint² Paper-Style Figure Generation"
echo "======================================================================"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "Output directory:  $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

python3 scripts/generate_paper_style_figures.py \
    --results "$RESULTS_DIR" \
    --output "$OUTPUT_DIR"

echo ""
echo "======================================================================"
echo "✓ Figures saved to $OUTPUT_DIR/"
echo "======================================================================"
echo ""
echo "Generated PDF figures:"
ls -lh "$OUTPUT_DIR"/*.pdf

echo ""
echo "Generated PNG figures:"
ls -lh "$OUTPUT_DIR"/*.png

echo ""
echo "Dataset metadata:"
cat "$OUTPUT_DIR/dataset_metadata.txt"
echo ""
