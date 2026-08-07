#!/bin/bash
# Generate all publication-quality figures

RESULTS_DIR="results/single_runs_35k"
OUTPUT_DIR="figures/publication"

echo "=============================================="
echo "Generating Publication Figures"
echo "=============================================="
echo ""

mkdir -p "$OUTPUT_DIR"

python3 scripts/generate_publication_figures.py \
    --results "$RESULTS_DIR" \
    --output "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "✓ Figures saved to $OUTPUT_DIR/"
echo "=============================================="
echo ""
ls -lh "$OUTPUT_DIR"/*.png
