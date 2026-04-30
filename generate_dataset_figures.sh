#!/bin/bash
# Generate dataset visualization figures

DATASET_PATH="/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres"
OUTPUT_DIR="figures"

echo "=============================================="
echo "Generating FHIBE Dataset Visualizations"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run visualization script (simple version - no CSV dependency)
python3 scripts/visualize_dataset_simple.py \
    --dataset "$DATASET_PATH" \
    --output "$OUTPUT_DIR" \
    --samples 24 \
    --seed 42

echo ""
echo "=============================================="
echo "✓ Figures saved to $OUTPUT_DIR/"
echo "=============================================="
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.png
