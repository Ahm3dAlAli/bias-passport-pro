#!/bin/bash
# Complete workflow: sync, generate, and download figures

set -e

echo "======================================================================"
echo "FingerPrint² Figure Generation Workflow"
echo "======================================================================"
echo ""

# Step 1: Upload scripts
echo "Step 1: Uploading scripts to rolf..."
rsync -avz --progress \
    scripts/generate_paper_style_figures.py \
    generate_paper_figures.sh \
    rolf:/local/scratch/alali/FingerPrint/

echo ""
echo "✓ Scripts uploaded"
echo ""

# Step 2: Generate figures on rolf
echo "Step 2: Generating figures on rolf..."
ssh rolf << 'EOF'
cd /local/scratch/alali/FingerPrint
chmod +x generate_paper_figures.sh
./generate_paper_figures.sh
EOF

echo ""
echo "✓ Figure generation complete"
echo ""

# Step 3: Download results
echo "Step 3: Downloading figures to local machine..."
mkdir -p figures/paper_style
rsync -avz --progress \
    rolf:/local/scratch/alali/FingerPrint/figures/paper_style/ \
    ./figures/paper_style/

echo ""
echo "======================================================================"
echo "✅ Complete! Figures downloaded to figures/paper_style/"
echo "======================================================================"
echo ""
ls -lh figures/paper_style/
echo ""
echo "View metadata:"
echo "  cat figures/paper_style/dataset_metadata.txt"
echo ""
