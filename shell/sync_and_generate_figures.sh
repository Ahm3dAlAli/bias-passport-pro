#!/bin/bash
# Sync code to rolf and generate research figures

echo "=============================================="
echo "Sync to Rolf and Generate Figures"
echo "=============================================="
echo ""

# 1. Sync the new scripts to rolf
echo "1. Syncing scripts to rolf..."
rsync -avz \
  scripts/generate_research_figures.py \
  generate_research_figures.sh \
  RESEARCH_FIGURES.md \
  rolf:/local/scratch/alali/FingerPrint/

echo ""
echo "2. Running figure generation on rolf..."
ssh rolf << 'EOF'
cd /local/scratch/alali/FingerPrint
chmod +x generate_research_figures.sh
./generate_research_figures.sh
EOF

echo ""
echo "3. Syncing generated figures back to local..."
mkdir -p figures/research
rsync -avz rolf:/local/scratch/alali/FingerPrint/figures/research/ ./figures/research/

echo ""
echo "=============================================="
echo "✓ Complete!"
echo "=============================================="
echo ""
echo "Generated figures:"
ls -lh figures/research/*.pdf 2>/dev/null || echo "No PDF files found"
