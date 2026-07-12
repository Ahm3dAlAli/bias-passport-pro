#!/bin/bash
# Check what data is actually on rolf

echo "======================================================================"
echo "Checking Data on Rolf"
echo "======================================================================"
echo ""

ssh rolf << 'EOF'
cd /local/scratch/alali/FingerPrint

echo "=== Results directory structure ==="
ls -lah results/ 2>/dev/null || echo "No results directory found"
echo ""

echo "=== Looking for .db files ==="
find results/ -name "*.db" -type f 2>/dev/null | head -20
echo ""

echo "=== Checking single_runs_35k ==="
if [ -d "results/single_runs_35k" ]; then
    echo "Directory exists. Contents:"
    ls -lh results/single_runs_35k/
else
    echo "Directory does not exist"
fi
echo ""

echo "=== All subdirectories in results/ ==="
find results/ -maxdepth 2 -type d 2>/dev/null
echo ""
EOF
