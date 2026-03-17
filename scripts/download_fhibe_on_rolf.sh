#!/bin/bash
# ============================================================================
# Download FHIBE Dataset on rolf
# ============================================================================
#
# This script downloads the FHIBE dataset from Sony AI directly to rolf's
# local scratch storage (fast SSD, 25TB available).
#
# Usage:
#   ssh alali@rolf.ifi.uzh.ch
#   screen -S fhibe_download
#   bash /local/scratch/alali/fingerprint_squared/FingerPrint/scripts/download_fhibe_on_rolf.sh
#
# ============================================================================

set -e

# Configuration
USERNAME=$(whoami)
DATA_DIR="/local/scratch/${USERNAME}/fhibe_data"
DOWNLOAD_URL="https://fairnessbenchmark.ai.sony/api/fhibe/get-dataset-download-redirect"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           FHIBE Dataset Download for rolf                     ║"
echo "║           Size: ~178 GB (downsampled version)                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo "Data directory: ${DATA_DIR}"
echo ""

# Create directory
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

# Set group permissions for AIML team
chgrp -R aiml "${DATA_DIR}" 2>/dev/null || true
chmod -R 775 "${DATA_DIR}" 2>/dev/null || true

# Check if already downloaded
if [ -f "${DATA_DIR}/fhibe_metadata.json" ] || [ -d "${DATA_DIR}/images" ]; then
    echo -e "${GREEN}FHIBE dataset already exists at ${DATA_DIR}${NC}"
    echo "To re-download, remove the directory first:"
    echo "  rm -rf ${DATA_DIR}"
    exit 0
fi

echo -e "${YELLOW}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  IMPORTANT: You need a valid download token from Sony        ║"
echo "║                                                              ║"
echo "║  1. Go to: https://fairnessbenchmark.ai.sony/download        ║"
echo "║  2. Request access and get approved                          ║"
echo "║  3. Copy the download URL with token                         ║"
echo "║  4. Paste it when prompted below                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo "Paste your FHIBE download URL (with token):"
read -r DOWNLOAD_URL

if [ -z "$DOWNLOAD_URL" ]; then
    echo -e "${RED}Error: No URL provided${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Starting download...${NC}"
echo "This will take a while (~178 GB). Run in a screen session!"
echo ""

# Download using curl with progress
curl -L -X GET "${DOWNLOAD_URL}" \
    -o "fhibe_downsampled.tar.gz" \
    --progress-bar

echo ""
echo -e "${GREEN}Download complete! Extracting...${NC}"
echo ""

# Extract
tar -xzf fhibe_downsampled.tar.gz --checkpoint=.1000

# Clean up archive to save space
echo ""
echo "Cleaning up archive..."
rm -f fhibe_downsampled.tar.gz

# Verify extraction
echo ""
echo "Verifying dataset structure..."

if [ -d "data/raw" ]; then
    IMAGE_COUNT=$(find data/raw -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
    echo -e "${GREEN}✓ Found ${IMAGE_COUNT} images in data/raw/${NC}"
elif [ -d "images" ]; then
    IMAGE_COUNT=$(find images -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
    echo -e "${GREEN}✓ Found ${IMAGE_COUNT} images in images/${NC}"
else
    echo -e "${YELLOW}Warning: Could not find images directory${NC}"
    echo "Contents of ${DATA_DIR}:"
    ls -la
fi

echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 FHIBE Download Complete!                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Dataset location: ${DATA_DIR}                               "
echo "║                                                              ║"
echo "║  Next steps:                                                 ║"
echo "║  1. Run the benchmark:                                       ║"
echo "║     cd /local/scratch/alali/fingerprint_squared/FingerPrint  ║"
echo "║     export OPENROUTER_API_KEY='your-key'                     ║"
echo "║     python scripts/run_fhibe_benchmark.py \\                 ║"
echo "║         --dataset ${DATA_DIR} \\                             "
echo "║         --models gpt-4o,claude-3.5-sonnet \\                 ║"
echo "║         --n-images 500                                       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
