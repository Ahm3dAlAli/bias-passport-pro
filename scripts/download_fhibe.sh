#!/bin/bash
# FHIBE Dataset Download Script
# Downloads the downsampled version (~178 GB)

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           FHIBE Dataset Download                             ║"
echo "║           Size: ~178 GB (downsampled version)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Create directory
mkdir -p ./fhibe_data
cd ./fhibe_data

echo ""
echo "Downloading FHIBE dataset..."
echo "This will take a while depending on your internet speed."
echo ""

# Download using curl (more reliable for large files)
curl -L -X GET \
  'https://fairnessbenchmark.ai.sony/api/fhibe/get-dataset-download-redirect?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJidWNrZXQiOiJzYWktZXRoaWNzLWVoY2lkLWZoaWJlLXRpYyIsImtleSI6ImZoaWJlLjIwMjUwNzE2LnUuZ1Q1X3JGVEFfZG93bnNhbXBsZWRfcHVibGljLnRhci5neiIsInVzZXIiOiJhaG1lZGFsaWFobWVkbW9oYW1lZC5hbC1hbGlAdXpoLmNoIiwic2x1ZyI6Ii9kb3dubG9hZCIsImlhdCI6MTc3MjgzNDQwOSwiZXhwIjoxNzcyOTIwODA5fQ.dzaUMAuV-1gfC75ncsy8ZQ7iJUuanssdelAl3mdEVtA' \
  -o 'fhibe_downsampled.tar.gz' \
  --progress-bar

echo ""
echo "Download complete! Extracting..."
echo ""

# Extract
tar -xzf fhibe_downsampled.tar.gz

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Download Complete!                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Dataset extracted to: ./fhibe_data/                         ║"
echo "║                                                              ║"
echo "║  Next steps:                                                 ║"
echo "║  1. Run benchmark:                                           ║"
echo "║     python scripts/run_benchmark.py --dataset ./fhibe_data   ║"
echo "║                                                              ║"
echo "║  2. Or start dashboard:                                      ║"
echo "║     python -m uvicorn fingerprint_squared.api.server:app     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
