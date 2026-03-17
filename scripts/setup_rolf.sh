#!/bin/bash
# Fingerprint² Setup Script for rolf (UZH AIML Server)
# Run this after SSHing into rolf

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Fingerprint² Setup for rolf (UZH AIML Server)            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Configuration
USERNAME=$(whoami)
PROJECT_DIR="/local/scratch/${USERNAME}/fingerprint_squared"
CONDA_ENV="fingerprint2"

echo ""
echo "Username: ${USERNAME}"
echo "Project directory: ${PROJECT_DIR}"
echo ""

# Step 1: Create project directory on local scratch
echo "[1/6] Creating project directory on /local/scratch..."
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

# Set group permissions for AIML team
chgrp -R aiml "${PROJECT_DIR}" 2>/dev/null || true
chmod -R 775 "${PROJECT_DIR}" 2>/dev/null || true

# Step 2: Check if Conda is installed
echo "[2/6] Checking Conda installation..."
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."

    # Download Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "${PROJECT_DIR}/miniconda"
    rm miniconda.sh

    # Initialize conda
    source "${PROJECT_DIR}/miniconda/etc/profile.d/conda.sh"
    conda init bash

    echo "Please restart your shell or run: source ~/.bashrc"
    echo "Then run this script again."
    exit 0
else
    echo "Conda found: $(conda --version)"
fi

# Step 3: Create conda environment
echo "[3/6] Creating conda environment '${CONDA_ENV}'..."
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Environment '${CONDA_ENV}' already exists. Activating..."
else
    conda create -n "${CONDA_ENV}" python=3.10 -y
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# Step 4: Install PyTorch with CUDA support (for rolf's CUDA 11.3+)
echo "[4/6] Installing PyTorch with CUDA support..."
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Step 5: Clone/setup Fingerprint² project
echo "[5/6] Setting up Fingerprint² project..."
if [ ! -d "${PROJECT_DIR}/FingerPrint" ]; then
    echo "Please copy your FingerPrint project to ${PROJECT_DIR}/"
    echo "You can use scp from your local machine:"
    echo "  scp -r /path/to/FingerPrint ${USERNAME}@rolf.ifi.uzh.ch:${PROJECT_DIR}/"
    echo ""
    echo "Or clone from git if you have a repository."
else
    cd "${PROJECT_DIR}/FingerPrint"
    pip install -e ".[dev]"
fi

# Step 6: Install additional dependencies
echo "[6/6] Installing additional dependencies..."
pip install roboflow rich aiohttp pillow

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Next steps:                                                 ║"
echo "║                                                              ║"
echo "║  1. Copy your project to rolf:                               ║"
echo "║     scp -r ~/Desktop/FingerPrint \\                          ║"
echo "║         ${USERNAME}@rolf.ifi.uzh.ch:${PROJECT_DIR}/          ║"
echo "║                                                              ║"
echo "║  2. Set API keys:                                            ║"
echo "║     export OPENROUTER_API_KEY='your-key'                     ║"
echo "║     export ROBOFLOW_API_KEY='your-key'                       ║"
echo "║                                                              ║"
echo "║  3. Activate environment:                                    ║"
echo "║     conda activate ${CONDA_ENV}                              ║"
echo "║                                                              ║"
echo "║  4. Run benchmark (in a screen session!):                    ║"
echo "║     screen -S fingerprint                                    ║"
echo "║     cd ${PROJECT_DIR}/FingerPrint                            ║"
echo "║     nice -n 20 python scripts/run_roboflow_benchmark.py \\   ║"
echo "║         --models gpt-4o,claude-3.5-sonnet --n-images 100     ║"
echo "║                                                              ║"
echo "║  5. Check GPU usage before running:                          ║"
echo "║     nvidia-smi                                               ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
