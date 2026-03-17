#!/bin/bash
# ============================================================================
# Fingerprint² Complete Setup & Run Script for rolf
# ============================================================================
#
# Usage:
#   1. Copy project to rolf:
#      rsync -avz --progress "/Users/ahmeda./Desktop/FingerPrint/" \
#          alali@rolf.ifi.uzh.ch:/local/scratch/alali/fingerprint_squared/FingerPrint/
#
#   2. SSH into rolf:
#      ssh alali@rolf.ifi.uzh.ch
#
#   3. Run in a screen session:
#      screen -S fingerprint
#      export OPENROUTER_API_KEY="your-key"
#      cd /local/scratch/alali/fingerprint_squared/FingerPrint
#      bash scripts/rolf_setup_and_run.sh
#
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"  # Set before running or edit here

# Benchmark settings
N_IMAGES=50
MODELS="gpt-4o,claude-3.5-sonnet"

# Paths
USERNAME=$(whoami)
BASE_DIR="/local/scratch/${USERNAME}"
PROJECT_DIR="${BASE_DIR}/fingerprint_squared"
CONDA_DIR="${BASE_DIR}/miniconda"
ENV_NAME="fingerprint2"

# ============================================================================
# COLORS
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          Fingerprint² Setup & Run on rolf                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP $1]${NC} $2"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print_header

echo "Username: ${USERNAME}"
echo "Project directory: ${PROJECT_DIR}"
echo ""

# ----------------------------------------------------------------------------
# Step 1: Check/Set API Keys
# ----------------------------------------------------------------------------
print_step "1/7" "Checking API keys..."

if [ -z "$OPENROUTER_API_KEY" ]; then
    print_error "OPENROUTER_API_KEY not set!"
    echo ""
    echo "Please set it before running this script:"
    echo "  export OPENROUTER_API_KEY='sk-or-v1-your-key-here'"
    echo ""
    echo "Get your key at: https://openrouter.ai/keys"
    exit 1
fi

echo "API key configured."

# ----------------------------------------------------------------------------
# Step 2: Create directories
# ----------------------------------------------------------------------------
print_step "2/7" "Creating directories..."

mkdir -p "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/results"
cd "${BASE_DIR}"

# Set group permissions
chgrp -R aiml "${BASE_DIR}" 2>/dev/null || true
chmod -R 775 "${BASE_DIR}" 2>/dev/null || true

echo "Directories created."

# ----------------------------------------------------------------------------
# Step 3: Install Miniconda if needed
# ----------------------------------------------------------------------------
print_step "3/7" "Checking Conda installation..."

if [ -d "${CONDA_DIR}" ]; then
    echo "Conda already installed at ${CONDA_DIR}"
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    echo "System Conda found."
    source $(conda info --base)/etc/profile.d/conda.sh
else
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
    rm /tmp/miniconda.sh
    source "${CONDA_DIR}/etc/profile.d/conda.sh"

    # Add to bashrc
    echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> ~/.bashrc
    echo "Miniconda installed."
fi

# ----------------------------------------------------------------------------
# Step 4: Create/activate conda environment
# ----------------------------------------------------------------------------
print_step "4/7" "Setting up Python environment..."

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' exists. Activating..."
    conda activate "${ENV_NAME}"
else
    echo "Creating environment '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" python=3.10 -y
    conda activate "${ENV_NAME}"

    # Install PyTorch with CUDA
    echo "Installing PyTorch with CUDA support..."
    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
fi

echo "Python environment ready: $(python --version)"

# ----------------------------------------------------------------------------
# Step 5: Setup Fingerprint² project
# ----------------------------------------------------------------------------
print_step "5/7" "Setting up Fingerprint² project..."

cd "${PROJECT_DIR}"

if [ -d "FingerPrint" ] && [ -f "FingerPrint/setup.py" ]; then
    echo "Project found. Installing dependencies..."
    cd FingerPrint
    pip install -e ".[dev]" -q
    pip install aiohttp rich pillow -q
    echo "Dependencies installed."
else
    print_error "Project not found at ${PROJECT_DIR}/FingerPrint"
    echo ""
    echo "Please copy your project from your local machine first:"
    echo ""
    echo "  rsync -avz --progress \"/Users/ahmeda./Desktop/FingerPrint/\" \\"
    echo "      alali@rolf.ifi.uzh.ch:${PROJECT_DIR}/FingerPrint/"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# ----------------------------------------------------------------------------
# Step 6: Check GPU availability
# ----------------------------------------------------------------------------
print_step "6/7" "Checking GPU availability..."

echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""

# Show who's using GPUs
echo "Current GPU users:"
nvidia-smi | tee /dev/stderr | awk '/ C / {print $5}' | xargs -r ps -up 2>/dev/null || echo "No active GPU processes"
echo ""

# ----------------------------------------------------------------------------
# Step 7: Run benchmark
# ----------------------------------------------------------------------------
print_step "7/7" "Running Fingerprint² benchmark..."

cd "${PROJECT_DIR}/FingerPrint"

echo ""
echo "Configuration:"
echo "  - Models: ${MODELS}"
echo "  - Images: ${N_IMAGES}"
echo "  - Using synthetic data (no external dataset required)"
echo ""

# Export API key for the Python script
export OPENROUTER_API_KEY

# Run with nice (low priority) using synthetic data
echo "Running benchmark with synthetic data..."
nice -n 20 python scripts/run_fhibe_benchmark.py \
    --models "${MODELS}" \
    --n-images "${N_IMAGES}" \
    --output "./results"

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Benchmark Complete!                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Results saved to: ${PROJECT_DIR}/FingerPrint/results/       ║"
echo "║                                                              ║"
echo "║  To view dashboard:                                          ║"
echo "║                                                              ║"
echo "║  1. On rolf (in a screen session):                           ║"
echo "║     conda activate ${ENV_NAME}                               ║"
echo "║     cd ${PROJECT_DIR}/FingerPrint                            ║"
echo "║     python -m uvicorn fingerprint_squared.api.server:app \\  ║"
echo "║         --port 8000                                          ║"
echo "║                                                              ║"
echo "║  2. On your local machine (SSH tunnel):                      ║"
echo "║     ssh -L 8000:localhost:8000 alali@rolf.ifi.uzh.ch         ║"
echo "║                                                              ║"
echo "║  3. Open in browser:                                         ║"
echo "║     http://localhost:8000                                    ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
