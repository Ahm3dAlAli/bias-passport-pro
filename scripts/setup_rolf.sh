#!/bin/bash
# =============================================================================
# setup_rolf.sh - Run this on ROLF to set up the environment and run benchmark
# =============================================================================

set -e

PROJ_DIR="/local/scratch/alali/FingerPrint"
DATASET_PATH="/local/scratch/alali/fhibe_data"
cd "$PROJ_DIR"

echo "=============================================="
echo "Setting up Fingerprint² on ROLF"
echo "=============================================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists"
fi

# Activate and install packages
echo "[2/4] Installing packages..."
source venv/bin/activate

pip install --upgrade pip

# First uninstall conflicting packages
pip uninstall -y torchaudio 2>/dev/null || true

# Install PyTorch with CUDA 11.8 (required for latest transformers)
# torch 2.6.0 pairs with torchvision 0.21.0
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Install transformers with Pixtral support (requires >= 4.45)
pip install "transformers>=4.45.0" accelerate bitsandbytes

# Core dependencies
pip install pillow tqdm vaderSentiment scikit-learn pandas numpy rich
pip install einops timm sentencepiece protobuf qwen-vl-utils num2words

# OpenCLIP for CLIP/FLAVA-style models
pip install open-clip-torch

# NOTE: OpenFlamingo requires torch 2.0.1 which conflicts with newer transformers
# Skip it for now - use IDEFICS2 as the Flamingo-style alternative instead

# Pixtral requires specific mistral packages
pip install mistral-common

# Flash attention for faster inference (optional, may fail on older GPUs)
pip install flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention not available"

echo "[3/4] Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo "[4/4] Checking dataset..."
if [ -d "$DATASET_PATH" ]; then
    echo "Dataset found: $(ls -1 $DATASET_PATH | wc -l) items in $DATASET_PATH"
else
    echo "WARNING: Dataset not found at $DATASET_PATH"
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To run the benchmark manually:"
echo "  source venv/bin/activate"
echo "  CUDA_VISIBLE_DEVICES=0 python scripts/run_full_evaluation.py \\"
echo "      --dataset $DATASET_PATH --lightweight"
echo ""
echo "Or run with screen for persistence:"
echo "  screen -S fingerprint_benchmark"
echo "  source venv/bin/activate"
echo "  CUDA_VISIBLE_DEVICES=0 python scripts/run_full_evaluation.py \\"
echo "      --dataset $DATASET_PATH --lightweight"
echo "  # Detach: Ctrl+A, D"
echo ""
