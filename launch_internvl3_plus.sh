#!/bin/bash
# Quick launcher for InternVL3 and additional models on free GPUs

cd /local/scratch/alali/FingerPrint

echo "=============================================="
echo "Launching Models on Free GPUs"
echo "=============================================="
echo ""

# GPU 0 - InternVL3-2B
echo "Starting InternVL3-2B on GPU 0..."
./scripts/start_single_model.sh 0 "OpenGVLab/InternVL3-2B"
sleep 2

# GPU 4 - Qwen3-VL-2B (newer Qwen model)
echo "Starting Qwen3-VL-2B on GPU 4..."
./scripts/start_single_model.sh 4 "Qwen/Qwen3-VL-2B"
sleep 2

# GPU 6 - LLaVA (popular model)
echo "Starting LLaVA-v1.6 on GPU 6..."
./scripts/start_single_model.sh 6 "llava-hf/llava-v1.6-vicuna-7b-hf"
sleep 2

# GPU 7 - PaliGemma (Google's model)
echo "Starting PaliGemma on GPU 7..."
./scripts/start_single_model.sh 7 "google/paligemma-3b-mix-448"

echo ""
echo "=============================================="
echo "All models launched!"
echo "=============================================="
echo ""
echo "Running models:"
echo "  GPU 0: InternVL3-2B"
echo "  GPU 4: Qwen3-VL-2B"
echo "  GPU 5: Qwen2.5-VL-3B (already running)"
echo "  GPU 6: LLaVA-v1.6-vicuna-7b"
echo "  GPU 7: PaliGemma-3b"
echo ""
echo "To monitor:"
echo "  screen -ls"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "To attach to InternVL3:"
echo "  screen -r fp_gpu0_OpenGVLab_InternVL3_2B"
echo "=============================================="
