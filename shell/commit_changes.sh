#!/bin/bash
# Commit script for Fingerprint² full dataset evaluation updates

set -e

cd /Users/ahmeda./Desktop/FingerPrint

echo "Adding files..."
git add scripts/run_fhibe_benchmark.py
git add scripts/run_full_evaluation.py
git add scripts/run_full_benchmark.sh
git add paper/fingerprint2_neurips_enhanced.tex

echo ""
echo "Files to be committed:"
git status --short

echo ""
echo "Committing..."
git commit -m "Add full 10K dataset evaluation with 27 SOTA VLMs

- Update benchmark to run on complete FHIBE corpus (10,318 images)
- Add 27 VLMs including Qwen3-VL, Llama-3.2-Vision, InternVL3,
  Pixtral, DeepSeek-VL2, Gemma 3, FLAVA, OpenFlamingo, IDEFICS
- Add model clients for new architectures (FLAVA, Flamingo,
  DeepSeek, Pixtral, Gemma3, IDEFICS)
- Add run_full_evaluation.py with --lightweight and --large options
- Add run_full_benchmark.sh for shell-based execution
- Update paper with full dataset statistics and robustness analysis"

echo ""
echo "Done! Commit created."
git log -1 --oneline
