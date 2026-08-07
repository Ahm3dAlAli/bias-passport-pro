#!/bin/bash
# Launch SmolVLM2-2.2B on GPU 4 (fastest, verified working)

cd /local/scratch/alali/FingerPrint
./scripts/start_single_model.sh 4 "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
