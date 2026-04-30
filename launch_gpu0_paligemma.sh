#!/bin/bash
# Launch IDEFICS2-8b on GPU 0 (reliable model)

cd /local/scratch/alali/FingerPrint
./scripts/start_single_model.sh 0 "HuggingFaceM4/idefics2-8b"
