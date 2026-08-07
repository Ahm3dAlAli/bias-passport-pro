#!/bin/bash
# Launch Phi-3.5-vision on GPU 0

cd /local/scratch/alali/FingerPrint
./scripts/start_single_model.sh 0 "microsoft/Phi-3.5-vision-instruct"
