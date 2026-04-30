# Multi-Model 35K Evaluation Guide

## Current Status

You have completed **SmolVLM2-2.2B-Instruct** evaluation:
- **175,945 probe results** (~35,189 images × 5 probes)
- Database: `gpu5_20260325.db` (134M)
- Location: `/local/scratch/alali/FingerPrint/results/gpu5_20260325.db`

## Running Additional Models

### Quick Start

```bash
# Sync and run all models
./run_multi_model_rolf.sh --run

# Or just sync for now
./run_multi_model_rolf.sh
```

### Manual Execution on Rolf

```bash
# 1. SSH to rolf
ssh rolf

# 2. Start screen session
screen -S fingerprint_multimodel

# 3. Run evaluation
cd /local/scratch/alali/FingerPrint
./scripts/run_multi_model_35k.sh /shares/fhibe/fullres 5

# 4. Detach from screen
# Press: Ctrl+A, then D
```

### Monitor Progress

```bash
# Attach to running screen session
ssh rolf
screen -r fingerprint_multimodel

# Or attach to existing session (from your output)
screen -r 268608.fingerprint

# List all screen sessions
screen -ls

# Check GPU usage
ssh rolf "nvidia-smi"

# View recent logs
ssh rolf "tail -100 /local/scratch/alali/FingerPrint/results/multi_model_35k_*/master_log.txt"
```

## Models Queue (10 models)

The script will evaluate these models sequentially:

### Small Models (2-4GB VRAM in 4-bit):
1. `Qwen/Qwen2.5-VL-3B-Instruct`
2. `OpenGVLab/InternVL2-2B`
3. `OpenGVLab/InternVL3-2B`
4. `google/paligemma-3b-mix-448`
5. `Qwen/Qwen3-VL-2B`

### Medium Models (6-10GB VRAM in 4-bit):
6. `llava-hf/llava-v1.6-vicuna-7b-hf`
7. `HuggingFaceM4/idefics2-8b`
8. `microsoft/Phi-3.5-vision-instruct`

### Larger Models (12-20GB VRAM in 4-bit):
9. `meta-llama/Llama-3.2-11B-Vision-Instruct`
10. `mistralai/Pixtral-12B-2409`

**Note:** SmolVLM2-2.2B-Instruct already completed, so you'll have **11 models total** when done.

## Expected Timeline

- **Per model:** ~24-48 hours
- **Total for 10 models:** ~360 hours (~15 days)
- Each model generates ~175,945 probe results

## Output Structure

```
results/multi_model_35k_YYYYMMDD_HHMMSS/
├── master_log.txt                          # Overall progress log
├── Qwen_Qwen2.5_VL_3B_Instruct_35k.db     # SQLite results
├── Qwen_Qwen2.5_VL_3B_Instruct_35k.json   # JSON results
├── Qwen_Qwen2.5_VL_3B_Instruct_dashboard.html
├── Qwen_Qwen2.5_VL_3B_Instruct.log        # Individual model log
├── OpenGVLab_InternVL2_2B_35k.db
├── ... (and so on for each model)
```

## Checking Results

### On Rolf (Quick Stats):

```bash
ssh rolf

# Count results in all databases
for db in /local/scratch/alali/FingerPrint/results/multi_model_35k_*/*.db; do
    echo "=== $(basename $db) ==="
    sqlite3 "$db" "SELECT COUNT(*) FROM probe_results;" 2>/dev/null
    sqlite3 "$db" "SELECT COUNT(DISTINCT image_id) FROM probe_results;" 2>/dev/null
    echo ""
done

# Check master log
tail -50 /local/scratch/alali/FingerPrint/results/multi_model_35k_*/master_log.txt
```

### Download Results to Local Machine:

```bash
# Download specific database
scp rolf:/local/scratch/alali/FingerPrint/results/multi_model_35k_*/Qwen*.db ./

# Download all results
rsync -avz --progress \
    rolf:/local/scratch/alali/FingerPrint/results/multi_model_35k_*/ \
    ./results/multi_model_35k_downloaded/
```

## Database Schema

Each `.db` file contains:

### `probe_results` table:
- `image_id`: Unique image identifier
- `model_name`: Model used for evaluation
- `probe_id`: P1-P5 (occupation, education, trust, lifestyle, neighbourhood)
- `response`: Model's text response
- `jurisdiction`, `age_group`, `gender_presentation`: Demographics

### `judge_scores` table:
- `valence`: Sentiment score
- `stereotype_alignment`: Stereotype bias score
- `confidence`: Model confidence
- `refusal`: Whether model refused to answer

## Troubleshooting

### If a model fails:
1. Check the individual model log: `results/multi_model_35k_*/ModelName.log`
2. The script continues with next model automatically
3. Failed models are marked in `master_log.txt`

### To skip a model:
Edit `scripts/run_multi_model_35k.sh` and comment out the model line with `#`

### To run only specific models:
Edit the `MODELS` array in `scripts/run_multi_model_35k.sh` to include only desired models

### Memory issues:
- All models use `--4bit` quantization to fit in single GPU
- If OOM errors occur, check GPU memory: `nvidia-smi`
- Consider reducing batch size in the benchmark script

## Customization

### Change dataset path:
```bash
./scripts/run_multi_model_35k.sh /path/to/dataset 5
```

### Change GPU:
```bash
./scripts/run_multi_model_35k.sh /shares/fhibe/fullres 0  # Use GPU 0
```

### Add more models:
Edit `scripts/run_multi_model_35k.sh` and add to the `MODELS` array

## Results Analysis

Once complete, you'll have 11 models evaluated on ~35K images:
- **Total probe responses:** ~1,935,395 (11 models × 175,945 probes)
- **Comprehensive bias comparison** across all SOTA VLMs
- **Publication-ready** benchmarking data

Next steps after completion:
1. Generate comparative analysis across all models
2. Create visualization dashboards
3. Statistical significance testing
4. Publication tables and figures
