#!/bin/bash
# =============================================================================
# run_dpe_final_eval.sh
# =============================================================================
# Final DPE evaluation: run each VLM at its OPTIMAL alpha (from the ablation),
# each on its own GPU, in its correct conda env, then compare against the stored
# baseline (before vs after). Produces the "with DPE" side; the baseline is the
# "without DPE" side you already have.
#
# Optimal alphas (from the balanced ablation):
#   idefics2  -> 0.25    InternVL2 -> 0.5    LLaVA -> 0.5
#
# Run ON rolf:
#   ./run_dpe_final_eval.sh
#
# Tunables (env vars):
#   PER_GROUP   balanced images per (gender x region) group  [default 300]
#               -> ~12 groups x 300 = ~3600 images (robust, ~half day/model).
#               Set PER_GROUP=0 and N_IMAGES=0 for the FULL 35k (~6 days/model!).
#   N_IMAGES    used only if PER_GROUP=0 (0 = all images)
#   GPUS        space-separated GPUs to use, one per model  [default: auto free]
#   DATASET_PATH  fullres FHIBE dir
# =============================================================================
set -e

PER_GROUP="${PER_GROUP:-300}"
BALANCE_BY="${BALANCE_BY:-region}"    # 'region' = cap PER_GROUP images per region (gender ignored)
N_IMAGES="${N_IMAGES:-0}"
DATASET_PATH="${DATASET_PATH:-/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres}"
BASE_DIR="results/single_runs_35k"
# OUT_DIR override => FIXED dir (resumable: rerun skips already-scored images).
# Leave unset for a fresh timestamped run.
OUT="${OUT_DIR:-results/dpe_final_$(date +%Y%m%d_%H%M%S)}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"

# model | optimal_alpha | conda_env | 4bit(1/0) | baseline_glob | screen_tag
JOBS=(
  "HuggingFaceM4/idefics2-8b|0.25|fingerprint|1|gpu*idefics2_8b*.db|idefics2"
  "OpenGVLab/InternVL2-2B|0.5|internvl|0|gpu*InternVL2_2B*.db|internvl2"
  "llava-hf/llava-v1.6-vicuna-7b-hf|0.5|internvl|1|gpu*llava*v1.6*vicuna*7b*.db|llava"
)

# --- helpers ---------------------------------------------------------------
resolve_baseline_db() {   # most-scored DB among glob matches
  local best="" bn=-1
  for c in $1; do
    [ -f "$c" ] || continue
    local n; n=$(sqlite3 "$c" "SELECT COUNT(*) FROM judge_scores WHERE valence IS NOT NULL;" 2>/dev/null || echo 0)
    [ "$n" -gt "$bn" ] && { bn="$n"; best="$c"; }
  done
  echo "$best"
}
gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }

# free GPU pool (either provided GPUS or auto-detected)
if [ -n "$GPUS" ]; then
  read -ra POOL <<< "$GPUS"
else
  read -ra POOL <<< "$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<500{print $1}' | tr '\n' ' ')"
fi

echo "=============================================="
echo "  DPE FINAL EVALUATION (optimal alpha per model)"
echo "=============================================="
echo "  Sampling:  $([ "$PER_GROUP" -gt 0 ] && echo "balanced ${PER_GROUP}/group" || echo "N_IMAGES=$N_IMAGES (0=full 35k)")"
echo "  Free GPUs: ${POOL[*]:-none}"
echo "  Output:    $OUT"
echo ""
if [ "${#POOL[@]}" -lt "${#JOBS[@]}" ]; then
  echo "⚠ Only ${#POOL[@]} free GPU(s) for ${#JOBS[@]} models — some will queue or skip."
  echo "  Provide GPUs explicitly: GPUS=\"0 1 2\" ./run_dpe_final_eval.sh"
fi
mkdir -p "$OUT"

# sampling flag (shared)
if [ "$PER_GROUP" -gt 0 ]; then
  SAMPLE_FLAG="--balanced-per-group $PER_GROUP"
elif [ "$N_IMAGES" -gt 0 ]; then
  SAMPLE_FLAG="--n-images $N_IMAGES"
else
  SAMPLE_FLAG=""   # full dataset
fi

i=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r model alpha env fourbit glob tag <<< "$job"
  baseline=$(resolve_baseline_db "$BASE_DIR/$glob")
  if [ -z "$baseline" ]; then echo "⚠ no baseline for $model — skip"; continue; fi

  gpu="${POOL[$i]}"; i=$((i+1))
  if [ -z "$gpu" ]; then echo "⚠ no GPU left for $model — skip (rerun later)"; continue; fi
  free=$(gpu_free_mib "$gpu")
  if [ "${free:-0}" -lt "$MIN_FREE_MIB" ]; then
    echo "⚠ GPU $gpu only ${free}MiB free — skipping $model"; continue
  fi

  fourbit_flag=""; [ "$fourbit" = "1" ] && fourbit_flag="--4bit"
  outdir="$OUT/$tag"; mkdir -p "$outdir"
  echo "→ $model  α=$alpha  env=$env  GPU=$gpu  4bit=$fourbit  (screen final_$tag)"

  screen -S "final_$tag" -X quit 2>/dev/null || true
  screen -dmS "final_$tag" bash -c "
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate $env
    export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd $(pwd)
    echo '=== $model  alpha=$alpha  env=$env  gpu=$gpu ==='
    python3 scripts/run_dpe_benchmark.py --model '$model' --baseline-db '$baseline' \
      --dataset-path '$DATASET_PATH' --out-db '$outdir/${tag}_dpe.db' \
      --alpha $alpha --gpu $gpu $fourbit_flag $SAMPLE_FLAG --balance-by $BALANCE_BY \
      --correction-axis region \
      2>&1 | tee '$outdir/run.log'
    python3 scripts/compare_dpe_baseline.py --baseline-db '$baseline' \
      --dpe-db '$outdir/${tag}_dpe.db' --out-dir '$outdir/compare' \
      2>&1 | tee '$outdir/compare.log'
    echo ''
    echo '=== final_$tag DONE. Ctrl+A then D to detach. ==='
    exec bash
  "
done

echo ""
echo "=============================================="
echo "✓ Launched. Watch:  tail -f $OUT/*/run.log"
echo "  Screens:  screen -ls | grep final_"
echo "=============================================="
echo ""
echo "When all finish, the before/after comparison is in each $OUT/<model>/compare/"
echo "  dpe_comparison_summary.json  + figures."
echo "Then pull to laptop:  ./sync_dpe_from_rolf.sh"
