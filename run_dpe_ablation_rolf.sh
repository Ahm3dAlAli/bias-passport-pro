#!/bin/bash
# =============================================================================
# run_dpe_ablation_rolf.sh
# =============================================================================
# Launch the per-model alpha ablation IN PARALLEL — one model per GPU, each in
# its own screen, looping over all alphas on a balanced sample with a chosen
# correction axis (region-only by default). Then per model prints an
# alpha-vs-disparity table so you can pick the headline alpha.
#
# Run ON rolf:
#   ./run_dpe_ablation_rolf.sh
#
# Tunables (env vars):
#   ALPHAS           default "0.25 0.5 0.75 1.0 1.5"
#   CORRECTION_AXIS  region (default) | intersectional | gender
#   PER_GROUP        balanced images per (gender x region) group (default 50)
#   DATASET_PATH     fullres FHIBE dir
#   GPUS             GPUs for the models, in order (default "3 4 5")
#
# Watch:  tail -f results/dpe_ablation_*/*/console.log
# Summaries print at the end of each model's console.log.
# =============================================================================
set -e

ALPHAS="${ALPHAS:-0.25 0.5 0.75 1.0 1.5}"
CORRECTION_AXIS="${CORRECTION_AXIS:-region}"
PER_GROUP="${PER_GROUP:-50}"
GPUS="${GPUS:-3 4 5}"
BASELINE_DIR="results/single_runs_35k"
OUT_BASE="results/dpe_ablation_$(date +%Y%m%d_%H%M%S)"
export DATASET_PATH="${DATASET_PATH:-/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres}"

read -ra GPU_ARR <<< "$GPUS"

# model  |  baseline-db glob  |  screen tag
# InternVL2 included but may fail at generation (transformers 5.x / GenerationMixin);
# if it does, its screen ends with the real error and the other two are unaffected.
JOBS=(
    "llava-hf/llava-v1.6-vicuna-7b-hf|gpu*llava*v1.6*vicuna*7b*.db|abl_llava"
    "HuggingFaceM4/idefics2-8b|gpu*idefics2_8b*.db|abl_idefics2"
    "OpenGVLab/InternVL2-2B|gpu*InternVL2_2B*.db|abl_internvl2"
)

resolve_baseline_db() {
    local best="" bn=-1
    for c in $1; do
        [ -f "$c" ] || continue
        local n; n=$(sqlite3 "$c" "SELECT COUNT(*) FROM judge_scores WHERE valence IS NOT NULL;" 2>/dev/null || echo 0)
        [ "$n" -gt "$bn" ] && { bn="$n"; best="$c"; }
    done
    echo "$best"
}

mkdir -p "$OUT_BASE"
echo "=============================================="
echo "  DPE alpha ablation (parallel, one model/GPU)"
echo "=============================================="
echo "  axis:       $CORRECTION_AXIS"
echo "  alphas:     $ALPHAS"
echo "  per_group:  $PER_GROUP (balanced)"
echo "  GPUs:       ${GPU_ARR[*]}"
echo "  out base:   $OUT_BASE"
echo ""

i=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r model glob tag <<< "$job"
    baseline=$(resolve_baseline_db "$BASELINE_DIR/$glob")
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    i=$((i+1))

    if [ -z "$baseline" ]; then
        echo "⚠ no baseline DB for $model (glob: $glob) — skipping"
        continue
    fi

    outdir="$OUT_BASE/$tag"
    mkdir -p "$outdir"
    echo "→ $model  GPU $gpu  baseline=$(basename "$baseline")  screen=$tag"

    screen -S "$tag" -X quit 2>/dev/null || true
    screen -dmS "$tag" bash -c "
        cd $(pwd)
        export DATASET_PATH='$DATASET_PATH'
        ./run_dpe_ablation_one.sh '$model' '$baseline' '$gpu' '$outdir' '$PER_GROUP' '$CORRECTION_AXIS' $ALPHAS 2>&1 | tee '$outdir/console.log'
        echo ''
        echo '=== $tag done. Ctrl+A then D to detach. ==='
        exec bash
    "
done

echo ""
echo "=============================================="
echo "✓ Launched. Screens: $(for j in "${JOBS[@]}"; do echo -n "$(echo "$j" | cut -d'|' -f3) "; done)"
echo "=============================================="
echo "Watch:   tail -f $OUT_BASE/*/console.log"
echo "Attach:  screen -r abl_idefics2   (Ctrl+A then D to detach)"
echo ""
echo "Each model's alpha-vs-disparity table prints at the end of its console.log."
echo "After picking alphas, sync back:  ./sync_dpe_from_rolf.sh"
