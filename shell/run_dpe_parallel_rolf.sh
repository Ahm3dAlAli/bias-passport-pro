#!/bin/bash
# =============================================================================
# run_dpe_parallel_rolf.sh
# =============================================================================
# Launch the DPE re-evaluation for the 3 working models IN PARALLEL, one model
# per GPU, each in its own detached screen session. Much faster than the
# sequential run_dpe_on_rolf.sh.
#
# Run ON rolf:
#   cd /local/scratch/alali/FingerPrint
#   chmod +x run_dpe_parallel_rolf.sh
#   ./run_dpe_parallel_rolf.sh
#
# Each model runs its own DPE inference + comparison + figures. Watch progress:
#   screen -ls                     # list the 3 sessions
#   screen -r dpe_llava            # attach (Ctrl+A then D to detach)
#   tail -f results/dpe_parallel_*/dpe_*/master_log.txt
#
# Tunables (env vars):
#   ALPHA=1.5        DPE correction strength
#   N_IMAGES=0       images per model (0 = all ~35k)
#   DATASET_PATH     fullres FHIBE directory
#   GPUS="0 1 2"     GPU indices to use (one per model, in order)
# =============================================================================

set -e

ALPHA="${ALPHA:-1.5}"
# Balanced sampling by default (full 35k is ~6 days/model on these GPUs).
# BALANCED_PER_GROUP images per (gender x region) group ≈ 18 groups × 60 ≈ 1000
# images × 5 probes ≈ 5000 generations ≈ 4-5h/model. Set to 0 + N_IMAGES for other modes.
BALANCED_PER_GROUP="${BALANCED_PER_GROUP:-60}"
N_IMAGES="${N_IMAGES:-0}"
DATASET_PATH="${DATASET_PATH:-/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres}"
GPUS="${GPUS:-0 1 2}"
SHARED_OUT="results/dpe_parallel_$(date +%Y%m%d_%H%M%S)"

# model-name-substring  <=>  short screen tag
# One entry per model; assigned to GPUs in order.
# InternVL2 excluded: its 2024 remote code is incompatible with transformers 5.x
# at the generation stage (GenerationMixin). Ship LLaVA + idefics2.
FILTERS=("llava" "idefics2")
TAGS=("dpe_llava" "dpe_idefics2")

read -ra GPU_ARR <<< "$GPUS"

echo "=============================================="
echo "  Parallel DPE launch (one model per GPU)"
echo "=============================================="
echo "  Models:   ${FILTERS[*]}"
echo "  GPUs:     ${GPU_ARR[*]}"
echo "  Alpha:    $ALPHA"
echo "  Sampling: balanced ${BALANCED_PER_GROUP}/group (0=use N_IMAGES=$N_IMAGES)"
echo "  Out base: $SHARED_OUT"
echo ""

mkdir -p "$SHARED_OUT"

for i in "${!FILTERS[@]}"; do
    only="${FILTERS[$i]}"
    tag="${TAGS[$i]}"
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    logdir="$SHARED_OUT/$tag"
    mkdir -p "$logdir"

    echo "→ $only  on GPU $gpu  (screen: $tag)"

    # Kill an old session with the same name, then launch detached.
    screen -S "$tag" -X quit 2>/dev/null || true
    screen -dmS "$tag" bash -c "
        cd $(pwd)
        export GPU=$gpu ONLY='$only' ALPHA=$ALPHA N_IMAGES=$N_IMAGES DATASET_PATH='$DATASET_PATH'
        export BALANCED_PER_GROUP=$BALANCED_PER_GROUP
        export OUT_DIR_OVERRIDE='$logdir'
        ./run_dpe_on_rolf.sh 2>&1 | tee '$logdir/console.log'
        echo ''
        echo '=== $tag finished. Press Enter to close. ==='
        read
    "
done

echo ""
echo "=============================================="
echo "✓ Launched ${#FILTERS[@]} screens on GPUs: ${GPU_ARR[*]}"
echo "=============================================="
echo ""
echo "Monitor:"
echo "  screen -ls"
for tag in "${TAGS[@]}"; do
    echo "  screen -r $tag        # attach (Ctrl+A then D to detach)"
done
echo ""
echo "Live logs:"
echo "  tail -f $SHARED_OUT/*/console.log"
echo ""
echo "When all 3 are done, sync results to your laptop:"
echo "  # (on laptop)  ./sync_dpe_from_rolf.sh"
echo ""
