#!/bin/bash
# =============================================================================
# run_dpe_on_rolf.sh
# =============================================================================
# Re-run the VLM evaluation WITH Demographic Positional Encoding (DPE) on rolf,
# then compare against the existing baseline result DBs.
#
# This does NOT recompute the baseline. It:
#   1. Reads bias correction vectors from your existing baseline DBs
#   2. Re-runs each model with DPE injected into the vision tower
#   3. Compares DPE vs baseline and writes figures + a summary JSON
#
# Run this ON rolf, ideally inside a screen session so it survives disconnects:
#   ssh rolf
#   cd /local/scratch/alali/FingerPrint
#   screen -S dpe
#   ./run_dpe_on_rolf.sh
#   # detach with Ctrl+A then D ;  reattach with: screen -r dpe
#
# Tunables (env vars or edit below):
#   GPU=0            GPU index to use
#   ALPHA=1.5        DPE correction strength
#   N_IMAGES=500     images per model (0 = all ~35k; 500 is enough for a result)
#   DATASET_PATH     fullres FHIBE directory
# =============================================================================

set -e
set -o pipefail   # so a Python crash piped to `tee` is detected, not masked

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GPU="${GPU:-0}"
ALPHA="${ALPHA:-1.5}"
USE_4BIT="${USE_4BIT:-1}"          # 1 = 4-bit quantization (match baseline). Set 0 for fp16 if 4bit errors on transformers 5.x.
N_IMAGES="${N_IMAGES:-0}"          # 0 = ALL data (full 35k). Override to subset for a quick test.
DATASET_PATH="${DATASET_PATH:-/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres}"
BASELINE_DIR="results/single_runs_35k"
OUT_DIR="${OUT_DIR_OVERRIDE:-results/dpe_$(date +%Y%m%d_%H%M%S)}"

# The 4 models with complete + fully-scored baseline data (175,945 rows each).
# Baseline DB is chosen by MOST scored rows among glob matches (see resolve below),
# so incomplete duplicates (e.g. InternVL2 gpu4, InternVL3) are skipped automatically.
# Format: "HF_MODEL_NAME|baseline_db_glob"
# NOTE: Llama-3.2-11B-Vision is EXCLUDED — its baseline DB is 100% "[ERROR]"
# responses (the baseline run crashed on every image), so there is nothing valid
# to debias or compare against. Re-run its baseline first, then add it back here.
MODELS=(
    "llava-hf/llava-v1.6-vicuna-7b-hf|${BASELINE_DIR}/gpu*llava*v1.6*vicuna*7b*.db"
    "OpenGVLab/InternVL2-2B|${BASELINE_DIR}/gpu*InternVL2_2B*.db"
    "HuggingFaceM4/idefics2-8b|${BASELINE_DIR}/gpu*idefics2_8b*.db"
)

# Optional: run only models whose HF name contains this substring (for per-GPU
# parallel launches). e.g. ONLY=llava  ONLY=InternVL2  ONLY=idefics2
ONLY="${ONLY:-}"
if [ -n "$ONLY" ]; then
    FILTERED=()
    for entry in "${MODELS[@]}"; do
        [[ "${entry%%|*}" == *"$ONLY"* ]] && FILTERED+=("$entry")
    done
    MODELS=("${FILTERED[@]}")
fi

# Pick the baseline DB with the most scored judge_scores rows among glob matches.
# Guards against incomplete/unscored duplicate DBs.
resolve_baseline_db() {
    local glob="$1"
    local best_db="" best_n=-1
    for cand in $glob; do
        [ -f "$cand" ] || continue
        local n
        n=$(sqlite3 "$cand" "SELECT COUNT(*) FROM judge_scores WHERE valence IS NOT NULL;" 2>/dev/null || echo 0)
        if [ "$n" -gt "$best_n" ]; then best_n="$n"; best_db="$cand"; fi
    done
    echo "$best_db"
}

# ---------------------------------------------------------------------------
# Preamble
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  DPE Re-Evaluation on rolf"
echo "========================================================================"
echo "  Host:        $(hostname)"
echo "  GPU:         $GPU"
echo "  Alpha:       $ALPHA"
echo "  N images:    $N_IMAGES  (0 = all)"
echo "  Dataset:     $DATASET_PATH"
echo "  Output dir:  $OUT_DIR"
echo "  Models:      ${#MODELS[@]}${ONLY:+  (filter: $ONLY)}"
echo ""

if [[ ! $(hostname) == *"rolf"* ]]; then
    echo "⚠️  Warning: hostname does not contain 'rolf'. Run this ON rolf."
    echo "    (Continuing anyway in case of a custom hostname...)"
    echo ""
fi

# Activate conda env (matches run_on_rolf.sh)
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | grep -q "fingerprint"; then
        echo "✓ Activating conda env: fingerprint"
        conda activate fingerprint
    fi
fi

# IMPORTANT: ignore ~/.local user-site packages. A newer transformers there
# shadows the env and demands PyTorch >= 2.4 (env has 2.0.1), which disables
# PyTorch and makes every model fail to load. The conda env's own transformers
# is the one that produced the baseline, so use it exclusively.
export PYTHONNOUSERSITE=1

echo "Python: $(which python3)  ($(python3 --version 2>&1))"

# Show the ACTUAL torch/transformers that will be used (post PYTHONNOUSERSITE)
python3 - <<'PYVER'
import importlib
for mod in ("torch", "transformers"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:14s} {getattr(m,'__version__','?'):12s} @ {m.__file__}")
    except Exception as e:
        print(f"  {mod:14s} IMPORT FAILED: {e}")
# Hard gate: torch must be importable AND expose CUDA
try:
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    print(f"  CUDA OK: {torch.cuda.get_device_name(0)}")
except Exception as e:
    raise SystemExit(f"\nFATAL: PyTorch/CUDA not usable ({e}).\n"
                     "Fix the env before running DPE — e.g. inside the fingerprint env:\n"
                     "  pip install 'transformers==4.44.2'   # compatible with torch 2.0.1\n"
                     "or upgrade torch to >= 2.4 to match the newer transformers.")
PYVER
echo "✓ torch + CUDA verified"
echo ""

# Dataset check
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Dataset not found: $DATASET_PATH"
    echo "   Set DATASET_PATH=... to the fullres FHIBE directory and re-run."
    exit 1
fi
echo "✓ Dataset directory exists"
echo ""

mkdir -p "$OUT_DIR"

# N_IMAGES flag (0 => omit --n-images, i.e. use all)
NIMG_FLAG=""
if [ "$N_IMAGES" -gt 0 ]; then
    NIMG_FLAG="--n-images $N_IMAGES"
fi

# 4-bit flag (toggle via USE_4BIT)
FOURBIT_FLAG=""
if [ "$USE_4BIT" = "1" ]; then
    FOURBIT_FLAG="--4bit"
fi

# ---------------------------------------------------------------------------
# Run each model: DPE benchmark -> compare
# ---------------------------------------------------------------------------
MASTER_LOG="$OUT_DIR/master_log.txt"
echo "DPE run started $(date)" | tee "$MASTER_LOG"

for entry in "${MODELS[@]}"; do
    model="${entry%%|*}"
    db_glob="${entry##*|}"

    # Resolve baseline DB = glob match with the most scored rows
    baseline_db=$(resolve_baseline_db "$db_glob")

    safe_name=$(echo "$model" | sed 's/\//_/g; s/-/_/g; s/\./_/g')
    out_db="$OUT_DIR/${safe_name}_dpe.db"
    cmp_dir="$OUT_DIR/compare_${safe_name}"
    model_log="$OUT_DIR/${safe_name}.log"

    echo "" | tee -a "$MASTER_LOG"
    echo "═══════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
    echo "Model: $model" | tee -a "$MASTER_LOG"
    echo "Baseline DB: ${baseline_db:-<none found>}" | tee -a "$MASTER_LOG"
    echo "Started: $(date)" | tee -a "$MASTER_LOG"
    echo "═══════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"

    if [ -z "$baseline_db" ] || [ ! -f "$baseline_db" ]; then
        echo "  ⚠ No baseline DB matched '$db_glob' — skipping $model" | tee -a "$MASTER_LOG"
        continue
    fi

    # --- Step 1: DPE inference ---
    echo "  [1/2] Running DPE benchmark..." | tee -a "$MASTER_LOG"
    if python3 scripts/run_dpe_benchmark.py \
        --model "$model" \
        --baseline-db "$baseline_db" \
        --dataset-path "$DATASET_PATH" \
        --out-db "$out_db" \
        --alpha "$ALPHA" \
        --gpu "$GPU" \
        $FOURBIT_FLAG \
        $NIMG_FLAG \
        2>&1 | tee "$model_log"; then
        echo "  ✓ DPE benchmark done" | tee -a "$MASTER_LOG"
    else
        echo "  ✗ DPE benchmark FAILED for $model (see $model_log)" | tee -a "$MASTER_LOG"
        continue
    fi

    # Guard: only compare if the DPE run actually produced scored rows
    dpe_scored=$(sqlite3 "$out_db" "SELECT COUNT(*) FROM judge_scores WHERE valence IS NOT NULL;" 2>/dev/null || echo 0)
    if [ "${dpe_scored:-0}" -eq 0 ]; then
        echo "  ⚠ DPE DB has 0 scored rows — skipping compare for $model" | tee -a "$MASTER_LOG"
        continue
    fi
    echo "  DPE produced $dpe_scored scored rows" | tee -a "$MASTER_LOG"

    # --- Step 2: Compare vs baseline ---
    echo "  [2/2] Comparing DPE vs baseline..." | tee -a "$MASTER_LOG"
    python3 scripts/compare_dpe_baseline.py \
        --baseline-db "$baseline_db" \
        --dpe-db "$out_db" \
        --out-dir "$cmp_dir" \
        2>&1 | tee -a "$model_log" || echo "  ⚠ Compare step failed" | tee -a "$MASTER_LOG"

    echo "  ✓ $model complete — results in $cmp_dir" | tee -a "$MASTER_LOG"
    echo "  Finished: $(date)" | tee -a "$MASTER_LOG"
done

echo "" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"
echo "✅ ALL DPE RUNS COMPLETE  ($(date))" | tee -a "$MASTER_LOG"
echo "========================================================================" | tee -a "$MASTER_LOG"
echo "Output: $OUT_DIR" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Sync results back to your laptop with:" | tee -a "$MASTER_LOG"
echo "  ./sync_dpe_from_rolf.sh   (or)" | tee -a "$MASTER_LOG"
echo "  rsync -avz rolf:${PWD}/$OUT_DIR/ ./$OUT_DIR/" | tee -a "$MASTER_LOG"
