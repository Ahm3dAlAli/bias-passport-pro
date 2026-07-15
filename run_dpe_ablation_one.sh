#!/bin/bash
# =============================================================================
# run_dpe_ablation_one.sh   (inner worker — usually launched by run_dpe_ablation_rolf.sh)
# =============================================================================
# Run the alpha ablation for ONE model on ONE GPU: loop over alphas, each a
# balanced DPE run + comparison, then print an alpha-vs-disparity summary.
#
# Args:  MODEL  BASELINE_DB  GPU  OUTDIR  PER_GROUP  AXIS  ALPHA [ALPHA ...]
# =============================================================================
set -e
set -o pipefail

MODEL="$1"; BASELINE_DB="$2"; GPU="$3"; OUTDIR="$4"; PER_GROUP="$5"; AXIS="$6"
shift 6
ALPHAS="$*"
DATASET_PATH="${DATASET_PATH:-/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres}"

# Env (same guards as the main runner)
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda env list | grep -q fingerprint && conda activate fingerprint
fi
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTDIR"
echo "=============================================="
echo "  Ablation: $MODEL"
echo "  GPU=$GPU  axis=$AXIS  per_group=$PER_GROUP"
echo "  alphas: $ALPHAS"
echo "  baseline: $BASELINE_DB"
echo "=============================================="

for a in $ALPHAS; do
    tag=$(echo "$a" | tr '.' 'p')
    echo ""
    echo "─────────── $MODEL  alpha=$a ───────────"
    python3 scripts/run_dpe_benchmark.py \
        --model "$MODEL" \
        --baseline-db "$BASELINE_DB" \
        --dataset-path "$DATASET_PATH" \
        --out-db "$OUTDIR/alpha_${tag}_dpe.db" \
        --alpha "$a" \
        --gpu "$GPU" \
        --4bit \
        --balanced-per-group "$PER_GROUP" \
        --correction-axis "$AXIS" \
        2>&1 | tee "$OUTDIR/alpha_${tag}.log"

    python3 scripts/compare_dpe_baseline.py \
        --baseline-db "$BASELINE_DB" \
        --dpe-db "$OUTDIR/alpha_${tag}_dpe.db" \
        --out-dir "$OUTDIR/compare_alpha_${tag}" \
        2>&1 | tee "$OUTDIR/compare_alpha_${tag}.log" || echo "compare failed for alpha=$a"
done

# --- alpha-vs-disparity summary -------------------------------------------
echo ""
echo "=============================================="
echo "  ABLATION SUMMARY: $MODEL  (axis=$AXIS)"
echo "=============================================="
python3 - "$OUTDIR" "$ALPHAS" <<'PYSUM'
import json, sqlite3, sys
from pathlib import Path
out = Path(sys.argv[1]); alphas = sys.argv[2].split()

def coherence(db):
    if not Path(db).exists(): return (None, None)
    try:
        rows = sqlite3.connect(db).execute(
            "SELECT response FROM probe_results WHERE response IS NOT NULL").fetchall()
    except Exception:
        return (None, None)
    if not rows: return (None, None)
    L = [len((r[0] or "").strip()) for r in rows]
    return (sum(L)/len(L), 100.0*sum(1 for x in L if x < 3)/len(L))

print(f"{'alpha':>6} | {'region_red%':>11} | {'gender_red%':>11} | {'avg_len':>7} | {'%empty':>6}")
print("-"*58)
for a in alphas:
    t = a.replace('.', 'p')
    s = out / f"compare_alpha_{t}" / "dpe_comparison_summary.json"
    db = out / f"alpha_{t}_dpe.db"
    rr = gr = None
    if s.exists():
        d = json.loads(s.read_text()).get("disparity", {})
        rr = d.get("jurisdiction_region", {}).get("pct_reduction")
        gr = d.get("gender_presentation", {}).get("pct_reduction")
    al, pe = coherence(str(db))
    f = lambda x, n=1: f"{x:.{n}f}" if isinstance(x, (int, float)) else "  --"
    print(f"{a:>6} | {f(rr):>11} | {f(gr):>11} | {f(al,0):>7} | {f(pe):>6}")
print()
print("Pick the alpha with the largest positive region_red% whose avg_len stays")
print("near baseline and %empty ~ 0.")
PYSUM

echo ""
echo "=== $MODEL ablation complete. Artifacts in $OUTDIR ==="
