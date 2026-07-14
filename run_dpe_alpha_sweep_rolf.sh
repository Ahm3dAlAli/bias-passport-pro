#!/bin/bash
# =============================================================================
# run_dpe_alpha_sweep_rolf.sh
# =============================================================================
# Sweep the DPE correction strength alpha on a MODERATE image sample for ONE
# model, and tabulate disparity reduction + output coherence per alpha, so you
# can pick the strength that debiases WITHOUT breaking the model — before
# committing GPUs to the full 35k run.
#
# Run ON rolf (in screen):
#   ./run_dpe_alpha_sweep_rolf.sh
#
# Tunables (env vars):
#   MODEL          HF model id (default: idefics2 — fastest to iterate)
#   BASELINE_DB    baseline DB glob (auto-picks most-scored match)
#   ALPHAS         space-separated strengths (default: "0.25 0.5 1.0 1.5")
#   PER_GROUP      balanced sample: images per (gender x region) group
#                  (default: 40 -> ~24 groups x 40 = ~1000 balanced images)
#   GPU            GPU index (default 0)
#   DATASET_PATH   fullres FHIBE dir
# =============================================================================

set -e
set -o pipefail

MODEL="${MODEL:-HuggingFaceM4/idefics2-8b}"
BASELINE_GLOB="${BASELINE_DB:-results/single_runs_35k/gpu*idefics2_8b*.db}"
ALPHAS="${ALPHAS:-0.25 0.5 1.0 1.5}"
PER_GROUP="${PER_GROUP:-40}"
GPU="${GPU:-0}"
DATASET_PATH="${DATASET_PATH:-/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres}"
OUT_DIR="results/dpe_sweep_$(date +%Y%m%d_%H%M%S)"

# Env fixes (same as main runner)
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda env list | grep -q fingerprint && conda activate fingerprint
fi
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resolve baseline DB = glob match with most scored rows
resolve_baseline_db() {
    local best="" best_n=-1
    for c in $1; do
        [ -f "$c" ] || continue
        local n; n=$(sqlite3 "$c" "SELECT COUNT(*) FROM judge_scores WHERE valence IS NOT NULL;" 2>/dev/null || echo 0)
        [ "$n" -gt "$best_n" ] && { best_n="$n"; best="$c"; }
    done
    echo "$best"
}
BASELINE_DB=$(resolve_baseline_db "$BASELINE_GLOB")

echo "=============================================="
echo "  DPE alpha sweep"
echo "=============================================="
echo "  Model:      $MODEL"
echo "  Baseline:   $BASELINE_DB"
echo "  Alphas:     $ALPHAS"
echo "  Balanced:   $PER_GROUP images/group (gender x region)"
echo "  Out:        $OUT_DIR"
echo ""
mkdir -p "$OUT_DIR"

for a in $ALPHAS; do
    tag=$(echo "$a" | tr '.' 'p')
    out_db="$OUT_DIR/alpha_${tag}_dpe.db"
    cmp_dir="$OUT_DIR/compare_alpha_${tag}"
    echo ""
    echo "───────────── alpha=$a ─────────────"

    python3 scripts/run_dpe_benchmark.py \
        --model "$MODEL" \
        --baseline-db "$BASELINE_DB" \
        --dataset-path "$DATASET_PATH" \
        --out-db "$out_db" \
        --alpha "$a" \
        --gpu "$GPU" \
        --4bit \
        --balanced-per-group "$PER_GROUP" \
        2>&1 | tee "$OUT_DIR/alpha_${tag}.log" | grep -E "hook fired|Completed|Scoring DPE" || true

    python3 scripts/compare_dpe_baseline.py \
        --baseline-db "$BASELINE_DB" \
        --dpe-db "$out_db" \
        --out-dir "$cmp_dir" \
        > /dev/null 2>&1 || echo "  compare failed for alpha=$a"
done

# ---------------------------------------------------------------------------
# Summarise the sweep: alpha vs disparity reduction vs coherence
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  SWEEP SUMMARY (pick alpha: high reduction, low %empty, sane avg_len)"
echo "=============================================="
python3 - "$OUT_DIR" "$ALPHAS" <<'PYSUM'
import json, sqlite3, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
alphas = sys.argv[2].split()

def coherence(db):
    if not Path(db).exists(): return (None, None)
    try:
        rows = sqlite3.connect(db).execute(
            "SELECT response FROM probe_results WHERE response IS NOT NULL").fetchall()
    except Exception:
        return (None, None)
    if not rows: return (None, None)
    L = [len((r[0] or '').strip()) for r in rows]
    return (sum(L)/len(L), 100.0*sum(1 for x in L if x < 3)/len(L))

print(f"{'alpha':>6} | {'gender_red%':>11} | {'region_red%':>11} | {'avg_len':>7} | {'%empty':>6}")
print("-"*58)
for a in alphas:
    tag = a.replace('.', 'p')
    summ = out_dir / f"compare_alpha_{tag}" / "dpe_comparison_summary.json"
    db   = out_dir / f"alpha_{tag}_dpe.db"
    gred = rred = None
    if summ.exists():
        d = json.loads(summ.read_text()).get("disparity", {})
        gred = d.get("gender_presentation", {}).get("pct_reduction")
        rred = d.get("jurisdiction_region", {}).get("pct_reduction")
    avg_len, pct_empty = coherence(str(db))
    def f(x, nd=1): return f"{x:.{nd}f}" if isinstance(x,(int,float)) else "  --"
    print(f"{a:>6} | {f(gred):>11} | {f(rred):>11} | {f(avg_len,0):>7} | {f(pct_empty):>6}")

print()
print("Guidance: choose the alpha with the LARGEST positive reduction whose")
print("avg_len stays close to the baseline and %empty stays near 0. If every")
print("alpha shows negative reduction, DPE needs a different correction rule,")
print("not just a smaller alpha — flag it.")
PYSUM

echo ""
echo "Sweep artifacts in: $OUT_DIR"
