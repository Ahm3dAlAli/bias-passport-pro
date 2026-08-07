#!/bin/bash
# =============================================================================
# reproduce_dpe.sh — regenerate every DPE table/figure from the local result DBs.
# Run from the repo root:  bash shell/reproduce_dpe.sh
# Requires: pip install -r requirements-dpe.txt, and the DBs under results/
# (see docs/REPRODUCE_DPE.md). Steps whose DBs are missing are skipped with a note.
# =============================================================================
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PYTHON:-python3}
ok=0; skip=0

run() { echo; echo ">>> $*"; if "$@"; then ok=$((ok+1)); else echo "!! step failed: $*"; fi; }

echo "=============================================="
echo " Reproducing DPE analysis   (repo: $ROOT)"
echo "=============================================="

# --- 1. baseline + Tier-1/2 (2000/region eval + baselines) -------------------
if [ -e results/single_runs_35k ] && ls results/dpe_final_20260720_125918/*/*_dpe.db >/dev/null 2>&1; then
  run "$PY" scripts/dpe_analysis.py all
else
  echo "SKIP dpe_analysis.py all — missing results/single_runs_35k or results/dpe_final_20260720_125918/*"
  echo "     (pull with: bash shell/sync_dpe_from_rolf.sh)"; skip=$((skip+1))
fi

# --- 2. corrected full-corpus + sampling-robustness (the headline) ----------
if ls results/dpe_full35k/*/*_dpe.db >/dev/null 2>&1; then
  run "$PY" scripts/dpe_sampling_robustness.py
else
  echo "SKIP dpe_sampling_robustness.py — missing results/dpe_full35k/<model>/<model>_dpe.db"
  echo "     (rsync with: rsync -avz rolf:/local/scratch/alali/FingerPrint/results/dpe_full35k/ results/dpe_full35k/)"
  skip=$((skip+1))
fi

# --- 3. copy the paper-facing artifacts into aaai_build/ ---------------------
copied=0
for f in dpe_results_fullcorpus.tex dpe_robustness_table.tex dpe_tier1_tables.tex \
         dpe_correction_norms.tex dpe_content_preservation.tex \
         fig_dpe_sigma_before_after.pdf fig_dpe_convergence.pdf fig_dpe_robustness.pdf \
         fig_dpe_visual_umap.pdf; do
  if [ -f "figures/dpe_aaai/$f" ]; then cp "figures/dpe_aaai/$f" aaai_build/ && copied=$((copied+1)); fi
done
echo; echo "copied $copied artifact(s) into aaai_build/"

echo
echo "=============================================="
echo " done: $ok step(s) ran, $skip skipped."
echo " outputs: figures/dpe_aaai/  (and copies in aaai_build/)"
echo "=============================================="
