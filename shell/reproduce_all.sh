#!/bin/bash
# =============================================================================
# reproduce_all.sh — master, guided reproduction of the FingerPrint / DPE paper.
#
#   bash shell/reproduce_all.sh
#
# It walks the full pipeline stage by stage. For each stage it checks whether the
# outputs already exist; if so it skips, otherwise it RUNS the step when that is
# feasible on the current machine, or PRINTS the exact command to run elsewhere
# (the dataset is gated and the model runs take ~weeks of cluster GPU — those
# cannot happen inside one laptop command, so the script guides you through them).
#
# Env overrides:
#   DATASET_PATH   path to the FHIBE full-res image dir
#   REMOTE         rolf ssh alias (default: rolf)   REMOTE_DIR  cluster repo path
#   RUN_GEN=1      attempt to launch GPU generation locally (only on the cluster)
# =============================================================================
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY=${PYTHON:-python3}
REMOTE=${REMOTE:-rolf}
REMOTE_DIR=${REMOTE_DIR:-/local/scratch/alali/FingerPrint}
DATASET_PATH=${DATASET_PATH:-/local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres}
B="\033[1m"; G="\033[32m"; Y="\033[33m"; R="\033[31m"; N="\033[0m"
say(){ echo -e "\n${B}== $* ==${N}"; }
todo(){ echo -e "  ${Y}NEXT:${N} $*"; }
ok(){ echo -e "  ${G}OK:${N} $*"; }

# is this the GPU cluster? (dataset present or nvidia-smi available)
ON_CLUSTER=0; { [ -d "$DATASET_PATH" ] || command -v nvidia-smi >/dev/null 2>&1; } && ON_CLUSTER=1

echo -e "${B}FingerPrint / DPE — guided reproduction${N}"
echo "repo: $ROOT   |   host: $([ $ON_CLUSTER = 1 ] && echo cluster || echo local)"
echo "See docs/REPRODUCE_DPE.md for the full data->script->artifact map."

# --- Stage 0: environment ----------------------------------------------------
say "Stage 0 — Python analysis environment"
if $PY -c "import numpy,pandas,matplotlib,scipy" 2>/dev/null; then ok "core analysis deps present"
else todo "pip install -r requirements-dpe.txt"; fi

# --- Stage 1: dataset (gated) ------------------------------------------------
say "Stage 1 — FHIBE dataset (35,189 consented images)"
if [ -d "$DATASET_PATH" ]; then ok "dataset found at $DATASET_PATH"
else
  echo -e "  ${R}Not found${N} at $DATASET_PATH."
  echo "  FHIBE is a gated, consent-based benchmark and CANNOT be auto-downloaded."
  todo "Request access at Sony AI's FHIBE release, then extract to a dir and set:"
  echo "        export DATASET_PATH=/path/to/fhibe_fullres"
  echo "  (Only needed to (re)generate the DBs; the analysis below runs from DBs.)"
fi

# --- Stage 2: baseline DBs (no DPE) -----------------------------------------
say "Stage 2 — baseline model runs (no DPE)"
if ls results/single_runs_35k/gpu*_*.db >/dev/null 2>&1; then ok "baseline DBs present (results/single_runs_35k/)"
else
  todo "on the cluster: python3 scripts/run_fhibe_benchmark.py per model (IDEFICS2/InternVL2/LLaVA),"
  echo "        or pull them:  rsync -avz $REMOTE:$REMOTE_DIR/results/single_runs_35k/ results/single_runs_35k/"
fi

# --- Stage 3: DPE full-corpus runs ------------------------------------------
say "Stage 3 — DPE runs at optimal alpha (IDEFICS2 0.25, InternVL2/LLaVA 0.5)"
if ls results/dpe_full35k/*/*_dpe.db >/dev/null 2>&1; then ok "full-corpus DPE DBs present (results/dpe_full35k/)"
else
  if [ "$ON_CLUSTER" = 1 ] && [ "${RUN_GEN:-0}" = 1 ] && [ -d "$DATASET_PATH" ]; then
    echo "  launching full-corpus DPE (this takes days per model) via shell/run_dpe_final_eval.sh ..."
    OUT_DIR=results/dpe_full35k PER_GROUP=0 N_IMAGES=0 DATASET_PATH="$DATASET_PATH" bash shell/run_dpe_final_eval.sh || true
  else
    todo "on the cluster (~2 weeks GPU): OUT_DIR=results/dpe_full35k PER_GROUP=0 N_IMAGES=0 bash shell/run_dpe_final_eval.sh"
    echo "        (select alpha* first with shell/run_dpe_ablation_rolf.sh), then pull:"
    echo "        rsync -avz $REMOTE:$REMOTE_DIR/results/dpe_full35k/ results/dpe_full35k/"
  fi
fi

# --- Stage 4: embeddings for the UMAP (optional) -----------------------------
say "Stage 4 — visual-token embeddings (optional, for the UMAP figure)"
if ls results/dpe_embeddings/*_emb.npz >/dev/null 2>&1; then ok "embedding dumps present"
else todo "on the cluster: python3 scripts/dump_visual_embeddings.py --model ... --alpha a* --device-map cuda  (per model)"; fi

# --- Stage 5: analysis -> all tables & figures -------------------------------
say "Stage 5 — regenerate all tables & figures from the DBs"
if ls results/dpe_full35k/*/*_dpe.db >/dev/null 2>&1 || ls results/single_runs_35k/gpu*_*.db >/dev/null 2>&1; then
  bash shell/reproduce_dpe.sh
else
  echo -e "  ${Y}skipped${N} — no DBs yet (finish Stage 2/3 or pull them), then re-run this script."
fi

# --- Stage 6: compile the paper ---------------------------------------------
say "Stage 6 — compile the paper (aaai_build/main.tex)"
if command -v pdflatex >/dev/null 2>&1; then
  ( cd aaai_build && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
      && bibtex main >/dev/null 2>&1; pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
      && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 )
  [ -f aaai_build/main.pdf ] && ok "built aaai_build/main.pdf" || todo "pdflatex ran with errors — see aaai_build/main.log"
else
  todo "install a LaTeX toolchain, then:  cd aaai_build && pdflatex main && bibtex main && pdflatex main && pdflatex main"
fi

say "Done"
echo "Headline DPE result (full corpus): IDEFICS2 -12% (genuine), InternVL2 ~0 (artifact), LLaVA ~0 (resistant)."
echo "Artifacts: figures/dpe_aaai/  +  aaai_build/ .  Full guide: docs/REPRODUCE_DPE.md"
