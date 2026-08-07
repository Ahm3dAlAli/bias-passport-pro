# Reproducing the DPE (Demographic Positional Encoding) results

This documents how to reproduce every DPE table and figure from the result
databases, and how those databases were generated on the GPU cluster.

DPE is a training-free, inference-time debiasing method for VLMs: it adds a
per-region correction vector (projected from score space into the vision-token
embedding space by a fixed orthonormal matrix) to the visual tokens at
generation time. The headline finding, **validated on the full 35,189-image
corpus**:

| Model | full-corpus σ reduction | random-2000/reg (5 seeds) | verdict |
|-------|------------------------|---------------------------|---------|
| IDEFICS2-8B  | **+12.0%** | **+11.8 ± 1.1%** | genuine — Africa lifted |
| InternVL2-2B | −2.1%      | −2.8 ± 5.9%      | first-N sampling artifact |
| LLaVA-v1.6-7B| −0.2%      | −0.2 ± 1.0%      | architecturally resistant |

σ = std of the six per-region mean valences (lower = more equitable). The large
InternVL2 reduction reported for the deterministic first-2000/region subsample
does **not** survive random-balanced or full-corpus evaluation — see the
sampling-robustness analysis.

---

## 1. Environment

**Analysis (local, this repo):**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dpe.txt
```

**Generation (GPU cluster "rolf"):** two conda envs, because InternVL2 needs an
older transformers:
- `fingerprint` — transformers 5.x, torch 2.7 (IDEFICS2, LLaVA figures)
- `internvl`    — cloned from `fingerprint`, `transformers==4.44.2` + `accelerate==0.34.2` (InternVL2, LLaVA generation)

---

## 2. Data (result databases)

The DBs are large (100–200 MB each) and **git-ignored** (`results/`). They must
be present locally to run the analysis. Layout:

| Path | What | Rows |
|------|------|------|
| `results/single_runs_35k/gpu0_*idefics2*.db`, `gpu6_*InternVL2*.db`, `gpu7_*llava*.db` | **baseline** (no DPE), full corpus | 175,945 each |
| `results/dpe_full35k/<model>/<model>_dpe.db` | **DPE at α★**, full 35,189-image corpus | 175,945 each |
| `results/dpe_final_20260720_125918/<model>/<model>_dpe.db` | DPE, region-balanced 2000/region (the initial eval) | ~39,520 each |
| `results/dpe_embeddings/<model>_emb.npz` | mean-pooled visual-token embeddings (before/after) for the UMAP | 2,486 imgs |

Each `judge_scores` row is one (image, probe) with `valence`,
`economic_valence`, `stereotype_alignment`, `refusal`, `jurisdiction_region`.

Optimal correction strengths: **IDEFICS2 α★=0.25, InternVL2 α★=0.5, LLaVA α★=0.5**.

Pull the DBs from the cluster:
```bash
bash shell/sync_dpe_from_rolf.sh          # results/dpe_* dirs
# baselines + full35k are large; rsync individually as needed, e.g.:
rsync -avz rolf:/local/scratch/alali/FingerPrint/results/dpe_full35k/ results/dpe_full35k/
```

---

## 3. Regenerate all tables/figures (local, ~1–2 min)

One command:
```bash
bash shell/reproduce_dpe.sh
```
It runs the two analysis entry points below and reports what it wrote.

### 3a. `scripts/dpe_analysis.py` — baseline + Tier-1/2 artifacts
Single CLI, run any step or `all` (dependency order):
```bash
python3 scripts/dpe_analysis.py all
```
| subcommand | outputs |
|------------|---------|
| `baseline-tables` | `figures/dpe_aaai/baseline_tables.tex`, `baseline_fig_data.json`, `baseline_appendix_rows.tex` |
| `paper-figs` | `aaai_build/fig{1,2,3,6,9}_*.pdf` (baseline benchmark figures) |
| `tier1` | `dpe_tier1_tables.tex`, `fig_dpe_sigma_before_after`, `fig_dpe_convergence` (2000/region) |
| `norms` | `dpe_correction_norms.tex` (per-region ‖α·ε_g‖ — LLaVA-resistance evidence) |
| `content` | `dpe_content_preservation.tex` (SBERT response similarity; precision-confound rebuttal) |
| `umap` | `fig_dpe_visual_umap.pdf` (needs `results/dpe_embeddings/*.npz`) |

### 3b. `scripts/dpe_sampling_robustness.py` — the corrected full-corpus results
This is the authoritative headline analysis (needs `results/dpe_full35k/`):
```bash
python3 scripts/dpe_sampling_robustness.py
```
Writes, reproducibly (sorted-then-seeded sampling):
- `dpe_results_fullcorpus.tex` — headline: full σ + random-5-seed (`tab:dpe_results`)
- `dpe_robustness_table.tex`   — first vs random vs full (`tab:dpe_robustness`)
- `fig_dpe_robustness.{pdf,png}` — σ-reduction by sampling (visualizes the artifact)
- regenerates `fig_dpe_sigma_before_after` and `fig_dpe_convergence` on the **full corpus**

Outputs land in `figures/dpe_aaai/` and are copied into `aaai_build/`.

---

## 4. Regenerate the DBs from scratch (GPU cluster)

Only needed to rebuild the underlying data. On rolf:

1. **Baseline** (no DPE): `scripts/run_fhibe_benchmark.py` per model → `single_runs_35k/`.
2. **Select α★**: `shell/run_dpe_ablation_rolf.sh` (α ∈ {0.25,0.5,0.75,1.0,1.5} on a balanced dev set), pick min-σ per model.
3. **DPE eval**: `scripts/run_dpe_benchmark.py --correction-axis region --alpha α★` — full corpus (no `--balanced-per-group`) → `dpe_full35k/`, or `--balanced-per-group 2000 --balance-by region` for the balanced eval. Runs in a screen per model on its GPU/env (`shell/run_dpe_final_eval.sh`). Resumable (skips already-scored images).
4. **Compare**: `scripts/compare_dpe_baseline.py --baseline-db … --dpe-db …` → `compare/dpe_comparison_summary.json`.
5. **Embeddings** (for the UMAP): `scripts/dump_visual_embeddings.py --model … --alpha α★ --device-map cuda` in the model's env → `results/dpe_embeddings/<model>_emb.npz`.

Core code: `fingerprint_squared/debiasing/{demographic_positional_encoder,dpe_hook}.py`
(the encoder builds δ_g = grand_mean − group_mean over valence/stereotype/confidence,
projects via a seeded orthonormal W, and injects α·ε_g via a forward hook on the
vision module).

---

## 5. Key methodological caution

The correction is confined to a ≤3-dim subspace of the vision-token space, so its
effect is a small targeted mean-shift (invisible in a full-embedding UMAP; see
`fig_dpe_visual_umap`). Evaluation is sensitive to **non-random subsampling**:
report region-balanced results as a mean±std over random seeds (not a single
`ORDER BY image_id LIMIT N` draw), and validate against the full corpus. The
`dpe_sampling_robustness.py` first/random/full comparison is the check that
separates a genuine reduction (IDEFICS2) from a sampling artifact (InternVL2).
