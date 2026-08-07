# shell/ — orchestration & sync scripts

All shell scripts now live here. **Invoke them from the repository root**, e.g.:

```bash
bash shell/sync_dpe_to_rolf.sh
bash shell/sync_dpe_from_rolf.sh
```

The self-locating scripts resolve the repo root as `$(dirname "$0")/..`, so paths
still work as long as the folder layout (`shell/`, `scripts/`, `fingerprint_squared/`,
`results/`) is intact.

## Active DPE pipeline
| script | runs on | purpose |
|---|---|---|
| `sync_dpe_to_rolf.sh`     | laptop | push DPE code + orchestration scripts to rolf (one OTP) |
| `sync_dpe_from_rolf.sh`   | laptop | pull `results/dpe_*` back to the laptop |
| `run_dpe_final_eval.sh`   | rolf   | region-balanced DPE eval at optimal α per model |
| `run_dpe_ablation_rolf.sh`, `run_dpe_ablation_one.sh` | rolf | α-sweep to select α★ |
| `run_dpe_on_rolf.sh`, `run_dpe_parallel_rolf.sh`, `run_dpe_alpha_sweep_rolf.sh` | rolf | earlier DPE run variants |

Note: `scripts/dump_visual_embeddings.py` (visual-token dump for the UMAP) is a
Python script under `scripts/`, run on rolf in a model's conda env — not here.

## Legacy / one-off
`launch_gpu*.sh`, `generate_*_figures.sh`, `sync_to_rolf.sh` / `sync_from_rolf.sh`
(superseded by the `*_dpe_*` variants), `RUN_*_WORKFLOW.sh`, `check_*.sh`,
`extract_paper_stats.sh`, `commit_changes.sh` — retained for reference; some use
hardcoded paths or reference sibling scripts by name and may need adjustment if reused.
