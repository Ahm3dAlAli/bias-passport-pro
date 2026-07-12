---
name: project-dpe-contribution
description: Novel DPE (Demographic Positional Encoding) implementation added to FingerPrint²
metadata:
  type: project
---

Implemented Demographic Positional Encoding (DPE) — a novel inference-time debiasing technique for VLMs.

**Why:** User requested a novel contribution to the paper: demographic-aware positional encodings that minimize bias.
**How to apply:** When discussing the paper contribution, this is the novel method. When extending it, the key files are in fingerprint_squared/debiasing/.

## Algorithm
1. Load `judge_scores` from baseline DBs, group by (gender_presentation, jurisdiction_region)
2. Compute grand mean μ and per-group mean μ_g across (valence, stereotype_alignment, confidence)
3. Correction vector: δ_g = μ − μ_g  (direction toward demographic parity)
4. Project to embedding space: ε_g = W @ δ_g  (W = fixed orthogonal random matrix, seeded)
5. At inference: visual_tokens += α * ε_g via PyTorch forward hook on vision tower

## Files created
- `fingerprint_squared/debiasing/__init__.py`
- `fingerprint_squared/debiasing/demographic_positional_encoder.py` — DemographicPositionalEncoder class
- `fingerprint_squared/debiasing/dpe_vlm.py` — DPEWrappedHuggingFaceVLM with forward hooks
- `scripts/run_dpe_benchmark.py` — Re-run evaluation pipeline with DPE applied
- `scripts/compare_dpe_baseline.py` — Statistical comparison + 4 publication-ready figures

## Hook targets by model family
- LLaVA: model.model.mm_projector (preferred) or model.model.vision_tower
- Qwen2-VL: model.visual
- InternVL: model.vision_model
- SmolVLM: model.model.vision_model

## Key numbers (LLaVA 7B baseline)
- Africa valence 0.44 vs grand mean 0.49 → correction +0.05
- Male valence 0.46 vs grand mean 0.49 → correction +0.03
- Region disparity σ=0.048, Gender disparity σ=0.073

## Rolf workflow (3 orchestration scripts at repo root)
- `sync_dpe_to_rolf.sh` — upload debiasing module + DPE scripts to rolf (does NOT touch baseline DBs)
- `run_dpe_on_rolf.sh` — run ON rolf (in screen). Loops all models: DPE benchmark then compare. Tunables: GPU, ALPHA, N_IMAGES, DATASET_PATH env vars.
- `sync_dpe_from_rolf.sh` — download results/dpe_* back to laptop

Rolf: alali@rolf.ifi.uzh.ch:/local/scratch/alali/FingerPrint, conda env "fingerprint".
Dataset on rolf: /local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres (files: main_<uid>.png; image_id == uid).
Baseline was run with --gpu N --4bit; DPE runner matches (has --gpu, --4bit flags).

Full run:
```bash
./sync_dpe_to_rolf.sh
ssh rolf 'cd /local/scratch/alali/FingerPrint && screen -dmS dpe ./run_dpe_on_rolf.sh'
./sync_dpe_from_rolf.sh   # after it finishes
```

## Direct usage
```bash
python scripts/run_dpe_benchmark.py --model llava-hf/llava-v1.6-vicuna-7b-hf \
  --baseline-db results/single_runs_35k/gpu7_llava_*.db \
  --dataset-path <fullres> --out-db results/dpe/llava_dpe.db \
  --alpha 1.5 --gpu 0 --4bit --n-images 500
python scripts/compare_dpe_baseline.py --baseline-db results/single_runs_35k/gpu7_llava_*.db \
  --dpe-db results/dpe/llava_dpe.db --out-dir results/dpe_comparison/llava
```
