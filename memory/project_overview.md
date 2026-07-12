---
name: project-overview
description: FingerPrint² project architecture, DB schema, model families, and evaluation pipeline
metadata:
  type: project
---

FingerPrint² is a VLM bias evaluation framework that runs 5 social inference probes (P1–P5) against HuggingFace and API models and scores responses on valence, stereotype_alignment, and confidence.

**Why:** Research project producing an AAAI submission on demographic bias in VLMs.
**How to apply:** When making changes, the main evaluation scripts are monolithic (run_fhibe_benchmark.py ~6000 lines). The actual result DBs are in results/single_runs_35k/ with schema: `probe_results` + `judge_scores` tables (NOT the fingerprint_squared SQLiteStorage schema which is separate).

## Key facts
- Real DB tables: `probe_results`, `judge_scores` (not `probe_responses`)
- Demographic columns: `gender_presentation`, `jurisdiction_region` (age_group is all "unknown")
- Probes: P1_occupation, P2_education, P3_trustworthiness, P4_lifestyle, P5_neighbourhood
- Models evaluated: LLaVA 1.6 7B, InternVL2 2B, InternVL3 2B, IDEFICS2 8B, LLaMA 3.2 11B
- Baseline results show Africa has lowest valence (~0.44), Europe highest (~0.57)
- Gender: female 0.52, male 0.46, non-binary 0.65 (small n)
- DB paths: results/single_runs_35k/gpu*.db

## Actual DB schema
```sql
probe_results: id, image_id, model_name, probe_id, prompt, response, latency_ms,
               jurisdiction, jurisdiction_region, age_group, gender_presentation, num_persons
judge_scores:  id, image_id, model_name, probe_id, valence, stereotype_alignment, confidence,
               refusal, economic_valence, reasoning, jurisdiction, jurisdiction_region,
               age_group, gender_presentation, num_persons
```
