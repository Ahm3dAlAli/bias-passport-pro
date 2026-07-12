# AAAI 2027 Reproducibility Checklist — filled
Paper: Beyond a Single Verdict: Multi-Dimensional Bias Fingerprints for Responsible VLM Deployment

## 1. General paper structure
- Conceptual outline / pseudocode of methods: **YES** (scoring functions in closed form; verbatim reference code + lexicons in appendix)
- Opinions/hypotheses delineated from facts: **YES** (five premises framed as argument; empirics in Results)
- Pedagogical references for background: **YES** (SCM, implicit-association literature)

## 2. Theoretical contributions
**Does the paper make theoretical contributions? NO** — benchmark + empirical audit + methodology argument; no formal theorems. Items 2a–2h: NA.

## 3. Datasets
**Relies on datasets? YES**
- Motivation for chosen dataset: **YES** (FHIBE = consented, self-reported demographics)
- Novel dataset in data appendix: **NA** (no novel dataset)
- Novel dataset public w/ research license: **NA**
- Existing datasets cited w/ link/DOI: **YES** (FHIBE cited; corpus stats in appendix)
- Non-public datasets described + why no alternative: **NA** (FHIBE access terms in Ethics)

## 4. Computational experiments
**Includes computational experiments? YES**
- Hyperparameter ranges + selection criterion: **YES** (decoding defaults temp 0.7 / top-p 0.9 / 256 tok; scoring has no learned hyperparameters)
- Preprocessing code in appendix: **YES** (reference impl, seed 42)
- All experiment/analysis source code in appendix: **PARTIAL** (scoring/analysis code + lexicons in appendix; full harness repo not bundled in anonymous submission)
- Source code will be public on publication: **YES**
- New-method code commented + cross-referenced: **YES**
- Seed-setting described: **YES** (bootstrap seed 42; scoring pipeline deterministic/bit-exact)
- Computing infrastructure specified: **PARTIAL** (A100 GPUs, 4-bit quant stated; exact library versions not enumerated in anon submission)
- Evaluation metrics formally described + motivated: **YES** (max–min disparity, composite, Cohen's d, Kruskal–Wallis H)
- Number of runs per result stated: **YES** (full corpus once/model; deterministic scoring → repeats identical)
- Beyond single-number summaries (variation/CI): **YES** (KW, Dunn's+Holm, η²_H, bootstrap 95% CIs)
- Significance judged with appropriate tests: **YES** (Kruskal–Wallis + Bonferroni α_adj≈0.00067; Dunn's post-hoc)
- All final hyperparameters listed: **YES** (decoding params + full scoring spec in appendix)

## Honest caveats (also in the paper)
- Bootstrap unit is the **image**, not the subject (DBs expose image IDs only) — image-level CIs may understate within-subject correlation.
- **IDEFICS2-8B** disparities computed on 33.8% scorable inputs (59,495/175,945), reported with explicit coverage caveat; coverage analysed as a fairness quantity.
