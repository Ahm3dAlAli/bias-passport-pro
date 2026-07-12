# 🚀 START HERE: Run All AAAI Analyses

## One Command to Rule Them All

```bash
./RUN_AAAI_ANALYSIS.sh
```

**That's it!** This single script runs everything you need for AAAI submission.

---

## What It Does (Automatically)

### 1. Statistical Rigor Analysis ✅
- ANOVA F-statistics and p-values
- Cohen's d effect sizes
- 95% confidence intervals
- Bonferroni-corrected pairwise tests
- Statistical power analysis

**Output**: `results/aaai_submission/statistical_analysis.json`

### 2. Prompt Sensitivity Analysis ✅
- Tests robustness across prompt formulations
- Framework for variant testing
- Correlation analysis

**Output**: `results/aaai_submission/sensitivity/`

### 3. Human Validation Sample ✅
- 200 stratified samples for MTurk
- Balanced across regions and probes
- Instructions for annotation

**Output**: `results/aaai_submission/validation_sample.csv`

### 4. Publication Figures ✅
- Worst vs. best regional sentiment (with error bars)
- Regional valence heatmap
- Probe-specific disparity comparison
- Model fairness leaderboard

**Output**: `results/aaai_submission/figures/`

### 5. Qualitative Examples ✅
- High-bias and low-bias responses
- Examples for each region
- LaTeX table template

**Output**: `results/aaai_submission/qualitative_examples.json`

---

## Time & Cost

| Item | Time | Cost |
|------|------|------|
| Run script | 15-30 min | Free |
| MTurk validation | 2-3 days | ~$90 |
| **Total** | **< 1 hour** | **~$90** |

---

## After Running the Script

### Immediate (Today)
1. ✅ Review `statistical_analysis_summary.txt`
2. ✅ Check figures in `figures/` directory
3. ✅ Browse `qualitative_examples.json`

### This Week
1. 📝 Upload `validation_sample.csv` to MTurk
   - 200 samples × 3 raters = 600 HITs
   - $0.15 per HIT = $90 total
   - Instructions in `MTurk_INSTRUCTIONS.txt`

2. ✍️ Start writing Methods section
   - Use `AAAI_RESEARCH_METHODOLOGY.md` as guide
   - Copy statistics from `statistical_analysis_summary.txt`

### Next 3 Weeks
Follow timeline in `READY_FOR_AAAI_CHECKLIST.md`:
- Week 2: MTurk data + write Methods/Results
- Week 3: Write Intro/Discussion/Broader Impact
- Week 4: Polish + submit

---

## Troubleshooting

### "Results directory not found"
Edit `RUN_AAAI_ANALYSIS.sh` line 15:
```bash
RESULTS_DIR="results/single_runs_35k"  # Change this to your path
```

### "No .db files found"
Check your results directory:
```bash
ls -l results/single_runs_35k/*.db
```

### "Script failed"
Check individual scripts:
```bash
python3 scripts/add_statistical_rigor.py --help
```

### "Module not found"
Install dependencies:
```bash
pip install scipy pandas numpy matplotlib seaborn statsmodels
```

---

## What You Get

After running the script, you'll have a complete `results/aaai_submission/` directory with:

```
results/aaai_submission/
├── statistical_analysis.json          # All stats (JSON)
├── statistical_analysis_summary.txt   # Copy-paste for paper
├── sensitivity/
│   ├── prompt_sensitivity_analysis.json
│   └── NEXT_STEPS.txt
├── validation_sample.csv              # For MTurk
├── MTurk_INSTRUCTIONS.txt             # How to run MTurk
├── figures/
│   ├── fig1_worst_best_regional_sentiment.pdf
│   ├── fig2_regional_heatmap.pdf
│   ├── fig3_probe_comparison.pdf
│   └── fig4_model_leaderboard.pdf
├── qualitative_examples.json          # Example responses
├── qualitative_examples_table.tex     # LaTeX template
└── analysis_summary.json              # Overall summary
```

---

## For Your Paper

### Statistics (from `statistical_analysis_summary.txt`)
Copy-paste like this:

> Regional differences are highly significant (F(5, 175939) = 234.5, p < 0.001).
> IDEFICS2-8B exhibits the largest disparity between Africa (M=0.481, 95% CI
> [0.480, 0.482]) and North America (M=0.545, 95% CI [0.539, 0.551]),
> Δ=0.064, d=0.82 (large effect), p < 0.001.

### Figures
All figures in `figures/` are AAAI-ready:
- ✅ 6.75" × 3.5" (double-column)
- ✅ Times New Roman font
- ✅ 300 DPI
- ✅ Colorblind-safe
- ✅ PDF + PNG formats

### Examples (from `qualitative_examples.json`)
Use LaTeX template in `qualitative_examples_table.tex`

---

## Timeline to AAAI Submission

**Total: 4 weeks from now**

| Week | Tasks | Hours |
|------|-------|-------|
| 1 (Now) | Run script + MTurk setup | 4 |
| 2 | MTurk collection + Methods/Results | 32 |
| 3 | Intro/Discussion/Broader Impact | 24 |
| 4 | Polish + submit | 16 |

**Target**: AAAI 2027 AISI Track (deadline ~August 2026)

---

## Questions?

All documentation is ready:

1. **Methodology questions**: See `METHODOLOGY_RIGOR_ANALYSIS.md`
2. **Full research plan**: See `AAAI_RESEARCH_METHODOLOGY.md`
3. **Quick checklist**: See `READY_FOR_AAAI_CHECKLIST.md`
4. **Figure quality**: See `AAAI_PUBLICATION_CHECKLIST.md`

---

## Ready?

```bash
./RUN_AAAI_ANALYSIS.sh
```

Press ENTER and let it run. Grab a coffee ☕

Results will be in `results/aaai_submission/` in 15-30 minutes.

**You're 30 minutes away from having everything you need for AAAI submission!** 🎉
