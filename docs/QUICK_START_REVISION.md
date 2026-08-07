# 🚀 Quick Start: AAAI Revision

## Run All Critical Analyses

```bash
./RUN_REVISION_ANALYSES.sh
```

**Runtime**: ~10-15 minutes
**Output**: `results/aaai_submission/aaai_figures/`

---

## What Gets Created

### 📊 New Analyses (Addressing Review)

1. **IDEFICS2 Coverage** (Question #2)
   - `idefics2_coverage_heatmap.pdf`
   - Coverage rates by region × probe
   - MNAR diagnostics

2. **Multi-Metric Benchmark** (Questions #4, #5, #6)
   - `multi_metric_dashboard.pdf`
   - 5 metrics: valence, economic, stereotype, probe-specific, refusal
   - Bootstrap 95% CIs

3. **Intersectional Analysis** (Question #7)
   - `interaction_heatmap_*.pdf` (per model)
   - `consistency_comparison.pdf`
   - Region × Probe interactions

4. **Baseline Figures**
   - All 15 original AAAI figures

---

## Review Results

```bash
# Open all figures
open results/aaai_submission/aaai_figures/

# Check multi-metric scores
cat results/aaai_submission/aaai_figures/multi_metric_results.json | jq

# Check intersectional findings
cat results/aaai_submission/aaai_figures/intersectional_analysis.json | jq
```

---

## What This Addresses

✅ **6/8 reviewer questions** answered
✅ **Multi-dimensional bias** (5 metrics)
✅ **Interaction effects** (Region × Probe)
✅ **Robustness** (bootstrap CIs)
✅ **Probe-specific** scoring (200+ terms)
✅ **IDEFICS2** coverage clarified

---

## Next Steps

### Week 2 (TODO):
1. **LLM-as-judge comparison** (Question #3)
2. **Qualitative neighbourhood audit** (Question #8)

### Paper Updates Needed:
1. Add multi-metric methods section
2. Add intersectional analysis results
3. Add IDEFICS2 coverage clarification
4. Update figures (use new 20+ PDFs)

See [REVISION_IMPLEMENTATION_SUMMARY.md](REVISION_IMPLEMENTATION_SUMMARY.md) for full details.

---

## 📋 Files Created

| File | Purpose |
|------|---------|
| `RUN_REVISION_ANALYSES.sh` | Master script (run this!) |
| `scripts/analyze_idefics2_coverage.py` | IDEFICS2 MNAR analysis |
| `scripts/multi_metric_benchmark.py` | 5-metric benchmark |
| `scripts/intersectional_analysis.py` | Region × Probe interactions |
| `REVISION_IMPLEMENTATION_SUMMARY.md` | Full technical docs |
| `REVIEW_RESPONSE_PLAN.md` | 3-week timeline |

---

## ⚡ TL;DR

1. Run: `./RUN_REVISION_ANALYSES.sh`
2. Wait: ~10-15 min
3. Review: `open results/aaai_submission/aaai_figures/`
4. Update paper with new analyses
5. Resubmit 🎉
