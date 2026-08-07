# ONE COMMAND TO RUN EVERYTHING

## Just Do This:

```bash
./RUN_AAAI_ANALYSIS.sh
```

That's it. Wait 30 minutes.

---

## What You Get:

```
results/aaai_submission/
├── statistical_analysis_summary.txt  ← Copy-paste stats for paper
├── figures/                          ← 4 publication-ready figures
│   ├── fig1_worst_best_regional_sentiment.pdf
│   ├── fig2_regional_heatmap.pdf
│   ├── fig3_probe_comparison.pdf
│   └── fig4_model_leaderboard.pdf
├── validation_sample.csv             ← Upload to MTurk ($90)
├── qualitative_examples.json         ← Example responses for paper
└── [more files...]
```

---

## Then What?

1. **This week**: Upload `validation_sample.csv` to MTurk (~$90)
2. **Next 3 weeks**: Write paper (see `READY_FOR_AAAI_CHECKLIST.md`)
3. **Week 4**: Submit to AAAI 2027 AISI Track

---

## Need Help?

- **Full details**: See `COMPLETE_IMPLEMENTATION_SUMMARY.md`
- **Research plan**: See `AAAI_RESEARCH_METHODOLOGY.md`
- **Timeline**: See `READY_FOR_AAAI_CHECKLIST.md`

---

## Troubleshooting

**"Results directory not found"**
→ Edit line 15 in `RUN_AAAI_ANALYSIS.sh` with your path

**"Module not found"**
→ `pip install scipy pandas numpy matplotlib seaborn statsmodels`

**Other errors**
→ Check `COMPLETE_IMPLEMENTATION_SUMMARY.md` for details

---

## That's All!

Run the script. Get results. Write paper. Submit.

**You're 30 minutes away from having everything for AAAI submission.** 🚀
