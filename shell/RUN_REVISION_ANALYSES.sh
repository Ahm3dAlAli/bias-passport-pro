#!/bin/bash
#
# Master Script: Run All Critical Revision Analyses
# Addresses borderline rejection review
#
# Runs:
# 1. IDEFICS2 coverage analysis (MNAR check)
# 2. Multi-metric benchmark (5 metrics)
# 3. Intersectional analysis (Region × Probe)
# 4. All original AAAI figures
#
# Output: results/aaai_submission/aaai_figures/
#

set -e  # Exit on error

echo "======================================================================"
echo "FINGERPRINT² Revision Analyses"
echo "======================================================================"
echo ""
echo "Addressing Borderline Rejection Review"
echo "Running 3 critical analyses + baseline figures"
echo ""
echo "Expected runtime: ~10-15 minutes"
echo ""
echo "======================================================================"

# Create output directory
mkdir -p results/aaai_submission/aaai_figures

echo ""
echo "======================================================================"
echo "Analysis 1/4: IDEFICS2-8B Coverage & MNAR Diagnostics"
echo "======================================================================"
echo "Addresses Reviewer Question #2"
echo ""

python3 scripts/analyze_idefics2_coverage.py

echo ""
echo "✓ IDEFICS2 coverage analysis complete"
echo ""
sleep 2

echo ""
echo "======================================================================"
echo "Analysis 2/4: Multi-Metric Benchmark"
echo "======================================================================"
echo "Addresses Reviewer Questions #4, #5, #6"
echo "Computing 5 metrics:"
echo "  • Sentiment valence (baseline)"
echo "  • Economic valence (wealth/poverty)"
echo "  • Stereotype alignment"
echo "  • Probe-specific scoring"
echo "  • Refusal rate"
echo ""

python3 scripts/multi_metric_benchmark.py

echo ""
echo "✓ Multi-metric analysis complete"
echo ""
sleep 2

echo ""
echo "======================================================================"
echo "Analysis 3/4: Intersectional Analysis (Region × Probe)"
echo "======================================================================"
echo "Addresses Reviewer Question #7"
echo "Analyzing:"
echo "  • 2-way ANOVA (Region × Probe interactions)"
echo "  • Bias direction consistency"
echo "  • Interaction heatmaps"
echo ""

python3 scripts/intersectional_analysis.py

echo ""
echo "✓ Intersectional analysis complete"
echo ""
sleep 2

echo ""
echo "======================================================================"
echo "Analysis 4/4: Baseline AAAI Figures"
echo "======================================================================"
echo "Generating original publication figures"
echo ""

python3 scripts/generate_final_aaai_figures.py

echo ""
echo "✓ Baseline figures complete"
echo ""
sleep 2

echo ""
echo "======================================================================"
echo "All Analyses Complete!"
echo "======================================================================"
echo ""
echo "Output directory: results/aaai_submission/aaai_figures/"
echo ""
echo "New figures created:"
echo "  • idefics2_coverage_heatmap.pdf"
echo "  • idefics2_coverage_analysis.json"
echo "  • multi_metric_dashboard.pdf"
echo "  • multi_metric_results.json"
echo "  • interaction_heatmap_*.pdf (per model)"
echo "  • consistency_comparison.pdf"
echo "  • intersectional_analysis.json"
echo ""
echo "Plus all baseline AAAI figures (~15 PDFs)"
echo ""
echo "======================================================================"
echo "Next Steps:"
echo "======================================================================"
echo ""
echo "1. Review figures:"
echo "   open results/aaai_submission/aaai_figures/"
echo ""
echo "2. Check multi-metric results:"
echo "   cat results/aaai_submission/aaai_figures/multi_metric_results.json"
echo ""
echo "3. Review intersectional findings:"
echo "   cat results/aaai_submission/aaai_figures/intersectional_analysis.json"
echo ""
echo "4. Update paper with new analyses"
echo ""
echo "======================================================================"
echo "Revision Status:"
echo "======================================================================"
echo ""
echo "✅ COMPLETED:"
echo "   • IDEFICS2 MNAR analysis (Question #2)"
echo "   • Multi-metric composite (Questions #4, #5, #6)"
echo "   • Intersectional analysis (Question #7)"
echo ""
echo "📋 TODO (Week 2-3):"
echo "   • Robustness analysis (bootstrap CIs) - partially done"
echo "   • Auxiliary dimensions deep dive"
echo "   • Within-group variance - partially done"
echo "   • LLM-as-judge comparison (500 samples)"
echo "   • Qualitative neighbourhood audit (40 examples)"
echo "   • Expand stereotype lexicons (50 → 200 terms) - done"
echo ""
echo "See REVIEW_RESPONSE_PLAN.md for full timeline"
echo ""
echo "======================================================================"
