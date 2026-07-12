# FingerPrint² Paper Figures - Informative Visualizations

Generated on: 2026-06-01

## Dataset Summary
- **35,189 images** from **61 jurisdictions**
- **6 regions**: Africa (45.7%), Asia (42.3%), Europe (6.6%), Americas (3.1%), N. America (1.6%), Oceania (0.7%)
- **5 social probes**: P1 (Occupation), P2 (Education), P3 (Trustworthiness), P4 (Lifestyle), P5 (Neighbourhood)
- **4 VLM models**: IDEFICS2-8B, LLaVA-v1.6-7B, Llama-3.2-11B, InternVL2-2B
- **703,780 total evaluations** (175,945 per model)

---

## Publication-Quality Figures

### Figure 1: Composite Disparity Leaderboard + Heatmap
**Files**: `fig1_composite_and_heatmap.pdf` / `.png`

Side-by-side visualization combining:
- **Left panel**: Ranked leaderboard showing composite disparity scores per model
- **Right panel**: Model × Probe heatmap with color-coded max-min valence gaps

This figure provides both high-level model comparison and detailed probe-specific insights.

---

### Figure 2: Radar Fingerprints
**Files**: `fig2_radar_fingerprints.pdf` / `.png`

Five individual radar plots (one per model) showing characteristic fairness fingerprints across all 5 socio-economic probes. Each model exhibits unique disparity patterns, visualized as filled polygons on P1-P5 axes.

---

### Figure 3: Variance by Probe + Region
**Files**: `fig3_variance_by_probe_and_region.pdf` / `.png`

Side-by-side grouped bar charts showing:
- **Left panel**: Valence gap (max-min) for each of the 5 probes across all models
- **Right panel**: Mean valence (averaged across probes) for each of the 6 geographic regions

This dual view reveals both probe-specific and region-specific bias patterns.

---

## File Formats
- **PDF**: Vector graphics, publication-ready
- **PNG**: 300 DPI raster graphics for presentations/slides

All figures use consistent color coding:
- IDEFICS2-8B: Green (#2ecc71)
- Llama-3.2-11B: Blue (#3498db)
- InternVL2-2B: Purple (#9b59b6)
- LLaVA-v1.6-7B: Orange (#e67e22)
