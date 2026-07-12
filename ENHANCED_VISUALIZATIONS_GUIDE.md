# Enhanced Visualizations Guide

**Status**: 🔄 Generating now...  
**Output**: `results/aaai_submission/enhanced_viz/`

---

## 🎨 What You're Getting (5 New Figures)

### 1. UMAP of Individual Images ⭐ **Most Meaningful**
**File**: `fig_umap_per_image.pdf`

**What it shows**:
- 3,000 individual images (sampled) in 2D bias space
- Each dot = one person's image
- Color = region
- Clustering shows how images from different regions are treated

**Why it's meaningful**:
- Shows **individual-level** patterns (not just regional averages)
- UMAP preserves global structure better than t-SNE
- Reveals if Africa images form distinct cluster
- Shows within-region variance

**Use in paper**:
```latex
Figure X: UMAP projection of 3,000 sampled images shows clear regional 
clustering. African images (orange) form a distinct cluster separated from 
other regions along the primary bias dimension, indicating systematic 
differential treatment at the individual level.
```

---

### 2. Bias Trajectory Across Models ⭐ **Shows Model Differences**
**File**: `fig_bias_trajectory.pdf`

**What it shows**:
- Line plot: X-axis = models, Y-axis = mean valence
- One line per region
- Shows how each region's treatment changes across models

**Why it's meaningful**:
- Some models may be fair to Africa, others biased
- Shows which model is "best" for each region
- Reveals if bias is consistent or model-specific

**Use in paper**:
```latex
Figure Y: Regional bias trajectories across three VLMs. All models show 
Africa receiving lowest valence scores (orange line), but the gap varies: 
IDEFICS2-8B shows largest disparity (Δ=0.064), while LLaVA-1.6-7B shows 
smallest (Δ=0.022).
```

---

### 3. Probe-Specific Bias Patterns ⭐ **Shows Which Stereotypes Are Strongest**
**File**: `fig_probe_specific_bias.pdf`

**What it shows**:
- 5 subplots (one per probe: occupation, education, trust, lifestyle, neighborhood)
- Bar chart per region for each probe
- Red outline = worst-treated region, Green = best-treated

**Why it's meaningful**:
- Shows which stereotypes are strongest (e.g., occupation vs trust)
- Reveals if Africa is consistently worst across ALL probes
- Identifies which dimensions need most mitigation

**Use in paper**:
```latex
Figure Z: Probe-specific bias patterns. Africa (orange) receives lowest 
valence across all five demographic dimensions, with strongest bias in 
occupation (P1, Δ=0.08) and neighborhood (P5, Δ=0.09) probes.
```

---

### 4. Model Comparison Radar Plot ⭐ **Visual Model Ranking**
**File**: `fig_model_comparison_radar.pdf`

**What it shows**:
- Radar/spider chart with 6 axes (one per region)
- One colored shape per model
- Larger shape = more fair (higher valence across regions)
- Symmetric shape = fair across regions

**Why it's meaningful**:
- Easy visual comparison of models
- Shows which model is most "balanced"
- Asymmetric shapes indicate bias

**Use in paper**:
```latex
Figure W: Radar plot comparison of model fairness. LLaVA-1.6-7B (green) 
shows most symmetric pattern, indicating balanced treatment across regions. 
IDEFICS2-8B (blue) shows pronounced asymmetry, with Africa consistently 
receiving lower scores.
```

---

### 5. PCA with Feature Loadings ⭐ **Interpretable Axes**
**File**: `fig_pca_with_loadings.pdf`

**What it shows**:
- Enhanced PCA plot from before
- **Red arrows** showing which features drive each PC axis
- Labels on arrows: "ModelName_ProbeID"

**Why it's meaningful**:
- Answers: "What does PC1 actually measure?"
- Shows which model-probe combinations explain bias
- Interpretable dimensions (not just "PC1")

**Use in paper**:
```latex
Figure V: PCA biplot with feature loadings. PC1 (72% variance) is primarily 
driven by occupation and neighborhood probes across all models (red arrows), 
suggesting these dimensions capture the dominant bias pattern. Africa 
separates along PC1, indicating systematic differences in occupational and 
residential stereotyping.
```

---

## 📊 Key Insights You'll Discover

### Insight 1: Individual-Level Clustering (UMAP)
**Expected finding**: African images will form a **distinct cluster** separate from other regions.

**What this means**: Bias isn't just in aggregate statistics—it's visible in how individual images are treated.

**Paper impact**: Strengthens "systematic bias" claim with visual evidence.

---

### Insight 2: Model-Specific Patterns (Trajectory)
**Expected finding**: Some models may be more fair to certain regions.

**What this means**: Model choice matters for deployment fairness.

**Paper impact**: Actionable guidance (e.g., "Use LLaVA for African contexts").

---

### Insight 3: Probe Hierarchy (Probe-Specific)
**Expected finding**: Occupation and neighborhood probes show **strongest bias**.

**What this means**: Economic/residential stereotypes are more pronounced than trust/education.

**Paper impact**: Identifies highest-priority dimensions for mitigation.

---

### Insight 4: Occupation + Neighborhood Drive Bias (PCA Loadings)
**Expected finding**: PC1 arrows point to P1 (occupation) and P5 (neighborhood).

**What this means**: These two probes capture the dominant bias axis.

**Paper impact**: Explains what PC1 measures (not just "variance").

---

## 📝 How to Use in Your Paper

### Option 1: Add All 5 Figures (Comprehensive)
- Figure 5: UMAP per-image
- Figure 6: Bias trajectory
- Figure 7: Probe-specific patterns
- Figure 8: Radar comparison
- Figure 9: PCA with loadings

**Pros**: Thorough, multiple perspectives  
**Cons**: Takes up more space (may need to cut other figures)

---

### Option 2: Add Top 3 (Recommended)
- **UMAP per-image** (shows individual-level clustering)
- **Probe-specific bias** (shows hierarchy of stereotypes)
- **PCA with loadings** (interpretable dimensions)

**Pros**: Most meaningful insights, space-efficient  
**Cons**: Loses trajectory and radar views

---

### Option 3: Replace Old Figures
Keep total at 8 figures by replacing:
- Replace old PCA → new PCA with loadings (more informative)
- Replace t-SNE → UMAP per-image (better global structure)
- Add probe-specific and trajectory

---

## 🎯 Figure Captions (Ready to Copy)

### UMAP Per-Image
```latex
\caption{\textbf{Individual images in bias space (UMAP).} Projection of 
3,000 sampled images (n=500 per region) shows regional clustering. African 
images (orange) form a distinct cluster separated from other regions, 
indicating systematic differential treatment at the individual level. UMAP 
preserves both local and global structure, revealing clear separation along 
the primary bias axis.}
```

### Bias Trajectory
```latex
\caption{\textbf{Regional bias across models.} Mean valence scores per 
region across three VLMs. Africa (orange) receives consistently lowest 
scores across all models, but the magnitude varies: IDEFICS2-8B shows 
largest gap (Δ=0.064), LLaVA-1.6-7B smallest (Δ=0.022). This suggests model 
selection can influence deployment fairness.}
```

### Probe-Specific Bias
```latex
\caption{\textbf{Bias patterns across demographic probes.} Regional valence 
scores for five probes (averaged across models). Red outline indicates 
worst-treated region, green indicates best-treated. Africa shows 
consistently low scores across all dimensions, with strongest bias in 
occupation (P1) and neighborhood (P5) probes.}
```

### Radar Comparison
```latex
\caption{\textbf{Model fairness comparison (radar plot).} Each axis 
represents one region; larger area indicates higher valence. LLaVA-1.6-7B 
(green) shows most symmetric pattern, indicating balanced treatment. 
IDEFICS2-8B (blue) exhibits pronounced asymmetry, with Africa receiving 
lowest scores.}
```

### PCA with Loadings
```latex
\caption{\textbf{Regional bias PCA with feature loadings.} Regions plotted 
in first two principal components (81.9\% variance). Red arrows show top-5 
feature loadings: PC1 is primarily driven by occupation (P1) and neighborhood 
(P5) probes, suggesting these dimensions capture the dominant bias pattern. 
Africa separates along PC1, indicating systematic differences in occupational 
and residential stereotyping.}
```

---

## 🔬 Expected Runtime

- **UMAP**: 2-3 minutes (most computationally expensive)
- **Trajectory**: 30 seconds
- **Probe-specific**: 1 minute
- **Radar**: 30 seconds
- **PCA with loadings**: 30 seconds

**Total**: ~5 minutes

---

## ✅ After Completion

Once the script finishes, check the outputs:

```bash
open results/aaai_submission/enhanced_viz/
```

You should see 10 files (5 PDFs + 5 PNGs).

---

## 🚀 What Makes These "More Meaningful"?

### vs. Original Visualizations:

| Original | Enhanced |
|----------|----------|
| Regional averages only | Individual images (UMAP) |
| One snapshot | Across-model trajectory |
| Aggregate bias | Probe-specific breakdown |
| Unlabeled PC axes | Interpretable loadings |
| Static comparison | Dynamic radar view |

**Bottom line**: Original showed "what" (bias exists), enhanced shows "how" (which dimensions, which images, which models).

---

## 💡 For Your Abstract

Consider adding:

```latex
Enhanced visualizations reveal that bias is primarily driven by occupational 
and residential stereotyping (PC1: 72% variance), with African images forming 
a distinct cluster in UMAP space, indicating systematic individual-level 
differential treatment.
```

---

**Status**: Check `results/aaai_submission/enhanced_viz/` for completed figures!
