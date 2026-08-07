# Generate Paper-Style Figures on Rolf

This guide shows how to generate publication figures matching your paper's exact style.

## What Gets Generated

The script creates 6 high-quality figures in **both PDF and PNG formats** (300 DPI):

1. **fig1_radar_fingerprints** (.pdf + .png) - 5-axis radar plot showing disparity by probe
2. **fig2_composite_leaderboard** (.pdf + .png) - Horizontal bar chart ranking models
3. **fig3_disparity_heatmap** (.pdf + .png) - Model × Probe heatmap with color coding
4. **fig4_regional_bias** (.pdf + .png) - Regional error rate analysis (6 subplots)
5. **fig5_effect_sizes** (.pdf + .png) - Grouped bar chart of effect sizes by probe
6. **fig6_worst_best_groups** (.pdf + .png) - Gap analysis between best/worst treated groups

Plus: **dataset_metadata.txt** - Complete dataset statistics for updating your paper

**Total**: 12 figure files (6 PDF + 6 PNG) + metadata

---

## Step 1: Upload Script to Rolf

```bash
# From your local machine
rsync -avz --progress \
    scripts/generate_paper_style_figures.py \
    generate_paper_figures.sh \
    rolf:/local/scratch/alali/FingerPrint/
```

---

## Step 2: SSH to Rolf and Generate Figures

```bash
ssh rolf
cd /local/scratch/alali/FingerPrint

# Make script executable
chmod +x generate_paper_figures.sh

# Run the generation
./generate_paper_figures.sh
```

Expected output:
```
======================================================================
FingerPrint² Paper-Style Figure Generation
======================================================================
Results directory: results/single_runs_35k
Output directory:  figures/paper_style

📊 Loading model data...
  Loading HuggingFaceM4_idefics2_8b...
    ✓ Loaded 175,945 results
  Loading meta_llama_Llama_3.2_11B_Vision_Instruct...
    ✓ Loaded 175,945 results
  ...

✓ Successfully loaded 5 models

🔢 Computing disparity scores...
  IDEFICS2-8B...
  Llama-3.2-11B...
  ...

🎨 Generating figures...

1. Creating radar fingerprints...
  ✓ Saved: fig1_radar_fingerprints.pdf + .png

2. Creating composite leaderboard...
  ✓ Saved: fig2_composite_leaderboard.pdf + .png

3. Creating disparity heatmap...
  ✓ Saved: fig3_disparity_heatmap.pdf + .png

4. Creating regional bias analysis...
  ✓ Saved: fig4_regional_bias.pdf + .png

5. Creating effect size analysis...
  ✓ Saved: fig5_effect_sizes.pdf + .png

6. Creating worst/best groups analysis...
  ✓ Saved: fig6_worst_best_groups.pdf + .png

📝 Generating dataset metadata...
  ✓ Saved: dataset_metadata.txt

======================================================================
✅ All figures generated successfully!
======================================================================
Output directory: figures/paper_style
Generated 6 PDF figures
Generated 6 PNG figures
Metadata saved to: figures/paper_style/dataset_metadata.txt
```

This should take 3-5 minutes.

---

## Step 3: Download Figures Locally

```bash
# Exit from rolf
exit

# Download all generated files
mkdir -p figures/paper_style
rsync -avz --progress \
    rolf:/local/scratch/alali/FingerPrint/figures/paper_style/ \
    ./figures/paper_style/
```

---

## Step 4: View Results

```bash
# List generated figures
ls -lh figures/paper_style/

# View metadata
cat figures/paper_style/dataset_metadata.txt

# Open figures
open figures/paper_style/fig1_radar_fingerprints.pdf
open figures/paper_style/fig2_composite_leaderboard.pdf
open figures/paper_style/fig3_disparity_heatmap.pdf
```

---

## Dataset Metadata for Paper

The `dataset_metadata.txt` file contains complete statistics:

- **Total images**: Exact count from your evaluation
- **Jurisdictions**: Number and list of jurisdictions
- **Regional distribution**: Image counts per region with percentages
- **Probes**: List of all evaluated probes
- **Age/Gender distribution**: If available in your data
- **Models evaluated**: All 5 models with full IDs
- **Total probe results**: Complete statistics

Use this to update your paper's dataset description and methods section.

---

## Figure Styles

All figures match your paper's style:

### Radar Plot (Fig 1)
- 5 axes: Occupation, Education, Trust, Lifestyle, Neighbourhood
- Filled areas with transparency
- Color-coded by model (green, blue, purple, orange, red)
- Scale 0-0.5
- Legend top-right

### Leaderboard (Fig 2)
- Horizontal bars sorted by composite score
- Color gradient from green (best) to red (worst)
- Values displayed at bar ends
- Threshold lines for medium/high severity
- "Lower is better" axis label

### Heatmap (Fig 3)
- Models (rows) × Probes (columns)
- Green-to-red colormap (0.0 to 0.6)
- White gridlines between cells
- Annotated with 2-decimal values
- Contrasting text colors (white on dark, black on light)

### Regional Bias (Fig 4)
- 6 subplots (one per model or region)
- Bar charts with error rates by region
- Color-coded by model
- Rotated x-axis labels

### Effect Sizes (Fig 5)
- Grouped bar chart
- 5 groups (one per probe)
- 5 bars per group (one per model)
- Threshold lines for medium/large effects

### Best/Worst Groups (Fig 6)
- Paired bars (worst vs best)
- Gap indicators with delta values
- Green for best, red for worst
- Connecting dashed lines

---

## Alternative: Direct Python Execution

If you prefer to run Python directly instead of the shell script:

```bash
ssh rolf
cd /local/scratch/alali/FingerPrint

python3 scripts/generate_paper_style_figures.py \
    --results results/single_runs_35k/ \
    --output figures/paper_style/
```

---

## Troubleshooting

### Missing Dependencies

If you see import errors:
```bash
pip install matplotlib seaborn pandas numpy scipy --user
```

### No Database Files Found

Check that your results directory exists:
```bash
ls -lh results/single_runs_35k/*.db
```

Should show 5 .db files (one per model).

### Connection Issues

If `rsync` fails, check VPN connection:
```bash
ping rolf.ifi.uzh.ch
ssh alali@rolf.ifi.uzh.ch
```

---

## Quick Command Reference

```bash
# Upload
rsync -avz scripts/generate_paper_style_figures.py generate_paper_figures.sh rolf:/local/scratch/alali/FingerPrint/

# Generate on rolf
ssh rolf "cd /local/scratch/alali/FingerPrint && ./generate_paper_figures.sh"

# Download
rsync -avz rolf:/local/scratch/alali/FingerPrint/figures/paper_style/ ./figures/paper_style/

# View metadata
cat figures/paper_style/dataset_metadata.txt
```
