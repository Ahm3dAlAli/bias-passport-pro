# Quick Start: Generate Research Figures

## Option 1: Run Locally (if you have the .db files)

If you've already synced the database files from rolf to your local machine:

```bash
# Make sure you have the dependencies
pip install numpy pandas matplotlib seaborn scipy

# Run the figure generation
python3 scripts/generate_research_figures.py \
    --results results/single_runs_35k/ \
    --output figures/research/
```

## Option 2: Run on Rolf (when connection works)

### First, sync the scripts to rolf:
```bash
rsync -avz \
    scripts/generate_research_figures.py \
    generate_research_figures.sh \
    rolf:/local/scratch/alali/FingerPrint/
```

### Then SSH to rolf and run:
```bash
ssh rolf
cd /local/scratch/alali/FingerPrint
chmod +x generate_research_figures.sh
./generate_research_figures.sh
```

### Finally, download the figures:
```bash
# Exit from rolf first (Ctrl+D or type 'exit')
mkdir -p figures/research
rsync -avz rolf:/local/scratch/alali/FingerPrint/figures/research/ ./figures/research/
```

## Option 3: Download DB files first, then generate locally

### Step 1: Download database files from rolf
```bash
mkdir -p results/single_runs_35k
rsync -avz --progress \
    rolf:/local/scratch/alali/FingerPrint/results/single_runs_35k/*.db \
    results/single_runs_35k/
```

This will download:
- gpu0_HuggingFaceM4_idefics2_8b_20260427_114159.db (~140MB)
- gpu3_meta_llama_Llama_3.2_11B_Vision_Instruct_20260427_125809.db (~140MB)
- gpu6_OpenGVLab_InternVL2_2B_20260421_145205.db (~140MB)
- gpu7_llava_hf_llava_v1.6_vicuna_7b_hf_20260421_145210.db (~140MB)
- Qwen_Qwen2.5_VL_3B_Instruct_35k.db (~140MB)

Total: ~700MB

### Step 2: Generate figures locally
```bash
python3 scripts/generate_research_figures.py \
    --results results/single_runs_35k/ \
    --output figures/research/
```

## Troubleshooting

### "Connection refused" error
- Check VPN connection
- Verify rolf server is accessible: `ping rolf.ifi.uzh.ch`
- Try connecting via UZH network or VPN

### "No such file or directory" for results
- Make sure database files are in `results/single_runs_35k/`
- Check files exist: `ls -lh results/single_runs_35k/*.db`

### "ImportError" for packages
```bash
pip install numpy pandas matplotlib seaborn scipy
```

### Script permission denied
```bash
chmod +x generate_research_figures.sh
```

## What You'll Get

After running successfully, you'll have these PDFs in `figures/research/`:

**Per-Model Radar Plots (10 files):**
- `radar_region_IDEFICS2-8B.pdf`
- `radar_region_Llama-3_2-11B.pdf`
- `radar_region_InternVL2-2B.pdf`
- `radar_region_LLaVA-v1_6-7B.pdf`
- `radar_region_Qwen2_5-VL-3B.pdf`
- `radar_probe_IDEFICS2-8B.pdf`
- `radar_probe_Llama-3_2-11B.pdf`
- `radar_probe_InternVL2-2B.pdf`
- `radar_probe_LLaVA-v1_6-7B.pdf`
- `radar_probe_Qwen2_5-VL-3B.pdf`

**Comparative Plots (6 files):**
- `gaps_heatmap.pdf`
- `regional_variance.pdf`
- `effect_sizes.pdf`
- `model_comparison.pdf`

Total: **16 publication-ready PDF figures**

## Expected Runtime

- Local (with downloaded DBs): ~3-5 minutes
- On Rolf server: ~3-5 minutes

## Next Steps

Once figures are generated:
1. Review figures in `figures/research/`
2. Read detailed descriptions in `RESEARCH_FIGURES.md`
3. Use figures in your paper (see suggested arrangement in documentation)
