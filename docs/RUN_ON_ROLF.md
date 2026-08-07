# Run Figure Generation on Rolf

## When rolf connection is available, follow these steps:

### Step 1: Upload the scripts to rolf
```bash
rsync -avz \
    scripts/generate_research_figures.py \
    generate_research_figures.sh \
    rolf:/local/scratch/alali/FingerPrint/
```

### Step 2: SSH to rolf and run the generation
```bash
ssh rolf
cd /local/scratch/alali/FingerPrint
chmod +x generate_research_figures.sh
./generate_research_figures.sh
```

Wait for it to complete (should take 3-5 minutes). You'll see output like:
```
======================================================================
FingerPrint² Research Figure Generation
======================================================================
Results: results/single_runs_35k
Output: figures/research

✓ Found completed: gpu0_HuggingFaceM4_idefics2_8b_20260427_114159 (175,945 results)
✓ Found completed: gpu3_meta_llama_Llama_3.2_11B_Vision_Instruct_20260427_125809 (175,945 results)
...

📊 Loading data from 5 models...
  Loading gpu0_HuggingFaceM4_idefics2_8b...
  Loading gpu3_meta_llama_Llama_3.2_11B_Vision_Instruct...
  ...

🎨 Generating figures...

1. Creating radar plots (by region)...
✓ Saved radar plot: figures/research/radar_region_IDEFICS2-8B.pdf
...

✅ All figures generated successfully!
======================================================================
Output directory: figures/research
Generated 16 PDF figures
```

### Step 3: Exit rolf and download figures
```bash
# Exit from rolf
exit

# Download figures to local machine
mkdir -p figures/research
rsync -avz --progress rolf:/local/scratch/alali/FingerPrint/figures/research/ ./figures/research/
```

### Step 4: Verify downloaded figures
```bash
ls -lh figures/research/
```

You should see 16 PDF files (~50KB each).

---

## Troubleshooting Connection

If you get "Connection refused":
1. Check if you're on UZH VPN
2. Test connection: `ping rolf.ifi.uzh.ch`
3. Try direct SSH: `ssh alali@rolf.ifi.uzh.ch`
4. Contact UZH IT if issue persists

## Alternative: One-liner approach

Once on rolf, you can also run the Python script directly:

```bash
ssh rolf
cd /local/scratch/alali/FingerPrint
python3 scripts/generate_research_figures.py \
    --results results/single_runs_35k/ \
    --output figures/research/
```

Then exit and rsync as above.
