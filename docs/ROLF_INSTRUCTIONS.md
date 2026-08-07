# Running AAAI Analyses on Rolf

## Three Simple Steps

### Step 1: Sync to Rolf
```bash
./sync_to_rolf.sh
```

This copies all scripts and database files to Rolf.

**Time**: 5-15 minutes (depending on network speed)

---

### Step 2: Run on Rolf

**Option A: Automatic (run from local machine)**
```bash
./run_on_rolf.sh
```

**Option B: Manual (SSH to Rolf first)**
```bash
ssh rolf
cd /local/scratch/alali/FingerPrint
./run_on_rolf.sh
```

**Time**: 15-30 minutes

---

### Step 3: Sync Results Back
```bash
./sync_from_rolf.sh
```

This copies all results back to your local machine.

**Time**: 2-5 minutes

---

## Complete One-Liner

```bash
./sync_to_rolf.sh && ./run_on_rolf.sh && ./sync_from_rolf.sh
```

**Total time**: 20-40 minutes (hands-off)

---

## What You Get

After running, you'll have:

```
results/aaai_submission/
├── statistical_analysis_summary.txt  ← For paper
├── figures/*.pdf                     ← 4 figures
├── validation_sample.csv             ← MTurk upload
├── qualitative_examples.json         ← Examples
└── [complete analysis]
```

---

## Troubleshooting

**"Cannot connect to Rolf"**
→ Check VPN, test: `ssh rolf`

**"Permission denied"**
→ `chmod +x *.sh`

**"Module not found"**
→ Script auto-installs packages

---

## Quick Reference

```bash
# Full workflow
./sync_to_rolf.sh && ./run_on_rolf.sh && ./sync_from_rolf.sh

# Check results
ls -la results/aaai_submission/
cat results/aaai_submission/statistical_analysis_summary.txt
```

**That's it!** Results ready for AAAI submission.
