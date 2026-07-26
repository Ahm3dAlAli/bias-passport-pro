#!/usr/bin/env python3
"""
UMAP of the true VLM visual-token embeddings, BEFORE vs AFTER the DPE correction,
from the npz files produced by scripts/dump_visual_embeddings.py (run on rolf).

Reads results/dpe_embeddings/{idefics2,internvl2,llava}_emb.npz.
Fits one UMAP per model on [before; after] jointly (shared coords) so the two
panels are comparable, colours by region, and draws each region's centroid shift
(before -> after) to illustrate the per-region correction vector.
"""
import glob, os, numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

SEED = 42
ROOT = "/Users/ahmeda./Desktop/FingerPrint"
OUT = f"{ROOT}/figures/dpe_aaai"
EMB = f"{ROOT}/results/dpe_embeddings"
REGIONS = ["Africa", "Asia", "Europe", "Americas", "Northern America", "Oceania"]
RCOL = {"Africa":"#d1193e","Asia":"#e8890c","Europe":"#128a5c",
        "Americas":"#1f4e9c","Northern America":"#7b1fa2","Oceania":"#00838f"}
NAME = {"idefics2":"IDEFICS2-8B","internvl2":"InternVL2-2B","llava":"LLaVA-v1.6-7B"}
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "pdf.fonttype":42,"ps.fonttype":42})

files = sorted(glob.glob(f"{EMB}/*_emb.npz"))
if not files:
    raise SystemExit(f"No embedding dumps in {EMB}. Run dump_visual_embeddings.py on rolf and sync back.")

fig, axes = plt.subplots(len(files), 2, figsize=(6.6, 3.0*len(files)), squeeze=False)
for row, f in enumerate(files):
    key = os.path.basename(f).replace("_emb.npz", "")
    d = np.load(f, allow_pickle=True)
    before, after, region = d["before"].astype(float), d["after"].astype(float), d["region"]
    X = np.vstack([before, after])
    X = StandardScaler().fit_transform(X)
    emb = umap.UMAP(n_neighbors=25, min_dist=0.25, random_state=SEED).fit_transform(X)
    eb, ea = emb[:len(before)], emb[len(before):]
    for col, (E, lab) in enumerate([(eb, "before (baseline)"), (ea, "after DPE")]):
        ax = axes[row, col]
        for r in REGIONS:
            mk = region == r
            if mk.any():
                ax.scatter(E[mk,0], E[mk,1], s=5, c=RCOL[r], alpha=0.5, edgecolors="none")
        # centroid-shift arrows (before->after) on the "after" panel
        if col == 1:
            for r in REGIONS:
                mk = region == r
                if mk.sum() < 3: continue
                cb, ca = eb[mk].mean(0), ea[mk].mean(0)
                ax.annotate("", xy=ca, xytext=cb,
                            arrowprops=dict(arrowstyle="-|>", color=RCOL[r], lw=1.3, alpha=0.9))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{NAME.get(key,key)} — {lab}", fontsize=8.5, color="black")
        for s in ax.spines.values(): s.set_color("#ccc")
handles = [plt.Line2D([0],[0], marker="o", ls="", mfc=RCOL[r], mec="none", ms=6,
                      label=r.replace("Northern America","N. America")) for r in REGIONS]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5,-0.01), handletextpad=0.2, columnspacing=1.0)
fig.suptitle("VLM visual-token embedding (UMAP), before vs. after DPE correction",
             fontsize=10, y=0.999)
fig.subplots_adjust(hspace=0.22, wspace=0.06, bottom=0.06, top=0.95)
fig.savefig(f"{OUT}/fig_dpe_visual_umap.pdf", bbox_inches="tight", dpi=400)
fig.savefig(f"{OUT}/fig_dpe_visual_umap.png", bbox_inches="tight", dpi=200)
print("wrote", f"{OUT}/fig_dpe_visual_umap.{{pdf,png}}  from", [os.path.basename(x) for x in files])
