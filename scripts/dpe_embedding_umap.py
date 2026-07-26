#!/usr/bin/env python3
"""
UMAP (and t-SNE) of the per-image deterministic-response embedding, before vs
after DPE correction, coloured by region.

Each image -> a 20-dim vector = its five probe responses scored on
(valence, economic_valence, stereotype_alignment, confidence). "Before" is the
archived baseline; "after" is the DPE run on the same images. If DPE reduces the
demographic signal, region clusters mix more in the "after" panel.

NOTE: this is the deterministic-SCORE (response) embedding, not the VLM's raw
visual-token embedding (those are not stored; a true visual-token UMAP needs a
hook-dump run on rolf -- see scripts/dump_visual_embeddings.py if generated).
"""
import sqlite3, numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

SEED = 42
CAP = 450   # images/region (balanced subsample) -> ~2.4k points/panel
ROOT = "/Users/ahmeda./Desktop/FingerPrint"
OUT = f"{ROOT}/figures/dpe_aaai"
FINAL = f"{ROOT}/results/dpe_final_20260720_125918"
REGIONS = ["Africa", "Asia", "Europe", "Americas", "Northern America", "Oceania"]
RCOL = {"Africa":"#d1193e","Asia":"#e8890c","Europe":"#128a5c",
        "Americas":"#1f4e9c","Northern America":"#7b1fa2","Oceania":"#00838f"}
DIMS = ["valence", "economic_valence", "stereotype_alignment", "confidence"]
PROBES = ["P1_occupation","P2_education","P3_trustworthiness","P4_lifestyle","P5_neighbourhood"]
MODELS = {
 "IDEFICS2-8B": (f"{ROOT}/results/single_runs_35k/gpu0_HuggingFaceM4_idefics2_8b_20260427_114159.db",
                 f"{FINAL}/idefics2/idefics2_dpe.db"),
 "InternVL2-2B":(f"{ROOT}/results/single_runs_35k/gpu6_OpenGVLab_InternVL2_2B_20260421_145205.db",
                 f"{FINAL}/internvl2/internvl2_dpe.db"),
 "LLaVA-v1.6-7B":(f"{ROOT}/results/single_runs_35k/gpu7_llava_hf_llava_v1.6_vicuna_7b_hf_20260421_145210.db",
                  f"{FINAL}/llava/llava_dpe.db"),
}

def feature_matrix(db):
    con = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT image_id, probe_id, jurisdiction_region AS region, "
        "valence, economic_valence, stereotype_alignment, confidence "
        "FROM judge_scores WHERE valence IS NOT NULL", con)
    con.close()
    df = df[df.region.isin(REGIONS) & df.probe_id.isin(PROBES)]
    piv = df.pivot_table(index="image_id", columns="probe_id", values=DIMS)
    piv.columns = [f"{d}_{p}" for d, p in piv.columns]
    reg = df.groupby("image_id").region.first()
    piv = piv.join(reg).dropna()
    return piv

def balanced(idx_region, rng):
    keep = []
    for r in REGIONS:
        ids = idx_region.index[idx_region == r].tolist()
        rng.shuffle(ids)
        keep += ids[:CAP]
    return keep

fig, axes = plt.subplots(len(MODELS), 2, figsize=(6.6, 9.0))
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "pdf.fonttype":42,"ps.fonttype":42})
rng = np.random.default_rng(SEED)
for row, (m, (bdb, ddb)) in enumerate(MODELS.items()):
    B = feature_matrix(bdb); D = feature_matrix(ddb)
    shared = B.index.intersection(D.index)
    B, D = B.loc[shared], D.loc[shared]
    ids = balanced(B.region, rng)
    B, D = B.loc[ids], D.loc[ids]
    feat = [c for c in B.columns if c != "region"]
    Xb, Xd = B[feat].to_numpy(float), D[feat].to_numpy(float)
    reg = B.region.to_numpy()
    # shared embedding space: fit UMAP on before+after together
    X = np.vstack([Xb, Xd])
    X = StandardScaler().fit_transform(X)
    emb = umap.UMAP(n_neighbors=25, min_dist=0.25, random_state=SEED, metric="euclidean").fit_transform(X)
    eb, ed = emb[:len(Xb)], emb[len(Xb):]
    for col, (E, lab) in enumerate([(eb, "before (baseline)"), (ed, "after DPE")]):
        ax = axes[row, col]
        for r in REGIONS:
            mask = reg == r
            ax.scatter(E[mask, 0], E[mask, 1], s=5, c=RCOL[r], alpha=0.55,
                       edgecolors="none", label=r if (row == 0 and col == 0) else None)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{m} — {lab}", fontsize=8.5, color="black")
        for s in ax.spines.values(): s.set_color("#ccc")
handles = [plt.Line2D([0],[0], marker="o", ls="", mfc=RCOL[r], mec="none", ms=6,
                      label=r.replace("Northern America","N. America")) for r in REGIONS]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.005), handletextpad=0.2, columnspacing=1.0)
fig.suptitle("Deterministic-response embedding (UMAP), before vs.\\ after DPE",
             fontsize=10, y=0.997)
fig.subplots_adjust(hspace=0.22, wspace=0.06, bottom=0.05, top=0.95)
fig.savefig(f"{OUT}/fig_dpe_umap.pdf", bbox_inches="tight", dpi=400)
fig.savefig(f"{OUT}/fig_dpe_umap.png", bbox_inches="tight", dpi=200)
print("wrote", f"{OUT}/fig_dpe_umap.{{pdf,png}}")
