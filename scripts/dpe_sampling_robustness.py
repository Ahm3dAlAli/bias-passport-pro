#!/usr/bin/env python3
"""
DPE sampling-robustness analysis (the key validity check).

For each model, compute the regional valence-disparity reduction under three
samplings of the SAME full-corpus DPE run:
  - first-2000/region   (ORDER BY image_id -- what the 2000/region eval used)
  - random-2000/region  (mean over several seeds)
  - full corpus         (all images -- the representative ground truth)

sigma = std of the six per-region mean valences (mean over all image-probe rows
in the region, matching compare_dpe_baseline.py). Reduction = (base-dpe)/base.

Finding: IDEFICS2's reduction holds under random + full (genuine); InternVL2's
large first-2000 reduction collapses under random/full (a first-N sampling
artifact); LLaVA is flat everywhere (resistant).

Writes figures/dpe_aaai/dpe_robustness_table.tex
Requires the completed full-corpus DPE DBs in results/dpe_full35k/<model>/.
"""
import sqlite3, numpy as np
from collections import defaultdict

ROOT = "/Users/ahmeda./Desktop/FingerPrint"
REGS = ["Africa", "Asia", "Europe", "Americas", "Northern America", "Oceania"]
SEEDS = [1, 2, 3, 4, 5]
CAP = 2000
MODELS = {
 "IDEFICS2-8B": ("results/single_runs_35k/gpu0_HuggingFaceM4_idefics2_8b_20260427_114159.db",
                 "results/dpe_full35k/idefics2/idefics2_dpe.db"),
 "InternVL2-2B":("results/single_runs_35k/gpu6_OpenGVLab_InternVL2_2B_20260421_145205.db",
                 "results/dpe_full35k/internvl2/internvl2_dpe.db"),
 "LLaVA-v1.6-7B":("results/single_runs_35k/gpu7_llava_hf_llava_v1.6_vicuna_7b_hf_20260421_145210.db",
                  "results/dpe_full35k/llava/llava_dpe.db"),
}

def load(db):
    c = sqlite3.connect(f"{ROOT}/{db}")
    r = c.execute("SELECT image_id, jurisdiction_region, valence FROM judge_scores "
                  "WHERE valence IS NOT NULL").fetchall()
    c.close()
    return [(i, rg, v) for i, rg, v in r if rg in REGS]

def region_sigma(rows_by_region_valence):
    means = [np.mean(rows_by_region_valence[r]) for r in REGS if rows_by_region_valence[r]]
    return float(np.std(means))

RES = {}
for m, (bdb, ddb) in MODELS.items():
    base = load(bdb); dpe = load(ddb)
    dpe_imgs = {i for i, _, _ in dpe}
    # baseline restricted to the images the DPE run covers (handles <100% runs)
    base = [(i, rg, v) for i, rg, v in base if i in dpe_imgs]
    reg = {i: rg for i, rg, _ in dpe}
    imgs_by_reg = defaultdict(list)
    for i in dpe_imgs:
        imgs_by_reg[reg[i]].append(i)
    # index rows by (image) for fast subset aggregation
    b_by_img = defaultdict(list); d_by_img = defaultdict(list)
    for i, rg, v in base: b_by_img[i].append(v)
    for i, rg, v in dpe: d_by_img[i].append(v)

    def sigma_for(imgset):
        b = defaultdict(list); d = defaultdict(list)
        for i in imgset:
            b[reg[i]].extend(b_by_img[i]); d[reg[i]].extend(d_by_img[i])
        return region_sigma(b), region_sigma(d)

    def subset(mode, seed=0):
        rng = np.random.default_rng(seed); keep = []
        for r in REGS:
            ids = sorted(imgs_by_reg[r])          # deterministic order (dpe_imgs is a set)
            if mode == "random": rng.shuffle(ids)  # reproducible: sorted list + seeded shuffle
            keep += ids[:CAP]
        return keep

    def red(sb, sd): return (sb - sd) / sb * 100 if sb > 1e-9 else float("nan")

    sb_f, sd_f = sigma_for(subset("first"))
    rr = [sigma_for(subset("random", s)) for s in SEEDS]
    rand_reds = [red(sb, sd) for sb, sd in rr]
    sb_all, sd_all = sigma_for(list(dpe_imgs))
    # full-corpus per-region baseline/DPE mean valence (for the convergence figure)
    rb = defaultdict(list); rd = defaultdict(list)
    for i in dpe_imgs:
        rb[reg[i]].extend(b_by_img[i]); rd[reg[i]].extend(d_by_img[i])
    reg_b = {r: float(np.mean(rb[r])) for r in REGS if rb[r]}
    reg_d = {r: float(np.mean(rd[r])) for r in REGS if rd[r]}
    RES[m] = dict(n=len(dpe_imgs),
                  first=(sb_f, sd_f, red(sb_f, sd_f)),
                  rand_mean=float(np.mean(rand_reds)), rand_std=float(np.std(rand_reds)),
                  full=(sb_all, sd_all, red(sb_all, sd_all)),
                  reg_b=reg_b, reg_d=reg_d)
    d = RES[m]
    print(f"{m:14s} n={d['n']:6d}  first {d['first'][2]:+5.1f}%  "
          f"random {d['rand_mean']:+5.1f}±{d['rand_std']:.1f}%  full {d['full'][2]:+5.1f}%")

# ---- LaTeX table ----
def pc(x): return f"{x:+.1f}\\%"
L = [r"\begin{table}[t]\centering\small",
     r"\caption{\textbf{Sampling robustness of the DPE disparity reduction.} Regional valence-disparity "
     r"$\sigma$ reduction for each model under three samplings of the \emph{same} full-corpus DPE run: the "
     r"deterministic first-$2000$/region set used by the two-stage evaluation, a random $2000$/region set "
     r"(mean$\pm$std over 5 seeds), and the full corpus. \model{IDEFICS2-8B}'s reduction holds under random "
     r"and full evaluation (genuine); \model{InternVL2-2B}'s large first-$N$ reduction collapses to $\approx0$ "
     r"under random/full sampling (an artifact of the deterministic image ordering); \model{LLaVA-v1.6-7B} is "
     r"unaffected throughout. Positive $=$ disparity reduced.}",
     r"\label{tab:dpe_robustness}",
     r"\begin{tabular}{lccc}", r"\toprule",
     r"Model & first-2000/reg. & random-2000/reg. & full corpus \\", r"\midrule"]
for m in MODELS:
    d = RES[m]
    note = "" if d["n"] >= 35189 else rf"\,\footnotesize({d['n']//1000}k)"
    L.append(f"\\model{{{m}}}{note} & {pc(d['first'][2])} & "
             f"${d['rand_mean']:+.1f}\\pm{d['rand_std']:.1f}\\%$ & {pc(d['full'][2])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
out = f"{ROOT}/figures/dpe_aaai/dpe_robustness_table.tex"
open(out, "w").write("\n".join(L) + "\n")
print("\nWrote", out)

# ---- headline results table: full-corpus sigma before/after + random 5-seed ----
DISP = {"IDEFICS2-8B":"genuine (holds under random \\& full)",
        "InternVL2-2B":"artifact (first-$N$ only; $\\approx0$ otherwise)",
        "LLaVA-v1.6-7B":"resistant"}
H = [r"\begin{table}[t]\centering\small",
     r"\caption{\textbf{DPE regional valence-disparity reduction (corrected).} Headline result on the full "
     r"$35{,}189$-image corpus, with the region-balanced $2000$/region evaluation reported as a mean$\pm$std "
     r"over 5 random seeds (rather than a single deterministic subsample). \model{IDEFICS2-8B} shows a "
     r"consistent reduction under both; \model{InternVL2-2B} and \model{LLaVA-v1.6-7B} show none.}",
     r"\label{tab:dpe_results}",
     r"\begin{tabular}{lccc}", r"\toprule",
     r"Model & $\sigma_{\text{base}}\!\to\!\sigma_{\text{DPE}}$ (full) & $\Delta\sigma$ full & $\Delta\sigma$ random-2000 (5 seeds) \\",
     r"\midrule"]
for m in MODELS:
    d = RES[m]; sb, sd, rd = d["full"]
    H.append(f"\\model{{{m}}} & ${sb:.4f}\\!\\to\\!{sd:.4f}$ & {rd:+.1f}\\% & "
             f"${d['rand_mean']:+.1f}\\pm{d['rand_std']:.1f}\\%$ \\\\")
H += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
hout = f"{ROOT}/figures/dpe_aaai/dpe_results_fullcorpus.tex"
open(hout, "w").write("\n".join(H) + "\n")
print("Wrote", hout)

# ---- regenerate fig_dpe_sigma_before_after with FULL-CORPUS sigma ----
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "mathtext.fontset":"stix","font.size":9,"axes.labelsize":9.5,"xtick.labelsize":8.5,
    "ytick.labelsize":8.5,"legend.fontsize":8.5,"axes.linewidth":0.8,"axes.edgecolor":"#333",
    "figure.dpi":150,"savefig.dpi":400,"savefig.bbox":"tight","savefig.pad_inches":0.02,
    "pdf.fonttype":42,"ps.fonttype":42})
MC = {"IDEFICS2-8B":"#1f4e9c","InternVL2-2B":"#c1440e","LLaVA-v1.6-7B":"#1a7a3c"}
mods = list(MODELS); x = np.arange(len(mods)); w = 0.36
sb = [RES[m]["full"][0] for m in mods]; sd = [RES[m]["full"][1] for m in mods]
fig, ax = plt.subplots(figsize=(6.2, 2.9))
ax.bar(x - w/2, sb, w, color="#b0b7c6", edgecolor="black", lw=0.5, label="baseline", zorder=3)
ax.bar(x + w/2, sd, w, color=[MC[m] for m in mods], edgecolor="black", lw=0.5, label="with DPE", zorder=3)
for i, m in enumerate(mods):
    rd = RES[m]["full"][2]
    lab = f"$\\downarrow${rd:.1f}%" if rd > 1 else r"$\approx$0"
    ax.text(i, max(sb[i], sd[i]) + 0.0012, lab, ha="center", va="bottom", fontsize=8,
            fontweight="bold", color="#222")
ax.set_xticks(x); ax.set_xticklabels(mods, fontsize=8.5)
ax.set_ylabel(r"regional valence disparity $\sigma$ (full corpus)")
ax.grid(True, axis="y", color="#eee", lw=0.7, zorder=0)
ax.spines[["top","right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", handlelength=1.4)
for ext in ("pdf", "png"):
    fig.savefig(f"{ROOT}/figures/dpe_aaai/fig_dpe_sigma_before_after.{ext}")
plt.close(fig)
print("Regenerated fig_dpe_sigma_before_after (full corpus)")

# ---- NEW: fig_dpe_robustness -- sigma-reduction by sampling (visualizes the artifact) ----
fig, ax = plt.subplots(figsize=(6.4, 3.0))
gx = np.arange(len(mods)); bw = 0.26
first = [RES[m]["first"][2] for m in mods]
rmean = [RES[m]["rand_mean"] for m in mods]; rstd = [RES[m]["rand_std"] for m in mods]
fullv = [RES[m]["full"][2] for m in mods]
ax.bar(gx - bw, first, bw, color="#c9c9c9", edgecolor="black", lw=0.4, label="first-2000/reg. (deterministic)", zorder=3)
ax.bar(gx, rmean, bw, yerr=rstd, capsize=3, color="#7ba0c9", edgecolor="black", lw=0.4,
       label="random-2000/reg. (5 seeds)", zorder=3, error_kw=dict(lw=0.9))
ax.bar(gx + bw, fullv, bw, color=[MC[m] for m in mods], edgecolor="black", lw=0.4, label="full corpus", zorder=3)
ax.axhline(0, color="#999", lw=0.7)
ax.set_xticks(gx); ax.set_xticklabels(mods, fontsize=8.5)
ax.set_ylabel(r"regional $\sigma$ reduction (\%)")
ax.grid(True, axis="y", color="#eee", lw=0.7, zorder=0)
ax.spines[["top","right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right", fontsize=7.6, handlelength=1.3)
ax.annotate("first-$N$ inflates the\napparent reduction", xy=(1-bw, first[1]), xytext=(1.15, 30),
            fontsize=7.3, color="#555", ha="left",
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0, connectionstyle="arc3,rad=-0.2"))
for ext in ("pdf", "png"):
    fig.savefig(f"{ROOT}/figures/dpe_aaai/fig_dpe_robustness.{ext}")
plt.close(fig); print("Wrote fig_dpe_robustness")

# ---- regenerate fig_dpe_convergence with FULL-CORPUS per-region means ----
RSHORT = {"Africa":"Afri","Asia":"Asia","Europe":"Euro","Americas":"Amer",
          "Northern America":"N.A","Oceania":"Ocea"}
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
xr = np.arange(len(REGS))
for ax, m in zip(axes, mods):
    d = RES[m]; b = [d["reg_b"][r] for r in REGS]; p = [d["reg_d"][r] for r in REGS]
    gm = np.mean(b)
    ax.axhline(gm, color="#999", lw=0.8, ls="--", zorder=1)
    ax.plot(xr, b, "o-", color="#b0b7c6", ms=4, lw=1.2, label="baseline", zorder=3)
    ax.plot(xr, p, "o-", color=MC[m], ms=4, lw=1.6, label="DPE", zorder=4)
    ax.plot(0, b[0], "o", mfc="none", mec="#c1440e", ms=9, mew=1.4, zorder=5)   # ring Africa
    ax.set_xticks(xr); ax.set_xticklabels([RSHORT[r] for r in REGS], fontsize=6.5, rotation=45, ha="right")
    ax.set_title(m, fontsize=8.5, color="black", pad=3)
    ax.grid(True, axis="y", color="#f0f0f0", lw=0.6, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    rd = d["full"][2]
    lbl = f"$\\sigma\\downarrow${rd:.1f}%" if rd > 1 else r"$\sigma\approx$ unch."
    ax.text(0.5, -0.42, lbl, transform=ax.transAxes, ha="center", fontsize=7.5, color="#222")
axes[0].set_ylabel("mean valence (full corpus)")
axes[0].legend(frameon=False, loc="best", fontsize=7, handlelength=1.3)
fig.subplots_adjust(wspace=0.32, bottom=0.28)
for ext in ("pdf", "png"):
    fig.savefig(f"{ROOT}/figures/dpe_aaai/fig_dpe_convergence.{ext}")
plt.close(fig); print("Regenerated fig_dpe_convergence (full corpus)")
