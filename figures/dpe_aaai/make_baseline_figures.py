#!/usr/bin/env python3
"""
AAAI-style baseline bias-characterization figures (pre-DPE).
Matches the DPE figure style. Outputs vector PDF + high-DPI PNG.

Figures:
  b1_probe_disparity   grouped bars: per-probe max-min disparity per model
  b2_fingerprints      radar: per-model bias fingerprint across the 5 probes
  b3_region_valence    mean valence by region per model  (needs REGION_VAL data)
  b4_region_probe_heat region x probe valence heatmaps    (needs REGION_PROBE data)
"""
import json, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman","DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333",
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
OUT = "/Users/ahmeda./Desktop/FingerPrint/figures/dpe_aaai"
MODEL_C = {"IDEFICS2-8B":"#1f4e9c", "InternVL2-2B":"#c1440e", "LLaVA-1.6-7B":"#1a7a3c"}

PROBES = ["Occupation","Education","Trust.","Lifestyle","Neighb."]
REGIONS = ["Africa","Asia","Europe","Americas","Northern America","Oceania"]

# Single source of truth: full-precision data exported by regenerate_baseline_tables.py
# from the complete DBs (judge valence). Regenerate with:
#   python3 scripts/regenerate_baseline_tables.py
_DATA = json.load(open(f"{OUT}/baseline_fig_data.json"))
_KEY = {"IDEFICS2-8B":"IDEFICS2-8B", "InternVL2-2B":"InternVL2-2B", "LLaVA-1.6-7B":"LLaVA-v1.6-7B"}
REGION_PROBE = {m: {r: _DATA["region_probe"][_KEY[m]][r] for r in REGIONS} for m in MODEL_C}
REGION_VAL   = {m: _DATA["region_val"][_KEY[m]] for m in MODEL_C}
DISP         = {m: _DATA["disparity"][_KEY[m]] for m in MODEL_C}

# ---------------------------------------------------------------------------
# B1 — per-probe disparity, grouped bars
# ---------------------------------------------------------------------------
def fig_probe_disparity():
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    models = list(DISP.keys())
    x = np.arange(len(PROBES)); w = 0.26
    for i, m in enumerate(models):
        off = (i - 1) * w
        bars = ax.bar(x + off, DISP[m], w, color=MODEL_C[m], edgecolor="black",
                      lw=0.4, label=m, zorder=3)
    # annotate the outlier probe: neighbourhood is worst for every model
    ax.annotate("neighbourhood: largest regional gap\nfor all three models",
                xy=(4 + w, DISP["InternVL2-2B"][4]), xytext=(1.7, 0.335),
                fontsize=7.5, color="#333", ha="left",
                arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.0,
                                connectionstyle="arc3,rad=-0.15"))
    ax.set_xticks(x); ax.set_xticklabels(PROBES)
    ax.set_ylabel(r"per-probe disparity (max$-$min gap)")
    ax.set_ylim(0, 0.40)
    ax.grid(True, axis="y", color="#eee", lw=0.7, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, handlelength=1.5, columnspacing=1.4)
    fig.savefig(f"{OUT}/b1_probe_disparity.pdf"); fig.savefig(f"{OUT}/b1_probe_disparity.png")
    plt.close(fig)

# ---------------------------------------------------------------------------
# B2 — bias fingerprints (radar), 3 models overlaid
# ---------------------------------------------------------------------------
def fig_fingerprints():
    N = len(PROBES)
    ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(3.7, 3.7), subplot_kw=dict(polar=True))
    for m in DISP:
        v = DISP[m] + DISP[m][:1]
        ax.plot(ang, v, color=MODEL_C[m], lw=2.0, label=m)
        ax.fill(ang, v, color=MODEL_C[m], alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(PROBES, fontsize=8.5)
    ax.set_yticks([0.10, 0.20, 0.30])
    ax.set_yticklabels(["0.10","0.20","0.30"], fontsize=7, color="#888")
    ax.set_ylim(0, 0.38)
    ax.tick_params(pad=1)
    ax.grid(color="#ddd", lw=0.7)
    ax.spines["polar"].set_color("#ccc")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=8, handlelength=1.4, columnspacing=1.2)
    fig.savefig(f"{OUT}/b2_fingerprints.pdf"); fig.savefig(f"{OUT}/b2_fingerprints.png")
    plt.close(fig)

# ---------------------------------------------------------------------------
# B3 — mean valence by region, grouped bars per model
# ---------------------------------------------------------------------------
def fig_region_valence():
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(REGIONS)); w = 0.26
    for i, m in enumerate(REGION_VAL):
        ax.bar(x + (i-1)*w, REGION_VAL[m], w, color=MODEL_C[m], edgecolor="black",
               lw=0.4, label=m, zorder=3)
    # mark Africa as the consistently lowest-valence region
    ax.annotate("Africa: lowest valence\nin every model", xy=(0, 0.02),
                xytext=(0.55, 0.30), fontsize=7.5, color="#333", ha="left",
                arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.0,
                                connectionstyle="arc3,rad=0.2"))
    ax.set_xticks(x); ax.set_xticklabels([r.replace("Northern ","N. ") for r in REGIONS], fontsize=8)
    ax.set_ylabel("mean valence (avg over probes)")
    ax.grid(True, axis="y", color="#eee", lw=0.7, zorder=0)
    ax.axhline(0, color="#999", lw=0.6, zorder=2)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, handlelength=1.5, columnspacing=1.4)
    fig.savefig(f"{OUT}/b3_region_valence.pdf"); fig.savefig(f"{OUT}/b3_region_valence.png")
    plt.close(fig)

# ---------------------------------------------------------------------------
# B4 — region x probe valence heatmaps, one panel per model
# ---------------------------------------------------------------------------
def fig_region_probe_heat():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9))
    rlab = [r.replace("Northern America","N. America") for r in REGIONS]
    for ax, m in zip(axes, REGION_PROBE):
        M = np.array([REGION_PROBE[m][r] for r in REGIONS])   # 6 regions x 5 probes
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(PROBES)))
        ax.set_xticklabels(PROBES, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(REGIONS)))
        ax.set_yticklabels(rlab if ax is axes[0] else [], fontsize=7)
        ax.set_title(m, fontsize=8.5, color=MODEL_C[m], pad=4)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                        color="white" if abs(v) > 0.55 else "#222")
        ax.set_xticks(np.arange(-.5, len(PROBES), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(REGIONS), 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("mean valence", fontsize=8); cbar.ax.tick_params(labelsize=7)
    fig.savefig(f"{OUT}/b4_region_probe_heat.pdf"); fig.savefig(f"{OUT}/b4_region_probe_heat.png")
    plt.close(fig)

if __name__ == "__main__":
    fig_probe_disparity()
    fig_fingerprints()
    fig_region_valence()
    fig_region_probe_heat()
    import os
    print("Wrote baseline figures to", OUT)
    for f in sorted(os.listdir(OUT)):
        if f.startswith("b") and f.endswith((".pdf",".png")):
            print("  ", f)
