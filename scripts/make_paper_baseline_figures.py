#!/usr/bin/env python3
"""
Regenerate the main-paper baseline figures from the COMPLETE runs (judge
valence), written under the paper's existing filenames into aaai_build/.

Conventions (per author request):
  - every axis / colorbar spans 0-1
  - VALENCE figures (fig9, fig6) use valence rescaled to [0,1] via (v+1)/2
  - DISPARITY figures (fig1, fig2, fig3) keep raw max-min gaps (match the tables),
    plotted on a fixed 0-1 axis
  - no titles, no text annotations
  - fig2 radar = three separate panels (one per model)
  - fig6 = single worst/best-group panel (per-probe half is fig3), legend at side

Source of truth: figures/dpe_aaai/baseline_fig_data.json
"""
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333",
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
ROOT = "/Users/ahmeda./Desktop/FingerPrint"
OUT = f"{ROOT}/aaai_build"
DATA = json.load(open(f"{ROOT}/figures/dpe_aaai/baseline_fig_data.json"))

MODELS = ["IDEFICS2-8B", "InternVL2-2B", "LLaVA-v1.6-7B"]
MODEL_C = {"IDEFICS2-8B": "#1f4e9c", "InternVL2-2B": "#c1440e", "LLaVA-v1.6-7B": "#1a7a3c"}
REGIONS = DATA["regions"]
RSHORT = {"Africa": "Africa", "Asia": "Asia", "Europe": "Europe", "Americas": "Americas",
          "Northern America": "N. America", "Oceania": "Oceania"}
PROBES = ["Occupation", "Education", "Trust.", "Lifestyle", "Neighb."]

r01 = lambda v: (v + 1.0) / 2.0                       # valence [-1,1] -> [0,1]
DISP = {m: DATA["disparity"][m] for m in MODELS}      # raw max-min gaps
RVAL01 = {m: [r01(v) for v in DATA["region_val"][m]] for m in MODELS}
COMP = {m: float(np.mean(DISP[m])) for m in MODELS}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig); print("  wrote", name)


# --- fig1: composite leaderboard + model x probe disparity heatmap -----------
def fig1():
    fig = plt.figure(figsize=(7.2, 2.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.5)
    ax1 = fig.add_subplot(gs[0])
    order = sorted(MODELS, key=lambda m: COMP[m])
    y = np.arange(len(order))
    ax1.barh(y, [COMP[m] for m in order], color=[MODEL_C[m] for m in order],
             edgecolor="black", lw=0.6, height=0.62, zorder=3)
    for i, m in enumerate(order):
        ax1.text(COMP[m] + 0.015, i, f"#{i+1}  {COMP[m]:.3f}", ha="left",
                 va="center", fontsize=8)
    ax1.set_yticks(y); ax1.set_yticklabels(order, fontsize=8.5)
    ax1.set_xlabel("composite disparity (lower is fairer)")
    ax1.set_xlim(0, 1.0); ax1.set_xticks(np.arange(0, 1.01, 0.25))
    ax1.grid(axis="x", color="#eee", lw=0.7, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.invert_yaxis()

    ax2 = fig.add_subplot(gs[1])
    M = np.array([DISP[m] for m in MODELS])
    im = ax2.imshow(M, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
    ax2.set_xticks(range(len(PROBES))); ax2.set_xticklabels(PROBES, fontsize=8)
    ax2.set_yticks(range(len(MODELS))); ax2.set_yticklabels(MODELS, fontsize=8.5)
    for e, s in ax2.spines.items(): s.set_visible(False)
    ax2.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
    ax2.set_yticks(np.arange(M.shape[0] + 1) - 0.5, minor=True)
    ax2.grid(which="minor", color="white", lw=2.2); ax2.tick_params(which="minor", size=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax2.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                     color="white" if M[i, j] > 0.5 else "#222", fontsize=7.5)
    cb = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.03, ticks=np.arange(0, 1.01, 0.25))
    cb.set_label("max$-$min valence gap", fontsize=8); cb.ax.tick_params(labelsize=7)
    save(fig, "fig1_leaderboard_heatmap")


# --- fig2: three separate radar panels (one per model) -----------------------
def fig2():
    N = len(PROBES)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); ang += ang[:1]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), subplot_kw=dict(polar=True))
    for ax, m in zip(axes, MODELS):
        v = DISP[m] + DISP[m][:1]
        ax.plot(ang, v, color=MODEL_C[m], lw=2.0)
        ax.fill(ang, v, color=MODEL_C[m], alpha=0.12)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(PROBES, fontsize=7)
        ax.set_ylim(0, 1.0); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6, color="#999")
        ax.tick_params(pad=0.5)
        ax.grid(color="#ddd", lw=0.7); ax.spines["polar"].set_color("#ccc")
        ax.text(0.5, -0.22, m, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color=MODEL_C[m])
    fig.subplots_adjust(wspace=0.45)
    save(fig, "fig2_radar_fingerprints")


# --- fig3: grouped per-probe disparity (0-1 axis, no annotation) -------------
def fig3():
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(PROBES)); w = 0.26
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1) * w, DISP[m], w, color=MODEL_C[m], edgecolor="black",
               lw=0.4, label=m, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(PROBES)
    ax.set_ylabel(r"per-probe disparity (max$-$min gap)")
    ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(True, axis="y", color="#eee", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, handlelength=1.5, columnspacing=1.4)
    save(fig, "fig3_probe_comparison")


# --- fig6: worst vs best regional group valence (rescaled 0-1, legend at side)-
def fig6():
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    xg = np.arange(len(MODELS)); wg = 0.36
    worst = [min(RVAL01[m]) for m in MODELS]; best = [max(RVAL01[m]) for m in MODELS]
    wlab = [RSHORT[REGIONS[int(np.argmin(RVAL01[m]))]] for m in MODELS]
    blab = [RSHORT[REGIONS[int(np.argmax(RVAL01[m]))]] for m in MODELS]
    ax.bar(xg - wg/2, worst, wg, color="#b0413e", edgecolor="black", lw=0.5,
           label="worst-treated region", zorder=3)
    ax.bar(xg + wg/2, best, wg, color="#3a6ea5", edgecolor="black", lw=0.5,
           label="best-treated region", zorder=3)
    for i in range(len(MODELS)):
        ax.text(xg[i] - wg/2, worst[i] + 0.015, wlab[i], ha="center", va="bottom", fontsize=6.5)
        ax.text(xg[i] + wg/2, best[i] + 0.015, blab[i], ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(xg); ax.set_xticklabels(MODELS, fontsize=8.5)
    ax.set_ylabel("mean valence (rescaled to $[0,1]$)")
    ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(True, axis="y", color="#eee", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
              handlelength=1.4)
    save(fig, "fig6_worstbest_group")


# --- fig9: mean valence by region (rescaled 0-1, no annotation) --------------
def fig9():
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(REGIONS)); w = 0.26
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1) * w, RVAL01[m], w, color=MODEL_C[m], edgecolor="black",
               lw=0.4, label=m, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([RSHORT[r] for r in REGIONS], fontsize=8)
    ax.set_ylabel("mean valence (rescaled to $[0,1]$)")
    ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(True, axis="y", color="#eee", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, handlelength=1.5, columnspacing=1.4)
    save(fig, "fig9_regional_comparison")


if __name__ == "__main__":
    print("Writing complete-run baseline figures to", OUT)
    fig1(); fig2(); fig3(); fig6(); fig9()
    print("Done. Composites:", {m: round(COMP[m], 3) for m in MODELS})
