#!/usr/bin/env python3
"""
AAAI-style publication figures for Demographic Positional Encoding (DPE).
Outputs vector PDF + high-DPI PNG for each figure.
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------------------------------
# AAAI-ish publication style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
})

OUT = "/Users/ahmeda./Desktop/FingerPrint/figures/dpe_aaai"

MODEL_C = {"IDEFICS2-8B": "#1f4e9c", "InternVL2-2B": "#c1440e", "LLaVA-1.6-7B": "#1a7a3c"}

# ----------------------------------------------------------------------------
# Data  (balanced FHIBE, region-only correction)
# ----------------------------------------------------------------------------
REGIONS = ["Africa", "Americas", "Asia", "Europe", "Northern America", "Oceania"]
RC = {  # region colors
    "Africa": "#d1193e", "Americas": "#e8890c", "Asia": "#128a5c",
    "Europe": "#1f6fd6", "Northern America": "#7b4fd0", "Oceania": "#5b6570",
}

# per model: alphas list (numeric; baseline handled separately), sigma, per-region valence
MODELS = {
    "IDEFICS2-8B": {
        "opt": 0.25,
        "alphas": [0.25, 0.5, 0.75, 1.0, 1.5],
        "sigma_base": 0.0183,
        "sigma": [0.0124, 0.0145, 0.0124, 0.0127, 0.0127],
        "reg_base": {"Africa":0.0267,"Americas":0.0695,"Asia":0.0477,"Europe":0.0726,"Northern America":0.0817,"Oceania":0.0650},
        "reg": {
            "Africa":[0.0442,0.0402,0.0402,0.0417,0.0408],
            "Americas":[0.0621,0.0639,0.0612,0.0623,0.0584],
            "Asia":[0.0469,0.0425,0.0492,0.0478,0.0484],
            "Europe":[0.0612,0.0673,0.0643,0.0661,0.0706],
            "Northern America":[0.0726,0.0764,0.0743,0.0746,0.0677],
            "Oceania":[0.0784,0.0751,0.0740,0.0755,0.0769],
        },
    },
    "InternVL2-2B": {
        "opt": 0.5,
        "alphas": [0.25, 0.5, 0.75, 1.0, 1.5],
        "sigma_base": 0.0338,
        "sigma": [0.0275, 0.0246, 0.0279, 0.0256, 0.0258],
        "reg_base": {"Africa":0.5583,"Americas":0.6117,"Asia":0.5967,"Europe":0.6207,"Northern America":0.5900,"Oceania":0.6693},
        "reg": {
            "Africa":[0.5272,0.5229,0.5227,0.5236,0.5173],
            "Americas":[0.5166,0.5211,0.5222,0.5155,0.5239],
            "Asia":[0.5689,0.5573,0.5684,0.5621,0.5613],
            "Europe":[0.5257,0.5226,0.5215,0.5216,0.5249],
            "Northern America":[0.4878,0.4884,0.4865,0.4877,0.4898],
            "Oceania":[0.5623,0.5611,0.5634,0.5584,0.5642],
        },
    },
    "LLaVA-1.6-7B": {
        "opt": 0.5,
        "alphas": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
        "sigma_base": 0.0381,
        "sigma": [0.0414, 0.0396, 0.0372, 0.0405, 0.0395, 0.0382],
        "reg_base": {"Africa":0.4688,"Americas":0.5558,"Asia":0.5200,"Europe":0.5760,"Northern America":0.5694,"Oceania":0.5701},
        "reg": {
            "Africa":[0.4599,0.4610,0.4683,0.4610,0.4662,0.4652],
            "Americas":[0.5526,0.5520,0.5514,0.5497,0.5586,0.5494],
            "Asia":[0.5165,0.5257,0.5248,0.5205,0.5187,0.5141],
            "Europe":[0.5821,0.5793,0.5765,0.5750,0.5773,0.5727],
            "Northern America":[0.5646,0.5640,0.5676,0.5691,0.5705,0.5652],
            "Oceania":[0.5681,0.5665,0.5678,0.5728,0.5692,0.5669],
        },
    },
}

def pct(x, p): return f"{100*x:.0f}%"

# ============================================================================
# FIGURE 1 — Disparity sigma vs alpha (all models), optimum starred
# ============================================================================
def fig_disparity():
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(4.1, 3.2))
    for name, M in MODELS.items():
        c = MODEL_C[name]
        a = np.array(M["alphas"]); s = np.array(M["sigma"])
        m = a > 0
        ax.plot(a[m], s[m], "-o", color=c, ms=5, mfc="white", mew=1.5, lw=2.0, label=name, zorder=3)
        ax.axhline(M["sigma_base"], color=c, ls=(0,(4,3)), lw=1.0, alpha=0.5, zorder=1)
        oi = M["alphas"].index(M["opt"])
        ax.plot(M["opt"], M["sigma"][oi], marker="*", color=c, ms=16, mec="black", mew=0.6, zorder=4)
    # directional cue placed in the empty gap between InternVL2 and LLaVA curves
    ax.annotate("", xy=(0.175, 0.029), xytext=(0.175, 0.0365),
                arrowprops=dict(arrowstyle="-|>", color="#2a8a4a", lw=1.2))
    ax.text(0.205, 0.0327, "more\nequitable", fontsize=6.6, color="#2a8a4a",
            ha="left", va="center", linespacing=0.95)
    ax.set_xlabel(r"correction strength  $\alpha$")
    ax.set_ylabel(r"regional disparity  $\sigma$")
    ax.set_xticks([0.25,0.5,0.75,1.0,1.5]); ax.set_xlim(0.13, 1.62)
    ax.set_ylim(0.010, 0.043)
    ax.grid(True, axis="y", color="#eeeeee", lw=0.7, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    # small style key (upper right) WITH a white frame so it reads over the curve
    key = [Line2D([0],[0], color="#777", ls=(0,(4,3)), lw=1.0, label="baseline (no DPE)"),
           Line2D([0],[0], color="#777", marker="*", ms=11, mec="black", mew=.5, ls="none", label="optimal $\\alpha$")]
    leg1 = ax.legend(handles=key, fontsize=7.0, loc="upper right",
                     handlelength=1.6, borderaxespad=0.4, labelspacing=0.35,
                     frameon=True, facecolor="white", edgecolor="#e2e2e2", framealpha=0.95)
    leg1.get_frame().set_linewidth(0.6)
    ax.add_artist(leg1)
    handles = [Line2D([0],[0], color=MODEL_C[n], lw=2.2, marker="o", mfc="white", mew=1.4, ms=5, label=n) for n in MODELS]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, handlelength=1.6, columnspacing=1.2)
    fig.savefig(f"{OUT}/fig1_disparity_vs_alpha.pdf")
    fig.savefig(f"{OUT}/fig1_disparity_vs_alpha.png")
    plt.close(fig)

# ============================================================================
# FIGURE 2 — Per-region valence convergence, 3 panels
# ============================================================================
def fig_perregion():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=False)
    red_pct = {}  # for the model annotation
    for ax, (name, M) in zip(axes, MODELS.items()):
        xs = ["base"] + [a for a in M["alphas"] if a > 0]
        xi = list(range(len(xs)))
        for reg in REGIONS:
            base = M["reg_base"][reg]
            vals = [M["reg"][reg][i] for i,a in enumerate(M["alphas"]) if a > 0]
            y = [base] + vals
            lw = 2.6 if reg == "Africa" else 1.1
            z = 6 if reg == "Africa" else 3
            ax.plot(xi, y, "-", color=RC[reg], lw=lw, zorder=z,
                    label=reg if ax is axes[0] else None)
            ax.plot(0, base, "o", color=RC[reg], ms=3.2, zorder=z)
        # shade + mark the optimum alpha column
        if M["opt"] in xs:
            oi = xs.index(M["opt"])
            ax.axvspan(oi-0.30, oi+0.30, color="#1f9d55", alpha=0.09, zorder=0)
            ax.text(oi, ax.get_ylim()[1], "optimal $\\alpha$", fontsize=6.6,
                    color="#177a43", ha="center", va="bottom")
        # subtle in-panel model label (NOT a title) with a soft box
        red = 100*(M["sigma_base"]-M["sigma"][M["alphas"].index(M["opt"])])/M["sigma_base"]
        ax.text(0.035, 0.965, name, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                color="#333", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#e2e2e2", lw=0.6, alpha=0.9))
        ax.text(0.035, 0.845, f"disparity $\\downarrow${red:.0f}%", transform=ax.transAxes,
                fontsize=7.2, color="#177a43", va="top", ha="left")
        ax.set_xticks(xi)
        ax.set_xticklabels([("base" if v=="base" else v) for v in xs], fontsize=7.5)
        ax.set_xlabel(r"correction strength  $\alpha$", fontsize=8)
        ax.grid(True, axis="y", color="#f2f2f2", lw=0.6, zorder=0)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("mean valence")
    # arrow annotation on idefics2 (clearest): Africa rising toward the pack
    a0 = axes[0]
    a0.annotate("Africa lifted\ntoward mean", xy=(1, 0.044), xytext=(2.2, 0.031),
                fontsize=6.8, color=RC["Africa"], ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=RC["Africa"], lw=1.1,
                                connectionstyle="arc3,rad=-0.25"))
    fig.subplots_adjust(bottom=0.30, wspace=0.26)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=8, handlelength=1.5, columnspacing=1.2)
    fig.savefig(f"{OUT}/fig2_perregion_convergence.pdf")
    fig.savefig(f"{OUT}/fig2_perregion_convergence.png")
    plt.close(fig)

# ============================================================================
# FIGURE 3 — Before vs after DPE (sigma) at optimum alpha, grouped bars
# ============================================================================
def fig_before_after():
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    names = list(MODELS.keys())
    x = np.arange(len(names)); w = 0.36
    base = [MODELS[n]["sigma_base"] for n in names]
    best = [MODELS[n]["sigma"][MODELS[n]["alphas"].index(MODELS[n]["opt"])] for n in names]
    b1 = ax.bar(x - w/2, base, w, color="#c7ccd1", edgecolor="#9aa0a6", lw=0.6, label="baseline")
    b2 = ax.bar(x + w/2, best, w, color="#1f4e9c", edgecolor="black", lw=0.4, label="DPE (opt $\\alpha$)")
    for i,n in enumerate(names):
        red = 100*(base[i]-best[i])/base[i]
        ax.text(x[i] + 0.14, max(base[i],best[i]) + 0.0034,
                f"$\\downarrow${red:.0f}%",
                ha="center", fontsize=8, color="#0b7a35", fontweight="bold")
        ax.text(x[i]-w/2, base[i]+0.0004, f"{base[i]:.4f}", ha="center", fontsize=6.3, color="#555")
        ax.text(x[i]+w/2, best[i]+0.0004, f"{best[i]:.4f}", ha="center", fontsize=6.3, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split("-")[0] for n in names], fontsize=8.5)
    ax.set_ylabel(r"regional disparity  $\sigma$")
    ax.set_ylim(0, 0.046)
    ax.grid(True, axis="y", color="#eeeeee", lw=0.7, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", handlelength=1.3)
    fig.savefig(f"{OUT}/fig3_before_after.pdf")
    fig.savefig(f"{OUT}/fig3_before_after.png")
    plt.close(fig)

# ============================================================================
# FIGURE 4 — Combined 2-panel main figure (for the paper body)
# ============================================================================
def fig_main_combined():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.7),
                                   gridspec_kw={"width_ratios":[1,1.05], "wspace":0.32})
    # LEFT: disparity vs alpha
    for name, M in MODELS.items():
        c = MODEL_C[name]; a=np.array(M["alphas"]); s=np.array(M["sigma"]); m=a>0
        axL.plot(a[m], s[m], "-o", color=c, ms=4, mfc="white", mew=1.3, lw=1.6, label=name)
        axL.axhline(M["sigma_base"], color=c, ls=(0,(4,3)), lw=0.9, alpha=0.5)
        oi=M["alphas"].index(M["opt"]); axL.plot(M["opt"], M["sigma"][oi], "*", color=c, ms=13, mec="black", mew=0.5)
    axL.set_xlabel(r"correction strength  $\alpha$"); axL.set_ylabel(r"disparity  $\sigma$")
    axL.set_xticks([0.25,0.5,0.75,1.0,1.5]); axL.set_ylim(0.010,0.042)
    axL.grid(True, axis="y", color="#eee", lw=0.7); axL.spines[["top","right"]].set_visible(False)
    axL.legend(frameon=False, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5,-0.22),
               ncol=3, handlelength=1.4, columnspacing=1.2)
    # RIGHT: idefics2 per-region convergence (clearest example)
    M = MODELS["IDEFICS2-8B"]
    xs=["base"]+[a for a in M["alphas"] if a>0]; xi=list(range(len(xs)))
    for reg in REGIONS:
        y=[M["reg_base"][reg]]+[M["reg"][reg][i] for i,a in enumerate(M["alphas"]) if a>0]
        lw=2.4 if reg=="Africa" else 1.1
        axR.plot(xi, y, "-", color=RC[reg], lw=lw, label=reg)
        axR.plot(0, M["reg_base"][reg], "o", color=RC[reg], ms=3)
    oi=xs.index(M["opt"]); axR.axvspan(oi-0.28,oi+0.28,color="#128a5c",alpha=0.08)
    axR.set_xticks(xi); axR.set_xticklabels(xs, fontsize=8); axR.set_xlabel(r"$\alpha$")
    axR.set_ylabel("mean valence"); axR.grid(True, axis="y", color="#f2f2f2", lw=0.6)
    axR.spines[["top","right"]].set_visible(False)
    axR.legend(frameon=False, fontsize=6.6, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5,-0.22), handlelength=1.2, columnspacing=0.9)
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(f"{OUT}/fig0_main.pdf")
    fig.savefig(f"{OUT}/fig0_main.png")
    plt.close(fig)

if __name__ == "__main__":
    fig_main_combined()
    fig_disparity()
    fig_perregion()
    fig_before_after()
    print("Wrote figures to", OUT)
    import os
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".pdf",".png")):
            print("  ", f, f"{os.path.getsize(os.path.join(OUT,f))//1024} KB")
