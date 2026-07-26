#!/usr/bin/env python3
"""
Tier-1 DPE analysis (reviewer rebuttals), from the local final-eval DBs
(results/dpe_final_20260720_125918) matched against the archived baselines.

Outputs (figures/dpe_aaai/):
  dpe_tier1_tables.tex          - 3 LaTeX tables:
     (1) per-dimension disparity sigma, baseline->DPE + %reduction  (#2,#9)
     (2) valence sigma robustness: sigma / MAD / bootstrap 95% CI    (#3)
     (3) overall valence delta + Cohen's d per model                 (#1)
  fig_dpe_sigma_before_after.{pdf,png}  - sigma before/after bars
  fig_dpe_convergence.{pdf,png}         - per-region valence baseline vs DPE
"""
import sqlite3, numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
ROOT = "/Users/ahmeda./Desktop/FingerPrint"
OUT = f"{ROOT}/figures/dpe_aaai"
FINAL = f"{ROOT}/results/dpe_final_20260720_125918"
REGIONS = ["Africa", "Asia", "Europe", "Americas", "Northern America", "Oceania"]
RSHORT = {"Africa":"Africa","Asia":"Asia","Europe":"Europe","Americas":"Americas",
          "Northern America":"N. America","Oceania":"Oceania"}
DIMS = [("valence","Sentiment valence"), ("economic_valence","Economic valence"),
        ("stereotype_alignment","Stereotype align."), ("refusal","Refusal rate")]
MODELS = {  # display -> (baseline db, dpe db, alpha)
 "IDEFICS2-8B": (f"{ROOT}/results/single_runs_35k/gpu0_HuggingFaceM4_idefics2_8b_20260427_114159.db",
                 f"{FINAL}/idefics2/idefics2_dpe.db", 0.25),
 "InternVL2-2B":(f"{ROOT}/results/single_runs_35k/gpu6_OpenGVLab_InternVL2_2B_20260421_145205.db",
                 f"{FINAL}/internvl2/internvl2_dpe.db", 0.5),
 "LLaVA-v1.6-7B":(f"{ROOT}/results/single_runs_35k/gpu7_llava_hf_llava_v1.6_vicuna_7b_hf_20260421_145210.db",
                  f"{FINAL}/llava/llava_dpe.db", 0.5),
}
COLS = "image_id, probe_id, jurisdiction_region AS region, valence, economic_valence, stereotype_alignment, refusal"

def load(db):
    con = sqlite3.connect(db)
    df = pd.read_sql_query(f"SELECT {COLS} FROM judge_scores WHERE valence IS NOT NULL", con)
    con.close()
    return df[df.region.isin(REGIONS)].copy()

def region_means(df, col):
    return np.array([df[df.region == r][col].mean() for r in REGIONS
                     if (df.region == r).any() and df[df.region == r][col].notna().any()])

def sigma(df, col):    # std of per-region means
    m = region_means(df, col); return float(np.std(m)) if len(m) else float("nan")
def mad(df, col):      # mean abs deviation of per-region means
    m = region_means(df, col); return float(np.mean(np.abs(m - m.mean()))) if len(m) else float("nan")

def _per_image(df, col):
    ridx = {r: i for i, r in enumerate(REGIONS)}
    d = df[df[col].notna()]
    per = d.groupby("image_id").agg(s=(col, "sum"), c=(col, "count"),
                                    reg=("region", "first")).reset_index()
    per = per[per.reg.isin(REGIONS)]
    return (per.image_id.to_numpy(), per.s.to_numpy(), per.c.to_numpy(),
            per.reg.map(ridx).to_numpy())

def _sigma_from(rg, s, c, idx, K=6):
    rsum = np.bincount(rg[idx], weights=s[idx], minlength=K)
    rcnt = np.bincount(rg[idx], weights=c[idx], minlength=K)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = rsum / rcnt
    return np.std(means[rcnt > 0])

def bootstrap_delta_ci(base, dpe, col, B=2000, ci=95):
    """Paired image-level bootstrap of the sigma REDUCTION (sigma_base - sigma_dpe):
    resample the shared image set once per iteration and difference the two arms,
    which controls for the common sampling and is far more powerful than
    comparing two independent CIs. Returns (delta_point, lo, hi, sig_base_ci, sig_dpe_ci)."""
    ib, sb, cb, rb = _per_image(base, col)
    idd, sd, cd, rd = _per_image(dpe, col)
    # align on the shared image ids so a resampled index means the same image in both arms
    order = {im: k for k, im in enumerate(ib)}
    keep = np.array([im in order for im in idd])
    idd, sd, cd, rd = idd[keep], sd[keep], cd[keep], rd[keep]
    pos = np.array([order[im] for im in idd])
    sb2, cb2, rb2 = sb[pos], cb[pos], rb[pos]     # baseline aggregates for the same images
    N = len(idd)
    d_point = _sigma_from(rb2, sb2, cb2, np.arange(N)) - _sigma_from(rd, sd, cd, np.arange(N))
    db = np.empty(B); s_b = np.empty(B); s_d = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, N, N)
        s_b[b] = _sigma_from(rb2, sb2, cb2, idx)
        s_d[b] = _sigma_from(rd, sd, cd, idx)
        db[b] = s_b[b] - s_d[b]
    q = lambda a: tuple(np.percentile(a, [(100-ci)/2, 100-(100-ci)/2]))
    return (float(d_point),) + q(db) + (q(s_b), q(s_d))

# ---- compute -------------------------------------------------------------
RES = {}      # model -> dict
for m, (bdb, ddb, alpha) in MODELS.items():
    base, dpe = load(bdb), load(ddb)
    # match baseline to the exact (image_id, probe_id) pairs evaluated with DPE
    keys = dpe[["image_id", "probe_id"]].drop_duplicates()
    base = base.merge(keys, on=["image_id", "probe_id"], how="inner")
    per_dim = {}
    for col, _ in DIMS:
        sb, sd = sigma(base, col), sigma(dpe, col)
        red = (sb - sd) / sb * 100 if sb and not np.isnan(sb) and sb > 1e-9 else float("nan")
        per_dim[col] = dict(sb=sb, sd=sd, red=red)
    # robustness on valence: paired bootstrap of the sigma reduction
    dpt, dlo, dhi, sbci, sdci = bootstrap_delta_ci(base, dpe, "valence")
    RES[m] = dict(alpha=alpha, per_dim=per_dim,
                  mad_b=mad(base,"valence"), mad_d=mad(dpe,"valence"),
                  d_point=dpt, d_lo=dlo, d_hi=dhi, ci_b=sbci, ci_d=sdci,
                  ov_b=float(base.valence.mean()), ov_d=float(dpe.valence.mean()),
                  reg_b={r: float(base[base.region==r].valence.mean()) for r in REGIONS},
                  reg_d={r: float(dpe[dpe.region==r].valence.mean()) for r in REGIONS},
                  n=len(dpe))
    d = RES[m]
    print(f"{m}: valence σ {d['per_dim']['valence']['sb']:.4f}->{d['per_dim']['valence']['sd']:.4f} "
          f"({d['per_dim']['valence']['red']:+.1f}%)  overallΔ={d['ov_d']-d['ov_b']:+.4f}")

# ---- LaTeX tables --------------------------------------------------------
def f4(x): return "---" if x is None or np.isnan(x) else f"{x:.4f}"
def pct(x, sb=None):
    if x is None or np.isnan(x): return "---"
    if sb is not None and sb < 1e-3: return r"$\approx\!0$"   # negligible σ → % is meaningless
    return f"{x:+.1f}\\%"
L = []
# (1) per-dimension disparity
L += [r"\begin{table}[t]\centering\small",
      r"\caption{\textbf{Per-dimension regional disparity} $\sigma$ (std of per-region means) at baseline "
      r"and under DPE at $\alpha^{\star}$, on the region-balanced evaluation. DPE targets \emph{valence}; the "
      r"auxiliary dimensions are reported to show the correction does not inflate them. ($\sigma$ raw units; "
      r"refusal is near-zero throughout.)}", r"\label{tab:dpe_multidim}",
      r"\begin{tabular}{llccc}", r"\toprule",
      r"Model & Dimension & $\sigma_{\text{base}}$ & $\sigma_{\text{DPE}}$ & $\Delta$ \\", r"\midrule"]
for m in MODELS:
    L.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{m}}} ($\alpha^{{\star}}{{=}}{RES[m]['alpha']}$)}} \\")
    for col, lab in DIMS:
        pd_ = RES[m]["per_dim"][col]
        L.append(f"\\quad {lab} & & {f4(pd_['sb'])} & {f4(pd_['sd'])} & {pct(pd_['red'], pd_['sb'])} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table}", ""]

# (2) robustness of valence sigma -- paired bootstrap on the REDUCTION
L += [r"\begin{table}[t]\centering\small",
      r"\caption{\textbf{Robustness of the valence disparity reduction.} $\sigma$ is the std of per-region "
      r"means (less outlier-sensitive than the range); MAD is the mean absolute deviation of those means. "
      r"$\Delta\sigma$ is the reduction with a \emph{paired} image-level bootstrap $95\%$ CI "
      r"($B{=}2000$, seed $42$; the same resampled image set scores both arms). ``Excl.\ 0'' marks a CI "
      r"that excludes zero (a significant reduction).}", r"\label{tab:dpe_robust}",
      r"\begin{tabular}{lccccc}", r"\toprule",
      r"Model & $\sigma_{\text{base}}$ & $\sigma_{\text{DPE}}$ & MAD$_{\text{b}}\!\to\!$MAD$_{\text{d}}$ & $\Delta\sigma$ [95\% CI] & Excl.\ 0 \\",
      r"\midrule"]
for m in MODELS:
    d = RES[m]; v = d["per_dim"]["valence"]
    excl = "Yes" if d["d_lo"] > 0 else "No"
    L.append(f"{m} & {f4(v['sb'])} & {f4(v['sd'])} & "
             f"{f4(d['mad_b'])}$\\to${f4(d['mad_d'])} & "
             f"{d['d_point']:.4f} [{d['d_lo']:.4f}, {d['d_hi']:.4f}] & {excl} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# (3) overall valence delta
L += [r"\begin{table}[t]\centering\small",
      r"\caption{\textbf{Overall valence shift under DPE.} DPE equalises regions; the overall mean should "
      r"ideally be preserved. \model{IDEFICS2-8B} shifts negligibly (equalises without degrading), whereas "
      r"\model{InternVL2-2B} converges partly by lowering overall valence.}", r"\label{tab:dpe_overall}",
      r"\begin{tabular}{lccc}", r"\toprule",
      r"Model & overall $v_{\text{base}}$ & overall $v_{\text{DPE}}$ & $\Delta v$ \\", r"\midrule"]
for m in MODELS:
    d = RES[m]
    L.append(f"{m} & {d['ov_b']:.4f} & {d['ov_d']:.4f} & {d['ov_d']-d['ov_b']:+.4f} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(f"{OUT}/dpe_tier1_tables.tex", "w").write("\n".join(L) + "\n")
print("Wrote", f"{OUT}/dpe_tier1_tables.tex")

# ---- figures -------------------------------------------------------------
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "mathtext.fontset":"stix","font.size":9,"axes.labelsize":9.5,"xtick.labelsize":8.5,
    "ytick.labelsize":8.5,"legend.fontsize":8.5,"axes.linewidth":0.8,"axes.edgecolor":"#333",
    "figure.dpi":150,"savefig.dpi":400,"savefig.bbox":"tight","savefig.pad_inches":0.02,
    "pdf.fonttype":42,"ps.fonttype":42})
MC = {"IDEFICS2-8B":"#1f4e9c","InternVL2-2B":"#c1440e","LLaVA-v1.6-7B":"#1a7a3c"}
mods = list(MODELS)

# fig A: sigma before/after
fig, ax = plt.subplots(figsize=(6.2, 2.9))
x = np.arange(len(mods)); w = 0.36
sb = [RES[m]["per_dim"]["valence"]["sb"] for m in mods]
sd = [RES[m]["per_dim"]["valence"]["sd"] for m in mods]
ax.bar(x - w/2, sb, w, color="#b0b7c6", edgecolor="black", lw=0.5, label="baseline", zorder=3)
ax.bar(x + w/2, sd, w, color=[MC[m] for m in mods], edgecolor="black", lw=0.5, label="with DPE", zorder=3)
def redlbl(red): return f"$\\downarrow${red:.1f}%" if red > 1 else r"$\approx$0"
for i, m in enumerate(mods):
    red = RES[m]["per_dim"]["valence"]["red"]
    ax.text(i, max(sb[i], sd[i]) + 0.0012, redlbl(red), ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#222")
ax.set_xticks(x); ax.set_xticklabels(mods, fontsize=8.5)
ax.set_ylabel(r"regional valence disparity $\sigma$")
ax.grid(True, axis="y", color="#eee", lw=0.7, zorder=0)
ax.spines[["top","right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", handlelength=1.4)
fig.savefig(f"{OUT}/fig_dpe_sigma_before_after.pdf"); fig.savefig(f"{OUT}/fig_dpe_sigma_before_after.png")
plt.close(fig); print("wrote fig_dpe_sigma_before_after")

# fig B: per-region convergence (1x3, per-panel scale)
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
xr = np.arange(len(REGIONS))
for ax, m in zip(axes, mods):
    d = RES[m]
    b = [d["reg_b"][r] for r in REGIONS]; p = [d["reg_d"][r] for r in REGIONS]
    gm = np.mean(b)
    ax.axhline(gm, color="#999", lw=0.8, ls="--", zorder=1)
    ax.plot(xr, b, "o-", color="#b0b7c6", mfc="#b0b7c6", ms=4, lw=1.2, label="baseline", zorder=3)
    ax.plot(xr, p, "o-", color=MC[m], mfc=MC[m], ms=4, lw=1.6, label="DPE", zorder=4)
    # highlight Africa
    ax.plot(0, b[0], "o", mfc="none", mec="#c1440e", ms=9, mew=1.4, zorder=5)
    ax.set_xticks(xr); ax.set_xticklabels([RSHORT[r][:4] for r in REGIONS], fontsize=6.5, rotation=45, ha="right")
    ax.set_title(m, fontsize=8.5, color="black", pad=3)
    ax.grid(True, axis="y", color="#f0f0f0", lw=0.6, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    red = d["per_dim"]["valence"]["red"]
    lbl = f"disparity $\\sigma\\downarrow${red:.1f}%" if red > 1 else r"disparity $\sigma\approx$ unch."
    ax.text(0.5, -0.42, lbl, transform=ax.transAxes, ha="center",
            fontsize=7.5, color="#222")
axes[0].set_ylabel("mean valence")
axes[0].legend(frameon=False, loc="best", fontsize=7, handlelength=1.3)
fig.subplots_adjust(wspace=0.32, bottom=0.28)
fig.savefig(f"{OUT}/fig_dpe_convergence.pdf"); fig.savefig(f"{OUT}/fig_dpe_convergence.png")
plt.close(fig); print("wrote fig_dpe_convergence")
