#!/usr/bin/env python3
"""
Regenerate ALL baseline (pre-DPE) analysis tables from the COMPLETE runs,
using the judge valence already stored in judge_scores (the SAME metric the
DPE evaluation uses) so the baseline tables, baseline figures, and DPE results
are all on one consistent scale.

Reads:  results/single_runs_35k/{gpu0 idefics2, gpu6 InternVL2, gpu7 llava}.db
Writes: figures/dpe_aaai/baseline_tables.tex  (all tables, ready to \input)

Statistics computed from the raw per-image judge scores:
  - Model leaderboard (composite = mean per-probe max-min gap; severity buckets)
  - Per-probe disparity + worst/best region
  - Multi-metric disparity + bootstrap 95% CI (valence, economic, stereotype,
    probe-specific, refusal)
  - Two-way structure: F_region, F_probe (one-way KW-free ANOVA F) + interaction var
  - Modal worst region + consistency
  - Region x probe mean valence
  - Kruskal-Wallis H, df, p, eta^2, significance at Bonferroni alpha_adj
  - Dunn post-hoc (manual, Holm) headline-contrast significance
  - Africa-as-extreme counts
  - Bootstrap 95% CI on neighbourhood disparity (B=10000, seed 42)
"""
import sqlite3, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import f_oneway, kruskal, norm

RNG = np.random.default_rng(42)
ROOT = "/Users/ahmeda./Desktop/FingerPrint"
DBS = {
    "IDEFICS2-8B":  f"{ROOT}/results/single_runs_35k/gpu0_HuggingFaceM4_idefics2_8b_20260427_114159.db",
    "InternVL2-2B": f"{ROOT}/results/single_runs_35k/gpu6_OpenGVLab_InternVL2_2B_20260421_145205.db",
    "LLaVA-v1.6-7B":f"{ROOT}/results/single_runs_35k/gpu7_llava_hf_llava_v1.6_vicuna_7b_hf_20260421_145210.db",
}
REGIONS = ["Africa","Asia","Europe","Americas","Northern America","Oceania"]
RSHORT  = {"Africa":"Africa","Asia":"Asia","Europe":"Europe","Americas":"Americas",
           "Northern America":"N. America","Oceania":"Oceania"}
PROBES  = ["P1_occupation","P2_education","P3_trustworthiness","P4_lifestyle","P5_neighbourhood"]
PLABEL  = {"P1_occupation":"Occupation","P2_education":"Education",
           "P3_trustworthiness":"Trustworthiness","P4_lifestyle":"Lifestyle",
           "P5_neighbourhood":"Neighbourhood"}
ALPHA_ADJ = 0.05 / 75.0   # Bonferroni family: 3 models x 5 probes x 5 metrics

def load(db):
    con = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT image_id, probe_id, valence, economic_valence, stereotype_alignment, "
        "refusal, jurisdiction_region AS region FROM judge_scores "
        "WHERE valence IS NOT NULL", con)
    con.close()
    return df[df.region.isin(REGIONS)].copy()

DATA = {m: load(db) for m, db in DBS.items()}

def region_means(df, col="valence"):
    return {r: df[df.region == r][col].mean() for r in REGIONS if (df.region == r).any()}

def bootstrap_disparity_ci(df, probe=None, col="valence", B=10000, ci=95):
    """Bootstrap CI on the max-min regional gap by observation-level resampling."""
    d = df if probe is None else df[df.probe_id == probe]
    # matrix of per-region values (list per region)
    per = {r: d[d.region == r][col].values for r in REGIONS}
    per = {r: v for r, v in per.items() if len(v)}
    gaps = np.empty(B)
    for b in range(B):
        ms = [RNG.choice(v, size=len(v), replace=True).mean() for v in per.values()]
        gaps[b] = max(ms) - min(ms)
    lo, hi = np.percentile(gaps, [(100-ci)/2, 100-(100-ci)/2])
    return float(lo), float(hi)

def bootstrap_meanprobe_gap_ci(df, B=2000, ci=95):
    """Bootstrap CI on the MEAN across probes of the per-probe max-min gap."""
    per = {(p, r): df[(df.probe_id == p) & (df.region == r)]["valence"].values
           for p in PROBES for r in REGIONS}
    per = {k: v for k, v in per.items() if len(v)}
    vals = np.empty(B)
    for b in range(B):
        means = {k: RNG.choice(v, size=len(v), replace=True).mean() for k, v in per.items()}
        gaps = []
        for p in PROBES:
            mr = [means[(p, r)] for r in REGIONS if (p, r) in means]
            gaps.append(max(mr) - min(mr))
        vals[b] = np.mean(gaps)
    lo, hi = np.percentile(vals, [(100-ci)/2, 100-(100-ci)/2])
    return float(lo), float(hi)

def dunn_headline_sig(df, probe):
    """Manual Dunn test; return True if the (worst->best) headline contrast is
    significant at Holm-adjusted p<0.05."""
    d = df[df.probe_id == probe]
    groups = {r: d[d.region == r]["valence"].values for r in REGIONS if (d.region == r).any()}
    labels = list(groups.keys())
    allv = np.concatenate([groups[r] for r in labels])
    ranks = stats.rankdata(allv)
    N = len(allv)
    # tie correction
    _, counts = np.unique(allv, return_counts=True)
    ties = (counts**3 - counts).sum()
    sigma2 = (N*(N+1)/12.0) - ties/(12.0*(N-1))
    # mean ranks per group
    idx = 0; mr = {}; n = {}
    for r in labels:
        k = len(groups[r]); mr[r] = ranks[idx:idx+k].mean(); n[r] = k; idx += k
    means = {r: groups[r].mean() for r in labels}
    worst = min(means, key=means.get); best = max(means, key=means.get)
    pairs = [(a, b) for i, a in enumerate(labels) for b in labels[i+1:]]
    pvals = {}
    for a, b in pairs:
        z = abs(mr[a]-mr[b]) / np.sqrt(sigma2*(1.0/n[a] + 1.0/n[b]))
        pvals[(a, b)] = 2*(1-norm.cdf(z))
    # Holm
    order = sorted(pvals, key=pvals.get)
    m = len(order); adj = {}; run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m-i)*pvals[k]); adj[k] = min(run, 1.0)
    key = (worst, best) if (worst, best) in adj else (best, worst)
    return adj[key] < 0.05

# ---------------------------------------------------------------------------
# compute everything
# ---------------------------------------------------------------------------
COMPOSITE, PROBE_DISP, RXP, KW, MULTI, ANOVA, MODAL, AFRICA = {}, {}, {}, {}, {}, {}, {}, {}
for m, df in DATA.items():
    rxp = {p: region_means(df[df.probe_id == p]) for p in PROBES}
    RXP[m] = rxp
    disps = {}
    for p in PROBES:
        rm = rxp[p]
        worst = min(rm, key=rm.get); best = max(rm, key=rm.get)
        disps[p] = dict(disp=rm[best]-rm[worst], worst=worst, best=best)
    PROBE_DISP[m] = disps
    comp = np.mean([d["disp"] for d in disps.values()])
    sev = ("Negligible" if comp < 0.05 else "Low" if comp < 0.08
           else "Moderate" if comp < 0.15 else "High")
    wp = max(disps, key=lambda p: disps[p]["disp"])
    COMPOSITE[m] = dict(composite=comp, severity=sev, worst_probe=wp,
                        worst_gap=disps[wp]["disp"], valid=100.0)
    # Kruskal-Wallis per probe
    kw = {}
    for p in PROBES:
        d = df[df.probe_id == p]
        groups = [d[d.region == r]["valence"].values for r in REGIONS if (d.region == r).any()]
        H, pv = kruskal(*groups)
        n = sum(len(g) for g in groups); k = len(groups)
        eta2 = (H - k + 1) / (n - k)
        kw[p] = dict(H=H, df=k-1, p=pv, eta2=eta2, sig=pv < ALPHA_ADJ,
                     dunn=dunn_headline_sig(df, p))
    KW[m] = kw
    # multi-metric disparities + bootstrap CI
    mm = {}
    for label, col in [("Sentiment valence","valence"),("Economic valence","economic_valence"),
                       ("Stereotype score","stereotype_alignment")]:
        if col not in df or df[col].isna().all():
            continue
        rm = region_means(df, col)
        worst = min(rm, key=rm.get); best = max(rm, key=rm.get)
        lo, hi = bootstrap_disparity_ci(df, col=col, B=2000)
        mm[label] = dict(disp=rm[best]-rm[worst], lo=lo, hi=hi, worst=worst, best=best)
    # probe-specific = mean per-probe gap; CI via bootstrap of the SAME quantity;
    # worst->best = modal worst region across probes -> modal best region
    ps = np.mean([disps[p]["disp"] for p in PROBES])
    lo, hi = bootstrap_meanprobe_gap_ci(df, B=2000)
    from collections import Counter
    mw = Counter(disps[p]["worst"] for p in PROBES).most_common(1)[0][0]
    mb = Counter(disps[p]["best"] for p in PROBES).most_common(1)[0][0]
    if mw == mb:   # degenerate (region-inconsistent) -> use pooled-valence extremes
        pm = region_means(df); mw = min(pm, key=pm.get); mb = max(pm, key=pm.get)
    mm["Probe-specific"] = dict(disp=ps, lo=lo, hi=hi, worst=mw, best=mb)
    # refusal rate disparity (higher refusal = worse); bootstrap CI on the gap
    if "refusal" in df and not df["refusal"].isna().all():
        rr = {r: df[df.region == r]["refusal"].mean() for r in REGIONS if (df.region == r).any()}
        worst = max(rr, key=rr.get); best = min(rr, key=rr.get)
        rlo, rhi = bootstrap_disparity_ci(df, col="refusal", B=2000)
        mm["Refusal rate"] = dict(disp=rr[worst]-rr[best], lo=rlo, hi=rhi,
                                  worst=worst, best=best)
    MULTI[m] = mm
    # two-way structure: F_region, F_probe (one-way F), interaction variance
    fr = f_oneway(*[df[df.region == r]["valence"].values for r in REGIONS if (df.region == r).any()])[0]
    fp = f_oneway(*[df[df.probe_id == p]["valence"].values for p in PROBES])[0]
    grand = df["valence"].mean()
    reff = {r: df[df.region == r]["valence"].mean()-grand for r in REGIONS}
    peff = {p: df[df.probe_id == p]["valence"].mean()-grand for p in PROBES}
    inter = []
    for p in PROBES:
        for r in REGIONS:
            if not rxp[p].get(r) is None and r in rxp[p]:
                inter.append(rxp[p][r] - (grand + reff[r] + peff[p]))
    ANOVA[m] = dict(F_region=fr, F_probe=fp, inter_var=float(np.var(inter)))
    # modal worst region
    worst_regions = [disps[p]["worst"] for p in PROBES]
    vals, cnts = np.unique(worst_regions, return_counts=True)
    top = cnts.max(); modal = ", ".join(RSHORT[v] for v, c in zip(vals, cnts) if c == top)
    MODAL[m] = dict(modal=modal, count=f"{top}/5", consistency=top/5.0)
    # africa extreme
    af_low = sum(1 for p in PROBES if disps[p]["worst"] == "Africa")
    af_gap = sum(1 for p in PROBES if "Africa" in (disps[p]["worst"], disps[p]["best"]))
    AFRICA[m] = dict(low=af_low, gap=af_gap)

# neighbourhood bootstrap CI (B=10000)
NEIGH = {m: dict(disp=PROBE_DISP[m]["P5_neighbourhood"]["disp"],
                 ci=bootstrap_disparity_ci(DATA[m], probe="P5_neighbourhood", B=10000))
         for m in DATA}

# ---------------------------------------------------------------------------
# emit LaTeX
# ---------------------------------------------------------------------------
def f3(x): return f"{x:.3f}"
def sci(p):
    if p == 0 or p < 1e-300: return r"$<10^{-300}$"
    e = int(np.floor(np.log10(p))); mant = p/10**e
    return rf"${mant:.1f}\times10^{{{e}}}$" if e < -3 else f"{p:.4f}"

L = []
order = sorted(COMPOSITE, key=lambda m: COMPOSITE[m]["composite"])

# --- Table: leaderboard ---
L += [r"\begin{table}[t]\centering\small", r"\caption{\textbf{Model leaderboard} (complete runs, judge valence). "
      r"Composite is the mean per-probe max--min regional gap (lower is better); all three models now score "
      r"$100\%$ valid on the complete runs.}", r"\label{tab:leaderboard}",
      r"\begin{tabular}{lccccl}", r"\toprule",
      r"Model & Composite $\downarrow$ & Valid & Severity & Worst Probe & Worst Gap \\", r"\midrule"]
for m in order:
    c = COMPOSITE[m]
    L.append(f"{m} & {f3(c['composite'])} & {c['valid']:.0f}\\% & {c['severity']} & "
             f"{PLABEL[c['worst_probe']]} & {f3(c['worst_gap'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# --- Table: per-probe disparity breakdown ---
L += [r"\begin{table}[t]\centering\small", r"\caption{Per-model, per-probe disparity breakdown "
      r"(max--min valence gap) with worst- and best-treated regions, from the complete runs.}",
      r"\label{tab:probe-breakdown}", r"\begin{tabular}{llccc}", r"\toprule",
      r"Model & Probe & Disparity & Worst Region & Best Region \\", r"\midrule"]
for m in order:
    L.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{m}}}}} \\")
    for i, p in enumerate(PROBES, 1):
        d = PROBE_DISP[m][p]
        L.append(f"\\quad P{i} {PLABEL[p]} & & {f3(d['disp'])} & {RSHORT[d['worst']]} & {RSHORT[d['best']]} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}", r"\end{table}", ""]

# --- Table: multi-metric disparity ---
L += [r"\begin{table}[t]\centering\small", r"\caption{Multi-metric regional disparity with bootstrap "
      r"$95\%$ CIs (image-level resampling). Worst$\rightarrow$best region shown per metric.}",
      r"\label{tab:multimetric}", r"\begin{tabular}{llcl}", r"\toprule",
      r"Model & Metric & Disparity & Worst $\rightarrow$ Best Region \\", r"\midrule"]
for m in order:
    L.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{m}}}}} \\")
    for label, mm in MULTI[m].items():
        ci = "" if mm["lo"] is None else f" [{mm['lo']:.3f}, {mm['hi']:.3f}]"
        L.append(f"\\quad {label} & & {f3(mm['disp'])}{ci} & "
                 f"{RSHORT[mm['worst']]} $\\rightarrow$ {RSHORT[mm['best']]} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}", r"\end{table}", ""]

# --- Table: two-way structure ---
L += [r"\begin{table}[t]\centering\small", r"\caption{Two-way structure of valence: one-way $F$ for region "
      r"and probe factors, and the variance of the region$\times$probe interaction effect.}",
      r"\label{tab:anova}", r"\begin{tabular}{lccc}", r"\toprule",
      r"Model & $F_{\text{region}}$ & $F_{\text{probe}}$ & Interaction Var. \\", r"\midrule"]
for m in order:
    a = ANOVA[m]
    L.append(f"{m} & {a['F_region']:.1f} & {a['F_probe']:.1f} & {a['inter_var']:.5f} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# --- Table: modal worst region ---
L += [r"\begin{table}[t]\centering\small", r"\caption{Modal worst-treated region per model across the five "
      r"probes, with the fraction of probes for which it is worst (consistency).}",
      r"\label{tab:modal}", r"\begin{tabular}{lccc}", r"\toprule",
      r"Model & Modal Worst Region & Worst Count & Consistency \\", r"\midrule"]
for m in order:
    md = MODAL[m]
    L.append(f"{m} & {md['modal']} & {md['count']} & {md['consistency']:.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# --- Table: region x probe mean valence ---
L += [r"\begin{table*}[t]\centering\small", r"\caption{Region$\times$probe mean judge valence for each model "
      r"(complete runs).}", r"\label{tab:rxp}", r"\begin{tabular}{ll" + "c"*6 + "}", r"\toprule",
      r"Model & Probe & " + " & ".join(RSHORT[r] for r in REGIONS) + r" \\", r"\midrule"]
for m in order:
    L.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{m}}}}} \\")
    for i, p in enumerate(PROBES, 1):
        row = " & ".join(f3(RXP[m][p][r]) for r in REGIONS)
        L.append(f"\\quad P{i} & {PLABEL[p]} & {row} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}", r"\end{table*}", ""]

# --- Table: Kruskal-Wallis ---
L += [r"\begin{table}[t]\centering\small", rf"\caption{{Kruskal--Wallis omnibus test per model$\times$probe "
      rf"(6 regions, $df{{=}}5$). Significance at Bonferroni $\alpha_{{\text{{adj}}}}\approx{ALPHA_ADJ:.5f}$.}}",
      r"\label{tab:kw}", r"\begin{tabular}{llcccc}", r"\toprule",
      r"Model & Probe & $H$ & $p$ & $\eta^2$ & Sig. \\", r"\midrule"]
for m in order:
    L.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{m}}}}} \\")
    for i, p in enumerate(PROBES, 1):
        k = KW[m][p]
        L.append(f"\\quad P{i} {PLABEL[p]} & & {k['H']:.2f} & {sci(k['p'])} & "
                 f"{k['eta2']:.4f} & {'***' if k['sig'] else 'n.s.'} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}", r"\end{table}", ""]

# --- Table: Dunn headline contrast ---
L += [r"\begin{table}[t]\centering\small", r"\caption{Dunn post-hoc (Holm-corrected) headline-contrast "
      r"significance for the worst$\rightarrow$best regional pair, restricted to cells with a significant "
      r"omnibus test. $\Delta\bar v$ is the max--min valence gap.}", r"\label{tab:dunn}",
      r"\begin{tabular}{llcc}", r"\toprule",
      r"Model & Probe & $\Delta\bar v$ & Contrast sig. \\", r"\midrule"]
for m in order:
    L.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{m}}}}} \\")
    for i, p in enumerate(PROBES, 1):
        k = KW[m][p]; d = PROBE_DISP[m][p]
        if not k["sig"]:
            L.append(f"\\quad P{i} {PLABEL[p]} & & --- & n.s. omnibus \\\\")
        else:
            L.append(f"\\quad P{i} {PLABEL[p]} & & {f3(d['disp'])} & {'Yes' if k['dunn'] else 'No'} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}", r"\end{table}", ""]

# --- Table: Africa extreme ---
L += [r"\begin{table}[t]\centering\small", r"\caption{Africa as regional extreme. ``Africa lowest'' counts "
      r"probes where the African mean is the minimum; ``Africa in max gap'' counts probes where Africa is an "
      r"endpoint of the largest pairwise gap.}", r"\label{tab:africa}", r"\begin{tabular}{lcc}", r"\toprule",
      r"Model & Africa lowest & Africa in max gap \\", r"\midrule"]
tl = tg = 0
for m in order:
    a = AFRICA[m]; tl += a["low"]; tg += a["gap"]
    L.append(f"{m} & {a['low']}/5 & {a['gap']}/5 \\\\")
L += [r"\midrule", f"Total & {tl}/15 & {tg}/15 \\\\", r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# --- Table: neighbourhood bootstrap ---
L += [r"\begin{table}[t]\centering\small", r"\caption{Bootstrap $95\%$ CI on the neighbourhood-probe max--min "
      r"disparity ($B{=}10{,}000$ image-level resamples, seed 42).}", r"\label{tab:neigh-boot}",
      r"\begin{tabular}{lccc}", r"\toprule",
      r"Model & Disparity & $95\%$ CI & Excl.\ 0 \\", r"\midrule"]
for m in order:
    nb = NEIGH[m]; lo, hi = nb["ci"]
    L.append(f"{m} & {f3(nb['disp'])} & [{lo:.3f}, {hi:.3f}] & {'Yes' if lo>0 else 'No'} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

out = f"{ROOT}/figures/dpe_aaai/baseline_tables.tex"
open(out, "w").write("\n".join(L) + "\n")
print("Wrote", out)

# single source of truth for the figures: full-precision region x probe means +
# derived per-probe disparities, so make_baseline_figures.py never drifts.
import json
fig_data = {
    "regions": REGIONS,
    "probes": [PLABEL[p] for p in PROBES],
    "region_probe": {m: {r: [RXP[m][p][r] for p in PROBES] for r in REGIONS} for m in DATA},
    "region_val": {m: [np.mean([RXP[m][p][r] for p in PROBES]) for r in REGIONS] for m in DATA},
    "disparity": {m: [PROBE_DISP[m][p]["disp"] for p in PROBES] for m in DATA},
    "composite": {m: COMPOSITE[m]["composite"] for m in DATA},
}
fig_json = f"{ROOT}/figures/dpe_aaai/baseline_fig_data.json"
open(fig_json, "w").write(json.dumps(fig_data, indent=1))
print("Wrote", fig_json)

# ---------------------------------------------------------------------------
# appendix-format tables, matching aaai_build/main.tex row layout exactly, so
# they can be pasted verbatim (no hand transcription). Column order for the
# group-means table is Africa, Americas, Asia, Europe, N. America, Oceania.
# ---------------------------------------------------------------------------
APP_COLS = ["Africa", "Americas", "Asia", "Europe", "Northern America", "Oceania"]
A = []
A.append("% ==== group means (raw judge valence, [-1,1]) ====")
for m in order:
    A.append(rf"\multirow{{5}}{{*}}{{\model{{{m}}}}}")
    for i, p in enumerate(PROBES, 1):
        row = " & ".join(f"{RXP[m][p][c]:.3f}" for c in APP_COLS)
        A.append(f"& \\probe{{{i}}} & {row} \\\\")
    A.append(r"\midrule" if m != order[-1] else r"\bottomrule")

A.append("\n% ==== Kruskal-Wallis (complete runs) ====")
for m in order:
    A.append(rf"\multirow{{5}}{{*}}{{\model{{{m}}}}}")
    for i, p in enumerate(PROBES, 1):
        k = KW[m][p]
        A.append(f"& \\probe{{{i}}} {PLABEL[p]} & {k['H']:.2f} & 5 & {sci(k['p'])} & "
                 f"{k['eta2']:.4f} & {'''\\texttt{***}''' if k['sig'] else '''\\texttt{n.s.}'''} \\\\")
    A.append(r"\midrule" if m != order[-1] else r"\bottomrule")

A.append("\n% ==== Dunn headline contrast (complete runs) ====")
for m in order:
    A.append(rf"\multirow{{5}}{{*}}{{\model{{{m}}}}}")
    for i, p in enumerate(PROBES, 1):
        k = KW[m][p]; d = PROBE_DISP[m][p]
        if not k["sig"]:
            A.append(f"& \\probe{{{i}}} {PLABEL[p]} & --- & \\emph{{n.s.\\ omnibus}} \\\\")
        else:
            A.append(f"& \\probe{{{i}}} {PLABEL[p]} & {d['disp']:.3f} & {'Yes' if k['dunn'] else 'No'} \\\\")
    A.append(r"\midrule" if m != order[-1] else r"\bottomrule")

A.append("\n% ==== Africa extreme (complete runs) ====")
tl = tg = 0
for m in order:
    a = AFRICA[m]; tl += a["low"]; tg += a["gap"]
    A.append(f"\\model{{{m}}} & {a['low']}/5 & {a['gap']}/5 \\\\")
A.append(f"\\textbf{{Total}} & \\textbf{{{tl}/15}} & \\textbf{{{tg}/15}} \\\\")

A.append("\n% ==== Neighbourhood bootstrap CI (B=10000) ====")
for m in order:
    nb = NEIGH[m]; lo, hi = nb["ci"]
    A.append(f"\\model{{{m}}} & \\probe{{5}} Neighbourhood & {nb['disp']:.3f} & "
             f"$[{lo:.3f},\\ {hi:.3f}]$ & {'Yes' if lo>0 else 'No'} \\\\")

# summary stats for prose
n_sig = sum(1 for m in order for p in PROBES if KW[m][p]["sig"])
A.append(f"\n% n_omnibus_sig = {n_sig}/15")
nonsig = [(m, PLABEL[p]) for m in order for p in PROBES if not KW[m][p]["sig"]]
A.append(f"% non-sig cells: {nonsig}")
contrast_no = [(m, PLABEL[p]) for m in order for p in PROBES
               if KW[m][p]["sig"] and not KW[m][p]["dunn"]]
A.append(f"% omnibus-sig but contrast NOT sig: {contrast_no}")
app_out = f"{ROOT}/figures/dpe_aaai/baseline_appendix_rows.tex"
open(app_out, "w").write("\n".join(A) + "\n")
print("Wrote", app_out)
print("\n=== KEY NUMBERS (judge valence, complete runs) ===")
for m in order:
    c = COMPOSITE[m]
    print(f"{m:15s} composite={c['composite']:.3f} ({c['severity']:9s}) "
          f"worst={PLABEL[c['worst_probe']]}({c['worst_gap']:.3f}) valid=100%")
