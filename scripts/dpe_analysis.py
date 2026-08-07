#!/usr/bin/env python3
"""
DPE analysis — consolidated pipeline for the AAAI Demographic Positional
Encoder (DPE) study.

WHAT IS DPE?
------------
The Demographic Positional Encoder (DPE) is an inference-time debiasing
correction for vision-language models (VLMs). It adds a small, per-group
correction vector to the model's visual tokens so that the model's *valence*
(sentiment) toward people from different world regions is equalised, without
retraining. The correction magnitude is scaled by a per-model strength
alpha* (IDEFICS2 0.25, InternVL2 0.5, LLaVA 0.5). Because the projection W is
orthonormal (isometric), the injected norm equals alpha * ||delta_g||, where
delta_g = grand_mean - group_mean over (valence, stereotype_alignment,
confidence).

This module regenerates every baseline (pre-DPE) and DPE (post-correction)
analysis table and figure used in the paper, from the raw judge scores and
response databases. Outputs are byte-for-byte equivalent to the original
standalone scripts this file consolidates.

PIPELINE ORDER (subcommands)
----------------------------
The subcommands should be run in this dependency order (this is what `all`
does):

  1. baseline-tables : Baseline (pre-DPE) statistics from the COMPLETE runs.
       Emits figures/dpe_aaai/baseline_tables.tex,
              figures/dpe_aaai/baseline_fig_data.json,
              figures/dpe_aaai/baseline_appendix_rows.tex
       (baseline_fig_data.json is the single source of truth consumed by
        paper-figs, so this MUST run first.)

  2. paper-figs      : Main-paper baseline figures, from baseline_fig_data.json.
       Emits aaai_build/fig1_leaderboard_heatmap.{pdf,png},
              aaai_build/fig2_radar_fingerprints.{pdf,png},
              aaai_build/fig3_probe_comparison.{pdf,png},
              aaai_build/fig6_worstbest_group.{pdf,png},
              aaai_build/fig9_regional_comparison.{pdf,png}

  3. tier1           : Tier-1 DPE analysis (reviewer rebuttals): per-dimension
       disparity sigma baseline->DPE, valence-sigma robustness (paired
       bootstrap), overall valence shift.
       Emits figures/dpe_aaai/dpe_tier1_tables.tex,
              figures/dpe_aaai/fig_dpe_sigma_before_after.{pdf,png},
              figures/dpe_aaai/fig_dpe_convergence.{pdf,png}

  4. norms           : DPE correction-vector norms per region per model
       (appendix). Builds the encoder from each baseline DB.
       Emits figures/dpe_aaai/dpe_correction_norms.tex

  5. content         : Content-preservation / task-fidelity under DPE
       (needs sentence-transformers).
       Emits figures/dpe_aaai/dpe_content_preservation.tex

  6. umap            : UMAP of the true VLM visual-token embeddings before vs
       after DPE (needs umap-learn + results/dpe_embeddings/*_emb.npz).
       Emits figures/dpe_aaai/fig_dpe_visual_umap.{pdf,png}

INPUT DATA LOCATIONS
--------------------
  results/single_runs_35k/*.db                       — baseline (pre-DPE) runs
  results/dpe_final_20260720_125918/*/*_dpe.db        — final DPE eval runs
  results/dpe_embeddings/*_emb.npz                    — dumped visual-token
                                                        embeddings (umap only)

NOTE ON dump_visual_embeddings.py
---------------------------------
scripts/dump_visual_embeddings.py stays SEPARATE and is NOT merged here. It
runs on the GPU cluster (rolf) in a different conda env to dump the
visual-token embeddings into results/dpe_embeddings/*_emb.npz. This module's
`umap` subcommand only *consumes* that .npz output.

EXAMPLE USAGE
-------------
  python3 scripts/dpe_analysis.py all              # full pipeline, dep order
  python3 scripts/dpe_analysis.py baseline-tables  # just baseline tables/json
  python3 scripts/dpe_analysis.py paper-figs       # main-paper baseline figs
  python3 scripts/dpe_analysis.py tier1
  python3 scripts/dpe_analysis.py norms
  python3 scripts/dpe_analysis.py content          # needs sentence-transformers
  python3 scripts/dpe_analysis.py umap             # needs umap-learn + .npz
"""
import argparse
import sqlite3
import json
import glob
import os
import sys
import importlib.util
import difflib
from collections import Counter

import numpy as np
import pandas as pd

# ===========================================================================
# SHARED CONFIG  (defined ONCE; reused by every subcommand)
# ===========================================================================
ROOT = "/Users/ahmeda./Desktop/FingerPrint"
OUT = f"{ROOT}/figures/dpe_aaai"
AAAI_BUILD = f"{ROOT}/aaai_build"
FINAL = f"{ROOT}/results/dpe_final_20260720_125918"
EMB = f"{ROOT}/results/dpe_embeddings"

# baseline (pre-DPE) DB path per display model name
BASELINE_DBS = {
    "IDEFICS2-8B":   f"{ROOT}/results/single_runs_35k/gpu0_HuggingFaceM4_idefics2_8b_20260427_114159.db",
    "InternVL2-2B":  f"{ROOT}/results/single_runs_35k/gpu6_OpenGVLab_InternVL2_2B_20260421_145205.db",
    "LLaVA-v1.6-7B": f"{ROOT}/results/single_runs_35k/gpu7_llava_hf_llava_v1.6_vicuna_7b_hf_20260421_145210.db",
}
# DPE final-eval DB path per display model name
DPE_DBS = {
    "IDEFICS2-8B":   f"{FINAL}/idefics2/idefics2_dpe.db",
    "InternVL2-2B":  f"{FINAL}/internvl2/internvl2_dpe.db",
    "LLaVA-v1.6-7B": f"{FINAL}/llava/llava_dpe.db",
}
# optimal correction strength alpha* per model
ALPHA = {"IDEFICS2-8B": 0.25, "InternVL2-2B": 0.5, "LLaVA-v1.6-7B": 0.5}
# canonical model display order
MODELS = ["IDEFICS2-8B", "InternVL2-2B", "LLaVA-v1.6-7B"]

REGIONS = ["Africa", "Asia", "Europe", "Americas", "Northern America", "Oceania"]
RSHORT = {"Africa": "Africa", "Asia": "Asia", "Europe": "Europe", "Americas": "Americas",
          "Northern America": "N. America", "Oceania": "Oceania"}
PROBES = ["P1_occupation", "P2_education", "P3_trustworthiness", "P4_lifestyle", "P5_neighbourhood"]
PLABEL = {"P1_occupation": "Occupation", "P2_education": "Education",
          "P3_trustworthiness": "Trustworthiness", "P4_lifestyle": "Lifestyle",
          "P5_neighbourhood": "Neighbourhood"}

# per-model brand colours (figures)
MODEL_C = {"IDEFICS2-8B": "#1f4e9c", "InternVL2-2B": "#c1440e", "LLaVA-v1.6-7B": "#1a7a3c"}
# per-region colours (umap)
RCOL = {"Africa": "#d1193e", "Asia": "#e8890c", "Europe": "#128a5c",
        "Americas": "#1f4e9c", "Northern America": "#7b1fa2", "Oceania": "#00838f"}

# shared matplotlib rcParams (used by paper-figs and tier1; superset is safe)
RCPARAMS = {
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333",
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
}
# minimal rcParams for the umap figure (matches original dpe_visual_umap.py)
RCPARAMS_UMAP = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                 "pdf.fonttype": 42, "ps.fonttype": 42}


# ===========================================================================
# cmd_baseline_tables()  <- regenerate_baseline_tables.py
# ===========================================================================
def cmd_baseline_tables():
    """
    Regenerate ALL baseline (pre-DPE) analysis tables from the COMPLETE runs,
    using the judge valence already stored in judge_scores (the SAME metric the
    DPE evaluation uses) so the baseline tables, baseline figures, and DPE results
    are all on one consistent scale.

    Reads:  results/single_runs_35k/{gpu0 idefics2, gpu6 InternVL2, gpu7 llava}.db
    Writes: figures/dpe_aaai/baseline_tables.tex  (all tables, ready to \\input)
    """
    from scipy import stats
    from scipy.stats import f_oneway, kruskal, norm

    RNG = np.random.default_rng(42)
    DBS = BASELINE_DBS
    RSHORT_L = RSHORT
    PROBES_L = PROBES
    PLABEL_L = PLABEL
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
               for p in PROBES_L for r in REGIONS}
        per = {k: v for k, v in per.items() if len(v)}
        vals = np.empty(B)
        for b in range(B):
            means = {k: RNG.choice(v, size=len(v), replace=True).mean() for k, v in per.items()}
            gaps = []
            for p in PROBES_L:
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
        rxp = {p: region_means(df[df.probe_id == p]) for p in PROBES_L}
        RXP[m] = rxp
        disps = {}
        for p in PROBES_L:
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
        for p in PROBES_L:
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
        for label, col in [("Sentiment valence", "valence"), ("Economic valence", "economic_valence"),
                           ("Stereotype score", "stereotype_alignment")]:
            if col not in df or df[col].isna().all():
                continue
            rm = region_means(df, col)
            worst = min(rm, key=rm.get); best = max(rm, key=rm.get)
            lo, hi = bootstrap_disparity_ci(df, col=col, B=2000)
            mm[label] = dict(disp=rm[best]-rm[worst], lo=lo, hi=hi, worst=worst, best=best)
        # probe-specific = mean per-probe gap; CI via bootstrap of the SAME quantity;
        # worst->best = modal worst region across probes -> modal best region
        ps = np.mean([disps[p]["disp"] for p in PROBES_L])
        lo, hi = bootstrap_meanprobe_gap_ci(df, B=2000)
        mw = Counter(disps[p]["worst"] for p in PROBES_L).most_common(1)[0][0]
        mb = Counter(disps[p]["best"] for p in PROBES_L).most_common(1)[0][0]
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
        fp = f_oneway(*[df[df.probe_id == p]["valence"].values for p in PROBES_L])[0]
        grand = df["valence"].mean()
        reff = {r: df[df.region == r]["valence"].mean()-grand for r in REGIONS}
        peff = {p: df[df.probe_id == p]["valence"].mean()-grand for p in PROBES_L}
        inter = []
        for p in PROBES_L:
            for r in REGIONS:
                if not rxp[p].get(r) is None and r in rxp[p]:
                    inter.append(rxp[p][r] - (grand + reff[r] + peff[p]))
        ANOVA[m] = dict(F_region=fr, F_probe=fp, inter_var=float(np.var(inter)))
        # modal worst region
        worst_regions = [disps[p]["worst"] for p in PROBES_L]
        vals, cnts = np.unique(worst_regions, return_counts=True)
        top = cnts.max(); modal = ", ".join(RSHORT_L[v] for v, c in zip(vals, cnts) if c == top)
        MODAL[m] = dict(modal=modal, count=f"{top}/5", consistency=top/5.0)
        # africa extreme
        af_low = sum(1 for p in PROBES_L if disps[p]["worst"] == "Africa")
        af_gap = sum(1 for p in PROBES_L if "Africa" in (disps[p]["worst"], disps[p]["best"]))
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
                 f"{PLABEL_L[c['worst_probe']]} & {f3(c['worst_gap'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # --- Table: per-probe disparity breakdown ---
    L += [r"\begin{table}[t]\centering\small", r"\caption{Per-model, per-probe disparity breakdown "
          r"(max--min valence gap) with worst- and best-treated regions, from the complete runs.}",
          r"\label{tab:probe-breakdown}", r"\begin{tabular}{llccc}", r"\toprule",
          r"Model & Probe & Disparity & Worst Region & Best Region \\", r"\midrule"]
    for m in order:
        L.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{m}}}}} \\")
        for i, p in enumerate(PROBES_L, 1):
            d = PROBE_DISP[m][p]
            L.append(f"\\quad P{i} {PLABEL_L[p]} & & {f3(d['disp'])} & {RSHORT_L[d['worst']]} & {RSHORT_L[d['best']]} \\\\")
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
                     f"{RSHORT_L[mm['worst']]} $\\rightarrow$ {RSHORT_L[mm['best']]} \\\\")
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
          r"Model & Probe & " + " & ".join(RSHORT_L[r] for r in REGIONS) + r" \\", r"\midrule"]
    for m in order:
        L.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{m}}}}} \\")
        for i, p in enumerate(PROBES_L, 1):
            row = " & ".join(f3(RXP[m][p][r]) for r in REGIONS)
            L.append(f"\\quad P{i} & {PLABEL_L[p]} & {row} \\\\")
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
        for i, p in enumerate(PROBES_L, 1):
            k = KW[m][p]
            L.append(f"\\quad P{i} {PLABEL_L[p]} & & {k['H']:.2f} & {sci(k['p'])} & "
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
        for i, p in enumerate(PROBES_L, 1):
            k = KW[m][p]; d = PROBE_DISP[m][p]
            if not k["sig"]:
                L.append(f"\\quad P{i} {PLABEL_L[p]} & & --- & n.s. omnibus \\\\")
            else:
                L.append(f"\\quad P{i} {PLABEL_L[p]} & & {f3(d['disp'])} & {'Yes' if k['dunn'] else 'No'} \\\\")
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
    fig_data = {
        "regions": REGIONS,
        "probes": [PLABEL_L[p] for p in PROBES_L],
        "region_probe": {m: {r: [RXP[m][p][r] for p in PROBES_L] for r in REGIONS} for m in DATA},
        "region_val": {m: [np.mean([RXP[m][p][r] for p in PROBES_L]) for r in REGIONS] for m in DATA},
        "disparity": {m: [PROBE_DISP[m][p]["disp"] for p in PROBES_L] for m in DATA},
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
        for i, p in enumerate(PROBES_L, 1):
            row = " & ".join(f"{RXP[m][p][c]:.3f}" for c in APP_COLS)
            A.append(f"& \\probe{{{i}}} & {row} \\\\")
        A.append(r"\midrule" if m != order[-1] else r"\bottomrule")

    A.append("\n% ==== Kruskal-Wallis (complete runs) ====")
    for m in order:
        A.append(rf"\multirow{{5}}{{*}}{{\model{{{m}}}}}")
        for i, p in enumerate(PROBES_L, 1):
            k = KW[m][p]
            A.append(f"& \\probe{{{i}}} {PLABEL_L[p]} & {k['H']:.2f} & 5 & {sci(k['p'])} & "
                     f"{k['eta2']:.4f} & {'''\\texttt{***}''' if k['sig'] else '''\\texttt{n.s.}'''} \\\\")
        A.append(r"\midrule" if m != order[-1] else r"\bottomrule")

    A.append("\n% ==== Dunn headline contrast (complete runs) ====")
    for m in order:
        A.append(rf"\multirow{{5}}{{*}}{{\model{{{m}}}}}")
        for i, p in enumerate(PROBES_L, 1):
            k = KW[m][p]; d = PROBE_DISP[m][p]
            if not k["sig"]:
                A.append(f"& \\probe{{{i}}} {PLABEL_L[p]} & --- & \\emph{{n.s.\\ omnibus}} \\\\")
            else:
                A.append(f"& \\probe{{{i}}} {PLABEL_L[p]} & {d['disp']:.3f} & {'Yes' if k['dunn'] else 'No'} \\\\")
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
    n_sig = sum(1 for m in order for p in PROBES_L if KW[m][p]["sig"])
    A.append(f"\n% n_omnibus_sig = {n_sig}/15")
    nonsig = [(m, PLABEL_L[p]) for m in order for p in PROBES_L if not KW[m][p]["sig"]]
    A.append(f"% non-sig cells: {nonsig}")
    contrast_no = [(m, PLABEL_L[p]) for m in order for p in PROBES_L
                   if KW[m][p]["sig"] and not KW[m][p]["dunn"]]
    A.append(f"% omnibus-sig but contrast NOT sig: {contrast_no}")
    app_out = f"{ROOT}/figures/dpe_aaai/baseline_appendix_rows.tex"
    open(app_out, "w").write("\n".join(A) + "\n")
    print("Wrote", app_out)
    print("\n=== KEY NUMBERS (judge valence, complete runs) ===")
    for m in order:
        c = COMPOSITE[m]
        print(f"{m:15s} composite={c['composite']:.3f} ({c['severity']:9s}) "
              f"worst={PLABEL_L[c['worst_probe']]}({c['worst_gap']:.3f}) valid=100%")


# ===========================================================================
# cmd_paper_figs()  <- make_paper_baseline_figures.py
# ===========================================================================
def cmd_paper_figs():
    """
    Regenerate the main-paper baseline figures from the COMPLETE runs (judge
    valence), written under the paper's existing filenames into aaai_build/.

    Source of truth: figures/dpe_aaai/baseline_fig_data.json
    """
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(RCPARAMS)
    OUT = AAAI_BUILD
    DATA = json.load(open(f"{ROOT}/figures/dpe_aaai/baseline_fig_data.json"))

    MODELS_L = ["IDEFICS2-8B", "InternVL2-2B", "LLaVA-v1.6-7B"]
    MODEL_C_L = MODEL_C
    REGIONS_L = DATA["regions"]
    RSHORT_L = {"Africa": "Africa", "Asia": "Asia", "Europe": "Europe", "Americas": "Americas",
                "Northern America": "N. America", "Oceania": "Oceania"}
    PROBES_L = ["Occupation", "Education", "Trust.", "Lifestyle", "Neighb."]

    r01 = lambda v: (v + 1.0) / 2.0                       # valence [-1,1] -> [0,1]
    DISP = {m: DATA["disparity"][m] for m in MODELS_L}      # raw max-min gaps
    RVAL01 = {m: [r01(v) for v in DATA["region_val"][m]] for m in MODELS_L}
    COMP = {m: float(np.mean(DISP[m])) for m in MODELS_L}

    def save(fig, name):
        fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png")
        plt.close(fig); print("  wrote", name)

    # --- fig1: composite leaderboard + model x probe disparity heatmap -----------
    def fig1():
        fig = plt.figure(figsize=(7.2, 2.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.5)
        ax1 = fig.add_subplot(gs[0])
        order = sorted(MODELS_L, key=lambda m: COMP[m])
        y = np.arange(len(order))
        ax1.barh(y, [COMP[m] for m in order], color=[MODEL_C_L[m] for m in order],
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
        M = np.array([DISP[m] for m in MODELS_L])
        im = ax2.imshow(M, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
        ax2.set_xticks(range(len(PROBES_L))); ax2.set_xticklabels(PROBES_L, fontsize=8)
        ax2.set_yticks(range(len(MODELS_L))); ax2.set_yticklabels(MODELS_L, fontsize=8.5)
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
        N = len(PROBES_L)
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); ang += ang[:1]
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), subplot_kw=dict(polar=True))
        for ax, m in zip(axes, MODELS_L):
            v = DISP[m] + DISP[m][:1]
            ax.plot(ang, v, color=MODEL_C_L[m], lw=2.0)
            ax.fill(ang, v, color=MODEL_C_L[m], alpha=0.12)
            ax.set_xticks(ang[:-1]); ax.set_xticklabels(PROBES_L, fontsize=7)
            ax.set_ylim(0, 1.0); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6, color="#999")
            ax.tick_params(pad=0.5)
            ax.grid(color="#ddd", lw=0.7); ax.spines["polar"].set_color("#ccc")
            ax.text(0.5, -0.22, m, transform=ax.transAxes, ha="center", va="top",
                    fontsize=8.5, color=MODEL_C_L[m])
        fig.subplots_adjust(wspace=0.45)
        save(fig, "fig2_radar_fingerprints")

    # --- fig3: grouped per-probe disparity (0-1 axis, no annotation) -------------
    def fig3():
        fig, ax = plt.subplots(figsize=(6.6, 2.9))
        x = np.arange(len(PROBES_L)); w = 0.26
        for i, m in enumerate(MODELS_L):
            ax.bar(x + (i - 1) * w, DISP[m], w, color=MODEL_C_L[m], edgecolor="black",
                   lw=0.4, label=m, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(PROBES_L)
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
        xg = np.arange(len(MODELS_L)); wg = 0.36
        worst = [min(RVAL01[m]) for m in MODELS_L]; best = [max(RVAL01[m]) for m in MODELS_L]
        wlab = [RSHORT_L[REGIONS_L[int(np.argmin(RVAL01[m]))]] for m in MODELS_L]
        blab = [RSHORT_L[REGIONS_L[int(np.argmax(RVAL01[m]))]] for m in MODELS_L]
        ax.bar(xg - wg/2, worst, wg, color="#b0413e", edgecolor="black", lw=0.5,
               label="worst-treated region", zorder=3)
        ax.bar(xg + wg/2, best, wg, color="#3a6ea5", edgecolor="black", lw=0.5,
               label="best-treated region", zorder=3)
        for i in range(len(MODELS_L)):
            ax.text(xg[i] - wg/2, worst[i] + 0.015, wlab[i], ha="center", va="bottom", fontsize=6.5)
            ax.text(xg[i] + wg/2, best[i] + 0.015, blab[i], ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(xg); ax.set_xticklabels(MODELS_L, fontsize=8.5)
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
        x = np.arange(len(REGIONS_L)); w = 0.26
        for i, m in enumerate(MODELS_L):
            ax.bar(x + (i - 1) * w, RVAL01[m], w, color=MODEL_C_L[m], edgecolor="black",
                   lw=0.4, label=m, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([RSHORT_L[r] for r in REGIONS_L], fontsize=8)
        ax.set_ylabel("mean valence (rescaled to $[0,1]$)")
        ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.grid(True, axis="y", color="#eee", lw=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=3, handlelength=1.5, columnspacing=1.4)
        save(fig, "fig9_regional_comparison")

    print("Writing complete-run baseline figures to", OUT)
    fig1(); fig2(); fig3(); fig6(); fig9()
    print("Done. Composites:", {m: round(COMP[m], 3) for m in MODELS_L})


# ===========================================================================
# cmd_tier1()  <- dpe_tier1_analysis.py
# ===========================================================================
def cmd_tier1():
    """
    Tier-1 DPE analysis (reviewer rebuttals), from the local final-eval DBs
    (results/dpe_final_20260720_125918) matched against the archived baselines.

    Outputs (figures/dpe_aaai/):
      dpe_tier1_tables.tex, fig_dpe_sigma_before_after.{pdf,png},
      fig_dpe_convergence.{pdf,png}
    """
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    RNG = np.random.default_rng(42)
    RSHORT_L = RSHORT
    DIMS = [("valence", "Sentiment valence"), ("economic_valence", "Economic valence"),
            ("stereotype_alignment", "Stereotype align."), ("refusal", "Refusal rate")]
    MODELS_L = {  # display -> (baseline db, dpe db, alpha)
        "IDEFICS2-8B": (BASELINE_DBS["IDEFICS2-8B"], DPE_DBS["IDEFICS2-8B"], ALPHA["IDEFICS2-8B"]),
        "InternVL2-2B": (BASELINE_DBS["InternVL2-2B"], DPE_DBS["InternVL2-2B"], ALPHA["InternVL2-2B"]),
        "LLaVA-v1.6-7B": (BASELINE_DBS["LLaVA-v1.6-7B"], DPE_DBS["LLaVA-v1.6-7B"], ALPHA["LLaVA-v1.6-7B"]),
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
    for m, (bdb, ddb, alpha) in MODELS_L.items():
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
                      mad_b=mad(base, "valence"), mad_d=mad(dpe, "valence"),
                      d_point=dpt, d_lo=dlo, d_hi=dhi, ci_b=sbci, ci_d=sdci,
                      ov_b=float(base.valence.mean()), ov_d=float(dpe.valence.mean()),
                      reg_b={r: float(base[base.region == r].valence.mean()) for r in REGIONS},
                      reg_d={r: float(dpe[dpe.region == r].valence.mean()) for r in REGIONS},
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
    for m in MODELS_L:
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
    for m in MODELS_L:
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
    for m in MODELS_L:
        d = RES[m]
        L.append(f"{m} & {d['ov_b']:.4f} & {d['ov_d']:.4f} & {d['ov_d']-d['ov_b']:+.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(f"{OUT}/dpe_tier1_tables.tex", "w").write("\n".join(L) + "\n")
    print("Wrote", f"{OUT}/dpe_tier1_tables.tex")

    # ---- figures -------------------------------------------------------------
    plt.rcParams.update(RCPARAMS)
    MC = MODEL_C
    mods = list(MODELS_L)

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
    ax.spines[["top", "right"]].set_visible(False)
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
        ax.set_xticks(xr); ax.set_xticklabels([RSHORT_L[r][:4] for r in REGIONS], fontsize=6.5, rotation=45, ha="right")
        ax.set_title(m, fontsize=8.5, color="black", pad=3)
        ax.grid(True, axis="y", color="#f0f0f0", lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        red = d["per_dim"]["valence"]["red"]
        lbl = f"disparity $\\sigma\\downarrow${red:.1f}%" if red > 1 else r"disparity $\sigma\approx$ unch."
        ax.text(0.5, -0.42, lbl, transform=ax.transAxes, ha="center",
                fontsize=7.5, color="#222")
    axes[0].set_ylabel("mean valence")
    axes[0].legend(frameon=False, loc="best", fontsize=7, handlelength=1.3)
    fig.subplots_adjust(wspace=0.32, bottom=0.28)
    fig.savefig(f"{OUT}/fig_dpe_convergence.pdf"); fig.savefig(f"{OUT}/fig_dpe_convergence.png")
    plt.close(fig); print("wrote fig_dpe_convergence")


# ===========================================================================
# cmd_norms()  <- dpe_correction_norms.py
# ===========================================================================
def cmd_norms():
    """
    Report the DPE correction-vector norms per region per model, for the appendix.

    Builds the encoder from each local baseline DB with correction_axis='region'
    (the same configuration used in the final evaluation).
    """
    # load the encoder module directly (bypass the package __init__, which pulls heavy deps)
    _spec = importlib.util.spec_from_file_location(
        "dpe_mod",
        "/Users/ahmeda./Desktop/FingerPrint/fingerprint_squared/debiasing/demographic_positional_encoder.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["dpe_mod"] = _mod            # dataclass introspection needs this
    _spec.loader.exec_module(_mod)
    DemographicPositionalEncoder = _mod.DemographicPositionalEncoder

    RSHORT_L = RSHORT
    # model -> (baseline db, optimal alpha)
    MODELS_L = {
        "IDEFICS2-8B":  (BASELINE_DBS["IDEFICS2-8B"], ALPHA["IDEFICS2-8B"]),
        "InternVL2-2B": (BASELINE_DBS["InternVL2-2B"], ALPHA["InternVL2-2B"]),
        "LLaVA-v1.6-7B": (BASELINE_DBS["LLaVA-v1.6-7B"], ALPHA["LLaVA-v1.6-7B"]),
    }

    NORM = {}   # model -> {region: (raw ||delta||, injected alpha*||delta||)}
    for m, (db, alpha) in MODELS_L.items():
        enc = DemographicPositionalEncoder.from_sqlite([db], correction_axis="region", alpha=alpha)
        NORM[m] = {}
        for r in REGIONS:
            vec = enc.get_correction_vector("*", r)
            raw = float(np.linalg.norm(vec)) if vec is not None else float("nan")
            NORM[m][r] = (raw, alpha * raw)

    # console summary
    print(f"{'region':<16}" + "".join(f"{m:>22}" for m in MODELS_L))
    print(f"{'':16}" + "".join(f"{'raw / injected(αδ)':>22}" for _ in MODELS_L))
    for r in REGIONS:
        row = f"{r:<16}"
        for m in MODELS_L:
            raw, inj = NORM[m][r]
            row += f"{raw:>10.4f} /{inj:>8.4f}  "
        print(row)

    # LaTeX appendix table (injected norm alpha*||delta|| = logged enc value)
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{\textbf{DPE correction-vector norms} $\lVert\alpha\,\varepsilon_g\rVert$ per region "
         r"(the injected magnitude added to the visual tokens; equal to $\alpha\lVert\delta_g\rVert$ since the "
         r"projection is orthonormal). Norms are non-trivial and of comparable magnitude across models---"
         r"in particular \model{LLaVA-v1.6-7B}'s are the \emph{largest}---so its near-zero disparity change "
         r"reflects genuine architectural resistance to the correction rather than a failure to inject it. "
         r"$\alpha^{\star}$: IDEFICS2 $0.25$, InternVL2 $0.5$, LLaVA $0.5$.}",
         r"\label{tab:dpe_norms}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Region & \model{IDEFICS2-8B} & \model{InternVL2-2B} & \model{LLaVA-v1.6-7B} \\", r"\midrule"]
    for r in REGIONS:
        L.append(f"{RSHORT_L[r]} & " + " & ".join(f"{NORM[m][r][1]:.4f}" for m in MODELS_L) + r" \\")
    # mean row
    L.append(r"\midrule")
    L.append("Mean & " + " & ".join(
        f"{np.mean([NORM[m][r][1] for r in REGIONS]):.4f}" for m in MODELS_L) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = f"{ROOT}/figures/dpe_aaai/dpe_correction_norms.tex"
    open(out, "w").write("\n".join(L) + "\n")
    print("\nWrote", out)


# ===========================================================================
# cmd_content()  <- dpe_content_preservation.py
# ===========================================================================
def cmd_content():
    """
    Content preservation / task fidelity under DPE (Tier-2 #5), plus a bonus rebuttal
    of the precision/hardware confound (#7).

    Local only -- no rolf.  Writes figures/dpe_aaai/dpe_content_preservation.tex
    """
    MODELS_L = {
        "IDEFICS2-8B": (BASELINE_DBS["IDEFICS2-8B"], DPE_DBS["IDEFICS2-8B"]),
        "InternVL2-2B": (BASELINE_DBS["InternVL2-2B"], DPE_DBS["InternVL2-2B"]),
        "LLaVA-v1.6-7B": (BASELINE_DBS["LLaVA-v1.6-7B"], DPE_DBS["LLaVA-v1.6-7B"]),
    }

    # SBERT (same encoder family as the judge); fall back to char-sim only if unavailable
    sbert = None
    try:
        from sentence_transformers import SentenceTransformer, util
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        print("SBERT loaded")
    except Exception as e:
        print("SBERT unavailable, char-similarity only:", type(e).__name__)

    def matched_pairs(bdb, ddb):
        con = sqlite3.connect(ddb); con.execute(f"ATTACH '{bdb}' AS b")
        rows = con.execute(
            "SELECT b.response, d.response FROM probe_results d "
            "JOIN b.probe_results b USING(image_id, probe_id) "
            "WHERE d.response IS NOT NULL AND b.response IS NOT NULL").fetchall()
        con.close()
        return [(str(a).strip(), str(c).strip()) for a, c in rows]

    RES = {}
    for m, (bdb, ddb) in MODELS_L.items():
        pairs = matched_pairs(bdb, ddb)
        n = len(pairs)
        ident = [p for p in pairs if p[0] == p[1]]
        changed = [p for p in pairs if p[0] != p[1]]
        pct_ident = 100.0 * len(ident) / n
        # char-level similarity on changed pairs
        charsim = np.mean([difflib.SequenceMatcher(None, a, b).ratio() for a, b in changed]) if changed else 1.0
        # semantic cosine on changed pairs (identical pairs are cosine 1 by definition)
        cos_changed = np.nan
        if sbert is not None and changed:
            A = sbert.encode([a for a, _ in changed], batch_size=256, show_progress_bar=False,
                             normalize_embeddings=True, convert_to_numpy=True)
            B = sbert.encode([b for _, b in changed], batch_size=256, show_progress_bar=False,
                             normalize_embeddings=True, convert_to_numpy=True)
            cos_changed = float(np.mean(np.sum(A * B, axis=1)))
        overall_cos = (len(ident)*1.0 + (cos_changed*len(changed) if not np.isnan(cos_changed) else 0)) / n \
                      if not np.isnan(cos_changed) else np.nan
        RES[m] = dict(n=n, pct_ident=pct_ident, n_changed=len(changed),
                      charsim=charsim, cos_changed=cos_changed, overall_cos=overall_cos)
        print(f"{m:14s} n={n:6d}  identical={pct_ident:5.1f}%  changed={len(changed):5d}  "
              f"char-sim(changed)={charsim:.3f}  SBERT-cos(changed)={cos_changed:.3f}  overall-cos={overall_cos:.3f}")

    # LaTeX table
    def f1(x): return "---" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}"
    def f3(x): return "---" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{\textbf{Content preservation under DPE.} Fraction of responses byte-identical to the "
         r"baseline, and semantic (SBERT cosine) / character (difflib) similarity on the responses that "
         r"changed. DPE leaves most responses untouched and changes the rest only slightly, preserving task "
         r"fidelity; the high identical fraction also shows the DPE run reproduces the archived baseline "
         r"verbatim where the correction does not act, so the with/without-DPE gap is the correction rather "
         r"than precision or hardware drift.}", r"\label{tab:dpe_content}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Model & \% identical & \# changed & SBERT cos (changed) & char sim (changed) \\", r"\midrule"]
    for m in MODELS_L:
        d = RES[m]
        L.append(f"{m} & {f1(d['pct_ident'])}\\% & {d['n_changed']} & {f3(d['cos_changed'])} & {f3(d['charsim'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(f"{ROOT}/figures/dpe_aaai/dpe_content_preservation.tex", "w").write("\n".join(L) + "\n")
    print("\nWrote figures/dpe_aaai/dpe_content_preservation.tex")


# ===========================================================================
# cmd_umap()  <- dpe_visual_umap.py
# ===========================================================================
def cmd_umap():
    """
    UMAP of the true VLM visual-token embeddings, BEFORE vs AFTER the DPE correction,
    from the npz files produced by scripts/dump_visual_embeddings.py (run on rolf).

    Reads results/dpe_embeddings/{idefics2,internvl2,llava}_emb.npz.
    """
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    import umap

    SEED = 42
    NAME = {"idefics2": "IDEFICS2-8B", "internvl2": "InternVL2-2B", "llava": "LLaVA-v1.6-7B"}
    plt.rcParams.update(RCPARAMS_UMAP)

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
                    ax.scatter(E[mk, 0], E[mk, 1], s=5, c=RCOL[r], alpha=0.5, edgecolors="none")
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
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=RCOL[r], mec="none", ms=6,
                          label=r.replace("Northern America", "N. America")) for r in REGIONS]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.2, columnspacing=1.0)
    fig.suptitle("VLM visual-token embedding (UMAP), before vs. after DPE correction",
                 fontsize=10, y=0.999)
    fig.subplots_adjust(hspace=0.22, wspace=0.06, bottom=0.06, top=0.95)
    fig.savefig(f"{OUT}/fig_dpe_visual_umap.pdf", bbox_inches="tight", dpi=400)
    fig.savefig(f"{OUT}/fig_dpe_visual_umap.png", bbox_inches="tight", dpi=200)
    print("wrote", f"{OUT}/fig_dpe_visual_umap.{{pdf,png}}  from", [os.path.basename(x) for x in files])


# ===========================================================================
# CLI dispatcher
# ===========================================================================
COMMANDS = {
    "baseline-tables": cmd_baseline_tables,
    "paper-figs": cmd_paper_figs,
    "tier1": cmd_tier1,
    "norms": cmd_norms,
    "content": cmd_content,
    "umap": cmd_umap,
}
# dependency order used by `all` (baseline-tables first: paper-figs reads its json)
ALL_ORDER = ["baseline-tables", "paper-figs", "tier1", "norms", "content", "umap"]


def main():
    ap = argparse.ArgumentParser(
        description="Consolidated DPE analysis pipeline (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=list(COMMANDS) + ["all"],
                    help="subcommand to run ('all' runs every step in dependency order)")
    args = ap.parse_args()

    if args.command == "all":
        for name in ALL_ORDER:
            print(f"\n=== [{name}] ===")
            COMMANDS[name]()
    else:
        COMMANDS[args.command]()


if __name__ == "__main__":
    main()
