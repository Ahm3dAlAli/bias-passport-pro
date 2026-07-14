#!/usr/bin/env python3
"""
DPE vs Baseline Comparison

Loads a baseline result DB and a DPE result DB, then produces:
  1. Per-demographic-group bias score comparison (valence, stereotype, confidence)
  2. Per-probe bias comparison
  3. Statistical significance tests (Cohen's d, Welch t-test, % reduction)
  4. Publication-ready figures saved to results/dpe_comparison/

Usage
-----
python scripts/compare_dpe_baseline.py \\
    --baseline-db results/single_runs_35k/gpu7_llava_hf_llava_v1.6_vicuna_7b_hf_*.db \\
    --dpe-db results/dpe/llava_dpe.db \\
    --out-dir results/dpe_comparison/llava

# Compare all alpha levels:
python scripts/compare_dpe_baseline.py \\
    --baseline-db results/single_runs_35k/gpu7_*.db \\
    --dpe-db results/dpe/llava_dpe_alpha1.db results/dpe/llava_dpe_alpha2.db \\
    --out-dir results/dpe_comparison/llava_alpha_sweep
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRICS = ["valence", "stereotype_alignment", "confidence"]
METRIC_LABELS = {
    "valence": "Valence",
    "stereotype_alignment": "Stereotype Alignment",
    "confidence": "Confidence",
}
PROBES = ["P1_occupation", "P2_education", "P3_trustworthiness", "P4_lifestyle", "P5_neighbourhood"]
PROBE_SHORT = {
    "P1_occupation": "Occupation",
    "P2_education": "Education",
    "P3_trustworthiness": "Trustworthiness",
    "P4_lifestyle": "Lifestyle",
    "P5_neighbourhood": "Neighbourhood",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(db_path: str) -> List[dict]:
    """Load all judge_scores rows from a result database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT image_id, valence, stereotype_alignment, confidence, refusal,
               economic_valence, gender_presentation, jurisdiction_region,
               probe_id
        FROM judge_scores
        WHERE valence IS NOT NULL
          AND stereotype_alignment IS NOT NULL
          AND confidence IS NOT NULL
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def group_by(rows: List[dict], keys: List[str]) -> Dict[str, List[dict]]:
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        key = "|".join(str(r.get(k) or "unknown") for k in keys)
        buckets[key].append(r)
    return dict(buckets)


def mean_scores(rows: List[dict]) -> Dict[str, float]:
    out = {}
    for m in METRICS:
        vals = [r[m] for r in rows if r.get(m) is not None]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    out["n"] = len(rows)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def welch_t_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Welch t-test p-value."""
    from scipy import stats as scipy_stats
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(p)


def bias_disparity(rows: List[dict], demographic_key: str) -> float:
    """
    Demographic disparity: std of per-group means across all groups for a
    given demographic dimension.  Lower = more equitable.
    """
    groups = group_by(rows, [demographic_key])
    group_means = []
    for g_rows in groups.values():
        vals = [r["valence"] for r in g_rows if r.get("valence") is not None]
        if len(vals) >= 5:
            group_means.append(np.mean(vals))
    if len(group_means) < 2:
        return float("nan")
    return float(np.std(group_means))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })
    return plt


def plot_gender_comparison(
    baseline_rows: List[dict],
    dpe_rows: List[dict],
    out_path: Path,
    metric: str = "valence",
):
    plt = _setup_matplotlib()
    import matplotlib.pyplot as mpl_plt

    genders = ["female", "male", "non-binary"]
    baseline_by_gender = group_by(baseline_rows, ["gender_presentation"])
    dpe_by_gender = group_by(dpe_rows, ["gender_presentation"])

    fig, axes = mpl_plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    for ax, gender in zip(axes, genders):
        b_rows = [r for k, v in baseline_by_gender.items() if gender in k for r in v]
        d_rows = [r for k, v in dpe_by_gender.items() if gender in k for r in v]

        b_vals = np.array([r[metric] for r in b_rows if r.get(metric) is not None])
        d_vals = np.array([r[metric] for r in d_rows if r.get(metric) is not None])

        x = np.arange(2)
        means = [np.mean(b_vals) if len(b_vals) else 0.0,
                 np.mean(d_vals) if len(d_vals) else 0.0]
        sems = [np.std(b_vals) / np.sqrt(max(1, len(b_vals))),
                np.std(d_vals) / np.sqrt(max(1, len(d_vals)))]

        colors = ["#e74c3c", "#2ecc71"]
        bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.8, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "DPE"], fontsize=9)
        ax.set_title(gender.capitalize(), fontsize=10, fontweight="bold")
        ax.set_ylabel(METRIC_LABELS[metric] if ax == axes[0] else "")

        # Annotate % change
        if means[0] != 0:
            pct = (means[1] - means[0]) / abs(means[0]) * 100
            sign = "+" if pct > 0 else ""
            ax.text(0.5, max(means) * 1.05, f"{sign}{pct:.1f}%", ha="center",
                    fontsize=8, color="#2980b9", fontweight="bold",
                    transform=ax.get_xaxis_transform())

    fig.suptitle(f"DPE vs Baseline — {METRIC_LABELS[metric]} by Gender", fontsize=12, fontweight="bold")
    mpl_plt.tight_layout()
    mpl_plt.savefig(out_path, bbox_inches="tight")
    mpl_plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_region_heatmap(
    baseline_rows: List[dict],
    dpe_rows: List[dict],
    out_path: Path,
):
    """
    Heatmap of % bias reduction (valence disparity) per region × probe.
    """
    plt = _setup_matplotlib()
    import matplotlib.pyplot as mpl_plt
    import matplotlib.colors as mcolors

    regions = sorted({r.get("jurisdiction_region") or "unknown" for r in baseline_rows + dpe_rows})
    probes = [p for p in PROBES if any(r.get("probe_id") == p for r in baseline_rows)]

    data = np.full((len(regions), len(probes)), np.nan)

    for ri, region in enumerate(regions):
        for pi, probe in enumerate(probes):
            b = [r["valence"] for r in baseline_rows
                 if r.get("jurisdiction_region") == region
                 and r.get("probe_id") == probe
                 and r.get("valence") is not None]
            d = [r["valence"] for r in dpe_rows
                 if r.get("jurisdiction_region") == region
                 and r.get("probe_id") == probe
                 and r.get("valence") is not None]
            if len(b) >= 5 and len(d) >= 5:
                data[ri, pi] = float(np.mean(d)) - float(np.mean(b))

    fig, ax = mpl_plt.subplots(figsize=(10, max(4, len(regions) * 0.6)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([PROBE_SHORT.get(p, p) for p in probes], rotation=30, ha="right")
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions)
    ax.set_title("Valence Change (DPE − Baseline) per Region × Probe\n"
                 "Green = more positive (corrected), Red = more negative", fontsize=11)

    # Add value annotations
    for ri in range(len(regions)):
        for pi in range(len(probes)):
            val = data[ri, pi]
            if not np.isnan(val):
                ax.text(pi, ri, f"{val:+.3f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < 0.15 else "white")

    mpl_plt.colorbar(im, ax=ax, label="ΔMEAN Valence (DPE − Baseline)")
    mpl_plt.tight_layout()
    mpl_plt.savefig(out_path, bbox_inches="tight")
    mpl_plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_bias_disparity_reduction(
    baseline_rows: List[dict],
    dpe_rows: List[dict],
    out_path: Path,
):
    """Bar chart of demographic disparity (std of group means) before/after DPE."""
    plt = _setup_matplotlib()
    import matplotlib.pyplot as mpl_plt

    demo_dims = ["gender_presentation", "jurisdiction_region"]
    dim_labels = {"gender_presentation": "Gender", "jurisdiction_region": "Region"}

    n_dims = len(demo_dims)
    fig, axes = mpl_plt.subplots(1, n_dims, figsize=(5 * n_dims, 4))
    if n_dims == 1:
        axes = [axes]

    for ax, dim in zip(axes, demo_dims):
        b_disp = bias_disparity(baseline_rows, dim)
        d_disp = bias_disparity(dpe_rows, dim)

        colors = ["#e74c3c", "#2ecc71"]
        x = [0, 1]
        bars = ax.bar(x, [b_disp, d_disp], color=colors, alpha=0.85, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "DPE"], fontsize=10)
        ax.set_ylabel("Valence Disparity (σ of group means)", fontsize=9)
        ax.set_title(f"{dim_labels[dim]} Bias Disparity", fontsize=10, fontweight="bold")

        # Annotate reduction
        if b_disp and not np.isnan(b_disp) and b_disp > 0:
            reduction = (b_disp - d_disp) / b_disp * 100
            ax.text(0.5, max(b_disp, d_disp) * 1.05,
                    f"{'↓' if reduction > 0 else '↑'} {abs(reduction):.1f}% disparity",
                    ha="center", fontsize=9, color="#2980b9", fontweight="bold",
                    transform=ax.transData)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    fig.suptitle("Demographic Bias Disparity: Baseline vs DPE", fontsize=12, fontweight="bold")
    mpl_plt.tight_layout()
    mpl_plt.savefig(out_path, bbox_inches="tight")
    mpl_plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_probe_comparison(
    baseline_rows: List[dict],
    dpe_rows: List[dict],
    out_path: Path,
):
    """Grouped bar chart: all 3 metrics × all 5 probes, baseline vs DPE."""
    plt = _setup_matplotlib()
    import matplotlib.pyplot as mpl_plt

    probes = [p for p in PROBES if any(r.get("probe_id") == p for r in baseline_rows)]
    n_probes = len(probes)

    fig, axes = mpl_plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))

    for ax, metric in zip(axes, METRICS):
        b_means, d_means = [], []
        for probe in probes:
            b = [r[metric] for r in baseline_rows
                 if r.get("probe_id") == probe and r.get(metric) is not None]
            d = [r[metric] for r in dpe_rows
                 if r.get("probe_id") == probe and r.get(metric) is not None]
            b_means.append(np.mean(b) if b else 0.0)
            d_means.append(np.mean(d) if d else 0.0)

        x = np.arange(n_probes)
        width = 0.35
        ax.bar(x - width / 2, b_means, width, label="Baseline", color="#e74c3c", alpha=0.8)
        ax.bar(x + width / 2, d_means, width, label="DPE", color="#2ecc71", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([PROBE_SHORT.get(p, p) for p in probes], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Probe-Level Metrics: Baseline vs DPE", fontsize=12, fontweight="bold")
    mpl_plt.tight_layout()
    mpl_plt.savefig(out_path, bbox_inches="tight")
    mpl_plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_and_save_summary(
    baseline_rows: List[dict],
    dpe_rows: List[dict],
    out_path: Path,
):
    """Print a statistical summary table and save as JSON."""
    has_scipy = True
    try:
        from scipy import stats as _
    except ImportError:
        has_scipy = False

    results = {
        "n_baseline": len(baseline_rows),
        "n_dpe": len(dpe_rows),
        "overall": {},
        "by_gender": {},
        "by_region": {},
        "by_probe": {},
        "disparity": {},
    }

    print("\n" + "=" * 70)
    print("SUMMARY: Baseline vs DPE")
    print("=" * 70)
    print(f"  Baseline samples: {len(baseline_rows):,}")
    print(f"  DPE samples:      {len(dpe_rows):,}")

    # Overall
    print("\n--- Overall Metric Means ---")
    print(f"{'Metric':<25} {'Baseline':>10} {'DPE':>10} {'Δ':>10} {'Cohen d':>10}")
    print("-" * 70)
    for metric in METRICS:
        b_vals = np.array([r[metric] for r in baseline_rows if r.get(metric) is not None])
        d_vals = np.array([r[metric] for r in dpe_rows if r.get(metric) is not None])
        b_mean = float(np.mean(b_vals)) if len(b_vals) else float("nan")
        d_mean = float(np.mean(d_vals)) if len(d_vals) else float("nan")
        delta = d_mean - b_mean
        cd = cohens_d(b_vals, d_vals) if (len(b_vals) > 1 and len(d_vals) > 1) else float("nan")
        pval = welch_t_pvalue(b_vals, d_vals) if has_scipy and len(b_vals) > 1 and len(d_vals) > 1 else float("nan")
        print(f"  {METRIC_LABELS[metric]:<23} {b_mean:>10.4f} {d_mean:>10.4f} {delta:>+10.4f} {cd:>10.4f}")
        results["overall"][metric] = {
            "baseline_mean": b_mean, "dpe_mean": d_mean,
            "delta": delta, "cohens_d": cd, "pvalue": pval,
        }

    # By gender
    print("\n--- Valence by Gender ---")
    print(f"{'Group':<30} {'Baseline':>10} {'DPE':>10} {'Δ':>10} {'%Δ':>8}")
    print("-" * 65)
    b_by_g = group_by(baseline_rows, ["gender_presentation"])
    d_by_g = group_by(dpe_rows, ["gender_presentation"])
    all_keys = sorted(set(b_by_g) | set(d_by_g))
    for key in all_keys:
        b_v = np.array([r["valence"] for r in b_by_g.get(key, []) if r.get("valence") is not None])
        d_v = np.array([r["valence"] for r in d_by_g.get(key, []) if r.get("valence") is not None])
        b_m = float(np.mean(b_v)) if len(b_v) else float("nan")
        d_m = float(np.mean(d_v)) if len(d_v) else float("nan")
        delta = d_m - b_m
        pct = (delta / abs(b_m) * 100) if b_m and not np.isnan(b_m) else float("nan")
        n_b, n_d = len(b_v), len(d_v)
        print(f"  {key:<28} {b_m:>10.4f} {d_m:>10.4f} {delta:>+10.4f} {pct:>+7.1f}%")
        results["by_gender"][key] = {
            "baseline_mean_valence": b_m, "dpe_mean_valence": d_m,
            "delta": delta, "pct_change": pct, "n_baseline": n_b, "n_dpe": n_d,
        }

    # By region
    print("\n--- Valence by Region ---")
    print(f"{'Region':<30} {'Baseline':>10} {'DPE':>10} {'Δ':>10} {'%Δ':>8}")
    print("-" * 65)
    b_by_r = group_by(baseline_rows, ["jurisdiction_region"])
    d_by_r = group_by(dpe_rows, ["jurisdiction_region"])
    for key in sorted(set(b_by_r) | set(d_by_r)):
        b_v = np.array([r["valence"] for r in b_by_r.get(key, []) if r.get("valence") is not None])
        d_v = np.array([r["valence"] for r in d_by_r.get(key, []) if r.get("valence") is not None])
        b_m = float(np.mean(b_v)) if len(b_v) else float("nan")
        d_m = float(np.mean(d_v)) if len(d_v) else float("nan")
        delta = d_m - b_m
        pct = (delta / abs(b_m) * 100) if b_m and not np.isnan(b_m) else float("nan")
        print(f"  {key:<28} {b_m:>10.4f} {d_m:>10.4f} {delta:>+10.4f} {pct:>+7.1f}%")
        results["by_region"][key] = {
            "baseline_mean_valence": b_m, "dpe_mean_valence": d_m,
            "delta": delta, "pct_change": pct,
        }

    # Disparity
    print("\n--- Demographic Disparity (σ of group means, valence) ---")
    for dim in ["gender_presentation", "jurisdiction_region"]:
        b_disp = bias_disparity(baseline_rows, dim)
        d_disp = bias_disparity(dpe_rows, dim)
        pct_red = (b_disp - d_disp) / b_disp * 100 if b_disp > 0 and not np.isnan(b_disp) else float("nan")
        print(f"  {dim:<30} Baseline={b_disp:.4f}  DPE={d_disp:.4f}  "
              f"Reduction={pct_red:+.1f}%")
        results["disparity"][dim] = {
            "baseline": b_disp, "dpe": d_disp, "pct_reduction": pct_red,
        }

    print("\n" + "=" * 70)

    # Save JSON
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: None if np.isnan(x) else x)
    print(f"\nSummary JSON saved to: {out_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare DPE vs baseline bias evaluation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline-db", required=True, nargs="+",
        help="Path(s) to baseline result DB(s). If multiple, rows are merged.",
    )
    parser.add_argument(
        "--dpe-db", required=True, nargs="+",
        help="Path(s) to DPE result DB(s). If multiple, rows are merged.",
    )
    parser.add_argument(
        "--out-dir", default="results/dpe_comparison",
        help="Directory to save figures and summary JSON.",
    )
    parser.add_argument(
        "--metric", default="valence", choices=METRICS,
        help="Primary metric for gender/region comparison figures.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline from: {args.baseline_db}")
    baseline_rows = []
    for db in args.baseline_db:
        baseline_rows.extend(load_scores(db))
    print(f"  → {len(baseline_rows):,} rows")

    print(f"Loading DPE results from: {args.dpe_db}")
    dpe_rows = []
    for db in args.dpe_db:
        dpe_rows.extend(load_scores(db))
    print(f"  → {len(dpe_rows):,} rows")

    if not baseline_rows:
        print("ERROR: No baseline scores found.")
        sys.exit(1)
    if not dpe_rows:
        print("ERROR: No DPE scores found.")
        sys.exit(1)

    # Restrict the baseline to the SAME (image_id, probe) pairs the DPE run
    # covered, so a subset/balanced DPE run is compared apples-to-apples against
    # the baseline on identical images (not the full 35k).
    dpe_keys = {(r.get("image_id"), r.get("probe_id")) for r in dpe_rows}
    n_before = len(baseline_rows)
    baseline_rows = [r for r in baseline_rows
                     if (r.get("image_id"), r.get("probe_id")) in dpe_keys]
    print(f"  Matched baseline to DPE image/probe set: "
          f"{n_before:,} → {len(baseline_rows):,} rows")
    if not baseline_rows:
        print("ERROR: no baseline rows overlap the DPE image set "
              "(image_id mismatch between DBs?).")
        sys.exit(1)

    # Generate figures
    print("\nGenerating figures...")
    plot_gender_comparison(
        baseline_rows, dpe_rows,
        out_dir / f"gender_comparison_{args.metric}.pdf",
        metric=args.metric,
    )
    plot_region_heatmap(
        baseline_rows, dpe_rows,
        out_dir / "region_probe_heatmap.pdf",
    )
    plot_bias_disparity_reduction(
        baseline_rows, dpe_rows,
        out_dir / "disparity_reduction.pdf",
    )
    plot_probe_comparison(
        baseline_rows, dpe_rows,
        out_dir / "probe_metric_comparison.pdf",
    )

    # Summary report
    print_and_save_summary(
        baseline_rows, dpe_rows,
        out_dir / "dpe_comparison_summary.json",
    )


if __name__ == "__main__":
    main()
