#!/usr/bin/env python3
"""
Plot the DPE alpha-ablation curve from an ablation run directory.

Reads results/dpe_ablation_<TS>/abl_<model>/compare_alpha_<a>/dpe_comparison_summary.json
for every model and alpha, then produces:
  - dpe_alpha_curve.pdf : disparity reduction (%) vs alpha, one line per model,
                          with a coherence (%empty) twin axis so degeneration is
                          visible. Region reduction is the primary series.
  - dpe_alpha_curve.json: the extracted (model, alpha) -> metrics table.

Usage
-----
python scripts/plot_dpe_alpha_curve.py --ablation-dir results/dpe_ablation_20260715_171750
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

MODEL_DISPLAY = {
    "llava": "LLaVA-v1.6-7B",
    "idefics2": "IDEFICS2-8B",
    "internvl2": "InternVL2-2B",
}


def _pretty(tag: str) -> str:
    t = tag.replace("abl_", "").lower()
    for k, v in MODEL_DISPLAY.items():
        if k in t:
            return v
    return tag


def _alpha_from_dirname(name: str):
    # compare_alpha_0p75 -> 0.75
    m = re.search(r"alpha_([0-9p]+)$", name)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def _coherence_pct_empty(db: Path):
    if not db.exists():
        return None
    try:
        rows = sqlite3.connect(str(db)).execute(
            "SELECT response FROM probe_results WHERE response IS NOT NULL").fetchall()
    except Exception:
        return None
    if not rows:
        return None
    L = [len((r[0] or "").strip()) for r in rows]
    return 100.0 * sum(1 for x in L if x < 3) / len(L)


def collect(ablation_dir: Path):
    """Return {model_display: [ {alpha, region_red, gender_red, pct_empty}, ... ]}."""
    out = defaultdict(list)
    for model_dir in sorted(ablation_dir.glob("abl_*")):
        model = _pretty(model_dir.name)
        for cmp_dir in sorted(model_dir.glob("compare_alpha_*")):
            alpha = _alpha_from_dirname(cmp_dir.name)
            summ = cmp_dir / "dpe_comparison_summary.json"
            if alpha is None or not summ.exists():
                continue
            d = json.loads(summ.read_text()).get("disparity", {})
            region_red = d.get("jurisdiction_region", {}).get("pct_reduction")
            gender_red = d.get("gender_presentation", {}).get("pct_reduction")
            # coherence from the matching dpe db
            tag = cmp_dir.name.replace("compare_alpha_", "")
            pe = _coherence_pct_empty(model_dir / f"alpha_{tag}_dpe.db")
            out[model].append({
                "alpha": alpha,
                "region_red": region_red,
                "gender_red": gender_red,
                "pct_empty": pe,
            })
        out[model].sort(key=lambda r: r["alpha"])
    return dict(out)


def plot(data: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax2 = ax.twinx()

    colors = ["#2980b9", "#e67e22", "#27ae60", "#8e44ad"]
    for i, (model, rows) in enumerate(sorted(data.items())):
        c = colors[i % len(colors)]
        alphas = [r["alpha"] for r in rows]
        region = [r["region_red"] if isinstance(r["region_red"], (int, float)) else float("nan") for r in rows]
        empty = [r["pct_empty"] if isinstance(r["pct_empty"], (int, float)) else float("nan") for r in rows]
        ax.plot(alphas, region, "-o", color=c, label=f"{model} (region)")
        ax2.plot(alphas, empty, "--x", color=c, alpha=0.5)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel(r"Correction strength $\alpha$")
    ax.set_ylabel("Region disparity reduction (%)  ↑ better")
    ax2.set_ylabel("% empty responses  (dashed)  ↓ better", color="gray")
    ax.set_title("DPE alpha ablation: debiasing vs. output coherence")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    abl = Path(args.ablation_dir)
    out_dir = Path(args.out_dir) if args.out_dir else abl / "paper_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(abl)
    if not data:
        print(f"No compare_alpha_*/dpe_comparison_summary.json found under {abl}")
        sys.exit(1)

    # Text table
    print(f"\n{'model':<16} {'alpha':>6} {'region_red%':>12} {'gender_red%':>12} {'%empty':>8}")
    print("-" * 58)
    for model, rows in sorted(data.items()):
        for r in rows:
            f = lambda x, n=1: f"{x:.{n}f}" if isinstance(x, (int, float)) else "  --"
            print(f"{model:<16} {r['alpha']:>6} {f(r['region_red']):>12} "
                  f"{f(r['gender_red']):>12} {f(r['pct_empty']):>8}")

    # Best alpha per model (max region_red with %empty < 5)
    print("\nSuggested alpha per model (max region_red%, %empty < 5):")
    best = {}
    for model, rows in sorted(data.items()):
        cand = [r for r in rows
                if isinstance(r["region_red"], (int, float))
                and (not isinstance(r["pct_empty"], (int, float)) or r["pct_empty"] < 5)]
        if cand:
            b = max(cand, key=lambda r: r["region_red"])
            best[model] = b["alpha"]
            print(f"  {model:<16} -> alpha={b['alpha']}  (region_red={b['region_red']:.1f}%)")

    (out_dir / "dpe_alpha_curve.json").write_text(json.dumps(
        {"data": data, "suggested_alpha": best}, indent=2))
    plot(data, out_dir / "dpe_alpha_curve.pdf")
    print(f"\n✓ Wrote {out_dir}/dpe_alpha_curve.pdf and .json")


if __name__ == "__main__":
    main()
