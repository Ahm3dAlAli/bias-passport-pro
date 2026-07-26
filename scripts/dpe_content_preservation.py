#!/usr/bin/env python3
"""
Content preservation / task fidelity under DPE (Tier-2 #5), plus a bonus rebuttal
of the precision/hardware confound (#7).

For each model, match baseline vs DPE responses on (image_id, probe_id) and report:
  - % responses byte-identical to baseline (DPE left them untouched)
  - mean semantic similarity (SBERT cosine) over the responses that DID change
  - mean character-level similarity (difflib) as a model-agnostic complement
High identical-% + high similarity => DPE preserves content and only nudges valence
on a minority of responses. The high identical-% also shows the DPE run reproduces
the archived baseline verbatim where the correction does not bite, so the
with/without-DPE difference is the correction, not precision/hardware drift.

Local only -- no rolf.  Writes figures/dpe_aaai/dpe_content_preservation.tex
"""
import sqlite3, numpy as np, difflib
ROOT = "/Users/ahmeda./Desktop/FingerPrint"
FINAL = f"{ROOT}/results/dpe_final_20260720_125918"
MODELS = {
 "IDEFICS2-8B": (f"{ROOT}/results/single_runs_35k/gpu0_HuggingFaceM4_idefics2_8b_20260427_114159.db",
                 f"{FINAL}/idefics2/idefics2_dpe.db"),
 "InternVL2-2B":(f"{ROOT}/results/single_runs_35k/gpu6_OpenGVLab_InternVL2_2B_20260421_145205.db",
                 f"{FINAL}/internvl2/internvl2_dpe.db"),
 "LLaVA-v1.6-7B":(f"{ROOT}/results/single_runs_35k/gpu7_llava_hf_llava_v1.6_vicuna_7b_hf_20260421_145210.db",
                  f"{FINAL}/llava/llava_dpe.db"),
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
for m, (bdb, ddb) in MODELS.items():
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
def f1(x): return "---" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.1f}"
def f3(x): return "---" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.3f}"
L = [r"\begin{table}[t]\centering\small",
     r"\caption{\textbf{Content preservation under DPE.} Fraction of responses byte-identical to the "
     r"baseline, and semantic (SBERT cosine) / character (difflib) similarity on the responses that "
     r"changed. DPE leaves most responses untouched and changes the rest only slightly, preserving task "
     r"fidelity; the high identical fraction also shows the DPE run reproduces the archived baseline "
     r"verbatim where the correction does not act, so the with/without-DPE gap is the correction rather "
     r"than precision or hardware drift.}", r"\label{tab:dpe_content}",
     r"\begin{tabular}{lcccc}", r"\toprule",
     r"Model & \% identical & \# changed & SBERT cos (changed) & char sim (changed) \\", r"\midrule"]
for m in MODELS:
    d = RES[m]
    L.append(f"{m} & {f1(d['pct_ident'])}\\% & {d['n_changed']} & {f3(d['cos_changed'])} & {f3(d['charsim'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(f"{ROOT}/figures/dpe_aaai/dpe_content_preservation.tex", "w").write("\n".join(L) + "\n")
print("\nWrote figures/dpe_aaai/dpe_content_preservation.tex")
