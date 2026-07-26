#!/usr/bin/env python3
"""
Report the DPE correction-vector norms per region per model, for the appendix.

Because the projection W is orthonormal (isometric), the norm of the injected
vector equals alpha * ||delta_g||, where delta_g = grand_mean - group_mean over
(valence, stereotype_alignment, confidence). This is exactly the `enc‖·‖` value
logged during generation. Non-trivial norms for LLaVA (comparable to InternVL2)
document that its ~0% disparity change is genuine architecture resistance, not a
failure to inject the correction.

Builds the encoder from each local baseline DB with correction_axis='region'
(the same configuration used in the final evaluation).
"""
import sys, importlib.util, numpy as np
# load the encoder module directly (bypass the package __init__, which pulls heavy deps)
_spec = importlib.util.spec_from_file_location(
    "dpe_mod",
    "/Users/ahmeda./Desktop/FingerPrint/fingerprint_squared/debiasing/demographic_positional_encoder.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["dpe_mod"] = _mod            # dataclass introspection needs this
_spec.loader.exec_module(_mod)
DemographicPositionalEncoder = _mod.DemographicPositionalEncoder

ROOT = "/Users/ahmeda./Desktop/FingerPrint"
REGIONS = ["Africa", "Asia", "Europe", "Americas", "Northern America", "Oceania"]
RSHORT = {"Africa": "Africa", "Asia": "Asia", "Europe": "Europe", "Americas": "Americas",
          "Northern America": "N. America", "Oceania": "Oceania"}
# model -> (baseline db, optimal alpha)
MODELS = {
    "IDEFICS2-8B":  (f"{ROOT}/results/single_runs_35k/gpu0_HuggingFaceM4_idefics2_8b_20260427_114159.db", 0.25),
    "InternVL2-2B": (f"{ROOT}/results/single_runs_35k/gpu6_OpenGVLab_InternVL2_2B_20260421_145205.db", 0.5),
    "LLaVA-v1.6-7B":(f"{ROOT}/results/single_runs_35k/gpu7_llava_hf_llava_v1.6_vicuna_7b_hf_20260421_145210.db", 0.5),
}

NORM = {}   # model -> {region: (raw ||delta||, injected alpha*||delta||)}
for m, (db, alpha) in MODELS.items():
    enc = DemographicPositionalEncoder.from_sqlite([db], correction_axis="region", alpha=alpha)
    NORM[m] = {}
    for r in REGIONS:
        vec = enc.get_correction_vector("*", r)
        raw = float(np.linalg.norm(vec)) if vec is not None else float("nan")
        NORM[m][r] = (raw, alpha * raw)

# console summary
print(f"{'region':<16}" + "".join(f"{m:>22}" for m in MODELS))
print(f"{'':16}" + "".join(f"{'raw / injected(αδ)':>22}" for _ in MODELS))
for r in REGIONS:
    row = f"{r:<16}"
    for m in MODELS:
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
    L.append(f"{RSHORT[r]} & " + " & ".join(f"{NORM[m][r][1]:.4f}" for m in MODELS) + r" \\")
# mean row
L.append(r"\midrule")
L.append("Mean & " + " & ".join(
    f"{np.mean([NORM[m][r][1] for r in REGIONS]):.4f}" for m in MODELS) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
out = f"{ROOT}/figures/dpe_aaai/dpe_correction_norms.tex"
open(out, "w").write("\n".join(L) + "\n")
print("\nWrote", out)
