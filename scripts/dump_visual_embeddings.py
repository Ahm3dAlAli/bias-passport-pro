#!/usr/bin/env python3
"""
Dump the VLM's visual-token embedding (mean-pooled at the DPE injection point)
per image, so we can UMAP the true embedding space BEFORE vs AFTER the DPE
correction. Run this ON ROLF (needs the model + GPU), in the model's conda env.

The DPE hook adds a region-constant vector enc = alpha*eps_g to every visual
token at `find_vision_module`'s output; therefore the mean-pooled "after"
embedding = mean-pooled "before" + enc. We capture "before" with a lightweight
hook (no injection) and add enc analytically -- one forward pass per image.

Reuses run_dpe_benchmark's loaders/build_client for exact parity.

Usage (per model, in its env):
  # idefics2  (env: fingerprint)
  python3 scripts/dump_visual_embeddings.py \
     --model HuggingFaceM4/idefics2-8b \
     --baseline-db results/single_runs_35k/gpu0_HuggingFaceM4_idefics2_8b_20260427_114159.db \
     --dataset-path /local/scratch/alali/fhibe_data/fhibe.20250716.u.gT5_rFTA_fullres \
     --alpha 0.25 --gpu 0 --4bit --per-group 450 \
     --out results/dpe_embeddings/idefics2_emb.npz
  # internvl2 (env: internvl)  --alpha 0.5  (no --4bit)
  # llava     (env: internvl)  --alpha 0.5  --4bit
"""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--baseline-db", required=True)
ap.add_argument("--dataset-path", required=True)
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--gpu", default="0")
ap.add_argument("--4bit", dest="fourbit", action="store_true")
ap.add_argument("--per-group", type=int, default=450, help="images/region (balance_by=region)")
ap.add_argument("--device-map", default="auto",
                help="'auto' (default) or 'cuda' to force the whole model onto the "
                     "single visible GPU (device_map={'':0}, no CPU offload)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

# pin the GPU BEFORE torch is imported (mirrors run_dpe_benchmark --gpu handling)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("PYTHONNOUSERSITE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch
import run_dpe_benchmark as R
from fingerprint_squared.debiasing.demographic_positional_encoder import DemographicPositionalEncoder
from fingerprint_squared.debiasing.dpe_hook import find_vision_module

os.makedirs(os.path.dirname(args.out), exist_ok=True)

# 1. encoder (region axis) -> gives enc = alpha*eps_g per region at the injection dim
enc_model = DemographicPositionalEncoder.from_sqlite(
    [args.baseline_db], correction_axis="region", alpha=args.alpha)

# 2. balanced image records (region-balanced, same loader as the eval)
records = R.load_balanced_image_records(
    args.baseline_db, per_group=args.per_group, exclude_genders=["unknown", "non-binary"],
    balance_by="region")
dataset_path = R.Path(args.dataset_path)
image_index = R.build_image_index(dataset_path)
print(f"{len(records)} balanced image records")

# 3. load model exactly like the eval (linspace-on-cpu patch during build)
fhibe = R._load_fhibe_module()
_orig = torch.linspace
torch.linspace = lambda *a, **k: (_orig(*a, **{**k, "device": k.get("device", "cpu")}))
# build_client's per-model classes special-case a "cuda:N" string (they convert it
# to {"":N} for 4-bit / force the whole model onto that GPU). CUDA_VISIBLE_DEVICES
# was pinned to --gpu, so the only visible device is cuda:0.
dev_map = "cuda:0" if args.device_map == "cuda" else "auto"
try:
    client = fhibe.build_client(args.model, device_map=dev_map, load_in_4bit=args.fourbit)
finally:
    torch.linspace = _orig
model = client.model
R._ensure_generation_mixin(model)
# Hedge: cap generation so that even if a client swallows _Stop, each image is
# still one vision forward + ~1 LM token (fast) rather than a full 256-token decode.
for gc_owner in (model, getattr(model, "language_model", None)):
    gc = getattr(gc_owner, "generation_config", None)
    if gc is not None:
        try: gc.max_new_tokens = 1; gc.min_new_tokens = 1
        except Exception: pass
# Strongest hedge: wrap generate() so EVERY call emits exactly 1 token, overriding
# whatever max_new_tokens the client passes. The vision forward (which fires the
# capture hook) still runs during prefill; only the LM decode is truncated.
def _cap_generate(gen):
    def wrapped(*a, **k):
        k["max_new_tokens"] = 1
        k.pop("min_new_tokens", None)
        return gen(*a, **k)
    return wrapped
for owner in (model, getattr(model, "language_model", None)):
    if owner is not None and hasattr(owner, "generate"):
        try: owner.generate = _cap_generate(owner.generate)
        except Exception: pass
vision_module, vpath = find_vision_module(model)
print(f"capture hook on model.{vpath}")

# 4. capture hook: mean-pool the visual tokens at the injection point (no injection).
# After capturing, raise to abort the forward BEFORE the expensive LM decode.
cap = {}
class _Stop(BaseException):   # BaseException so client `except Exception` can't swallow it
    pass
def _pool(output):
    t = output
    if hasattr(t, "last_hidden_state"): t = t.last_hidden_state
    if isinstance(t, (tuple, list)): t = t[0]
    if not (isinstance(t, torch.Tensor) and t.is_floating_point() and t.ndim >= 2):
        return None
    return t.reshape(-1, t.shape[-1]).float().mean(0).detach().cpu().numpy()
def hook(m, i, o):
    v = _pool(o)
    if v is not None:
        cap["v"] = v
        raise _Stop()          # image already encoded; skip LM generation
handle = vision_module.register_forward_hook(hook)

load_pil = fhibe.load_pil_image
PROMPT = next(iter(R.PROBES.values()))   # any single probe; vision tokens are image-only

# 5. loop: one forward per image, capture "before" pooled embedding
before, regions, ids = [], [], []
for i, rec in enumerate(records):
    iid = rec["image_id"]; region = rec.get("jurisdiction_region") or "unknown"
    p = R.resolve_image_path(iid, image_index, dataset_path)
    if p is None: continue
    try:
        img = load_pil(str(p))
    except Exception as e:
        print(f"  skip load {iid[:12]}: {type(e).__name__}"); continue
    cap.pop("v", None)
    try:
        with torch.no_grad():
            client.generate(img, PROMPT)   # hook raises _Stop after vision encoding
    except _Stop:
        pass                               # expected: aborted right after vision encoding
    except Exception:
        pass                               # client-wrapped error -> use captured value if any
    if "v" not in cap: continue            # hook didn't fire this image
    before.append(cap["v"]); regions.append(region); ids.append(iid)
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(records)}] captured={len(before)} dim={before[-1].shape[0]}")
handle.remove()

before = np.stack(before)                     # [N, D]
regions = np.array(regions); ids = np.array(ids)
D = before.shape[1]
# 6. enc per region at this dim; after = before + enc (uniform per-token add -> mean shifts by enc)
REGS = ["Africa","Asia","Europe","Americas","Northern America","Oceania"]
encs = {}
for r in REGS:
    e = enc_model.get_embedding(gender="*", region=r, embedding_dim=D, alpha=args.alpha)
    encs[r] = (e.detach().cpu().numpy() if hasattr(e, "detach") else np.asarray(e))
after = np.stack([before[k] + encs[regions[k]] for k in range(len(before))])

np.savez_compressed(args.out, before=before, after=after, region=regions, image_id=ids,
                    dim=D, alpha=args.alpha,
                    enc_matrix=np.stack([encs[r] for r in REGS]), enc_regions=np.array(REGS))
print(f"\nSaved {len(before)} embeddings (dim={D}) -> {args.out}")
print("per-region counts:", {r: int((regions == r).sum()) for r in REGS})
