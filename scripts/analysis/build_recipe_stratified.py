"""Rebuild the candidate recipe with ARCHITECTURAL DIVERSITY (not just top-by-val, which
collapses to eff/muon). Stratify by block_class and ensure optimizer coverage + depth/width
spread + AutoResearch novelty, so the empirical greedy can decide if diversity helps.
"""

import json, glob
import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BAKE = f"{REPO}/outputs/hp_step1_bakeoff_e100"
SRC_RES = [
    "genomic",
    "motif_planted_v2",
    "evoaug_heavy",
    "dinuc_shuffle",
    "gc_matched",
    "phylogenetic_zoonomia",
    "uncertainty_guided",
    "diversity_guided",
]
CFG_KEYS = [
    "lr",
    "batch_size",
    "conv_dropout",
    "dense_dropout",
    "n_layers",
    "width_base",
    "width_jitter",
    "block_class",
    "ks",
    "pct_start",
    "optimizer",
    "weight_decay",
    "use_shift_aug",
    "shift_max",
    "use_evoaug",
    "activation",
    "loss",
]

# gather all valid (val_pearson, hp) across reservoirs + AutoResearch
pool = []
srcs = [f"{BAKE}/k562_{r}_d30000/seed42_0" for r in SRC_RES]
srcs.append(f"{REPO}/outputs/autoresearch_compare/k562_genomic_d30000/divt")
srcs.append(f"{REPO}/outputs/autoresearch_compare/k562_genomic_d30000/free")
for src in srcs:
    for mp in glob.glob(f"{src}/*/r*_meta.json") + glob.glob(f"{src}/r*_meta.json"):
        try:
            d = json.load(open(mp))
        except Exception:
            continue
        vp = d.get("val_pearson")
        if vp is None or not np.isfinite(vp) or "hp" not in d:
            continue
        pool.append((float(vp), d["hp"], "auto" if "autoresearch" in src else "bake"))
print(f"pool: {len(pool)} valid models", flush=True)


def keep(hp):
    return {k: hp[k] for k in CFG_KEYS if k in hp}


def dedup_key(hp):
    return (hp.get("block_class"), hp.get("optimizer"), hp.get("n_layers"), hp.get("width_base"))


cands, seen = [], set()


def add(hp):
    k = dedup_key(hp)
    if k in seen:
        return False
    seen.add(k)
    cands.append(keep(hp))
    return True


# 1) per block_class: top-3 by val (depth/width diversity via dedup)
for bc in ["eff", "ag", "plain"]:
    sub = sorted([p for p in pool if p[1].get("block_class") == bc], key=lambda x: -x[0])
    n = 0
    for vp, hp, _ in sub:
        if add(hp):
            n += 1
        if n >= 3:
            break
    print(f"  block {bc}: added {n}", flush=True)

# 2) ensure each optimizer represented (best of each if missing)
for opt in ["adam", "adamw", "muon"]:
    if any(c.get("optimizer") == opt for c in cands):
        continue
    sub = sorted([p for p in pool if p[1].get("optimizer") == opt], key=lambda x: -x[0])
    if sub:
        add(sub[0][1])
        print(f"  +optimizer {opt}", flush=True)

# 3) top-2 AutoResearch configs (may carry novel activation/loss axes)
auto = sorted([p for p in pool if p[2] == "auto"], key=lambda x: -x[0])
na = 0
for vp, hp, _ in auto:
    if add(hp):
        na += 1
    if na >= 2:
        break
print(f"  +autoresearch: {na}", flush=True)

json.dump(cands, open(f"{REPO}/configs/deploy_recipes/deploy_recipe_d30000.json", "w"), indent=1)
print(f"\nRECIPE: {len(cands)} configs", flush=True)
from collections import Counter

print("  block_class:", dict(Counter(c.get("block_class") for c in cands)), flush=True)
print("  optimizer:", dict(Counter(c.get("optimizer") for c in cands)), flush=True)
for c in cands:
    print(
        f"  {c.get('block_class')}/{c.get('optimizer')} L{c.get('n_layers')} "
        f"w{c.get('width_base')} lr{c.get('lr'):.1e} act={c.get('activation', '-')}",
        flush=True,
    )
print(f"WROTE deploy_recipe_d30000.json ({len(cands)} configs)", flush=True)
