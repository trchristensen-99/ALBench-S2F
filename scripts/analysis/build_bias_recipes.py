"""Build source-restricted DIVERSE recipes for the bias retraining test, from SLOPE cells
(5 reservoirs, available at D=10k/30k/100k). For held-out X=dinuc_shuffle:
  without_X: diverse configs sourced from {genomic, motif, evoaug}
  with_X:    diverse configs sourced from {genomic, motif, dinuc}
Stratified by block_class + optimizer so the candidate pool is architecturally diverse
(NOT the motif-collapsed free menu). BIAS_D env picks the D (default 30000).
"""

import os, json, glob
import numpy as np
from collections import Counter

D = int(os.environ.get("BIAS_D", "30000"))
X = os.environ.get("BIAS_X", "dinuc_shuffle")
# with_X sources include X; without_X swaps X for a control (a different 3rd reservoir), so the
# two differ only in whether X was in the candidate source. Both are later retrained on X's data.
BASE = ["genomic", "motif_planted_v2"]
POOL3 = ["dinuc_shuffle", "evoaug_heavy", "gc_matched"]
control = next(r for r in POOL3 if r != X)
SETS = {"without": BASE + [control], "with": BASE + [X]}
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
RECIPE_DIR = "/grid/wsbs/home_norepl/christen/ALBench-S2F/configs/deploy_recipes"


def load_pool(reservoirs):
    pool = []
    for r in reservoirs:
        for mp in glob.glob(f"outputs/overnight/slope_{r}_d{D}_s*_*/r*_meta.json"):
            try:
                d = json.load(open(mp))
            except Exception:
                continue
            vp = d.get("val_pearson")
            if vp is None or not np.isfinite(vp) or "hp" not in d:
                continue
            pool.append((float(vp), d["hp"]))
    return pool


def stratified(pool, n_per_block=3):
    cands, seen = [], set()

    def add(hp):
        k = (hp.get("block_class"), hp.get("optimizer"), hp.get("n_layers"), hp.get("width_base"))
        if k in seen:
            return False
        # emit the FULL hp dict (deploy_train filters to HPConfig keys); a filtered subset drops
        # required fields like 'seed' -> "missing required HP fields: ['seed']".
        seen.add(k)
        cands.append(dict(hp))
        return True

    for bc in ["eff", "ag", "plain"]:
        sub = sorted([p for p in pool if p[1].get("block_class") == bc], key=lambda x: -x[0])
        n = 0
        for vp, hp in sub:
            if add(hp):
                n += 1
            if n >= n_per_block:
                break
    for opt in ["adam", "adamw", "muon"]:
        if any(c.get("optimizer") == opt for c in cands):
            continue
        sub = sorted([p for p in pool if p[1].get("optimizer") == opt], key=lambda x: -x[0])
        if sub:
            add(sub[0][1])
    return cands


for name, res in SETS.items():
    pool = load_pool(res)
    rec = stratified(pool)
    out = f"{RECIPE_DIR}/bias_{name}_{X}_d{D}.json"
    json.dump(rec, open(out, "w"), indent=1)
    print(f"{name}: {len(rec)} configs from {res} (pool={len(pool)}) -> {out}", flush=True)
    print(
        f"  blocks={dict(Counter(c.get('block_class') for c in rec))} "
        f"opt={dict(Counter(c.get('optimizer') for c in rec))}",
        flush=True,
    )
print("DONE bias recipes", flush=True)
