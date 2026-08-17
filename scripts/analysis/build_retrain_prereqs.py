"""Prereqs for the retraining-composition experiment:
  1. Extract a candidate RECIPE (~8-10 raw diverse configs) from multi-reservoir HP-opt +
     AutoResearch -> configs/deploy_recipes/deploy_recipe_d30000.json
  2. Build MIXED reservoir caches (equal-parts concatenation of oracle-labeled component caches)
     -> outputs/reservoir_cache/k562_mix{3,5}_d1000000_seed42.npz
Both are CPU-only. Mixes become first-class training compositions for deploy_train.py.
"""
import os, json, glob
import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BAKE = f"{REPO}/outputs/hp_step1_bakeoff_e100"
RC = f"{REPO}/outputs/reservoir_cache"
RECIPE_DIR = f"{REPO}/configs/deploy_recipes"
os.makedirs(RECIPE_DIR, exist_ok=True)

# ---- 1. candidate recipe: top configs across reservoirs + AutoResearch, deduped, diverse ----
SRC_RES = ["genomic", "motif_planted_v2", "evoaug_heavy", "dinuc_shuffle", "gc_matched"]
TOP_PER = 2
CFG_KEYS = ["lr", "batch_size", "conv_dropout", "dense_dropout", "n_layers", "width_base",
            "width_jitter", "block_class", "ks", "pct_start", "optimizer", "weight_decay",
            "use_shift_aug", "shift_max", "use_evoaug"]


def sig(hp):  # coarse de-dup key
    return (hp.get("block_class"), hp.get("optimizer"), hp.get("n_layers"),
            hp.get("width_base"), round(np.log10(max(hp.get("lr", 1e-3), 1e-9)), 1))


cands, seen = [], set()
sources = [f"{BAKE}/k562_{r}_d30000/seed42_0" for r in SRC_RES]
sources.append(f"{REPO}/outputs/autoresearch_compare/k562_genomic_d30000/divt")
for src in sources:
    metas = []
    for mp in glob.glob(f"{src}/*/r*_meta.json") + glob.glob(f"{src}/r*_meta.json"):
        try:
            d = json.load(open(mp))
        except Exception:
            continue
        vp = d.get("val_pearson")
        if vp is None or not np.isfinite(vp) or "hp" not in d:
            continue
        metas.append((vp, d["hp"]))
    metas.sort(key=lambda x: -x[0])
    added = 0
    for vp, hp in metas:
        s = sig(hp)
        if s in seen:
            continue
        seen.add(s)
        cands.append({k: hp[k] for k in CFG_KEYS if k in hp})
        added += 1
        if added >= TOP_PER:
            break
print(f"recipe: {len(cands)} diverse candidate configs", flush=True)
for c in cands:
    print(f"  {c.get('block_class')}/{c.get('optimizer')} L{c.get('n_layers')} "
          f"w{c.get('width_base')} lr{c.get('lr'):.1e}", flush=True)
json.dump(cands, open(f"{RECIPE_DIR}/deploy_recipe_d30000.json", "w"), indent=1)
print(f"WROTE {RECIPE_DIR}/deploy_recipe_d30000.json", flush=True)

# ---- 2. mixed caches ----
GENOMIC = f"{REPO}/outputs/chr_split_cache/chr_train_ref_only.npz"
COMP = {
    "genomic": GENOMIC,
    "motif_planted_v2": f"{RC}/k562_motif_planted_v2_d1000000_seed42.npz",
    "evoaug_heavy": f"{RC}/k562_evoaug_heavy_d1000000_seed42.npz",
    "dinuc_shuffle": f"{RC}/k562_dinuc_shuffle_d1000000_seed42.npz",
    "gc_matched": f"{RC}/k562_gc_matched_d1000000_seed42.npz",
}
MIXES = {"mix3": ["genomic", "motif_planted_v2", "evoaug_heavy"],
         "mix5": ["genomic", "motif_planted_v2", "evoaug_heavy", "dinuc_shuffle", "gc_matched"]}
TOTAL = 1_000_000


def load_comp(name):
    z = np.load(COMP[name], allow_pickle=True)
    seqs = z["sequences"]
    lab = z["oracle_labels"].astype(np.float32)
    oid = str(z["oracle_id"]) if "oracle_id" in z.files else "unknown"
    return seqs, lab, oid


for mix, comps in MIXES.items():
    k = len(comps)
    n_each = TOTAL // k
    S, L, oids = [], [], set()
    for name in comps:
        seqs, lab, oid = load_comp(name)
        oids.add(oid)
        take = min(n_each, len(seqs))
        S.append(np.asarray(seqs[:take], dtype=object))
        L.append(lab[:take])
        print(f"  {mix} <- {name}: {take} seqs (oid={oid})", flush=True)
    seqs = np.concatenate(S)
    labs = np.concatenate(L)
    oid = "full856k_clean" if oids <= {"full856k_clean", "unknown"} else "MIXED"
    out = f"{RC}/k562_{mix}_d1000000_seed42.npz"
    np.savez(out, sequences=seqs, oracle_labels=labs, oracle_id=oid)
    print(f"WROTE {out}  n={len(seqs)}  oid={oid}", flush=True)
print("DONE prereqs", flush=True)
