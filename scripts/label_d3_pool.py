"""Label a collaborator pool with oracle_v3 and emit NESTED 10k/30k/100k subsets.

Nested (10k subset of 30k subset of 100k) rather than independent draws: a scaling
curve compares the SAME sequences plus more, so a difference between points is the
extra data, not a different sample. Only 100k sequences are ever scored.

Handles both collaborator provenance shapes: the D3 pools carry a single JSON
``provenance`` blob, the Evo2 pool carries separate ``prov_*`` scalar keys.
"""
from __future__ import annotations
import json, sys
import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)


def _read_provenance(z):
    """Collaborator pools stamp provenance differently: the D3 pools carry one JSON
    'provenance' blob, the Evo2 pool carries separate 'prov_*' scalar keys. Capture
    either shape so upstream metadata is never silently dropped from the cache."""
    if "provenance" in z.files:
        return json.loads(str(z["provenance"]))
    out = {}
    for k in z.files:
        if not k.startswith("prov_"):
            continue
        v = z[k]
        try:
            v = v.item()
        except (ValueError, AttributeError):
            v = v.tolist() if hasattr(v, "tolist") else str(v)
        out[k[len("prov_") :]] = v
    return out


src, arm, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
z = np.load(src, allow_pickle=True)
seqs = [str(s) for s in z["sequences"]]
prov = _read_provenance(z)
print(f"  source {arm}: {len(seqs):,} sequences", flush=True)

# Use the SAME canonical subsetting the collaborator uses
# (albench.core.pools.nested_subsets: permute the whole pool once with PCG64 and
# take sorted prefixes). seed 0 is his replicate 0, so our 10k/30k/100k are the
# IDENTICAL sequences to his draw and the numbers are directly comparable. Drawing
# our own 100k first would have been nested but a different sample.
from albench.core.pools import nested_subsets, verify_nesting

SIZES = [10_000, 30_000, 100_000]
subs = nested_subsets(len(seqs), SIZES, seed=seed)
verify_nesting(subs)
top = [seqs[i] for i in subs[100_000]]
pos = {s: np.searchsorted(subs[100_000], subs[s]) for s in SIZES}
print(f"  nested_subsets(seed={seed}) -> " + ", ".join(f"{s}:{len(subs[s])}" for s in SIZES), flush=True)

# Same loader build_pool.py uses, so these labels are byte-comparable with every
# other pool in the benchmark. _load_oracle lives in exp1_1_scaling, not in
# generate_labeled_pools.
from experiments.exp1_1_scaling import _load_oracle
from scripts.generate_labeled_pools import _label_sequences

# Canonical AG_S2 oracle = oracle_v3 (outputs/oracle_v3/s2), the dir _load_oracle
# resolves to by default. The old scripts.build_pool._pin_ag_s2_oracle helper was
# dropped in the repo consolidation (import was dead); pin the dir explicitly so the
# hardcoded oracle_id="oracle_v3" stamp cannot drift from the checkpoint that scores.
import os as _os

_os.environ.setdefault("AG_S2_ORACLE_DIR", f"{REPO}/outputs/oracle_v3/s2")
oracle = _load_oracle("k562", oracle_type="ag_s2")

labels = np.asarray(_label_sequences(oracle, top), dtype=np.float32).ravel()
assert len(labels) == len(top) and np.isfinite(labels).all(), "bad labels"
print(f"  labelled {len(labels):,}  mean={labels.mean():.3f} sd={labels.std():.3f}", flush=True)

for D in (100_000, 30_000, 10_000):
    out = f"{REPO}/outputs/grid_pools/k562_cumulative/{arm}_d{D}.npz"
    sel = pos[D]            # where this size's canonical indices sit inside the 100k
    np.savez_compressed(
        out,
        sequences=np.array([top[i] for i in sel], dtype=object),
        oracle_labels=labels[sel],
        oracle_id=np.array("oracle_v3"),
        strategy=np.array(arm),
        generation_seed=np.array(seed),
        params=np.array(json.dumps({"source": src, "subsetting": "albench.core.pools.nested_subsets", "replicate_seed": seed,
                                    "upstream_provenance": prov})),
    )
    print(f"  wrote {arm}_d{D}.npz  n={len(sel)} mean={labels[sel].mean():.3f}", flush=True)
