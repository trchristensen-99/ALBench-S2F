"""Compose training corpora that mix reservoirs, for the mixture sweep (30k total, components taken as nested prefixes).

Each component contributes the first round(fraction * D) sequences of its replicate-1 (or canonical nested) pool, so a mixture at size D
is a subset of the same mixture at any larger size. Writes outputs/grid_pools/k562_mix/{name}_d30000.npz.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GP = REPO / "outputs/grid_pools"
D = 30_000
COMP = {"G": "k562_reps/genomic_r1_d30000.npz", "R": "k562_reps/random_r1_d30000.npz", "M": "k562_reps/motif_ct_enriched_r1_d30000.npz",
        "E": "k562_reps/evoaug_paper2023_r1_d30000.npz", "MS": "k562_nested/motif_shared_core_d30000.npz", "MY": "k562_nested/motif_syntax_core_d30000.npz",
        "ZO": "k562_nested/zoonomia_orthologs_d30000.npz", "MU": "k562_nested/mutagenesis_d30000.npz"}
MIXES = {
    # pairs, 50/50
    "GR": {"G": .5, "R": .5}, "GM": {"G": .5, "M": .5}, "GE": {"G": .5, "E": .5}, "RM": {"R": .5, "M": .5}, "RE": {"R": .5, "E": .5}, "ME": {"M": .5, "E": .5},
    # trios and the quad, equal
    "GRM": {"G": 1 / 3, "R": 1 / 3, "M": 1 / 3}, "GRE": {"G": 1 / 3, "R": 1 / 3, "E": 1 / 3}, "GME": {"G": 1 / 3, "M": 1 / 3, "E": 1 / 3}, "RME": {"R": 1 / 3, "M": 1 / 3, "E": 1 / 3},
    "GRME": {"G": .25, "R": .25, "M": .25, "E": .25},
    # uneven
    "G75M25": {"G": .75, "M": .25}, "G25M75": {"G": .25, "M": .75}, "G75R25": {"G": .75, "R": .25}, "G25R75": {"G": .25, "R": .75}, "G50M25R25": {"G": .5, "M": .25, "R": .25},
    # within-family and diverse
    "MMM": {"M": 1 / 3, "MS": 1 / 3, "MY": 1 / 3}, "MZ": {"M": .5, "ZO": .5}, "GMU": {"G": .5, "MU": .5}, "GMZU": {"G": .25, "M": .25, "ZO": .25, "MU": .25},
    "DIV8": {k: 1 / 8 for k in ("G", "R", "M", "E", "MS", "MY", "ZO", "MU")},
}


def main() -> None:
    out = GP / "k562_mix"
    out.mkdir(parents=True, exist_ok=True)
    pools = {}
    manifest = {}
    for name, spec in MIXES.items():
        seqs, labs, comp = [], [], {}
        for key, f in spec.items():
            if key not in pools:
                z = np.load(GP / COMP[key], allow_pickle=True)
                pools[key] = (z["sequences"], z["oracle_labels"].astype(np.float32))
            n = int(round(f * D))
            seqs.append(pools[key][0][:n])
            labs.append(pools[key][1][:n])
            comp[key] = n
        s, l = np.concatenate(seqs), np.concatenate(labs)
        if len(s) != D:  # rounding: top up from the first component
            k0 = next(iter(spec))
            extra = D - len(s)
            s = np.concatenate([s, pools[k0][0][comp[k0]:comp[k0] + extra]])
            l = np.concatenate([l, pools[k0][1][comp[k0]:comp[k0] + extra]])
            comp[k0] += extra
        uniq = len(set(s.tolist()))
        dst = out / f"{name}_d{D}.npz"
        tmp = str(dst).replace(".npz", f".tmp{os.getpid()}.npz")
        np.savez(tmp, sequences=np.asarray(s, dtype=object), oracle_labels=l, oracle_id="oracle_v3", strategy=f"mix:{name}", seed=np.int64(0))
        os.replace(tmp, dst)
        manifest[name] = dict(components=comp, n=len(s), unique=uniq, label_mean=float(l.mean()), label_sd=float(l.std()))
        print(f"{name:10s} n={len(s)} unique={uniq} label mean {l.mean():.3f} sd {l.std():.3f} {comp}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    sys.exit(main())
