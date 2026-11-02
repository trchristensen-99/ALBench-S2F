#!/usr/bin/env python
"""Assemble a validation set of chosen composition from the rows of V0, V1 and V2 (oracle_v3 labels, similarity-guarded), in the format `--val_cache` takes.

Why this exists: `analyze_val_compositions.py` found that any broad mixture of reservoir families fits the stack about equally well (+0.002 mean over the 18 test sets, +0.005 on OOD against
genomic V0c), and one member per family is as good as three. B1 is that composition: 7 reservoir-family groups, one member each, equal counts. The rows already carry their guard
(no shared 12-mers with any training pool or evaluation set), so no new sequence is generated.

    python scripts/val_variants/build_val_composition.py --name B1 [--per_group 2400]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VV = Path("outputs/val_variants")
COMPOSITIONS = {
    "B1": {
        "genomic": "genomic_val",
        "perturb": "prm_0.05",
        "evoaug": "evoaug_paper2023",
        "random": "random_gc",
        "motif": "motif_ct_enriched",
        "zoonomia": "zoonomia_orthologs_valchr",
        "oracle": "ise_oracle_maximize",
    }
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="B1", choices=sorted(COMPOSITIONS))
    ap.add_argument(
        "--per_group",
        type=int,
        default=2400,
        help="default = the smallest member pool (motif_ct_enriched)",
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    seqs, y, fam, sub, parent, sim, src_set = [], [], [], [], [], [], []
    for v in ("V0", "V1", "V2"):
        z = np.load(VV / f"val_{v}.npz", allow_pickle=True)
        seqs += [str(s) for s in z["sequences"]]
        y.append(z["oracle_labels"].astype(np.float32))
        fam += [str(f) for f in z["family"]]
        sub += [str(s) for s in z["subfamily"]]
        parent.append(z["parent_v0_idx"])
        sim.append(z["max_sim"])
        src_set += [v] * len(z["oracle_labels"])
        oracle_id = str(z["oracle_id"])
    y, parent, sim = np.concatenate(y), np.concatenate(parent), np.concatenate(sim)
    fam = np.array(fam)
    keep, seen = [], set()
    for group, member in COMPOSITIONS[a.name].items():
        pool = [i for i in np.where(fam == member)[0] if seqs[i] not in seen]
        take = rng.choice(pool, min(a.per_group, len(pool)), replace=False)
        seen |= {seqs[i] for i in take}
        keep += list(take)
        print(f"{group:9s} {member:28s} pool {len(pool):6d} -> {len(take)}")
    keep = np.array(sorted(keep))
    prov = json.dumps(
        {
            "name": a.name,
            "rule": "equal counts from one member of each of 7 reservoir-family groups, rows of V0/V1/V2 (outputs/val_variants)",
            "members": COMPOSITIONS[a.name],
            "per_group": a.per_group,
            "seed": a.seed,
        }
    )
    out = VV / f"val_{a.name}.npz"
    np.savez(
        out,
        sequences=np.array([seqs[i] for i in keep], dtype=object),
        oracle_labels=y[keep],
        oracle_id=np.array(oracle_id),
        family=np.array(fam[keep]),
        subfamily=np.array([sub[i] for i in keep], dtype=object),
        parent_v0_idx=parent[keep],
        max_sim=sim[keep],
        provenance=np.array(prov),
    )
    print("wrote", out, len(keep), "sequences")


if __name__ == "__main__":
    main()
