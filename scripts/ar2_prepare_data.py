"""Manifests for the ar2 searches on our pools: the validation split (val_loop = chr19 + chr21, val_select = chrX) and one training manifest per cell.

Why: ar2 (AutoResearch v2) needs two disjoint validation sets, the loop one that drives all feedback and a held-back one read once at the end. Our V0c validation set
(29,449 real chr19/21/X sequences relabelled by oracle_v3) carries no chromosome column, so the chromosome is recovered by exact sequence match against the Gosai table
(`chr` column). Sequences that cannot be matched stop the script: guessing a split would silently void the held-back set.

    python scripts/ar2_prepare_data.py --out outputs/ar2/data            # cpuq
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
POOLS = {  # cell -> pool file (replicate 1, the pools the earlier searches used)
    "genomic_10k": ("genomic", 10_000, R / "outputs/grid_pools/k562_reps/genomic_r1_d10000.npz"),
    "genomic_30k": ("genomic", 30_000, R / "outputs/grid_pools/k562_reps/genomic_r1_d30000.npz"),
    "genomic_100k": ("genomic", 100_000, R / "outputs/grid_pools/k562_reps/genomic_r1_d100000.npz"),
    "random_30k": ("random", 30_000, R / "outputs/grid_pools/k562_reps/random_r1_d30000.npz"),
    "motif_30k": (
        "motif_ct_enriched",
        30_000,
        R / "outputs/grid_pools/k562_reps/motif_ct_enriched_r1_d30000.npz",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "outputs/ar2/data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = np.load(R / "outputs/val_variants/val_V0.npz", allow_pickle=True)
    seqs = [str(s) for s in v["sequences"]]
    tab = pd.read_csv(
        R / "data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", usecols=["chr", "sequence"]
    )
    tab = tab[tab["chr"].astype(str).isin(["19", "21", "X"])]
    chrom = {}
    for sq, c in zip(tab["sequence"].astype(str), tab["chr"].astype(str)):
        chrom[sq] = c
        if (
            len(sq) < 200
        ):  # shorter table entries were padded with N to 200 nt in the validation set (right, left or centred)
            chrom.setdefault(sq.ljust(200, "N"), c)
            chrom.setdefault(sq.rjust(200, "N"), c)
            chrom.setdefault(sq.center(200, "N"), c)
    cs = np.array([chrom.get(s, "?") for s in seqs])
    n_missing = int((cs == "?").sum())
    print(
        "validation sequences:",
        len(seqs),
        "chromosome counts:",
        {c: int((cs == c).sum()) for c in np.unique(cs)},
    )
    if n_missing:
        raise SystemExit(
            f"{n_missing} validation sequences could not be matched to a chromosome; stopping"
        )
    loop_idx, sel_idx = np.where(cs != "X")[0], np.where(cs == "X")[0]
    assert len(loop_idx) and len(sel_idx) and not set(loop_idx) & set(sel_idx)
    np.save(out / "val_loop_idx.npy", loop_idx)
    np.save(out / "val_select_idx.npy", sel_idx)
    base = {
        "sha256": None,
        "label_source": "oracle_v3",
        "tasks": ["K562"],
        "seq_key": "sequences",
        "label_keys": {"K562": "oracle_labels"},
        "chrom_key": None,
    }
    val = str(R / "outputs/val_variants/val_V0.npz")
    json.dump(
        {
            **base,
            "path": val,
            "n": int(len(loop_idx)),
            "role": "val_loop",
            "pool_id": "V0c_chr19_21",
            "reservoir": "genomic_val",
            "generation_seed": 0,
            "D": 0,
            "index_file": str(out / "val_loop_idx.npy"),
        },
        open(out / "val_loop.json", "w"),
        indent=1,
    )
    json.dump(
        {
            **base,
            "path": val,
            "n": int(len(sel_idx)),
            "role": "val_select",
            "pool_id": "V0c_chrX",
            "reservoir": "genomic_val",
            "generation_seed": 0,
            "D": 0,
            "index_file": str(out / "val_select_idx.npy"),
        },
        open(out / "val_select.json", "w"),
        indent=1,
    )
    for cell, (res, D, f) in POOLS.items():
        z = np.load(f, allow_pickle=True)
        assert str(z["oracle_id"]) == "oracle_v3" and len(z["sequences"]) >= D
        json.dump(
            {
                **base,
                "path": str(f),
                "n": D,
                "role": "train",
                "pool_id": f"{res}_r1_d{D}",
                "reservoir": res,
                "generation_seed": 43,
                "D": D,
                "index_file": None,
            },
            open(out / f"train_{cell}.json", "w"),
            indent=1,
        )
    print("wrote manifests to", out, "val_loop", len(loop_idx), "val_select", len(sel_idx))


if __name__ == "__main__":
    main()
