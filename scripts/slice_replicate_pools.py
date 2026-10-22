"""Turn the raw replicate pools into per-size nested prefix files, and audit them.

Design (the same one the scaling curves use): every independent run gets its OWN labelled draw from the reservoir
(replicate r), and every dataset size inside that run is a NESTED prefix of the replicate's pool, so a size
comparison varies only the amount of data. Synthesised strategies get independent generation seeds, which gives
disjoint pools (measured, not assumed: the pairwise overlaps are written to the manifest). The genomic reservoir is a
bounded set of real CREs, so its replicates are different permutations of the same labelled set and share sequences
by construction; the manifest records the overlap so reports can say so.

    python scripts/slice_replicate_pools.py [--reps 1 2 3]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GP = REPO / "outputs/grid_pools"
SIZES = [10_000, 30_000, 100_000, 300_000]
SEED0 = {1: 43, 2: 44, 3: 45}
SYNTH = ("random", "motif_ct_enriched", "evoaug_paper2023")


def gc(s):
    return float(np.mean([(x.count("G") + x.count("C")) / max(1, len(x)) for x in s[:20000]]))


def cpg(s):
    return float(np.mean([x.count("CG") / max(1, len(x) - 1) for x in s[:20000]]))


def save(path, seqs, labels, oracle_id, strategy, seed):
    tmp = str(path).replace(".npz", f".tmp{os.getpid()}.npz")
    np.savez(tmp, sequences=np.asarray(seqs, dtype=object), oracle_labels=np.asarray(labels, np.float32), oracle_id=oracle_id, strategy=strategy, seed=np.int64(seed))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    out = GP / "k562_reps"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"sizes": SIZES, "strategies": {}}
    pools = {}
    for strat in SYNTH:
        for r in args.reps:
            f = glob.glob(str(GP / f"k562_reps_raw/{strat}__g{SEED0[r]}__*.npz"))
            if not f:
                print(f"  missing raw pool: {strat} rep{r}")
                continue
            z = np.load(f[0], allow_pickle=True)
            pools[(strat, r)] = (list(z["sequences"]), z["oracle_labels"].astype(np.float32), str(z["oracle_id"]), SEED0[r])
    gz = np.load(GP / "k562_labelled/genomic_full/genomic.npz", allow_pickle=True)
    gs, gl = list(gz["sequences"]), gz["oracle_labels"].astype(np.float32)
    for r in args.reps:
        perm = np.random.default_rng(SEED0[r]).permutation(len(gs))
        pools[("genomic", r)] = ([gs[i] for i in perm[:300_000]], gl[perm[:300_000]], str(gz["oracle_id"]) if "oracle_id" in gz.files else "oracle_v3", SEED0[r])
    for strat in ("genomic",) + SYNTH:
        info = {"replicates": {}, "overlap_300k": {}}
        for r in args.reps:
            if (strat, r) not in pools:
                continue
            seqs, lab, oid, seed = pools[(strat, r)]
            assert len(seqs) >= 300_000 and np.isfinite(lab).all() and oid == "oracle_v3", (strat, r, len(seqs), oid)
            n_unique = len(set(seqs[:300_000]))
            for D in SIZES:
                save(out / f"{strat}_r{r}_d{D}.npz", seqs[:D], lab[:D], oid, strat, seed)
            info["replicates"][f"r{r}"] = dict(n=len(seqs), unique_300k=n_unique, gc=gc(seqs), cpg=cpg(seqs), label_mean=float(lab.mean()), label_sd=float(lab.std()))
        rs = sorted(info["replicates"])
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = set(pools[(strat, int(rs[i][1:]))][0][:300_000]), set(pools[(strat, int(rs[j][1:]))][0][:300_000])
                info["overlap_300k"][f"{rs[i]}-{rs[j]}"] = len(a & b) / 300_000
        legacy = GP / f"k562_cumulative/{strat}_d100000.npz"
        if legacy.exists():
            lz = np.load(legacy, allow_pickle=True)
            info["legacy_100k"] = dict(gc=gc(list(lz["sequences"])), cpg=cpg(list(lz["sequences"])), label_mean=float(lz["oracle_labels"].mean()), label_sd=float(lz["oracle_labels"].std()))
        manifest["strategies"][strat] = info
        print(strat, json.dumps({k: v for k, v in info.items()}, default=float)[:900], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    print("wrote", out / "manifest.json")


if __name__ == "__main__":
    main()
