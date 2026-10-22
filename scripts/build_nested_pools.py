"""Build NESTED canonical pools (replicate r0, generation seed 42) for every strategy that has labelled increments.

Why: the legacy per-size files in outputs/grid_pools/k562_cumulative are independent draws at each size (not prefixes of
one another), and for genomic the sizes even differ in label distribution (mean 0.459 / 0.462 / 0.418 / 0.497 vs 0.468 for
the full set). A scaling comparison on those files varies the data draw as well as the amount of data.

The labelled increments (inc_10k, seg_10k_30k, seg_30k_100k, seg_100k_300k) are consecutive slices of one 1M pool, so
their concatenations are nested by construction; this script verifies the slice claim against the raw pool before trusting it,
then writes outputs/grid_pools/k562_nested/{strategy}_d{D}.npz. Genomic (a bounded set of real CREs) is a seed-42 permutation
of the full labelled set instead.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GP = REPO / "outputs/grid_pools"
SEGS = [("inc_10k_seed42", 0, 10_000), ("seg_10k_30k_seed42", 10_000, 30_000), ("seg_30k_100k_seed42", 30_000, 100_000), ("seg_100k_300k_seed42", 100_000, 300_000)]
SIZES = [10_000, 30_000, 100_000, 300_000]


def save(path, seqs, labels, oracle_id, strategy, seed):
    tmp = str(path).replace(".npz", f".tmp{os.getpid()}.npz")
    np.savez(tmp, sequences=np.asarray(seqs, dtype=object), oracle_labels=np.asarray(labels, np.float32), oracle_id=oracle_id, strategy=strategy, seed=np.int64(seed))
    os.replace(tmp, path)


def main() -> None:
    out = GP / "k562_nested"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    strategies = sorted(p.stem for p in (GP / "k562_labelled/seg_100k_300k_seed42").glob("*.npz"))
    for st in strategies:
        raw = GP / f"k562/{st}/pool_1M_seed42.npz"
        parts = []
        for name, lo, hi in SEGS:
            f = GP / f"k562_labelled/{name}/{st}.npz"
            if not f.exists():
                parts = None
                break
            z = np.load(f, allow_pickle=True)
            parts.append((lo, hi, z["sequences"], z["oracle_labels"].astype(np.float32), str(z["oracle_id"])))
        if parts is None or not raw.exists():
            manifest[st] = {"nested": False, "reason": "missing increments or raw pool"}
            print(f"{st:26s} SKIP (increments or raw pool missing)", flush=True)
            continue
        rz = np.load(raw, allow_pickle=True)
        rs = rz["sequences"] if "sequences" in rz.files else rz[rz.files[0]]
        ok = all(len(s) == hi - lo and (s[:200] == rs[lo:lo + 200]).all() and (s[-200:] == rs[hi - 200:hi]).all()
                 and (s[::997] == rs[lo:hi:997]).all() for lo, hi, s, _, _ in parts)
        if not ok or any(p[4] != "oracle_v3" for p in parts):
            manifest[st] = {"nested": False, "reason": "increments are not consecutive slices of the raw 1M pool, or not oracle_v3"}
            print(f"{st:26s} SKIP (slice check failed)", flush=True)
            continue
        seqs = np.concatenate([p[2] for p in parts])
        lab = np.concatenate([p[3] for p in parts])
        assert len(set(seqs.tolist())) == len(seqs) and np.isfinite(lab).all()
        for D in SIZES:
            save(out / f"{st}_d{D}.npz", seqs[:D], lab[:D], "oracle_v3", st, 42)
        means = [round(float(lab[:D].mean()), 3) for D in SIZES]
        manifest[st] = {"nested": True, "source": "consecutive slices of pool_1M_seed42", "label_mean_by_D": means}
        print(f"{st:26s} nested OK  label mean by D {means}", flush=True)
    gz = np.load(GP / "k562_labelled/genomic_full/genomic.npz", allow_pickle=True)
    perm = np.random.default_rng(42).permutation(len(gz["sequences"]))[:300_000]
    gs, gl = gz["sequences"][perm], gz["oracle_labels"][perm].astype(np.float32)
    for D in SIZES:
        save(out / f"genomic_d{D}.npz", gs[:D], gl[:D], "oracle_v3", "genomic", 42)
    manifest["genomic"] = {"nested": True, "source": "seed-42 permutation of the full labelled CRE set", "label_mean_by_D": [round(float(gl[:D].mean()), 3) for D in SIZES]}
    print("genomic nested OK", manifest["genomic"]["label_mean_by_D"])
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", out / "MANIFEST.json")


if __name__ == "__main__":
    sys.exit(main())
