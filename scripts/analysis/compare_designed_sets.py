#!/usr/bin/env python
"""How different are the K562-designed and HepG2-designed sequence sets, and is each one actually active in the cell type it was designed for?

This matters for the two-head oracle: each designed set carries a measurement in its own cell type only, so if the two sets were near-duplicates we could pair them, and if they were not actually active in
their target cell type they would not be the high-activity anchor the designed axis is supposed to provide. Reports exact overlap, composition and 4-mer distance between the sets and against the measured
library, the design methods declared in each file, and each set's activity against its own cell type's library background.

    python scripts/analysis/compare_designed_sets.py      (cpuq)
"""

from __future__ import annotations

import collections
import itertools
import json
import os

import numpy as np
import pandas as pd

ROOT = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
FILES = {
    "K562": (f"{ROOT}/data/k562/test_sets/test_ood_designed_k562.tsv", "K562_log2FC"),
    "HepG2": (f"{ROOT}/data/hepg2/test_sets/test_ood_designed_hepg2.tsv", "HepG2_log2FC"),
}
K = 4


def kmer_vec(seqs: list, k: int = K) -> np.ndarray:
    idx = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=k))}
    v = np.zeros(len(idx))
    for s in seqs:
        for i in range(len(s) - k + 1):
            j = idx.get(s[i : i + k])
            if j is not None:
                v[j] += 1
    return v / max(v.sum(), 1)


def gc(seqs: list) -> float:
    tot = sum(len(s) for s in seqs)
    return sum(c in "GC" for s in seqs for c in s) / max(tot, 1)


def main() -> None:
    lib = pd.read_csv(
        f"{ROOT}/data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False
    )
    sets, res = {}, {}
    for cell, (path, col) in FILES.items():
        d = pd.read_csv(path, sep="\t")
        d = d[np.isfinite(d[col].astype(float))]
        sets[cell] = d
        print(f"\n== {cell}-designed: {len(d):,} sequences with a {col} measurement")
        if "method" in d:
            print("  methods:", dict(collections.Counter(d["method"]).most_common(6)))
        if "target_cell" in d:
            print("  target_cell:", dict(collections.Counter(d["target_cell"]).most_common()))
        y = d[col].astype(float).values
        bg = lib[col].astype(float).values
        bg = bg[np.isfinite(bg)]
        p90, p99 = np.percentile(bg, [90, 99])
        res[cell] = {
            "n": int(len(d)),
            "mean": float(y.mean()),
            "sd": float(y.std()),
            "library_mean": float(bg.mean()),
            "library_p90": float(p90),
            "frac_above_library_p90": float((y > p90).mean()),
            "frac_above_library_p99": float((y > p99).mean()),
            "gc": gc(d["sequence"].tolist()[:20000]),
        }
        print(
            f"  activity in {cell}: mean {y.mean():+.3f} (library {bg.mean():+.3f}), sd {y.std():.3f}; "
            f"above the library's 90th pct {(y > p90).mean() * 100:5.1f}%, 99th {(y > p99).mean() * 100:5.1f}%"
        )

    a, b = (set(sets[c]["sequence"]) for c in ("K562", "HepG2"))
    both = a & b
    print(
        f"\n== overlap: {len(both):,} sequences in both designed sets ({len(both) / min(len(a), len(b)) * 100:.2f}% of the smaller)"
    )
    res["overlap_n"] = len(both)
    if both:
        kk = sets["K562"].set_index("sequence")["K562_log2FC"].astype(float)
        hh = sets["HepG2"].set_index("sequence")["HepG2_log2FC"].astype(float)
        s = sorted(both)
        x, y = kk.loc[s].values, hh.loc[s].values
        print(
            f"  on those, r(K562, HepG2) = {np.corrcoef(x, y)[0, 1]:.3f}; mean difference {np.mean(x - y):+.3f}, sd {np.std(x - y):.3f}"
        )
        res["overlap_r"] = float(np.corrcoef(x, y)[0, 1])

    vk = kmer_vec(sets["K562"]["sequence"].tolist()[:20000])
    vh = kmer_vec(sets["HepG2"]["sequence"].tolist()[:20000])
    vl = kmer_vec(lib["sequence"].tolist()[:20000])
    d = lambda u, v: float(np.abs(u - v).sum() / 2)  # noqa: E731  total variation distance
    print(
        f"\n== composition: GC K562-designed {res['K562']['gc']:.3f}, HepG2-designed {res['HepG2']['gc']:.3f}, library {gc(lib['sequence'].tolist()[:20000]):.3f}"
    )
    print(
        f"  {K}-mer total-variation distance: the two designed sets {d(vk, vh):.4f}; K562-designed vs library {d(vk, vl):.4f}; HepG2-designed vs library {d(vh, vl):.4f}"
    )
    res["tv_designed_pair"] = d(vk, vh)
    res["tv_k562_vs_library"] = d(vk, vl)
    res["tv_hepg2_vs_library"] = d(vh, vl)
    os.makedirs(f"{ROOT}/outputs/justification_oct22", exist_ok=True)
    json.dump(
        res, open(f"{ROOT}/outputs/justification_oct22/data_designed_sets.json", "w"), indent=1
    )


if __name__ == "__main__":
    main()
