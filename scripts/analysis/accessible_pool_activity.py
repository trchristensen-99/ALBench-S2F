#!/usr/bin/env python
"""How much of the genomic reservoir is open chromatin in K562 or HepG2, and how does that change the activity distribution?

The question from the Nov 4 meeting: most of the Gosai library is variant-centric, so most windows are closed in the assayed cell type and carry little regulatory grammar. If an accessible-only subset is much
more active, it is the better base for mutagenesis, EvoAug and motif strategies. This reports, for the whole library and for the genomic nested pool (the base of every genomic arm), the accessible share and the
activity distribution of accessible against closed windows, in oracle_v3 labels and in the measured log2FC of the same cell type, split by the library's own projects (UKBB and GTEX variants against CRE tiles).

Accessible = the 200-bp window (variant position +- 100, the library's own construction) overlaps a merged ENCODE DNase peak of that cell type (`data/encode_accessibility/{K562,HepG2}_peaks_merged.bed`).

    python scripts/analysis/accessible_pool_activity.py      (cpuq)
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

ROOT = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
TABLE = f"{ROOT}/data/k562/DATA-Table_S2__MPRA_dataset.txt"
PEAKS = {c: f"{ROOT}/data/encode_accessibility/{c}_peaks_merged.bed" for c in ("K562", "HepG2")}
POOL = f"{ROOT}/outputs/grid_pools/k562_nested/genomic_d300000.npz"
WIN = 100


def load_peaks(path: str) -> dict:
    """chrom -> (starts, ends), sorted and merged on disk."""
    by: dict = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith(("#", "track")):
                continue
            f = line.split()
            by.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    out = {}
    for c, iv in by.items():
        iv.sort()
        out[c] = (np.array([a for a, _ in iv]), np.array([b for _, b in iv]))
    return out


def overlaps(peaks: dict, chrom: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """True where [lo, hi) meets a peak. Peaks are disjoint, so the candidate is the last one starting at or before hi."""
    out = np.zeros(len(lo), bool)
    for c in np.unique(chrom):
        key = c if c.startswith("chr") else f"chr{c}"
        if key not in peaks:
            continue
        starts, ends = peaks[key]
        m = chrom == c
        j = np.searchsorted(starts, hi[m], side="right") - 1
        ok = j >= 0
        idx = np.where(m)[0][ok]
        out[idx] = ends[j[ok]] > lo[m][ok]
    return out


def describe(name: str, y: np.ndarray, acc: np.ndarray) -> dict:
    """Activity of accessible against closed windows."""
    a, b = y[acc], y[~acc]
    q = lambda v: [float(np.percentile(v, p)) for p in (10, 50, 90)] if len(v) else []  # noqa: E731
    d = {
        "n_accessible": int(acc.sum()),
        "frac_accessible": float(acc.mean()),
        "mean_accessible": float(a.mean()) if len(a) else None,
        "mean_closed": float(b.mean()) if len(b) else None,
        "sd_accessible": float(a.std()) if len(a) else None,
        "sd_closed": float(b.std()) if len(b) else None,
        "q10_50_90_accessible": q(a),
        "q10_50_90_closed": q(b),
    }
    if len(a) and len(b):
        d["cohens_d"] = float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
        d["frac_of_top_decile_that_is_accessible"] = float(acc[y >= np.percentile(y, 90)].mean())
    print(
        f"  {name:34s} n={len(y):7d} accessible {d['frac_accessible'] * 100:5.2f}%  "
        f"mean {d['mean_accessible'] if d['mean_accessible'] is None else round(d['mean_accessible'], 3)} vs {d['mean_closed'] if d['mean_closed'] is None else round(d['mean_closed'], 3)}"
        + (
            f"  d={d['cohens_d']:+.2f}  top-decile accessible {d['frac_of_top_decile_that_is_accessible'] * 100:.1f}%"
            if "cohens_d" in d
            else ""
        )
    )
    return d


def main() -> None:
    rows = list(csv.DictReader(open(TABLE), delimiter="\t"))
    seqs = np.array([r["sequence"] for r in rows])
    chrom = np.array([r["chr"] for r in rows])
    pos = np.array([int(r["IDs"].split(":")[1]) for r in rows])
    proj = np.array([r["data_project"] for r in rows])
    meas = {c: np.array([float(r[f"{c}_log2FC"] or "nan") for r in rows]) for c in PEAKS}
    lo, hi = pos - WIN, pos + WIN
    acc = {c: overlaps(load_peaks(p), chrom, lo, hi) for c, p in PEAKS.items()}
    res: dict = {"library": {}, "pool": {}}
    print(
        f"library: {len(rows)} sequences, projects {dict(zip(*np.unique(proj, return_counts=True)))}"
    )
    for c in PEAKS:
        print(f"\n== {c} accessibility, measured {c} log2FC (whole library)")
        res["library"][c] = {"all": describe("all", meas[c], acc[c])}
        for p in np.unique(proj):
            m = proj == p
            res["library"][c][p] = describe(f"project {p}", meas[c][m], acc[c][m])
    both = acc["K562"] & acc["HepG2"]
    k_only = acc["K562"] & ~acc["HepG2"]
    print(
        f"\nopen in both {both.mean() * 100:.2f}%, K562 only {k_only.mean() * 100:.2f}%, open in K562 at all {acc['K562'].mean() * 100:.2f}%"
    )
    res["library"]["open_both_frac"] = float(both.mean())
    res["library"]["k562_only_frac"] = float(k_only.mean())

    z = np.load(POOL, allow_pickle=True)
    ps = np.array([str(s) for s in z["sequences"]])
    y = np.asarray(z["oracle_labels"], float)
    idx = {s: i for i, s in enumerate(seqs)}
    hit = np.array([idx.get(s, -1) for s in ps])
    found = hit >= 0
    print(
        f"\n== genomic nested pool, {len(ps)} sequences, matched to the library {found.mean() * 100:.1f}%"
    )
    for c in PEAKS:
        a = acc[c][hit[found]]
        print(f"  {c} oracle labels:")
        res["pool"][c] = {"oracle": describe(f"{c} oracle", y[found], a)}
        mm = meas[c][hit[found]]
        ok = np.isfinite(mm)
        res["pool"][c]["measured"] = describe(f"{c} measured", mm[ok], a[ok])
    os.makedirs(f"{ROOT}/outputs/justification_oct22", exist_ok=True)
    json.dump(
        res, open(f"{ROOT}/outputs/justification_oct22/data_accessible_pool.json", "w"), indent=1
    )


if __name__ == "__main__":
    main()
