"""Does our assembled Tewhey corpus agree with Gosai's published measurements?

762,160 of Gosai's 798,064 sequences also appear in the Tewhey/Siraj ENCODE deposition, and
three cell lines are common to both (K562, HepG2, SK-N-SH). That overlap is a free external
check on everything the corpus build does: the element-reference join that gave us sequence
at all, the inverse-variance combination across library batches, and the cell-column
assignment.

The two depositions are different experiments, so perfect agreement is not expected and
would itself be suspicious. What IS expected is that agreement approaches the MEASUREMENT
CEILING -- sqrt(1 - E[SE^2]/Var) computed from both sides' standard errors. Reporting a bare
correlation would leave us unable to tell "the join is wrong" from "these are two noisy
measurements of the same thing".

Three specific failure modes this would catch:
  * a cell-column swap -- the off-diagonal would beat the diagonal;
  * a bad sequence join -- r collapses to ~0 while the ceiling stays high;
  * a scale or sign error from the log2FC combination -- slope far from 1, or negative r.

    python3 -I validate_corpus_vs_gosai.py --corpus outputs/tewhey/corpus_chrom.npz \
        --gosai data/k562/DATA-Table_S2__MPRA_dataset.txt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

CELLS = ["K562", "HepG2", "SK-N-SH"]
GCOL = {"K562": "K562", "HepG2": "HepG2", "SK-N-SH": "SKNSH"}


def ceiling(v: np.ndarray, se: np.ndarray) -> float:
    """sqrt(1 - E[SE^2]/Var): the correlation two perfect replicas of this assay would hit."""
    var = float(np.var(v))
    noise = float(np.mean(np.square(se)))
    return float(np.sqrt(max(0.0, 1.0 - noise / var))) if var > 0 else float("nan")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gosai", required=True)
    ap.add_argument("--out", default="outputs/tewhey/validation_vs_gosai.json")
    a = ap.parse_args()

    z = np.load(a.corpus, allow_pickle=True)
    have = [str(c) for c in z["cells"]]
    seqs = np.array([str(s).upper() for s in z["sequences"]], dtype=object)
    y, se, mask = z["y"], z["se"], z["mask"]
    print(f"corpus: {len(seqs):,} oligos, cells {have}", flush=True)

    g = pd.read_csv(a.gosai, sep="\t", low_memory=False)
    gseq = g["sequence"].astype(str).str.upper().to_numpy()
    print(f"gosai:  {len(gseq):,} sequences", flush=True)

    pos = {s: i for i, s in enumerate(seqs)}
    gi, ci = [], []
    for k, s in enumerate(gseq):
        j = pos.get(s)
        if j is not None:
            gi.append(k)
            ci.append(j)
    gi, ci = np.array(gi), np.array(ci)
    print(f"joined by sequence: {len(gi):,} rows\n", flush=True)
    if len(gi) < 1000:
        raise SystemExit("join produced almost nothing -- the sequence sets do not match")

    report: dict = {"n_joined": int(len(gi)), "cells": {}, "offdiagonal": {}}

    print("=== per cell: ours (Tewhey/ENCODE) vs Gosai Table S2, same sequences")
    print("%-9s %9s %8s %8s %8s %8s %8s" % ("cell", "n", "r", "ceiling", "gap", "slope", "d_mean"))
    for cell in CELLS:
        j = have.index(cell)
        ours = y[ci, j]
        ourse = se[ci, j]
        ok = mask[ci, j]
        theirs = g[f"{GCOL[cell]}_log2FC"].to_numpy(np.float64)[gi]
        theirse = g[f"{GCOL[cell]}_lfcSE"].to_numpy(np.float64)[gi]
        m = ok & np.isfinite(ours) & np.isfinite(theirs) & np.isfinite(theirse)
        if m.sum() < 50:
            print("%-9s %9s  no overlap" % (cell, int(m.sum())))
            continue
        u, t = ours[m], theirs[m]
        r = pearson(u, t)
        # The ceiling for a comparison of two noisy measurements uses BOTH error terms.
        var = float(np.var(np.r_[u, t]))
        noise = float(np.mean(np.square(ourse[m])) + np.mean(np.square(theirse[m]))) / 2
        c = float(np.sqrt(max(0.0, 1.0 - noise / var))) if var > 0 else float("nan")
        slope = float(np.polyfit(t, u, 1)[0])
        print(
            "%-9s %9d %8.4f %8.4f %8.4f %8.3f %8.3f"
            % (cell, int(m.sum()), r, c, c - r, slope, float(np.mean(u - t)))
        )
        report["cells"][cell] = {
            "n": int(m.sum()),
            "r": round(r, 4),
            "ceiling": round(c, 4),
            "gap": round(c - r, 4),
            "slope": round(slope, 4),
            "mean_difference": round(float(np.mean(u - t)), 4),
            "our_ceiling": round(ceiling(u, ourse[m]), 4),
            "their_ceiling": round(ceiling(t, theirse[m]), 4),
        }

    print("\n=== off-diagonal: our cell X vs their cell Y (the diagonal must win)")
    print("%-9s %-9s %8s" % ("ours", "theirs", "r"))
    best_wrong = []
    for c1 in CELLS:
        j = have.index(c1)
        ours = y[ci, j]
        ok = mask[ci, j]
        row = {}
        for c2 in CELLS:
            theirs = g[f"{GCOL[c2]}_log2FC"].to_numpy(np.float64)[gi]
            m = ok & np.isfinite(ours) & np.isfinite(theirs)
            r = pearson(ours[m], theirs[m])
            row[c2] = round(r, 4)
            print("%-9s %-9s %8.4f%s" % (c1, c2, r, "   <-- diagonal" if c1 == c2 else ""))
        report["offdiagonal"][c1] = row
        if row and max(row, key=row.get) != c1:
            best_wrong.append((c1, max(row, key=row.get)))

    print()
    if best_wrong:
        print("FAIL: a cell agrees better with the WRONG cell: %s" % best_wrong)
        report["verdict"] = "cell columns look swapped"
    else:
        print("PASS: every cell agrees best with its own counterpart in Gosai.")
        report["verdict"] = "ok"

    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
