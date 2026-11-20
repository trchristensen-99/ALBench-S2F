"""Build the 7-cell Tewhey training corpus: one row per sequence, one column per cell line.

The element-reference FASTAs supply the sequence for 96.5-97.7% of measured rows in every
cell line, so the whole Tewhey/Siraj deposition becomes trainable rather than merely
countable. This produces the wide matrix the conditioning trainer wants -- sequence once,
activity and its standard error per cell, plus a mask -- and answers the question that has
been open since the Siraj paper came up: how much of it is sequence Gosai already covers.

Rows for the same (sequence, cell) appear in several library batches and several trait
projects. They are combined by INVERSE-VARIANCE weighting rather than a plain mean, because
the batches differ a lot in depth and a plain mean would let a shallow batch drag a deep
one; the combined SE is the usual 1/sqrt(sum 1/se^2).

    python3 -I scratchpad/nov20/build_tewhey_corpus.py \
        --elements data/external/tewhey_elements --quant data/external/tewhey_encode \
        --gosai data/k562/DATA-Table_S2__MPRA_dataset.txt --out outputs/tewhey/corpus.npz
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

CELLS = ["K562", "HepG2", "SK-N-SH", "A549", "GM12878", "HCT116", "Jurkat"]


def _open(path: str):
    with open(path, "rb") as probe:
        gz = probe.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if gz else open(path)


def read_fasta(path: str):
    name, buf = None, []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line)
    if name is not None:
        yield name, "".join(buf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--gosai", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=200)
    a = ap.parse_args()

    seqs: dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(a.elements, "*.fa"))):
        for name, s in read_fasta(p):
            if len(s) >= a.min_len:
                seqs.setdefault(name, s[: a.min_len].upper())
    print(f"{len(seqs):,} design oligos at >= {a.min_len} bp", flush=True)

    # (sequence, cell) -> running inverse-variance sums
    num: dict[tuple[str, int], float] = collections.defaultdict(float)
    den: dict[tuple[str, int], float] = collections.defaultdict(float)
    cidx = {c: i for i, c in enumerate(CELLS)}
    projects: dict[str, str] = {}

    for p in sorted(glob.glob(os.path.join(a.quant, "*.tsv"))):
        cell = os.path.basename(p).split("__")[0]
        if cell not in cidx:
            continue
        cols = pd.read_csv(p, sep="\t", nrows=0).columns
        lfc = "log2FoldChange" if "log2FoldChange" in cols else None
        sec = "lfcSE" if "lfcSE" in cols else None
        if "ID" not in cols or lfc is None:
            continue
        use = [c for c in ("ID", lfc, sec, "project") if c]
        d = pd.read_csv(p, sep="\t", usecols=use, low_memory=False)
        d = d[d["ID"].astype(str).isin(seqs.keys())]
        v = d[lfc].to_numpy(np.float64)
        e = d[sec].to_numpy(np.float64) if sec else np.ones(len(d))
        ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
        ids = d["ID"].astype(str).to_numpy()
        proj = d["project"].astype(str).to_numpy() if "project" in d else None
        j = cidx[cell]
        w = 1.0 / np.square(e[ok])
        for k, oid in enumerate(ids[ok]):
            num[(oid, j)] += v[ok][k] * w[k]
            den[(oid, j)] += w[k]
        if proj is not None:
            for k, oid in enumerate(ids):
                projects.setdefault(oid, proj[k])
        print(f"  {os.path.basename(p)}: +{int(ok.sum()):,} ({cell})", flush=True)

    used = sorted({oid for oid, _ in num})
    pos = {o: i for i, o in enumerate(used)}
    n, c = len(used), len(CELLS)
    y = np.full((n, c), np.nan, np.float32)
    se = np.full((n, c), np.nan, np.float32)
    for (oid, j), s in den.items():
        i = pos[oid]
        y[i, j] = num[(oid, j)] / s
        se[i, j] = 1.0 / np.sqrt(s)

    mask = np.isfinite(y)
    seq_arr = np.array([seqs[o] for o in used], dtype=object)
    print(
        f"\n{n:,} oligos with at least one measurement; "
        f"{int(mask.sum()):,} (oligo, cell) observations",
        flush=True,
    )
    for j, cell in enumerate(CELLS):
        print(f"   {cell:9s} {int(mask[:, j].sum()):>9,}")
    ncells = mask.sum(1)
    print(
        "   measured in k cells:",
        {int(k): int(v) for k, v in zip(*np.unique(ncells, return_counts=True), strict=True)},
    )

    overlap = {}
    if a.gosai and os.path.exists(a.gosai):
        g = pd.read_csv(a.gosai, sep="\t", usecols=["sequence"], low_memory=False)
        gs = set(g["sequence"].astype(str).str.upper())
        inter = sum(1 for s in seq_arr if s in gs)
        overlap = {
            "gosai_sequences": len(gs),
            "tewhey_sequences": int(n),
            "shared": int(inter),
            "pct_of_tewhey_new": round(100 * (1 - inter / max(1, n)), 2),
        }
        print("\n=== overlap with the Gosai 798k library")
        print(json.dumps(overlap, indent=1))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(
        a.out,
        sequences=seq_arr,
        oligo_id=np.array(used, dtype=object),
        y=y,
        se=se,
        mask=mask,
        cells=np.array(CELLS, dtype=object),
        project=np.array([projects.get(o, "") for o in used], dtype=object),
    )
    with open(a.out.replace(".npz", "_summary.json"), "w") as fh:
        json.dump(
            {
                "cells": CELLS,
                "n_oligos": int(n),
                "per_cell": {c: int(mask[:, j].sum()) for j, c in enumerate(CELLS)},
                "overlap_with_gosai": overlap,
            },
            fh,
            indent=1,
        )
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
