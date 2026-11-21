"""Attach a chromosome to every row of the 7-cell Tewhey corpus.

Without this the corpus cannot be trained on at all. chr7 and chr13 are test and chr19,
chr21 and chrX are validation across this whole study, and a corpus with no coordinate
cannot honour that -- it would quietly put held-out sequence in the training set, which is
exactly the failure that `MultiCellData.split()` now raises on.

Three sources, in order of directness:
  1. the oligo ID, which for the variant library is ``chr:pos:ref:alt:allele:window``;
  2. the ``chr`` column of the quantification tables, for IDs that do not parse;
  3. the Gosai library, joined by SEQUENCE, which covers the genomic oligos the two
     depositions share.
Anything still unplaced is a designed (non-genomic) oligo and is labelled ``designed``;
those carry no chromosome to hold out and are an eval set, never a training split.

    python3 -I add_corpus_chrom.py --corpus outputs/tewhey/corpus.npz \
        --quant data/external/tewhey_encode --gosai data/k562/DATA-Table_S2__MPRA_dataset.txt
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import re

import numpy as np
import pandas as pd

ID_RE = re.compile(r"^(chr)?([0-9]{1,2}|[XYxy]|MT?):(\d+):")
VALID = {"chr%d" % i for i in range(1, 23)} | {"chrX", "chrY", "chrM"}


def norm(c: str) -> str:
    c = str(c).strip()
    if not c:
        return ""
    if not c.lower().startswith("chr"):
        c = "chr" + c
    c = "chr" + c[3:].upper() if c[3:].lower() in ("x", "y", "m", "mt") else c
    return c if c in VALID else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--gosai", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    z = np.load(a.corpus, allow_pickle=True)
    ids = np.array([str(x) for x in z["oligo_id"]], dtype=object)
    seqs = np.array([str(x) for x in z["sequences"]], dtype=object)
    n = len(ids)
    chrom = np.full(n, "", dtype=object)
    src = np.full(n, "", dtype=object)

    # 1. straight from the ID
    for i, v in enumerate(ids):
        m = ID_RE.match(v)
        if m:
            c = norm(m.group(2))
            if c:
                chrom[i], src[i] = c, "id"
    print("from oligo ID:      %7d / %d" % ((src == "id").sum(), n), flush=True)

    # 2. the chr column of the quantification tables
    todo = {v: i for i, v in enumerate(ids) if not chrom[i]}
    if todo:
        for p in sorted(glob.glob(os.path.join(a.quant, "*.tsv"))):
            cols = pd.read_csv(p, sep="\t", nrows=0).columns
            if "ID" not in cols or "chr" not in cols:
                continue
            d = pd.read_csv(p, sep="\t", usecols=["ID", "chr"], low_memory=False)
            for v, c in zip(d["ID"].astype(str), d["chr"].astype(str), strict=True):
                i = todo.get(v)
                if i is not None and not chrom[i]:
                    cc = norm(c)
                    if cc:
                        chrom[i], src[i] = cc, "quant"
            if not any(not chrom[i] for i in todo.values()):
                break
    print("from quant tables: +%7d" % (src == "quant").sum(), flush=True)

    # 3. the Gosai library, by sequence
    g = pd.read_csv(a.gosai, sep="\t", usecols=["sequence", "chr"], low_memory=False)
    gmap = {}
    for s, c in zip(g["sequence"].astype(str).str.upper(), g["chr"].astype(str), strict=True):
        cc = norm(c)
        if cc:
            gmap[s] = cc
    for i in range(n):
        if not chrom[i]:
            c = gmap.get(seqs[i].upper())
            if c:
                chrom[i], src[i] = c, "gosai"
    print("from Gosai seq:    +%7d" % (src == "gosai").sum(), flush=True)

    unplaced = ~np.array([bool(c) for c in chrom])
    chrom[unplaced] = "designed"
    src[unplaced] = "none"
    print("unplaced -> designed: %7d (%.1f%%)" % (unplaced.sum(), 100 * unplaced.mean()))

    cnt = collections.Counter(chrom.tolist())
    print("\ntop chromosomes:", cnt.most_common(6))
    te = int(sum(cnt[c] for c in ("chr7", "chr13")))
    va = int(sum(cnt[c] for c in ("chr19", "chr21", "chrX")))
    placed = n - int(unplaced.sum())
    print(
        "holdout: test(chr7,13)=%d  val(chr19,21,X)=%d  train=%d  (of %d placed)"
        % (te, va, placed - te - va, placed)
    )
    if not (te and va):
        raise SystemExit("holdout is empty -- refusing to write a corpus that cannot hold out")

    out = a.out or a.corpus.replace(".npz", "_chrom.npz")
    payload = {k: z[k] for k in z.files}
    payload["chrom"] = chrom
    payload["chrom_source"] = src
    np.savez(out, **payload)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
