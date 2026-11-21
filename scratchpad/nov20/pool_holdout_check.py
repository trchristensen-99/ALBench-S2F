"""Do the genomic-derived pools actually exclude chr7/13/19/21/X?

The pool .npz carries sequence and oracle label but no coordinate, so the chromosome has
to come back from the library table by sequence identity. Only the reservoirs that draw
real genomic sequence can be checked this way; a synthetic reservoir has no chromosome to
breach, and says so rather than silently passing.
"""

import glob
import os

import numpy as np
import pandas as pd

TEST = {"chr7", "chr13"}
VAL = {"chr19", "chr21", "chrX"}
TBL = "data/k562/DATA-Table_S2__MPRA_dataset.txt"

d = pd.read_csv(TBL, sep="\t", usecols=["sequence", "chr"], low_memory=False)
chrom = {}
for s, c in zip(d["sequence"].astype(str).str.upper(), d["chr"].astype(str), strict=True):
    c = c.strip()
    chrom[s] = c if c.lower().startswith("chr") else "chr" + c
print("%d library sequences with a chromosome" % len(chrom))

root = "outputs/grid_pools/k562_nested"
for p in sorted(
    glob.glob(os.path.join(root, "*_d*.npz")),
    key=lambda q: (q.rsplit("_d", 1)[0], int(q.rsplit("_d", 1)[1][:-4])),
):
    z = np.load(p, allow_pickle=True)
    seq = [str(s).upper() for s in z["sequences"]]
    hit = [chrom[s] for s in seq if s in chrom]
    if not hit:
        continue
    cnt = len(hit)
    te = sum(1 for c in hit if c in TEST)
    va = sum(1 for c in hit if c in VAL)
    flag = "  <<< BREACH" if (te or va) else ""
    print(
        "%-42s matched=%6d/%-7d  test=%-5d val=%-5d (%.2f%%)%s"
        % (os.path.basename(p), cnt, len(seq), te, va, 100 * (te + va) / cnt, flag)
    )
