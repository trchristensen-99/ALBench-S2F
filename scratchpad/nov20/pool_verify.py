"""Spot-check the pools being handed to a collaborator: keys, nesting, holdout chromosomes.

Handing out a path is a claim about what is in it. The two claims that matter are the ones
AGENTS.md makes -- every size is a prefix of the next, and nothing touches chr7/chr13/
chr19/chr21/chrX -- and both have been violated silently before (k562_cumulative is not
nested; the zoonomia reservoir drew 7.37% of its regions from validation chromosomes).
"""

import glob
import os

import numpy as np

TEST = {"chr7", "chr13"}
VAL = {"chr19", "chr21", "chrX"}


def norm(v):
    s = str(v).strip()
    return s if s.lower().startswith("chr") else "chr" + s


root = "outputs/grid_pools/k562_nested"
one = sorted(glob.glob(os.path.join(root, "*.npz")))[0]
z = np.load(one, allow_pickle=True)
print("schema of %s:" % os.path.basename(one))
for k in z.files:
    a = z[k]
    print("   %-16s %-10s %s" % (k, str(a.dtype)[:10], a.shape))

print("\nnesting + holdout, per strategy")
strategies = [
    "genomic",
    "random",
    "motif_shared_core",
    "motif_ct_enriched",
    "zoonomia",
    "zoonomia_orthologs",
    "zoonomia_hx",
    "hg38_random",
    "gc_matched",
    "mutagenesis",
    "dinuc_shuffle",
    "evoaug_paper2023",
]
for s in strategies:
    ps = sorted(
        glob.glob(os.path.join(root, s + "_d*.npz")), key=lambda p: int(p.rsplit("_d", 1)[1][:-4])
    )
    if not ps:
        continue
    prev = None
    nest_ok, bad = True, []
    for p in ps:
        z = np.load(p, allow_pickle=True)
        seq = z["sequences"] if "sequences" in z.files else z[z.files[0]]
        if prev is not None and not np.array_equal(seq[: len(prev)], prev):
            nest_ok = False
        prev = seq
        for key in ("chrom", "chr", "chromosome"):
            if key in z.files:
                c = {norm(x) for x in np.unique(z[key])}
                bad += sorted((c & TEST) | (c & VAL))
                break
    print(
        "   %-22s sizes=%-2d nested=%-5s holdout_hits=%s  dup=%.1f%%"
        % (
            s,
            len(ps),
            nest_ok,
            sorted(set(bad)) or "none",
            100 * (1 - len(set(map(str, prev))) / len(prev)),
        )
    )
