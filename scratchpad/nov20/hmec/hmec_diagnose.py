"""Why does our HMEC activity correlate with nothing?

Published cells agree with each other at 0.4-0.8 on the joined loci; our HMEC sits at
0.00-0.15 against all of them. Either the join is pairing the wrong oligos, or the log-ratio
is not measuring activity. These are distinguishable.

  A. Normalisation. If activity tracks plasmid abundance, the DNA denominator is not doing
     its job and the ratio is an abundance artefact -- which would still be reproducible
     across replicates, because every replicate shares the same plasmid pool.
  B. The join. Restrict to designs whose WHOLE 101-mer sits inside the 200-mer, where the
     locus correspondence is certain rather than inferred from one 40-mer.
  C. Does it correlate with anything real? GC content tracks MPRA activity in every library
     we have; if our HMEC has no GC relationship either, the values are not activity.
"""

import gzip
import sys

import numpy as np

K = 40


def _open(p):
    with open(p, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    return gzip.open(p, "rt") if gz else open(p)


def gc(s):
    s = str(s).upper()
    return (s.count("G") + s.count("C")) / max(1, len(s))


def r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


act = np.load(sys.argv[1], allow_pickle=True)
seqs = [str(s) for s in act["sequences"]]
lfc = act["log2FC"].astype(float)
rna = act["rna_counts"].astype(float)
dna = act["dna_counts"].astype(float)
ok = np.isfinite(lfc)

print("=== A. normalisation")
dsum = dna.sum(0)
rsum = rna.sum(0)
print("   r(log2FC, log DNA abundance) = %+.4f" % r(lfc[ok], np.log2(dsum[ok] + 1)))
print("   r(log2FC, log RNA abundance) = %+.4f" % r(lfc[ok], np.log2(rsum[ok] + 1)))
print("   r(log RNA, log DNA)          = %+.4f" % r(np.log2(rsum[ok] + 1), np.log2(dsum[ok] + 1)))
print(
    "   DNA depth per design: median %d, IQR %d-%d"
    % (np.median(dsum), np.percentile(dsum, 25), np.percentile(dsum, 75))
)
print(
    "   RNA depth per design: median %d, IQR %d-%d"
    % (np.median(rsum), np.percentile(rsum, 25), np.percentile(rsum, 75))
)

print("\n=== C. does activity relate to GC at all?")
g = np.array([gc(s) for s in seqs])
print("   r(log2FC, GC) = %+.4f   (library GC %.3f)" % (r(lfc[ok], g[ok]), g.mean()))

print("\n=== B. join quality")
z = np.load(sys.argv[2], allow_pickle=True)
big = [str(s).upper() for s in z["sequences"]]
cells = [str(c) for c in z["cells"]]
y, mask = z["y"], z["mask"]
index = {}
for i, s in enumerate(big):
    for j in range(0, len(s) - K + 1):
        index.setdefault(s[j : j + K], i)
comp = str.maketrans("ACGTN", "TGCAN")

full, partial = [], []
for hi, s in enumerate(seqs):
    if not ok[hi]:
        continue
    su = s.upper()
    c = (len(su) - K) // 2
    p = su[c : c + K]
    bi = index.get(p)
    rc = False
    if bi is None:
        bi = index.get(p.translate(comp)[::-1])
        rc = True
    if bi is None:
        continue
    hay = big[bi]
    needle = su.translate(comp)[::-1] if rc else su
    (full if needle in hay else partial).append((hi, bi))

print("   whole 101-mer inside the 200-mer: %d   only the 40-mer: %d" % (len(full), len(partial)))
for label, pairs in (("FULL containment", full), ("partial", partial)):
    if len(pairs) < 200:
        print("   %s: too few (%d)" % (label, len(pairs)))
        continue
    h = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    out = []
    for j, cell in enumerate(cells):
        m = mask[b, j]
        if m.sum() < 200:
            continue
        out.append("%s %.3f (n=%d)" % (cell, r(lfc[h][m], y[b, j][m]), int(m.sum())))
    print("   %s: %s" % (label, "; ".join(out)))
