"""The honest reproducibility of our HMEC activity: two disjoint halves of the experiment.

The replicate check reported earlier (r = 0.84-0.91) is not evidence of anything. Every
replicate's log-ratio was computed against the SAME pooled plasmid denominator, so
lfc_i = R_i - D and lfc_j = R_j - D share the term D; with D contributing most of the
variance (log RNA and log DNA correlate at 0.83), two columns of pure noise would still
have come out strongly correlated.

This splits BOTH sides -- RNA reps 1,2 with DNA reps 1,2 against RNA reps 3,4,5 with DNA
reps 3,4,5 -- so the two estimates share no reads at all. Their correlation is the real
precision of the measurement, and it is the number that decides whether this dataset is
usable.
"""

import sys

import numpy as np

PSEUDO = 1.0
act = np.load(sys.argv[1], allow_pickle=True)
rna = act["rna_counts"].astype(float)
dna = act["dna_counts"].astype(float)
seqs = [str(s) for s in act["sequences"]]


def lfc_of(ri, di, minreads=20):
    r = rna[ri].sum(0)
    d = dna[di].sum(0)
    rc = r / r.sum() * 1e6
    dc = d / d.sum() * 1e6
    v = np.log2((rc + PSEUDO) / (dc + PSEUDO))
    v[(d < minreads) | (r <= 0)] = np.nan
    return v


def r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


print("RNA reps: %d, DNA reps: %d" % (rna.shape[0], dna.shape[0]))

a = lfc_of([0, 1], [0, 1])
b = lfc_of([2, 3, 4], [2, 3, 4])
val, n = r(a, b)
print("\nDISJOINT halves (RNA1,2/DNA1,2  vs  RNA3,4,5/DNA3,4,5): r = %.4f  (n=%d)" % (val, n))

sh = r(np.log2(rna[0] / rna[0].sum() * 1e6 + 1), np.log2(rna[1] / rna[1].sum() * 1e6 + 1))
print("raw RNA abundance, rep1 vs rep2:                          r = %.4f  (n=%d)" % sh)
sh = r(np.log2(dna[0] / dna[0].sum() * 1e6 + 1), np.log2(dna[1] / dna[1].sum() * 1e6 + 1))
print("raw DNA abundance, rep1 vs rep2:                          r = %.4f  (n=%d)" % sh)

# The shared-denominator artefact, demonstrated: same split but ONE pooled denominator.
pooled = dna.sum(0)
pc = pooled / pooled.sum() * 1e6
ra = rna[[0, 1]].sum(0)
rb = rna[[2, 3, 4]].sum(0)
va = np.log2((ra / ra.sum() * 1e6 + 1) / (pc + 1))
vb = np.log2((rb / rb.sum() * 1e6 + 1) / (pc + 1))
m = (pooled >= 20) & (ra > 0) & (rb > 0)
print(
    "same split but SHARING one denominator:                   r = %.4f  (n=%d)"
    % (float(np.corrcoef(va[m], vb[m])[0, 1]), int(m.sum()))
)

print("\nsd of the disjoint-half estimates: %.3f and %.3f" % (np.nanstd(a), np.nanstd(b)))
