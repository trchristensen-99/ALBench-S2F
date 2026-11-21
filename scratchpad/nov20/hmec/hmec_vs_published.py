"""Check HMEC activity, which we counted ourselves, against published activity.

No published quantification exists for the HMEC library -- every experiment using it (HMEC,
K562, HepG2, SK-N-SH, GM12878, HEK293) released reads only -- and its rsID-style names do
not join to the coordinate-style IDs of the 200-bp libraries. The only remaining anchor is
sequence: where a 101-bp oligo is embedded in a 200-bp oligo, the two designs carry the same
locus and the published activity for that 200-mer is a legitimate comparison.

It is a CROSS-CELL, CROSS-DESIGN comparison, so a modest correlation is the correct result.
What it can catch is the failure that matters: if our counting or log-ratio were wrong, the
correlation would be ~0 rather than merely modest.
"""

import collections
import gzip
import sys

import numpy as np

K = 40


def _open(p):
    with open(p, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    return gzip.open(p, "rt") if gz else open(p)


def read_fasta(p):
    name, buf = None, []
    with _open(p) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name:
                    yield name, "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line)
    if name:
        yield name, "".join(buf)


act = np.load(sys.argv[1], allow_pickle=True)
hm_seq = {str(s): i for i, s in enumerate(act["sequences"])}
lfc = act["log2FC"]
print("HMEC: %d designs, %d with activity" % (len(lfc), int(np.isfinite(lfc).sum())))

z = np.load(sys.argv[2], allow_pickle=True)
big = [str(s).upper() for s in z["sequences"]]
cells = [str(c) for c in z["cells"]]
y, mask = z["y"], z["mask"]

index = {}
for i, s in enumerate(big):
    for j in range(0, len(s) - K + 1):
        index.setdefault(s[j : j + K], i)

comp = str.maketrans("ACGTN", "TGCAN")
pairs = []
for s, hi in hm_seq.items():
    c = (len(s) - K) // 2
    p = s[c : c + K]
    bi = index.get(p)
    if bi is None:
        bi = index.get(p.translate(comp)[::-1])
    if bi is not None and np.isfinite(lfc[hi]):
        pairs.append((hi, bi))
print("%d HMEC designs joined to a 200-mer by a central %d-mer\n" % (len(pairs), K))

hi = np.array([p[0] for p in pairs])
bi = np.array([p[1] for p in pairs])
ours = lfc[hi]

print("HMEC (ours, counted from fastq) vs published activity on the joined loci")
print("%-10s %8s %9s" % ("cell", "n", "r"))
for j, cell in enumerate(cells):
    m = mask[bi, j] & np.isfinite(ours) & np.isfinite(y[bi, j])
    if m.sum() < 200:
        continue
    r = float(np.corrcoef(ours[m], y[bi, j][m])[0, 1])
    print("%-10s %8d %9.4f" % (cell, int(m.sum()), r))

# How much of that is just "both measure the same locus"? Compare published cells to each
# other on the SAME joined subset, as the scale a cross-cell correlation lives on.
print("\nfor scale -- published cell vs published cell on the same joined loci:")
done = set()
for j1, c1 in enumerate(cells):
    for j2, c2 in enumerate(cells):
        if j1 >= j2 or (c1, c2) in done:
            continue
        m = mask[bi, j1] & mask[bi, j2]
        if m.sum() < 200:
            continue
        r = float(np.corrcoef(y[bi, j1][m], y[bi, j2][m])[0, 1])
        print("   %-9s %-9s %7d %8.4f" % (c1, c2, int(m.sum()), r))
        done.add((c1, c2))
