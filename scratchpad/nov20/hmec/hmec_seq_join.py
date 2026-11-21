"""Can the 101-bp HMEC oligos be joined to the 200-bp libraries by SEQUENCE?

No published activity exists anywhere for the HMEC library -- every experiment that uses it
released reads only -- and its rsID-style names do not join to the coordinate-style IDs of
the 200-bp libraries. A sequence-level join is the one remaining route to an external check,
and it works only if the two designs centre the same variant in overlapping windows.

Tests the cheap version first: is a central k-mer of each 101-mer present in some 200-mer?
"""

from __future__ import annotations

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


hmec = {}
for p in sys.argv[1:3]:
    for n, s in read_fasta(p):
        hmec[n] = s.upper()
print("%d HMEC designs" % len(hmec), flush=True)

z = np.load(sys.argv[3], allow_pickle=True)
big = [str(s).upper() for s in z["sequences"]]
print("%d corpus 200-mers" % len(big), flush=True)

comp = str.maketrans("ACGTN", "TGCAN")
index = {}
for i, s in enumerate(big):
    for j in range(0, len(s) - K + 1):
        index.setdefault(s[j : j + K], i)
print("%d distinct %d-mers indexed" % (len(index), K), flush=True)

hit = collections.Counter()
examples = []
for n, s in list(hmec.items()):
    c = (len(s) - K) // 2
    probe = s[c : c + K]
    i = index.get(probe)
    orient = "fwd"
    if i is None:
        probe = probe.translate(comp)[::-1]
        i = index.get(probe)
        orient = "rc"
    if i is None:
        hit["miss"] += 1
        continue
    hit[orient] += 1
    if len(examples) < 5:
        examples.append((n, i, orient, s in big[i] or s.translate(comp)[::-1] in big[i]))

tot = sum(hit.values())
print(
    "\ncentral %d-mer found in a 200-mer: fwd %d, rc %d, miss %d  (%.1f%% joined)"
    % (K, hit["fwd"], hit["rc"], hit["miss"], 100 * (tot - hit["miss"]) / max(1, tot))
)
print("\nexamples (name, corpus row, orientation, whole 101-mer contained?):")
for e in examples:
    print("  ", e)
