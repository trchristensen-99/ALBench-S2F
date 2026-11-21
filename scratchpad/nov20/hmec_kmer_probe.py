"""Is a 30-mer enough to tell one HMEC design oligo from another?

The library is a variant library: ref and alt oligos of the same SNP differ at ONE base, so
most of their 30-mers are identical and an anchor that lands away from the variant cannot
say which allele the read came from. Counting reads against an ambiguous anchor would
silently average the two alleles and destroy exactly the signal the assay exists to
measure, so the discriminating fraction is measured before the counter is written.
"""

from __future__ import annotations

import collections
import gzip
import sys

K = 30


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


oligos = {}
for p in sys.argv[1:]:
    for n, s in read_fasta(p):
        oligos[n] = s.upper()
print(
    "%d oligos; length histogram %s"
    % (len(oligos), collections.Counter(len(s) for s in oligos.values()).most_common(4))
)

suffix = collections.Counter()
for n in oligos:
    parts = n.split("_")
    suffix[parts[1] if len(parts) > 1 else "?"] += 1
print("name field 2:", suffix.most_common(6))

owners = collections.defaultdict(set)
for i, (n, s) in enumerate(oligos.items()):
    for j in range(0, len(s) - K + 1):
        owners[s[j : j + K]].add(i)
tot = len(owners)
uniq = sum(1 for v in owners.values() if len(v) == 1)
print(
    "\n%d distinct %d-mers; %d (%.1f%%) belong to exactly one oligo"
    % (tot, K, uniq, 100 * uniq / tot)
)

# Per oligo: how many of its own k-mers are discriminating, and are they contiguous?
per = []
for i, (n, s) in enumerate(list(oligos.items())[:4000]):
    c = sum(1 for j in range(len(s) - K + 1) if len(owners[s[j : j + K]]) == 1)
    per.append(c)
per.sort()
nk = len(next(iter(oligos.values()))) - K + 1
print(
    "per-oligo discriminating k-mers (of %d): min=%d p10=%d median=%d p90=%d max=%d"
    % (nk, per[0], per[len(per) // 10], per[len(per) // 2], per[9 * len(per) // 10], per[-1])
)
print("oligos with ZERO discriminating k-mer: %d of %d" % (sum(1 for c in per if c == 0), len(per)))
