"""Which HMEC design sequences have no identifying k-mer, and why?

29% of the distinct sequences share every one of their 30-mers with another sequence, so a
read-based counter cannot assign them. That is a big enough hole to matter, and the fix
depends entirely on the cause: a shifted window of the same locus can be rescued by raising
K or by counting at the locus level, whereas a true near-duplicate design cannot be rescued
at all and has to be reported as unassignable.
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, "scratchpad/nov20/hmec")
import numpy as np  # noqa: E402
from count_oligos import _CODE, encode_kmers, read_fasta  # noqa: E402

K = 30

by_seq: dict[str, list[str]] = collections.defaultdict(list)
for p in sys.argv[1:]:
    for n, s in read_fasta(p):
        by_seq[s.upper()].append(n)
seqs = sorted(by_seq)
names = [by_seq[s] for s in seqs]

lens = np.array([len(s) for s in seqs], np.int32)
width = int(lens.max())
flat = bytearray()
for s in seqs:
    flat += s.encode("ascii", "replace").ljust(width, b"N")
codes = _CODE[np.frombuffer(bytes(flat), np.uint8).reshape(len(seqs), width)]

keys_l, owners_l = [], []
row = np.arange(len(seqs), dtype=np.int32)
for j in range(width - K + 1):
    q = encode_kmers(codes, j)
    ok = (lens >= j + K) & (q != np.uint64(0xFFFFFFFFFFFFFFFF))
    keys_l.append(q[ok])
    owners_l.append(row[ok])
keys = np.concatenate(keys_l)
owners = np.concatenate(owners_l).astype(np.int32)

order = np.argsort(keys, kind="stable")
keys, owners = keys[order], owners[order]
first = np.r_[True, keys[1:] != keys[:-1]]
gid = np.cumsum(first) - 1
mx = np.zeros(gid[-1] + 1, np.int32)
np.maximum.at(mx, gid, owners)
mn = np.full(gid[-1] + 1, np.iinfo(np.int32).max, np.int32)
np.minimum.at(mn, gid, owners)
sel = first & (mx == mn)[gid]
per = np.bincount(owners[sel], minlength=len(seqs))

dead = np.flatnonzero(per == 0)
print("%d of %d sequences have no identifying %d-mer\n" % (len(dead), len(seqs), K))

# Who do they collide with? Pair each dead sequence with the sequence sharing most k-mers.
pos_of = {}
for i, g in zip(owners, gid, strict=True):
    pos_of.setdefault(int(g), []).append(int(i))
partner = collections.Counter()
examples = []
for i in dead[:400]:
    s = seqs[i]
    share = collections.Counter()
    for j in range(len(s) - K + 1):
        q = encode_kmers(codes[i : i + 1], j)[0]
        p = int(np.searchsorted(keys, q))
        g = int(gid[min(p, len(gid) - 1)])
        for o in pos_of.get(g, []):
            if o != i:
                share[o] += 1
    if not share:
        continue
    j, n_shared = share.most_common(1)[0]
    partner[n_shared] += 1
    if len(examples) < 8:
        a, b = seqs[i], seqs[j]
        diff = sum(1 for x, y in zip(a, b, strict=False) if x != y)
        examples.append((names[i][0], names[j][0], len(a), len(b), diff, n_shared))

print("name-field histogram of the unassignable set:")
h = collections.Counter()
for i in dead:
    parts = names[i][0].split("_")
    h[parts[1] if len(parts) > 1 else "?"] += 1
print("  ", h.most_common(8))

print("\nexample collisions (dead sequence vs its top partner):")
print("  %-26s %-26s %5s %5s %6s %7s" % ("dead", "partner", "lenA", "lenB", "hamm", "shared"))
for a, b, la, lb, d, n in examples:
    print("  %-26s %-26s %5d %5d %6s %7d" % (a[:26], b[:26], la, lb, d if la == lb else "-", n))
