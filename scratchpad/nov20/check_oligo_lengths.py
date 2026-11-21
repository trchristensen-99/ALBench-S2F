"""Are the corpus oligos intact 200-mers, or did the build truncate longer designs?

build_tewhey_corpus.py keeps designs of >= 200 bp and stores ``s[:200]``. That is correct
only if the designs ARE 200 bp. If any library uses longer oligos -- 230 bp with adapters,
say -- taking the first 200 silently shifts the variable region off centre for every one of
them, and nothing downstream would show it.
"""

import collections
import glob
import gzip
import os
import sys

import numpy as np


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


print("=== design lengths per element-reference file")
overall = collections.Counter()
longer = 0
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.fa"))):
    c = collections.Counter(len(s) for _, s in read_fasta(p))
    overall.update(c)
    top = c.most_common(3)
    gt = sum(v for k, v in c.items() if k > 200)
    longer += gt
    print("  %-34s n=%7d  lengths %s  >200bp: %d" % (os.path.basename(p), sum(c.values()), top, gt))

print("\noverall length histogram (top 10): %s" % overall.most_common(10))
print("designs longer than 200 bp: %d of %d" % (longer, sum(overall.values())))

z = np.load(sys.argv[2], allow_pickle=True)
L = collections.Counter(len(str(s)) for s in z["sequences"])
print("\ncorpus stored lengths: %s" % L.most_common(5))
