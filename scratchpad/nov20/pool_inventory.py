"""Inventory the labelled pools, so a collaborator gets paths that exist and sizes that are real."""

import collections
import glob
import os
import re
import time

ROOT = "outputs/grid_pools"


def inv(sub, pat):
    out = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, sub, "*.npz"))):
        b = os.path.basename(p)
        m = re.match(pat, b)
        if not m:
            continue
        g = m.groupdict()
        out[g["s"]].append((int(g["d"]), g.get("r"), p, os.path.getmtime(p)))
    return out


for sub, pat, label in [
    ("k562_nested", r"(?P<s>.+?)_d(?P<d>\d+)\.npz$", "CANONICAL nested (seed 42)"),
    ("k562_reps", r"(?P<s>.+?)_r(?P<r>\d+)_d(?P<d>\d+)\.npz$", "replicate draws r1-r3"),
    ("k562_mixtures", r"(?P<s>.+?)_d(?P<d>\d+)\.npz$", "mixtures"),
    ("k562_mix", r"(?P<s>.+?)_d(?P<d>\d+)\.npz$", "mix"),
    ("k562_hepg2", r"(?P<s>.+?)_d(?P<d>\d+)\.npz$", "HepG2-labelled"),
]:
    d = inv(sub, pat)
    if not d:
        continue
    print("\n=== %s  (%s/%s)" % (label, ROOT, sub))
    for s in sorted(d):
        rows = d[s]
        ds = sorted({x[0] for x in rows})
        reps = sorted({x[1] for x in rows if x[1]})
        newest = time.strftime("%Y-%m-%d", time.localtime(max(x[3] for x in rows)))
        print(
            "  %-28s D=%s%s  n=%d  newest=%s"
            % (
                s,
                ",".join("%dk" % (x // 1000) if x >= 1000 else str(x) for x in ds),
                ("  reps=" + ",".join(reps)) if reps else "",
                len(rows),
                newest,
            )
        )
