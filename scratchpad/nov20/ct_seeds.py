"""Combine ct42 (seed 0) and ct43 (seeds 1,2) into per-arm means with a spread.

The point of the replicates: single-run sd here is about 0.008, which is the size of the
gaps being resolved. Any ranking read off one seed is not a ranking.
"""

import collections
import glob
import json

import numpy as np

from albench.celltype import conditioning as C

rows = collections.defaultdict(list)
for p in sorted(glob.glob("outputs/celltype/ct4[23]_*/metrics.json")):
    d = json.load(open(p))
    a = d.get("args", {})
    mode, depth = a.get("mode"), a.get("depth")
    if mode is None:
        continue
    gt = d.get("genomic_test", {})
    des = d.get("designed", {})
    rows[(mode, depth)].append(
        {
            "seed": a.get("seed", 0),
            "ho": {c: s["r"] for c, s in gt.items() if s.get("held_out")},
            "des_sknsh": (des.get("SK-N-SH") or {}).get("r"),
        }
    )

cells = sorted({c for v in rows.values() for r in v for c in r["ho"]})
print("held-out cells: %s   (n seeds per arm in brackets)\n" % cells)
hdr = "%-15s %-8s %-7s %-4s" % ("mode", "depth", "zeroshot", "n")
for c in cells:
    hdr += "  %18s" % c
hdr += "  %18s" % "SK-N-SH designed"
print(hdr)
print("-" * len(hdr))


def ms(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "        -         "
    if len(v) == 1:
        return "  %7.4f          " % v[0]
    return "  %7.4f +- %-6.4f" % (np.mean(v), np.std(v, ddof=1))


order = sorted(rows, key=lambda k: (not C.can_describe_novel(k[0]), k[0]))
for mode, depth in order:
    v = rows[(mode, depth)]
    tag = "yes" if C.can_describe_novel(mode) else ("floor" if mode == "agnostic" else "NO")
    line = "%-15s %-8s %-7s %-4d" % (mode, depth, tag, len(v))
    for c in cells:
        line += ms([r["ho"].get(c) for r in v])
    line += ms([r["des_sknsh"] for r in v])
    print(line)

print("\n--- the comparison that matters: text vs its own control, per held-out cell")
t = rows.get(("text", "deep"), [])
s = rows.get(("text_shuffled", "deep"), [])
ag = rows.get(("agnostic", "readout"), [])
for c in cells + ["designed"]:
    key = (lambda r: r["des_sknsh"]) if c == "designed" else (lambda r: r["ho"].get(c))
    tv = [key(r) for r in t if key(r) is not None]
    sv = [key(r) for r in s if key(r) is not None]
    av = [key(r) for r in ag if key(r) is not None]
    if not tv or not sv:
        continue
    d = np.mean(tv) - np.mean(sv)
    pooled = np.sqrt((np.var(tv, ddof=1) + np.var(sv, ddof=1)) / 2) if len(tv) > 1 else float("nan")
    print(
        "  %-10s text %.4f  shuffled %.4f  delta %+.4f   (seed sd ~%.4f)  floor %.4f"
        % (c, np.mean(tv), np.mean(sv), d, pooled, np.mean(av) if av else float("nan"))
    )
