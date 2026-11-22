"""The full cell-type-aware architecture comparison: what each mechanism buys, where.

Three axes, because they answer different questions and a mechanism can win one and lose
another:
  SEEN   -- cells the model trained on. "Does a per-cell mechanism help at all?"
  HELD   -- a cell it never trained on, genomic test chromosomes. "Does it transfer?"
  DESIGN -- the designed libraries, where the cell-type DIFFERENCE has a ceiling near 0.9
            instead of near 0. "Does it transfer SPECIFICITY, not just level?"
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
    if a.get("mode") is None:
        continue
    gt = d.get("genomic_test", {})
    des = d.get("designed", {})
    seen_gt = [s["r"] for s in gt.values() if not s.get("held_out")]
    seen_de = [s["r"] for s in des.values() if not s.get("held_out")]
    rows[(a["mode"], a["depth"])].append(
        {
            "seen_gen": np.mean(seen_gt) if seen_gt else None,
            "held_gen": np.mean([s["r"] for s in gt.values() if s.get("held_out")]) or None,
            "seen_des": np.mean(seen_de) if seen_de else None,
            "held_des": (des.get("SK-N-SH") or {}).get("r"),
        }
    )


def f(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "     -     "
    if len(v) == 1:
        return " %6.3f    " % v[0]
    return " %6.3f+-%.3f" % (np.mean(v), np.std(v, ddof=1))


print("Tewhey 7-cell corpus. Train K562/HepG2/A549/GM12878; held out HCT116, SK-N-SH, Jurkat.")
print("SEEN = cells with labels. HELD = mean over the three held-out cells.\n")
h = "%-15s %-8s %-6s %-3s %-13s %-13s %-13s %-13s" % (
    "mechanism",
    "depth",
    "0-shot",
    "n",
    "SEEN genomic",
    "HELD genomic",
    "SEEN design",
    "HELD design",
)
print(h)
print("-" * len(h))
for k in sorted(rows, key=lambda k: (not C.can_describe_novel(k[0]), k[0])):
    v = rows[k]
    tag = "yes" if C.can_describe_novel(k[0]) else ("floor" if k[0] == "agnostic" else "NO")
    print(
        "%-15s %-8s %-6s %-3d%s%s%s%s"
        % (
            k[0],
            k[1],
            tag,
            len(v),
            f([r["seen_gen"] for r in v]),
            f([r["held_gen"] for r in v]),
            f([r["seen_des"] for r in v]),
            f([r["held_des"] for r in v]),
        )
    )

ag = rows.get(("agnostic", "readout"), [])
if ag:
    base = {
        kk: np.mean([r[kk] for r in ag if r[kk] is not None])
        for kk in ("seen_gen", "held_gen", "seen_des", "held_des")
    }
    print("\nmargin over the cell-blind floor:")
    for k in sorted(rows, key=lambda k: (not C.can_describe_novel(k[0]), k[0])):
        if k[0] == "agnostic":
            continue
        v = rows[k]
        d = []
        for kk in ("seen_gen", "held_gen", "seen_des", "held_des"):
            x = [r[kk] for r in v if r[kk] is not None]
            d.append("%+.3f" % (np.mean(x) - base[kk]) if x else "   -  ")
        print(
            "   %-15s %-8s  seen_gen %s  held_gen %s  seen_des %s  held_des %s"
            % (k[0], k[1], d[0], d[1], d[2], d[3])
        )
