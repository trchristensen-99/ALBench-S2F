"""Report a conditioning grid with the held-out cells separated and flagged.

The earlier one-line-per-run summary hid two things that matter: there are THREE held-out
cells here, not one, and for `embed`/`onehot` a held-out score is not a transfer result at
all -- those modes have no vector for a cell they never trained on.
"""

import glob
import json
import os
import sys

from albench.celltype import conditioning as C

pat = sys.argv[1] if len(sys.argv) > 1 else "outputs/celltype/ct42_*/metrics.json"
rows = []
for p in sorted(glob.glob(pat)):
    d = json.load(open(p))
    a = d.get("args", {})
    mode = a.get("mode", "?")
    hist = [h for h in d.get("history", []) if h.get("val_r_seen_mean") is not None]
    best = max(hist, key=lambda h: h["val_r_seen_mean"]) if hist else {}
    seen = {c: s for c, s in best.get("val", {}).items() if not s.get("held_out")}
    gt = d.get("genomic_test", {})
    des = d.get("designed", {})
    rows.append(
        {
            "mode": mode,
            "depth": a.get("depth"),
            "seed": a.get("seed", 0),
            "valid": C.can_describe_novel(mode),
            "seen": sum(s["r"] for s in seen.values()) / max(1, len(seen)),
            "ho": {c: s["r"] for c, s in gt.items() if s.get("held_out")},
            "des": {c: s["r"] for c, s in des.items() if s.get("held_out")},
            "des_seen": {c: s["r"] for c, s in des.items() if not s.get("held_out")},
        }
    )

ho_cells = sorted({c for r in rows for c in r["ho"]})
print("held-out cells: %s\n" % ho_cells)
hdr = "%-15s %-8s %-5s %-7s %8s | " % ("mode", "depth", "seed", "zeroshot", "seen_r")
hdr += " ".join("%9s" % c for c in ho_cells) + " | %9s" % "SKNSH_des"
print(hdr)
print("-" * len(hdr))
for r in sorted(rows, key=lambda r: (not r["valid"], r["mode"], r["seed"])):
    line = "%-15s %-8s %-5s %-7s %8.4f | " % (
        r["mode"],
        r["depth"],
        r["seed"],
        "yes" if r["valid"] else ("floor" if r["mode"] == "agnostic" else "NO"),
        r["seen"],
    )
    line += " ".join("%9.4f" % r["ho"].get(c, float("nan")) for c in ho_cells)
    sk = r["des"].get("SK-N-SH")
    line += " | %9s" % ("%.4f" % sk if sk is not None else "-")
    print(line)

print("\n'zeroshot=NO' means the mode has no vector for a cell it never trained on:")
print("the number is an untrained row's behaviour, not transfer. Do not rank it against 'yes'.")
