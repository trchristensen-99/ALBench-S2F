#!/usr/bin/env python3
"""Per-training-size motif enrichment, because comparing a 100k arm with a 30k arm
is not a statement about architecture."""

from __future__ import annotations

import json
import re
from collections import defaultdict

SRC = "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad/nov11/arch_motifs.json"
MOTIFS = ["recurrent", "attention", "grouped_conv", "dilated", "depthwise",
          "global_pool", "inv_residual", "channel_shuf"]


def size_of(arm: str) -> str:
    m = re.search(r"_(\d+)k(?:_|$)", arm)
    return (m.group(1) + "k") if m else "?"


def main() -> None:
    rows = [r for r in json.load(open(SRC)) if r.get("val") is not None]
    byD = defaultdict(list)
    for r in rows:
        byD[size_of(r["arm"])].append(r)

    for D in ["10k", "30k", "100k", "300k"]:
        rs = byD.get(D, [])
        if len(rs) < 50:
            continue
        rs.sort(key=lambda r: -r["val"])
        k = max(25, len(rs) // 20)          # top 5%, never fewer than 25
        top = rs[:k]
        print("\n=== D=%s  n=%d  (top %d by val) ===" % (D, len(rs), k))
        print("  %-14s %8s %8s %7s" % ("motif", "all%", "top%", "ratio"))
        for m in MOTIFS:
            a = 100.0 * sum(m in r["motifs"] for r in rs) / len(rs)
            t = 100.0 * sum(m in r["motifs"] for r in top) / len(top)
            ratio = (t / a) if a > 0.5 else float("nan")
            print("  %-14s %7.1f%% %7.1f%% %6.2fx" % (m, a, t, ratio))

        # does the reference the arm was handed predict the winner's structure?
        print("  -- best proposal per reference class at this D --")
        best = {}
        for r in rs:
            best.setdefault(r["ref"], r)
        for ref, r in sorted(best.items(), key=lambda kv: -kv[1]["val"]):
            print("     ref=%-10s val %.4f test %s  %s  motifs=%s"
                  % (ref, r["val"],
                     ("%.4f" % r["test"]) if r["test"] is not None else "   -  ",
                     r["arm"], ",".join(m for m in r["motifs"] if m != "squeeze_exc")))

        # recurrence specifically: it is the claim Peter is most likely to probe
        rec = [r for r in rs if "recurrent" in r["motifs"]]
        norec = [r for r in rs if "recurrent" not in r["motifs"]]
        if rec and norec:
            rec.sort(key=lambda r: -r["val"])
            norec.sort(key=lambda r: -r["val"])
            print("  -- recurrence: best %.4f (n=%d) vs non-recurrent best %.4f (n=%d) --"
                  % (rec[0]["val"], len(rec), norec[0]["val"], len(norec)))


if __name__ == "__main__":
    main()
