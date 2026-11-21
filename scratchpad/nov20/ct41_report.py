"""Summarise the ct41 conditioning bake-off: one row per arm."""

import glob
import json
import os

rows = []
for p in sorted(glob.glob("outputs/celltype/*/metrics.json")):
    tag = os.path.basename(os.path.dirname(p))
    d = json.load(open(p))
    a = d.get("args", {})
    hist = d.get("history", [])
    best = (
        max(
            hist,
            key=lambda h: h.get("val_r_seen_mean") if h.get("val_r_seen_mean") is not None else -9,
        )
        if hist
        else {}
    )
    v = best.get("val", {})
    ho = [c for c, s in v.items() if s.get("held_out")]
    seen = [c for c, s in v.items() if not s.get("held_out")]
    des = d.get("designed", {})
    dif = d.get("designed_differences", {})
    rows.append(
        {
            "tag": tag,
            "mode": a.get("mode"),
            "depth": a.get("depth"),
            "ep": best.get("epoch"),
            "nep": len(hist),
            "seen_r": round(sum(v[c]["r"] for c in seen) / len(seen), 4) if seen else None,
            "ho_r": round(v[ho[0]]["r"], 4) if ho else None,
            "ho_gap": round(v[ho[0]]["gap"], 4) if ho else None,
            "des_ho_r": round(des[ho[0]]["r"], 4) if ho and ho[0] in des else None,
            "dif_ho": round(sum(s["r"] for k, s in dif.items() if ho and ho[0] in k) / 2, 4)
            if dif and ho
            else None,
            "min": d.get("minutes"),
        }
    )

hdr = "%-22s %-14s %-8s %-7s %-8s %-8s %-9s %-10s %-8s %s"
print(
    hdr
    % ("tag", "mode", "depth", "epoch", "seen_r", "HO_r", "HO_gap", "des_HO_r", "difHO_r", "min")
)
for r in sorted(rows, key=lambda r: -(r["ho_r"] or -9)):
    print(
        hdr
        % (
            r["tag"],
            r["mode"],
            r["depth"],
            "%s/%s" % (r["ep"], r["nep"]),
            r["seen_r"],
            r["ho_r"],
            r["ho_gap"],
            r["des_ho_r"],
            r["dif_ho"],
            round(r["min"] or 0, 1),
        )
    )
