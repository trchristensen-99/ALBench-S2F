#!/usr/bin/env python
"""Does a longer schedule help every evaluation set, or only the genomic test? Single-model mean r per set for the bank's epoch arms, the patience-16 early stops and the collaborator's architectures.

The worry being tested: more epochs on 10k-100k sequences should overfit, and overfitting would show first on the sets furthest from the training distribution (OOD, random, the GC sweep, the
large structural perturbations). Arms are the published MPRA-LegNet at the tuned weight decay of each size (5 seeds per arm, 20 for e100 and the tuned arm; `docs/TUNED_LEGNET.md`), the same tuned
schedule stopped by patience 16 (`*_es16`), and the five re-implemented collaborator architectures at 60-80 epochs (`pi38_base`, Muon) against 300 epochs (`mrr_pi41_ownmuon`, Muon; C209/C218 are
missing from that arm because they diverged in fp16). Writes `outputs/justification_oct22/data_epochs_per_set.json`.

    python scripts/analysis/analyze_epochs_per_set.py      (cpuq)
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from compare_legnet_vs_loop import MR, SETS, load, pearson  # noqa: E402

ARMS = {
    "10k": ["e100", "e200_wd4", "e300_wd4", "e500_wd4", "e800_wd4", "e800_wd4_es16"],
    "30k": ["e100", "e150_wd2", "e200_wd2", "e300_wd2", "e500_wd2", "e800_wd2", "e500_wd2_es16"],
    "100k": ["e100", "e150", "e200", "e300", "e500", "e500_es16"],
}
TUNED = {"10k": "e800_wd4", "30k": "e500_wd2", "100k": "e500"}
COLS = (*SETS, "snv_effect")


def per_model(c: dict) -> dict:
    """Per-set r of every model of a cell, SNV effect included (alt minus ref against the oracle's alt minus ref)."""
    y = c["y"]
    out = {s: [] for s in COLS}
    for _, t, _ in c["ms"]:
        for s in SETS:
            out[s].append(pearson(t[s], y[s]))
        out["snv_effect"].append(pearson(t["snv_alt"] - t["snv_ref"], y["snv_alt"] - y["snv_ref"]))
    return {s: np.array(v) for s, v in out.items()}


def summary(r: dict) -> dict:
    return {s: float(np.nanmean(v)) for s, v in r.items()} | {"n": int(len(r["genomic"]))}


def diff(a: dict, b: dict) -> dict:
    """a minus b per set, with the unpaired SE (seeds are independent draws)."""
    out = {}
    for s in COLS:
        d = float(np.nanmean(a[s]) - np.nanmean(b[s]))
        se = float(
            np.sqrt(np.nanvar(a[s], ddof=1) / len(a[s]) + np.nanvar(b[s], ddof=1) / len(b[s]))
        )
        out[s] = (d, se)
    return out


def main() -> None:
    res: dict = {}
    for size, arms in ARMS.items():
        cells = {}
        for arm in arms:
            c = load(f"{MR}/bench38_ln_{arm}_{size}_c20", "r*_menu_*.npz")
            if c and len(c["ms"]) >= 2:
                cells[arm] = per_model(c)
        res[size] = {"arms": {a: summary(r) for a, r in cells.items()}}
        base, tuned = cells.get("e100"), cells.get(TUNED[size])
        es = cells.get(f"{TUNED[size]}_es16")
        print(f"\n== {size}: single-model mean r per set (n seeds)")
        print("arm           n  " + " ".join(f"{s[:8]:>8s}" for s in COLS))
        for a, r in cells.items():
            sm = summary(r)
            print(f"{a:13s} {sm['n']:2d}  " + " ".join(f"{sm[s]:8.4f}" for s in COLS))
        for name, a, b in (("tuned - e100", tuned, base), ("tuned - es16", tuned, es)):
            if a is None or b is None:
                continue
            d = diff(a, b)
            up = sum(1 for s in COLS if d[s][0] > 0)
            sig_down = [s for s in COLS if d[s][0] < -2 * d[s][1]]
            print(
                f"{name}: improves {up} of {len(COLS)} sets; significantly worse on {sig_down or 'none'}\n  "
                + " ".join(f"{s[:8]}={d[s][0]:+.3f}" for s in COLS)
            )
            res[size][name] = {"improved": up, "of": len(COLS), "sig_worse": sig_down, "diff": d}
        if size == "30k" and "e800_wd2" in cells and tuned is not None:
            d = diff(cells["e800_wd2"], tuned)
            print("e800 - e500 (30k): " + " ".join(f"{s[:8]}={d[s][0]:+.3f}" for s in COLS))
            res[size]["e800 - e500"] = d

    # the collaborator's architectures: 60-80 epochs (pi38, Muon) against 300 epochs (pi41 own Muon recipe)
    short = load(f"{MR}/pi38_base_genomic_30k", "r*_menu_*.npz")
    long_ = load(f"{MR}/mrr_pi41_ownmuon_30k", "r*_menu_*.npz")
    if short and long_:

        def by_arch(c, d):
            groups: dict = {}
            for v, t, mid in c["ms"]:
                meta = json.load(open(f"{d}/{mid}_meta.json"))
                m = re.search(r"c\d{3}", meta.get("menu_entry", ""))
                if m:
                    groups.setdefault(m.group(0), {"ms": [], "y": c["y"]})["ms"].append((v, t, mid))
            return {k: per_model(g) for k, g in groups.items()}

        s_arch = by_arch(short, f"{MR}/pi38_base_genomic_30k")
        l_arch = by_arch(long_, f"{MR}/mrr_pi41_ownmuon_30k")
        res["pi_30k"] = {}
        print(
            "\n== collaborator architectures, 30k, Muon: 300 epochs minus 60-80 epochs (single-model mean)"
        )
        for k in sorted(set(s_arch) & set(l_arch)):
            d = {s: float(np.nanmean(l_arch[k][s]) - np.nanmean(s_arch[k][s])) for s in COLS}
            up = sum(1 for s in COLS if d[s] > 0)
            res["pi_30k"][k] = {"improved": up, "of": len(COLS), "diff": d}
            print(
                f"{k}: improves {up} of {len(COLS)}  "
                + " ".join(f"{s[:8]}={d[s]:+.3f}" for s in COLS)
            )
        epochs = {}
        for d, c in ((f"{MR}/pi38_base_genomic_30k", short), (f"{MR}/mrr_pi41_ownmuon_30k", long_)):
            epochs[d.split("/")[-1]] = [
                json.load(open(f"{d}/{m[2]}_meta.json")).get("epochs_trained") for m in c["ms"]
            ]
        print("epochs trained:", epochs)
        res["pi_30k"]["epochs_trained"] = epochs

    os.makedirs("outputs/justification_oct22", exist_ok=True)
    json.dump(res, open("outputs/justification_oct22/data_epochs_per_set.json", "w"), indent=1)


if __name__ == "__main__":
    main()
