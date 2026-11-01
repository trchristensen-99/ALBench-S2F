#!/usr/bin/env python
"""The five re-implemented architectures of the collaborator's open-world search (pi38_ cells) and the recipe transplant (tr38_ cells), per architecture rather than per cell.

For every cell and every menu entry: single-model validation r (V0c), genomic-test r, OOD r and random-sequence r (oracle labels), averaged over the seeds that trained; the stack of all trained models
of the cell (positive ElasticNet on V0c). Sweep arms are paired with the base arm on (entry, seed) and the transplant arms with the menu's own recipe on the entry. Entries that ran out of memory are
simply absent, so a comparison only ever uses entries that exist on both sides.

    python scripts/analysis/analyze_pi38.py [--out outputs/justification_oct22]      (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from analyze_extras38 import MR, SETS, pearson  # noqa: E402

from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

STATED = {"genomic": (0.852, 0.855), "ood": (0.309, 0.326), "random_32k": (0.765, 0.771)}


def models(cell: str) -> dict:
    """{(entry, round): {val, t}} for the models of a cell that trained to finite predictions."""
    p = f"{MR}/{cell}"
    if not os.path.exists(f"{p}/labels.npz"):
        return {}
    lab = np.load(f"{p}/labels.npz")
    out = {"_lab": lab}
    for mf in sorted(glob.glob(f"{p}/r*_menu_*_meta.json")):
        try:
            meta = json.load(open(mf))
            z = np.load(mf.replace("_meta.json", ".npz"))
            v = z["val_pred"].astype(np.float64)
            t = {s: z[f"test_pred_{s}"].astype(np.float64) for s in SETS}
        except Exception:
            continue
        if not np.isfinite(v).all() or any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
        ):
            continue
        out[(meta["menu_entry"], meta["round"])] = {"v": v, "t": t}
    return out


def scores(ms: dict) -> dict:
    lab = ms["_lab"]
    yv = lab["val_labels"].astype(np.float64)
    y = {s: lab[f"oracle_{s}"].astype(np.float64) for s in SETS}
    return {
        k: {"val": pearson(m["v"], yv), **{s: pearson(m["t"][s], y[s]) for s in SETS}}
        for k, m in ms.items()
        if k != "_lab"
    }


def stack(ms: dict) -> dict:
    keys = [k for k in ms if k != "_lab"]
    if len(keys) < 3:
        return {}
    lab = ms["_lab"]
    st = fit_zscored_stack(
        np.stack([ms[k]["v"] for k in keys]), lab["val_labels"].astype(np.float64)
    )
    return {
        s: pearson(
            st.predict(np.stack([ms[k]["t"][s] for k in keys])),
            lab[f"oracle_{s}"].astype(np.float64),
        )
        for s in SETS
    }


def per_entry(sc: dict) -> dict:
    by = defaultdict(list)
    for (e, _), s in sc.items():
        by[e].append(s)
    return {
        e: {"n": len(v), **{k: float(np.mean([x[k] for x in v])) for k in ("val", *SETS)}}
        for e, v in sorted(by.items())
    }


def paired(a: dict, b: dict, key=lambda k: k) -> dict:
    bb = {key(k): v for k, v in b.items()}
    d = defaultdict(list)
    for k, v in a.items():
        if key(k) in bb:
            for m in ("val", *SETS):
                d[m].append(v[m] - bb[key(k)][m])
    return {
        m: {
            "n": len(x),
            "mean": float(np.mean(x)),
            "se": float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"),
        }
        for m, x in d.items()
        if x
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification_oct22")
    a = ap.parse_args()
    res: dict = {"cells": {}, "paired": {}}
    print(
        "stated by the collaborator at 30k (test r): "
        + ", ".join(f"{k} {lo}-{hi}" for k, (lo, hi) in STATED.items())
    )
    cache: dict = {}
    names = sorted(
        os.path.basename(p)
        for p in glob.glob(f"{MR}/pi38_*") + glob.glob(f"{MR}/tr38_*") + glob.glob(f"{MR}/ep38_*")
    )
    for c in names:
        cache[c] = models(c)
        sc = scores(cache[c]) if cache[c] else {}
        res["cells"][c] = {
            "n_models": len(sc),
            "stack": stack(cache[c]) if sc else {},
            "entries": per_entry(sc),
        }
        print(
            f"\n{c}: {len(sc)} models, stack { ({k: round(v, 4) for k, v in res['cells'][c]['stack'].items()}) }",
            flush=True,
        )
        for e, v in res["cells"][c]["entries"].items():
            print(
                f"   {e:28s} n={v['n']} val {v['val']:.4f} genomic {v['genomic']:.4f} ood {v['ood']:.4f} random {v['random_32k']:.4f}"
            )
    pairs = [
        (c, f"pi38_base_genomic_30k")
        for c in names
        if c.startswith("pi38_")
        and c.endswith("_30k")
        and "base" not in c
        and "x05" not in c
        and "x2" not in c
    ]
    pairs += [(f"tr38_v2e100_genomic_{k}k", f"v2m38_genomic_{k}k") for k in (10, 30, 100)]
    pairs += [(f"ep38_v2x3_genomic_{k}k", f"v2m38_genomic_{k}k") for k in (10, 30)]
    pairs += [
        ("tr38_pie100_genomic_30k", "pi38_base_genomic_30k"),
        ("tr38_pie100_genomic_100k", "pi38_base_genomic_100k"),
    ]
    for x, y in pairs:
        if y not in cache:
            cache[y] = models(y)
        if not cache.get(x) or not cache.get(y):
            continue
        sx, sy = scores(cache[x]), scores(cache[y])
        if x.startswith(
            "tr38_v2e100"
        ):  # transplant arms rename the entries, and the menu has one seed
            r = paired(
                {(e.removesuffix("_e100"), 0): v for (e, _), v in sx.items()},
                {(e, 0): v for (e, _), v in sy.items()},
            )
        elif x.startswith(
            "ep38_"
        ):  # same menu entries, own recipe against 3x epochs, one seed each
            r = paired(
                {(e, 0): v for (e, _), v in sx.items()}, {(e, 0): v for (e, _), v in sy.items()}
            )
        elif x.startswith(
            "tr38_pie100"
        ):  # the same architecture under each of its two original recipes in the base cell
            for rec in ("muon", "adamw"):
                r = paired(
                    {(e.split("_")[1], rd): v for (e, rd), v in sx.items()},
                    {(e.split("_")[1], rd): v for (e, rd), v in sy.items() if e.endswith(rec)},
                )
                res["paired"][f"{x} - {y} [{rec} entries]"] = r
                print(
                    f"PAIRED {x} - {y} [{rec}]: "
                    + " ".join(
                        f"{m} {v['mean']:+.4f} (SE {v['se']:.4f}, n={v['n']})"
                        for m, v in r.items()
                        if m != "val"
                    ),
                    flush=True,
                )
            continue
        else:
            r = paired(sx, sy)
        res["paired"][f"{x} - {y}"] = r
        print(
            f"PAIRED {x} - {y}: "
            + " ".join(
                f"{m} {v['mean']:+.4f} (SE {v['se']:.4f}, n={v['n']})"
                for m, v in r.items()
                if m != "val"
            ),
            flush=True,
        )
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/data_pi38.json", "w"), indent=1)


if __name__ == "__main__":
    main()
