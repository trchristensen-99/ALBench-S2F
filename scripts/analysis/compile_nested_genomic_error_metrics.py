#!/usr/bin/env python
"""Reservoirs (genomic by default; ``--reservoirs`` for more) at 10k, 30k and 100k from NESTED pools, on all 19 evaluation sets, with r and MSE: the replacement for the genomic line of the
Oct 13-14 scaling table.

Why: in the Oct 13-14 cells every D had its own independent draw of genomic sequences, so 100k could (and did) score below 30k on several sets. Here
the three sizes are prefixes of one pool (``outputs/grid_pools/k562_nested/genomic_d{10000,30000,100000}.npz``), so each smaller training set is a
subset of the larger one. The script *measures* that (pool prefixes, identical test and validation labels) and refuses to write anything otherwise.

The cells are retrains of the menu (``menu_retrain.py`` outputs under ``outputs/menu_retrain_v0/<cell>/``). To keep the three sizes comparable the
ensemble at every size uses the SAME menu models: the (seed, menu index) pairs present at all three sizes. Series:

  sw38_base    sw38_<D>k_genomic_base   bare 200-bp students (like every other reservoir of the old table), first menu entries, canonical nested pools
  sw38_c20     sw38_<D>k_genomic_c20    the same with the study's input standard (20 real flank bases per side, shift +-10)
  sc31         sc31_<D>k_genomic        menu v1 (20 entries at 10k and 30k, 6 at 100k), bare students

Output rows have the schema of ``compile_oct14_error_metrics.py`` plus ``source`` (the series name), so ``plot_scaling_error_metrics.py --override``
can swap them in for the genomic reservoir.

    python scripts/analysis/compile_nested_genomic_error_metrics.py [--out outputs/compile_oct14_scaling]     (cpuq)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compile_oct14_error_metrics import cell_rows, write_rows  # noqa: E402
from compile_oct14_scaling import EVALS, pear  # noqa: E402

MR = "outputs/menu_retrain_v0"
POOLS = "outputs/grid_pools/k562_nested/{res}_d{D}.npz"
DS = (10000, 30000, 100000)
SERIES = {
    "sw38_base": "sw38_{k}k_{res}_base",
    "sw38_c20": "sw38_{k}k_{res}_c20",
    "sc31": "sc31_{k}k_{res}",
    "sc32": "sc32_{k}k_{res}",
}
FAMILY = {
    "genomic": "genomic",
    "zoonomia_orthologs": "genomic",
    "evoaug_paper2023": "perturbation",
    "mutagenesis_mr10": "perturbation",
    "motif_ct_enriched": "motif",
    "motif_shared_core": "motif",
    "random": "random",
    "random_gc_sweep": "random",
    "d3_steps64": "model",
    "d3_steps256": "model",
    "evo2": "model",
}


def check_pools(template: str = POOLS, res: str = "genomic") -> None:
    """Fail unless each pool is the exact prefix of the next (sequences and labels)."""
    pools = {}
    for D in DS:
        z = np.load(template.format(D=D, res=res), allow_pickle=True)
        pools[D] = (
            np.asarray(z["sequences"]).astype(str),
            np.asarray(z["oracle_labels"], np.float64),
        )
        if len(pools[D][0]) != D:
            raise SystemExit(f"pool for D={D} has {len(pools[D][0])} sequences")
    for small, big in zip(DS[:-1], DS[1:]):
        s, b = pools[small], pools[big]
        if not (np.array_equal(s[0], b[0][:small]) and np.allclose(s[1], b[1][:small])):
            raise SystemExit(
                f"NOT NESTED: the {small} pool is not a prefix of the {big} pool; nothing written"
            )
    print(
        f"{res}: pools nested: 10k is a prefix of 30k, 30k a prefix of 100k (sequences and labels)",
        flush=True,
    )


def load_menu_cell(cell: str, mr: str = MR):
    """(models, val labels, test labels) of a menu_retrain cell; each model has valr / val / test as the old compile expects."""
    p = f"{mr}/{cell}"
    if not os.path.exists(f"{p}/labels.npz"):
        return None
    lab = np.load(f"{p}/labels.npz")
    yval = lab["val_labels"].astype(np.float64)
    yte = {e: lab[f"oracle_{e}"].astype(np.float64) for e in EVALS}
    ms = []
    for mf in sorted(glob.glob(f"{p}/r*_menu_*_meta.json")):
        try:
            meta = json.load(open(mf))
            z = np.load(mf.replace("_meta.json", ".npz"))
            if not all(f"test_pred_{e}" in z.files for e in EVALS) or "val_pred" not in z.files:
                continue
            t = {e: z[f"test_pred_{e}"].astype(np.float64) for e in EVALS}
            v = z["val_pred"].astype(np.float64)
        except Exception:
            continue
        if not np.isfinite(v).all() or any(
            not np.isfinite(x).all() or np.abs(x).max() > 1e3 for x in t.values()
        ):
            continue
        idx = int(meta.get("menu_index", re.search(r"menu_(\d+)", mf).group(1)))
        ms.append(
            {
                "id": meta["model_id"],
                "key": (int(meta.get("round", 0)), idx),
                "val": v,
                "test": t,
                "valr": pear(v, yval),
            }
        )
    ms.sort(key=lambda m: m["key"])
    return ms, yval, yte


def series_rows(name: str, template: str, mr: str, rng, res: str = "genomic"):
    cells = {}
    for D in DS:
        c = load_menu_cell(template.format(k=D // 1000, res=res), mr)
        if c is None or not c[0]:
            print(f"{res} {name}: cell for D={D} missing or empty; series skipped", flush=True)
            return []
        cells[D] = c
    for D in DS[1:]:  # the same labels at every size, or the sizes are not comparable
        if not np.allclose(cells[D][1], cells[DS[0]][1]) or any(
            not np.allclose(cells[D][2][e], cells[DS[0]][2][e]) for e in EVALS
        ):
            raise SystemExit(
                f"{res} {name}: validation or test labels differ between D={DS[0]} and D={D}"
            )
    common = set.intersection(*[{m["key"] for m in cells[D][0]} for D in DS])
    if len(common) < 4:
        print(
            f"{res} {name}: only {len(common)} models at all three sizes; series skipped",
            flush=True,
        )
        return []
    print(
        f"{res} {name}: {len(common)} models at every size (of "
        + "/".join(str(len(cells[D][0])) for D in DS)
        + ")",
        flush=True,
    )
    rows = []
    for D in DS:
        ms, yval, yte = cells[D]
        keep = [m for m in ms if m["key"] in common]
        for r in cell_rows(FAMILY.get(res, res), res, D, keep, yval, yte, rng):
            r["source"] = name
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/compile_oct14_scaling")
    ap.add_argument("--menu_retrain", default=MR)
    ap.add_argument("--pools", default=POOLS, help="pool path template with {D} (and {res})")
    ap.add_argument("--series", nargs="+", default=list(SERIES), choices=list(SERIES))
    ap.add_argument("--reservoirs", nargs="+", default=["genomic"])
    ap.add_argument(
        "--skip_non_nested",
        action="store_true",
        help="skip (and say so) a reservoir whose pools are not nested instead of stopping",
    )
    ap.add_argument("--stem", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for res in a.reservoirs:
        try:
            check_pools(a.pools, res)
        except (SystemExit, OSError, KeyError) as e:
            if not a.skip_non_nested:
                raise
            print(f"{res}: SKIPPED ({e})", flush=True)
            continue
        for name in a.series:
            rows += series_rows(name, SERIES[name], a.menu_retrain, rng, res)
    if not rows:
        raise SystemExit("no complete series found")
    stem = a.stem or (
        "scaling_error_cells_genomic_nested"
        if a.reservoirs == ["genomic"]
        else "scaling_error_cells_latest_nested"
    )
    write_rows(rows, a.out, stem=stem)


if __name__ == "__main__":
    main()
