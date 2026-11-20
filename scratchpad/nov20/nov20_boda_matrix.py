"""The designed sequences ARE cross-measured: build the 3-cell matrix and score it.

Every BODA block in the 2024-04-25 Tewhey deposition appears with an identical oligo-ID
set in HepG2, K562 and SK-N-SH. That makes the designed-set cell-type contrast a MEASURED
endpoint, not an oracle extrapolation -- the thing we had written off. This builds the
(oligo x cell) activity matrix per block, reports the measurement ceiling of each cell and
of each pairwise difference, and checks whether these oligos are the ones already staged
in outputs/designed3 / Table S2.

    python3 -I scratchpad/nov20_boda_matrix.py --dir data/external/tewhey_encode --out scratchpad/nov20_boda_matrix.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = "/grid/wsbs/home_norepl/christen/ALBench-S2F"


def ceiling(v, se):
    m = np.isfinite(v) & np.isfinite(se)
    v, se = v[m], se[m]
    if v.size < 20:
        return None
    var, noise = float(v.var()), float(np.mean(se**2))
    return round(float(np.sqrt(max(0.0, 1 - noise / var))), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    frames: dict[str, list[pd.DataFrame]] = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(a.dir, "*.tsv"))):
        cell = os.path.basename(p).split("__")[0]
        cols = pd.read_csv(p, sep="\t", nrows=0).columns
        if "project" not in cols or "log2FoldChange" not in cols:
            continue
        d = pd.read_csv(
            p,
            sep="\t",
            usecols=[c for c in ("ID", "project", "log2FoldChange", "lfcSE") if c in cols],
            low_memory=False,
        )
        d = d[d["project"].astype(str).str.startswith("BODA")]
        if len(d):
            d["cell"] = cell
            frames[cell].append(d)

    big = pd.concat([x for v in frames.values() for x in v], ignore_index=True)
    print("BODA rows total:", len(big), "| example IDs:", big["ID"].head(4).tolist())

    out = {"blocks": {}}
    for proj, g in big.groupby("project"):
        piv = g.pivot_table(index="ID", columns="cell", values="log2FoldChange", aggfunc="mean")
        pse = g.pivot_table(index="ID", columns="cell", values="lfcSE", aggfunc="mean")
        cells = [c for c in ("K562", "HepG2", "SK-N-SH") if c in piv.columns]
        if len(cells) < 2:
            continue
        rec = {"n_oligos": int(len(piv)), "cells": cells, "per_cell": {}, "pairs": {}}
        for c in cells:
            v, s = piv[c].to_numpy(float), pse[c].to_numpy(float)
            m = np.isfinite(v)
            rec["per_cell"][c] = {
                "n": int(m.sum()),
                "mean": round(float(v[m].mean()), 4),
                "sd": round(float(v[m].std()), 4),
                "p95": round(float(np.quantile(v[m], 0.95)), 4),
                "ceiling": ceiling(v, s),
            }
        for i, c1 in enumerate(cells):
            for c2 in cells[i + 1 :]:
                v1, v2 = piv[c1].to_numpy(float), piv[c2].to_numpy(float)
                s1, s2 = pse[c1].to_numpy(float), pse[c2].to_numpy(float)
                m = np.isfinite(v1) & np.isfinite(v2)
                if m.sum() < 50:
                    continue
                dif = v1[m] - v2[m]
                sdif = np.sqrt(s1[m] ** 2 + s2[m] ** 2)
                rec["pairs"][f"{c1}-{c2}"] = {
                    "n": int(m.sum()),
                    "r_activity": round(float(np.corrcoef(v1[m], v2[m])[0, 1]), 4),
                    "mean_diff": round(float(dif.mean()), 4),
                    "sd_diff": round(float(dif.std()), 4),
                    "ceiling_diff": ceiling(dif, sdif),
                }
        out["blocks"][proj] = rec

    # ------------------------------- do we already have these oligos?
    have = {}
    for cell in ("k562", "hepg2", "sknsh"):
        p = f"{ROOT}/outputs/designed3/designed_{cell}.npz"
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            ids = {str(x) for x in z["ids"]}
            have[cell] = {"n": len(ids), "example": sorted(ids)[:3]}
    boda_ids = {p: set(g["ID"].astype(str)) for p, g in big.groupby("project")}
    out["staged_designed"] = have
    out["id_overlap_with_staged"] = {
        f"{proj} vs designed_{cell}": len(
            ids
            & {
                str(x)
                for x in np.load(
                    f"{ROOT}/outputs/designed3/designed_{cell}.npz", allow_pickle=True
                )["ids"]
            }
        )
        for proj, ids in boda_ids.items()
        if proj.startswith("BODA:")
        for cell in ("k562", "hepg2", "sknsh")
        if os.path.exists(f"{ROOT}/outputs/designed3/designed_{cell}.npz")
    }

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
