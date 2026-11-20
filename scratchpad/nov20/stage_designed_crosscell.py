"""Join table: every Gosai designed oligo, measured in all three cell lines.

Table S2 carries each designed block only in the cell it was designed for, which is why
we had written the cross-cell designed contrast off as unmeasurable. The ENCODE 2024-04-25
Tewhey deposition carries the same oligo IDs in HepG2, K562 and SK-N-SH, so the contrast
is measured after all. This keys the three measurements (and their SEs) by oligo ID so
de41's oracle predictions can be scored against a 3-cell matrix rather than one column.

    python3 -I scratchpad/stage_designed_crosscell.py \
        --dir data/external/tewhey_encode --out outputs/designed3/crosscell_measured.npz
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

CELLS = {"K562": "k562", "HepG2": "hepg2", "SK-N-SH": "sknsh"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    frames = []
    for p in sorted(glob.glob(os.path.join(a.dir, "*.tsv"))):
        cell = os.path.basename(p).split("__")[0]
        if cell not in CELLS:
            continue
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
            d["cell"] = CELLS[cell]
            frames.append(d)

    big = pd.concat(frames, ignore_index=True)
    lfc = big.pivot_table(index="ID", columns="cell", values="log2FoldChange", aggfunc="mean")
    se = big.pivot_table(index="ID", columns="cell", values="lfcSE", aggfunc="mean")
    proj = big.drop_duplicates("ID").set_index("ID")["project"].reindex(lfc.index)

    ids = lfc.index.to_numpy(dtype=object)
    out = {"ids": ids, "project": proj.to_numpy(dtype=object)}
    for c in ("k562", "hepg2", "sknsh"):
        out[f"measured_{c}"] = (
            lfc[c].to_numpy(np.float32) if c in lfc else np.full(len(ids), np.nan, np.float32)
        )
        out[f"measured_se_{c}"] = (
            se[c].to_numpy(np.float32) if c in se else np.full(len(ids), np.nan, np.float32)
        )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, **out)

    print(f"{a.out}: {len(ids):,} designed oligos")
    print(big.groupby(["project", "cell"]).size().unstack(fill_value=0).to_string())
    full = (
        np.isfinite(out["measured_k562"])
        & np.isfinite(out["measured_hepg2"])
        & np.isfinite(out["measured_sknsh"])
    )
    print(f"\nmeasured in all three cells: {int(full.sum()):,}")


if __name__ == "__main__":
    main()
