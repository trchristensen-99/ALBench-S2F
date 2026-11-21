"""How well do two INDEPENDENT library batches agree on the same (oligo, cell)?

The Gosai cross-check came back at r = 0.993-0.995, which is ABOVE the measurement ceiling
for two independent measurements. That is not a triumph -- it means the two sources are not
independent, and the ENCODE deposition is largely a re-deposition of the same underlying
data. It verifies the join, the cell assignment and the scale, and it cannot say anything
about reproducibility.

This does. The Tewhey deposition contains the same oligos measured in genuinely separate
library batches (different experiment accessions, different dates). Comparing one batch
against another for the same (oligo, cell) gives a real replication estimate and a real
measurement ceiling, and it is the number that should be quoted for how noisy this assay is.

    python3 -I validate_batch_repro.py --quant data/external/tewhey_encode \
        --elements data/external/tewhey_elements
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True)
    ap.add_argument("--out", default="outputs/tewhey/validation_batch_repro.json")
    ap.add_argument("--min-overlap", type=int, default=2000)
    a = ap.parse_args()

    # One table per (cell, experiment accession). Same cell, different accession = a
    # different library batch, which is the comparison we want.
    byexp: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for p in sorted(glob.glob(os.path.join(a.quant, "*.tsv"))):
        base = os.path.basename(p).split("__")
        if len(base) < 2:
            continue
        cell, exp = base[0], base[1]
        cols = pd.read_csv(p, sep="\t", nrows=0).columns
        if "ID" not in cols or "log2FoldChange" not in cols:
            continue
        use = [c for c in ("ID", "log2FoldChange", "lfcSE") if c in cols]
        d = pd.read_csv(p, sep="\t", usecols=use, low_memory=False)
        v = d["log2FoldChange"].to_numpy(np.float64)
        e = d["lfcSE"].to_numpy(np.float64) if "lfcSE" in d else np.full(len(d), np.nan)
        ids = d["ID"].astype(str).to_numpy()
        ok = np.isfinite(v)
        byexp[(cell, exp)] = {ids[k]: (v[k], e[k]) for k in np.flatnonzero(ok)}
        print(f"  {cell:9s} {exp}  {int(ok.sum()):>8,} rows", flush=True)

    bycell: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for (cell, exp), m in byexp.items():
        bycell[cell].append((exp, m))

    print("\n=== same oligo, same cell, DIFFERENT library batch")
    print(
        "%-9s %-12s %-12s %8s %8s %8s %8s"
        % ("cell", "batch A", "batch B", "n", "r", "ceiling", "gap")
    )
    rows = []
    for cell, exps in sorted(bycell.items()):
        for i in range(len(exps)):
            for j in range(i + 1, len(exps)):
                ea, ma = exps[i]
                eb, mb = exps[j]
                shared = ma.keys() & mb.keys()
                if len(shared) < a.min_overlap:
                    continue
                ids = sorted(shared)
                va = np.array([ma[k][0] for k in ids])
                vb = np.array([mb[k][0] for k in ids])
                sa = np.array([ma[k][1] for k in ids])
                sb = np.array([mb[k][1] for k in ids])
                r = pearson(va, vb)
                var = float(np.var(np.r_[va, vb]))
                noise = float(np.nanmean(np.square(sa)) + np.nanmean(np.square(sb))) / 2
                c = float(np.sqrt(max(0.0, 1 - noise / var))) if var > 0 else float("nan")
                print(
                    "%-9s %-12s %-12s %8d %8.4f %8.4f %8.4f" % (cell, ea, eb, len(ids), r, c, c - r)
                )
                rows.append(
                    {
                        "cell": cell,
                        "batch_a": ea,
                        "batch_b": eb,
                        "n": len(ids),
                        "r": round(r, 4),
                        "ceiling": round(c, 4),
                        "gap": round(c - r, 4),
                    }
                )

    if rows:
        rr = [x["r"] for x in rows]
        cc = [x["ceiling"] for x in rows if np.isfinite(x["ceiling"])]
        print(
            "\n%d batch pairs | r: min %.4f median %.4f max %.4f | median ceiling %.4f"
            % (len(rows), min(rr), float(np.median(rr)), max(rr), float(np.median(cc)))
        )
        print(
            "This is the honest reproducibility of the assay. A corpus-vs-Gosai r far ABOVE\n"
            "this is a sign of shared provenance, not of a better measurement."
        )
    with open(a.out, "w") as fh:
        json.dump({"pairs": rows}, fh, indent=1)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
