"""Why the staged Tewhey corpus has far more rows than variants, and what a row IS.

Answers three things the earlier scan only counted:
  1. the multiplicity: rows per (variant, cell) and what the duplicating axes are
     (file/experiment, window, strand, allele) -- decomposed so the product is visible
  2. the row vocabulary: every column, with an example row printed verbatim
  3. whether the 2023 and 2024 depositions are the same variants re-measured or a
     different library, per cell line

    python3 -I scratchpad/nov20_tewhey_anatomy.py --dir data/external/tewhey_encode
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.tsv")))

    # ---------------------------------------------------------------- 1. the schema
    head = pd.read_csv(files[0], sep="\t", nrows=3)
    print("=== columns of", os.path.basename(files[0]))
    print(list(head.columns))
    print("\n=== first row, verbatim")
    print(head.iloc[0].to_string())

    # ------------------------------------------------- 2. per cell x deposition anatomy
    per: dict[tuple, dict] = collections.defaultdict(
        lambda: {
            "files": [],
            "rows": 0,
            "snp": set(),
            "oligo_ids": set(),
            "windows": collections.Counter(),
            "strands": collections.Counter(),
            "alleles": collections.Counter(),
            "projects": collections.Counter(),
            "comb": set(),
        }
    )
    man = {r["file"]: r for r in json.load(open(os.path.join(a.dir, "MANIFEST.json")))["files"]}

    for p in files:
        base = os.path.basename(p)
        cell, exp, facc = base[:-4].split("__")
        rel = man.get(facc, {}).get("released", "?")
        d = pd.read_csv(p, sep="\t", low_memory=False)
        s = per[(cell, rel)]
        s["files"].append(facc)
        s["rows"] += len(d)
        for col, key in (
            ("window", "windows"),
            ("strand", "strands"),
            ("allele", "alleles"),
            ("project", "projects"),
        ):
            if col in d.columns:
                s[key].update(d[col].astype(str).value_counts().to_dict())
        if "SNP" in d.columns:
            s["snp"].update(d["SNP"].astype(str).unique().tolist())
        for idc in ("ID", "name", "oligo", "Oligo"):
            if idc in d.columns:
                s["oligo_ids"].update(d[idc].astype(str).unique().tolist())
                break
        # the full duplicating key, to see if rows are unique once you name all axes
        keys = [c for c in ("SNP", "allele", "window", "strand", "project") if c in d.columns]
        if keys:
            s["comb"].update(map(tuple, d[keys].astype(str).itertuples(index=False, name=None)))

    print("\n=== per cell line x deposition")
    rows = []
    for (cell, rel), s in sorted(per.items()):
        rows.append(
            {
                "cell": cell,
                "released": rel,
                "files": len(s["files"]),
                "rows": s["rows"],
                "variants": len(s["snp"]),
                "oligo_ids": len(s["oligo_ids"]),
                "rows/variant": round(s["rows"] / max(1, len(s["snp"])), 2),
                "distinct(SNP,allele,window,strand,project)": len(s["comb"]),
                "windows": dict(s["windows"].most_common(6)),
                "strands": dict(s["strands"].most_common(4)),
                "alleles": dict(s["alleles"].most_common(4)),
                "projects": dict(s["projects"].most_common(6)),
            }
        )
    print(json.dumps(rows, indent=1))

    # ----------------------------------- 3. is 2024 the same variant set re-measured?
    print("\n=== overlap of variant sets between depositions and between cells")
    sets = {f"{c}/{r}": s["snp"] for (c, r), s in per.items()}
    names = sorted(sets)
    print(f"{'':22s}" + "".join(f"{n.split('/')[0][:7]:>9s}" for n in names))
    for n in names:
        line = f"{n:22s}"
        for m in names:
            inter = len(sets[n] & sets[m])
            line += f"{inter / max(1, len(sets[n])) * 100:8.1f}%"
        print(line)

    # ---------------------------------------------- 4. skew precision, per cell x dep
    print("\n=== skew precision")
    for p in files:
        pass
    out = {}
    for cell, rel in sorted(per):
        sk, se, lf, lse = [], [], [], []
        for p in files:
            base = os.path.basename(p)
            c, _e, facc = base[:-4].split("__")
            if c != cell or man.get(facc, {}).get("released") != rel:
                continue
            h = pd.read_csv(p, sep="\t", nrows=0).columns
            use = [x for x in ("Log2Skew", "Skew_SE", "A_log2FC", "A_log2FC_SE") if x in h]
            if not use:
                continue
            d = pd.read_csv(p, sep="\t", usecols=use, low_memory=False)
            if "Log2Skew" in d:
                sk.append(d["Log2Skew"].to_numpy(np.float64))
            if "Skew_SE" in d:
                se.append(d["Skew_SE"].to_numpy(np.float64))
            if "A_log2FC" in d:
                lf.append(d["A_log2FC"].to_numpy(np.float64))
            if "A_log2FC_SE" in d:
                lse.append(d["A_log2FC_SE"].to_numpy(np.float64))

        def ceil(v, e):
            if not len(v) or not len(e) or len(v) != len(e):
                return None
            m = np.isfinite(v) & np.isfinite(e)
            v, e = v[m], e[m]
            if v.size < 10:
                return None
            var, noise = float(v.var()), float(np.mean(e**2))
            return {
                "n": int(v.size),
                "sd": round(float(v.std()), 4),
                "se_med": round(float(np.median(e)), 4),
                "noise/signal": round(noise / var, 4),
                "ceiling": round(float(np.sqrt(max(0.0, 1 - noise / var))), 4),
            }

        out[f"{cell}/{rel}"] = {
            "skew": ceil(
                np.concatenate(sk) if sk else np.array([]),
                np.concatenate(se) if se else np.array([]),
            ),
            "activity": ceil(
                np.concatenate(lf) if lf else np.array([]),
                np.concatenate(lse) if lse else np.array([]),
            ),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
