"""Export the Gosai K562 measurements with split labels, and audit the splits for systematic differences.

The yeast dataset had a real confound of this shape (test sequences measured more deeply than training
ones), so this checks the same thing here before anyone trains on it: per split, the measurement
standard error (lfcSE, which falls with read depth), the activity distribution, length and GC.
"""

import gzip
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

SRC = "data/k562/DATA-Table_S2__MPRA_dataset.txt"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs/exports"
TEST, VAL = {"7", "13"}, {"19", "21", "X"}

df = pd.read_csv(SRC, sep="\t", dtype={"chr": str})
df["chr"] = df["chr"].astype(str).str.replace("^chr", "", regex=True)


def split_of(c):
    if c in TEST:
        return "test"
    if c in VAL:
        return "validation"
    return "train" if c.isdigit() or c in {"X", "Y", "M"} else "designed_or_unplaced"


df["split"] = df["chr"].map(split_of)
df["len"] = df["sequence"].str.len()
df["gc"] = df["sequence"].str.count("[GCgc]") / df["len"]

print(f"{len(df):,} rows, columns: {list(df.columns)}")
print("\n== data_project x split")
print(pd.crosstab(df["data_project"], df["split"]))

print("\n== per split: K562 activity and its standard error")
hdr = f"{'split':<22}{'n':>9}{'K562 log2FC':>14}{'lfcSE mean':>12}{'lfcSE med':>11}{'len':>7}{'GC':>7}"
print(hdr)
for s, g in df.groupby("split"):
    print(
        f"{s:<22}{len(g):>9,}{g['K562_log2FC'].mean():>14.4f}"
        f"{g['K562_lfcSE'].mean():>12.4f}{g['K562_lfcSE'].median():>11.4f}"
        f"{g['len'].mean():>7.1f}{g['gc'].mean():>7.3f}"
    )

# the comparison that matters is genomic-only: designed sequences differ by construction
gen = df[df["split"].isin(["train", "validation", "test"])]
print("\n== genomic only, lfcSE quartiles (lower = measured more precisely)")
for s, g in gen.groupby("split"):
    q = g["K562_lfcSE"].quantile([0.25, 0.5, 0.75]).values
    print(f"  {s:<12} n={len(g):>8,}  q25 {q[0]:.4f}  median {q[1]:.4f}  q75 {q[2]:.4f}")

tr = gen[gen["split"] == "train"]["K562_lfcSE"]
for s in ("validation", "test"):
    o = gen[gen["split"] == s]["K562_lfcSE"]
    d = (o.mean() - tr.mean()) / tr.std()
    print(f"  {s} vs train: lfcSE difference {o.mean() - tr.mean():+.4f} ({d:+.3f} train SD)")

os.makedirs(OUT_DIR, exist_ok=True)
cols = [
    "IDs",
    "chr",
    "split",
    "data_project",
    "class",
    "sequence",
    "K562_log2FC",
    "K562_lfcSE",
    "HepG2_log2FC",
    "HepG2_lfcSE",
    "SKNSH_log2FC",
    "SKNSH_lfcSE",
]
out = os.path.join(OUT_DIR, "gosai_k562_mpra_with_splits.csv.gz")
df[cols].to_csv(out, index=False, compression="gzip")
h = hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]
print(f"\nwrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB, sha256:{h})")
json.dump(
    {
        "rows": len(df),
        "sha256_16": h,
        "source": SRC,
        "test_chroms": sorted(TEST),
        "val_chroms": sorted(VAL),
    },
    open(os.path.join(OUT_DIR, "gosai_export_manifest.json"), "w"),
    indent=1,
)
