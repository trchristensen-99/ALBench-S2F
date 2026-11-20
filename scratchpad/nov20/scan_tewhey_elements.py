"""Do the ENCODE element FASTAs actually give us trainable sequence for the Tewhey corpus?

Three checks, in the order that decides whether the corpus is usable:
  1. how many distinct oligo sequences the design files hold, and how long they are
  2. whether the FASTA headers join onto the `ID` column of the element-quantification
     tables -- if they do not, the sequences are unusable no matter how many there are
  3. how much of each cell line's measured rows end up with a sequence

    python3 -I scratchpad/nov20/scan_tewhey_elements.py \
        --elements data/external/tewhey_elements --quant data/external/tewhey_encode
"""

from __future__ import annotations

import argparse
import collections
import glob
import os

import pandas as pd


def _open(path: str):
    """ENCODE serves these gzipped regardless of the .fa name the href implies."""
    import gzip

    with open(path, "rb") as probe:
        gzipped = probe.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if gzipped else open(path)


def read_fasta(path: str):
    name, buf = None, []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line)
    if name is not None:
        yield name, "".join(buf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", required=True)
    ap.add_argument("--quant", required=True)
    a = ap.parse_args()

    seqs: dict[str, str] = {}
    lens: collections.Counter = collections.Counter()
    per_file = []
    for p in sorted(glob.glob(os.path.join(a.elements, "*.fa"))):
        n = 0
        for name, s in read_fasta(p):
            seqs.setdefault(name, s)
            lens[len(s)] += 1
            n += 1
        per_file.append((os.path.basename(p), n))

    print(f"=== {len(per_file)} design files, {len(seqs):,} distinct oligo IDs")
    for f, n in per_file:
        print(f"   {f:44s} {n:>8,}")
    print("\nlengths:", lens.most_common(6))
    ex = list(seqs.items())[:3]
    for k, v in ex:
        print(f"   header: {k!r}  seq[:40]={v[:40]}")

    print("\n=== join onto the measured tables")
    hit_total = collections.Counter()
    row_total = collections.Counter()
    for p in sorted(glob.glob(os.path.join(a.quant, "*.tsv"))):
        cell = os.path.basename(p).split("__")[0]
        cols = pd.read_csv(p, sep="\t", nrows=0).columns
        if "ID" not in cols:
            continue
        d = pd.read_csv(p, sep="\t", usecols=["ID"], low_memory=False)
        ids = d["ID"].astype(str)
        row_total[cell] += len(ids)
        hit_total[cell] += int(ids.isin(seqs.keys()).sum())
    for cell in sorted(row_total):
        n, h = row_total[cell], hit_total[cell]
        print(f"   {cell:10s} {h:>9,} / {n:>9,} rows get a sequence  ({h / max(1, n) * 100:5.1f}%)")


if __name__ == "__main__":
    main()
