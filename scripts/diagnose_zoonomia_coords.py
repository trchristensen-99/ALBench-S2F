"""Diagnose how Gosai oligo sequences map onto GRCh38 in the Zoonomia H5.

The first extraction attempt assumed the 200 bp oligo is centred on the variant position parsed from
the ID (`chr:pos:ref:alt:allele:suffix`, 1-based) and is on the plus strand. That produced a 100%
human-row mismatch, so at least one of those assumptions is wrong. Candidates, in order of
likelihood: the oligo is offset rather than centred; it is reverse-complemented; the ID coordinate
is hg19 rather than hg38; or the oligo is not raw genomic sequence at all.

Rather than guess, this searches a window around the stated position for the oligo in BOTH
orientations and reports the offset distribution. A tight, consistent offset means the oligos are
simply anchored differently; no hit in either orientation over +/-1 kb points at an assembly
mismatch, which would need a liftOver instead.
"""

import argparse

import numpy as np
import pandas as pd

H5 = "/grid/koo/home/shared/d3/data/zoonomia/zoonomia_241.h5"
CODE = np.array(list("NACGT"))
_COMP = str.maketrans("ACGTN", "TGCAN")


def rc(s):
    return s.translate(_COMP)[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=H5)
    ap.add_argument("--table", default="data/k562/DATA-Table_S2__MPRA_dataset.txt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--window", type=int, default=1000)
    args = ap.parse_args()

    import h5py

    df = pd.read_csv(args.table, sep="\t", usecols=["IDs", "chr", "sequence"], low_memory=False)
    p = df["IDs"].astype(str).str.split(":", expand=True)
    df["pos1"] = pd.to_numeric(p[1], errors="coerce")
    df["ref"], df["alt"], df["allele"] = p[2], p[3], p[4]
    df = df[(df.allele == "R") & df.pos1.notna() & df.sequence.notna()]
    df = df[df["chr"].astype(str).isin([str(i) for i in range(1, 23)] + ["X"])]
    sub = df.sample(n=min(args.n, len(df)), random_state=0)

    with h5py.File(args.h5, "r") as f:
        ref_ok = 0
        offsets, orients, nofind = [], [], 0
        oligo_len = None
        for _, r in sub.iterrows():
            chrom = "chr" + str(r["chr"])
            if chrom not in f:
                continue
            human_ds = f[chrom]["seq"]
            pos0 = int(r["pos1"]) - 1
            oligo = str(r["sequence"]).upper()
            oligo_len = oligo_len or len(oligo)

            # (a) does the ID's ref allele match the human base at pos0?
            base = CODE[int(human_ds[0, pos0])]
            if base == str(r["ref"]).upper():
                ref_ok += 1

            # (b) locate the oligo in a window, both orientations
            s = max(0, pos0 - args.window)
            e = min(human_ds.shape[1], pos0 + args.window)
            ctx = "".join(CODE[human_ds[0, s:e]])
            hit = ctx.find(oligo)
            orient = "+"
            if hit < 0:
                hit = ctx.find(rc(oligo))
                orient = "-"
            if hit < 0:
                nofind += 1
                continue
            start0 = s + hit
            offsets.append(pos0 - start0)  # variant offset within the oligo
            orients.append(orient)

        n = len(sub)
        print(f"sampled {n} REF oligos (length {oligo_len})")
        print(f"  ref allele matches human base at pos-1 : {ref_ok}/{n} ({ref_ok / n:.0%})")
        print(
            f"  oligo located in +/-{args.window} bp window: {len(offsets)}/{n}; not found {nofind}"
        )
        if offsets:
            o = np.array(offsets)
            print(
                f"  variant offset within oligo: median={np.median(o):.0f} "
                f"min={o.min()} max={o.max()} unique={len(set(offsets))}"
            )
            print(f"  orientation: + {orients.count('+')}  - {orients.count('-')}")
            if len(set(offsets)) == 1:
                print(
                    f"\n  => oligos are anchored at a FIXED offset of {offsets[0]}; "
                    f"use start0 = pos0 - {offsets[0]}"
                )
            else:
                print(
                    "\n  => offset VARIES, so oligos are not anchored on the variant; "
                    "coordinates must come from a locate-by-sequence step, not arithmetic."
                )
        else:
            print(
                "\n  => no hits in either orientation. Likely an ASSEMBLY mismatch "
                "(IDs may be hg19); a liftOver to GRCh38 is required before rate extraction."
            )


if __name__ == "__main__":
    main()
