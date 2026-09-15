"""Build ref/alt variant pairs from the Gosai Table S2 master table, split by chromosome.

Why: delta-supervision (contrastive ref/alt) is the highest-leverage lever for making MPRA data
informative *back* to the genome, but the only paired file we had (snv_oracle.npz) is chr7/13 --
exactly the held-out TEST chromosomes. Training on it would leak test sequence into every battery
metric. Table S2 contains every oligo, so we can build pairs on the TRAIN chromosomes instead and
put the val-chromosome pairs into the val set, leaving the test set untouched.

IDs look like `chr:pos:ref:alt:allele:suffix`, allele in {R,A}. A variant_key (chr:pos:ref:alt) with
exactly one R and one A row is MONOALLELIC; positions carrying several alt alleles are POLYALLELIC.
The canonical test set was filtered to monoallelic because polyallelic pairs inflated performance, so
we flag both here and can train/evaluate on either subset to quantify that effect.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

TEST_CHROMS = {"7", "13"}
VAL_CHROMS = {"19", "21", "X"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/k562/DATA-Table_S2__MPRA_dataset.txt")
    ap.add_argument("--out_dir", default="outputs/delta_pairs")
    ap.add_argument("--label_col", default="K562_log2FC")
    args = ap.parse_args()

    df = pd.read_csv(args.table, sep="\t", usecols=["IDs", "chr", "OL", args.label_col, "sequence"])
    parts = df["IDs"].astype(str).str.split(":", expand=True)
    if parts.shape[1] < 5:
        raise ValueError(f"unexpected ID format: {df['IDs'].iloc[0]}")
    df["chrom"] = parts[0]
    df["variant_key"] = parts[0] + ":" + parts[1] + ":" + parts[2] + ":" + parts[3]
    df["allele"] = parts[4]
    df["pos_key"] = parts[0] + ":" + parts[1]
    df = df[df["allele"].isin(["R", "A"])].dropna(subset=[args.label_col, "sequence"])

    # Pair ref/alt WITHIN the same oligo context (OL), not just by variant_key. The canonical
    # definition (scripts/build_chrsplit_snv_mono.py) is: strict-mono = exactly ONE (ref,alt) pair
    # per variant_key; a variant appearing in SEVERAL oligo contexts is "multi-context", and those
    # repeats are what inflated measured performance. Collapsing with drop_duplicates would hide the
    # distinction entirely, so we keep every context and flag it.
    refs = df[df["allele"] == "R"][["variant_key", "OL", "chrom", "sequence", args.label_col]]
    alts = df[df["allele"] == "A"][["variant_key", "OL", "sequence", args.label_col]]
    merged = refs.merge(alts, on=["variant_key", "OL"], suffixes=("_ref", "_alt"))
    ctx_per_variant = merged.groupby("variant_key")["OL"].transform("nunique")
    merged["monoallelic"] = ctx_per_variant == 1
    print(
        f"[pairs] {len(merged)} (ref,alt) pairs over {merged['variant_key'].nunique()} variants; "
        f"strict-mono {int(merged['monoallelic'].sum())}, multi-context {int((~merged['monoallelic']).sum())}"
    )

    out = pd.DataFrame(
        {
            "variant_key": merged["variant_key"].values,
            "chrom": merged["chrom"].values,
            "ref_sequence": merged["sequence_ref"].values,
            "alt_sequence": merged["sequence_alt"].values,
            "ref_label": merged[args.label_col + "_ref"].values.astype(np.float32),
            "alt_label": merged[args.label_col + "_alt"].values.astype(np.float32),
            "monoallelic": merged["monoallelic"].values,
        }
    )
    out["delta"] = out["alt_label"] - out["ref_label"]

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for split, sel in (
        ("train", ~out["chrom"].isin(TEST_CHROMS | VAL_CHROMS)),
        ("val", out["chrom"].isin(VAL_CHROMS)),
        ("test", out["chrom"].isin(TEST_CHROMS)),
    ):
        d = out[sel]
        p = os.path.join(args.out_dir, f"delta_pairs_{split}.npz")
        np.savez_compressed(
            p,
            variant_key=d["variant_key"].values.astype(str),
            chrom=d["chrom"].values.astype(str),
            ref_sequences=d["ref_sequence"].values.astype(str),
            alt_sequences=d["alt_sequence"].values.astype(str),
            ref_label=d["ref_label"].values,
            alt_label=d["alt_label"].values,
            true_delta=d["delta"].values,
            monoallelic=d["monoallelic"].values,
        )
        summary[split] = {
            "n": int(len(d)),
            "mono": int(d["monoallelic"].sum()),
            "poly": int((~d["monoallelic"]).sum()),
            "chroms": sorted(d["chrom"].unique())[:6],
        }
        print(
            f"  {split:<6} n={len(d):>7}  mono={int(d['monoallelic'].sum()):>7}  "
            f"poly={int((~d['monoallelic']).sum()):>7}"
        )
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(
        f"[pairs] wrote -> {args.out_dir}  (labels = REAL {args.label_col}; oracle scoring is a separate step)"
    )


if __name__ == "__main__":
    main()
