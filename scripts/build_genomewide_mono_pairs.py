"""Build a GENOME-WIDE monoallelic ref/alt pair set for a better-powered out-of-fold delta estimate.

Why chr7/13 is the wrong restriction for the ORACLE. The oracle's 10 folds are a RANDOM permutation
of the whole 856,252-sequence pool with no chromosome exclusion, so chromosome membership carries no
meaning for it - only fold membership does. Restricting the delta evaluation to the chr7/13 battery
was inherited from the student-model test sets, and it costs power: a delta needs BOTH alleles held
out by the SAME fold, which happens for ~1/10 of pairs, leaving only 2,593 usable pairs.

Building monoallelic pairs genome-wide from Table S2 gives ~461k candidates, so the same-fold subset
should be ~46k - roughly 18x the sample size, and drawn from the whole genome rather than two
chromosomes, which also makes it distributionally closer to the 35,226-pair sets used in earlier
model comparisons.

Output is written in the battery npz layout so the existing foldmap/predict stages run unchanged:
just point --battery_dir at the new directory.

Note this measures ORACLE fidelity on sequences no fold saw. It is still not the same question as
"how well does a chromosome-held-out student generalise", because every oracle fold trained on
chr7/13 sequences outside its own validation slice.
"""

import argparse
import os

import numpy as np
import pandas as pd

CELL = "K562"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/k562/DATA-Table_S2__MPRA_dataset.txt")
    ap.add_argument("--out_dir", default="data/k562/test_sets_mono_genomewide")
    args = ap.parse_args()

    lab, sec = f"{CELL}_log2FC", f"{CELL}_lfcSE"
    df = pd.read_csv(
        args.table, sep="\t", usecols=["IDs", "chr", "OL", "sequence", lab, sec], low_memory=False
    )
    p = df["IDs"].astype(str).str.split(":", expand=True)
    df["vk"] = p[0] + ":" + p[1] + ":" + p[2] + ":" + p[3]
    df["allele"] = p[4]
    df = df[df.allele.isin(["R", "A"])].dropna(subset=[lab, sec, "sequence"])

    refs = df[df.allele == "R"][["vk", "OL", "chr", "sequence", lab, sec]]
    alts = df[df.allele == "A"][["vk", "OL", "sequence", lab, sec]]
    m = refs.merge(alts, on=["vk", "OL"], suffixes=("_r", "_a"))
    # strict-mono: the variant appears in exactly ONE oligo context, matching the canonical
    # definition used by the chr7/13 battery
    nctx = m.groupby("vk")["OL"].transform("nunique")
    m = m[nctx == 1].drop_duplicates(subset=["vk"])
    m["delta"] = m[f"{lab}_a"] - m[f"{lab}_r"]
    print(f"[pairs] {len(m):,} strict-mono pairs genome-wide over {m['chr'].nunique()} chromosomes")
    held = m["chr"].astype(str).isin(["7", "13"]).sum()
    print(f"        chr7/13 {held:,} ({held / len(m):.1%}); rest {len(m) - held:,}")
    print(f"        delta sd={m['delta'].std():.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "snv_oracle.npz")
    np.savez_compressed(
        out,
        pair_keys=m["vk"].to_numpy(),
        ref_sequences=m["sequence_r"].to_numpy(),
        alt_sequences=m["sequence_a"].to_numpy(),
        true_ref_label=m[f"{lab}_r"].to_numpy(np.float32),
        true_alt_label=m[f"{lab}_a"].to_numpy(np.float32),
        true_delta=m["delta"].to_numpy(np.float32),
        ref_se=m[f"{sec}_r"].to_numpy(np.float32),
        alt_se=m[f"{sec}_a"].to_numpy(np.float32),
        chrom=m["chr"].to_numpy(),
        n_pairs=len(m),
        test_set_version=np.str_("snv_mono_genomewide_v1"),
        monoallelic=True,
        # placeholders so the report's deployed-vs-OOF comparison does not crash; the deployed
        # ensemble has not been run on this set
        ref_mean=np.full(len(m), np.nan, np.float32),
        alt_mean=np.full(len(m), np.nan, np.float32),
        delta_mean=np.full(len(m), np.nan, np.float32),
    )
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
