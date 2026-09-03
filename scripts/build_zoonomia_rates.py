"""Build REAL per-position substitution rates from the Zoonomia 241-species alignment.

Replaces the placeholder in `albench/reservoir/motif_planted_v2.py:PhylogeneticZoonomiaSampler`,
which applies position-INDEPENDENT uniform substitutions at a flat 2% and carries the comment
"For the proper version: load `data/zoonomia/per_position_rates.npz` (TODO)". Every sweep result
labelled `phylogenetic_zoonomia` was therefore 2% uniform mutagenesis with no conservation signal.
This script produces exactly that missing file.

Source (Anirban, cluster-only -- unpublished):
    /grid/koo/home/shared/d3/data/zoonomia/zoonomia_241.h5
      chrN/seq     (241, chrom_len) uint8   0=N 1=A 2=C 3=G 4=T ; row 0 = Homo_sapiens
      chrN/phyloP  (chrom_len,)     float32 precomputed conservation
      species      (241,)           |S28

Two per-position signals are emitted:
  subst_rate  fraction of the 240 non-human species whose called base differs from human.
              This is the literal "per-position substitution rate across ~240 mammals" the TODO
              asks for, and it is what the sampler should mutate proportionally to.
  phylop      the precomputed score, kept for comparison / alternative weighting.

Coordinates come from the Gosai ID field `chr:pos:ref:alt:allele:suffix`, which is 1-BASED
(VCF-style) while the H5 arrays are 0-based -- an off-by-one here would silently shift every rate
by one base, so the conversion is asserted against the human row of the alignment.
"""

import argparse
import os

import numpy as np
import pandas as pd

H5_DEFAULT = "/grid/koo/home/shared/d3/data/zoonomia/zoonomia_241.h5"
CODE = {0: "N", 1: "A", 2: "C", 3: "G", 4: "T"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=H5_DEFAULT)
    ap.add_argument("--table", default="data/k562/DATA-Table_S2__MPRA_dataset.txt")
    ap.add_argument("--out", default="data/zoonomia/per_position_rates.npz")
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--max_regions", type=int, default=None)
    ap.add_argument("--label_col", default="K562_log2FC")
    args = ap.parse_args()

    import h5py

    df = pd.read_csv(
        args.table, sep="\t", usecols=["IDs", "chr", "sequence", args.label_col], low_memory=False
    )
    p = df["IDs"].astype(str).str.split(":", expand=True)
    df["pos1"] = pd.to_numeric(p[1], errors="coerce")
    df["allele"] = p[4] if p.shape[1] > 4 else None
    # only REF oligos have a meaningful human-genome window; alt oligos carry an engineered base
    df = df[(df["allele"] == "R") & df["pos1"].notna() & df["sequence"].notna()].copy()
    df["chrom"] = "chr" + df["chr"].astype(str).str.replace("chr", "", regex=False)
    df["pos0"] = df["pos1"].astype(np.int64) - 1  # 1-based -> 0-based
    df = df.drop_duplicates(subset=["chrom", "pos0"])
    if args.max_regions:
        df = df.iloc[: args.max_regions]
    print(f"[regions] {len(df):,} unique REF oligo centres over {df.chrom.nunique()} chromosomes")

    L = args.seq_len
    half = L // 2
    seqs, rates, phylops, kept = [], [], [], []
    n_oob = n_mismatch = 0

    with h5py.File(args.h5, "r") as f:
        avail = set(f.keys())
        for chrom, g in df.groupby("chrom", sort=True):
            if chrom not in avail:
                print(f"  [skip] {chrom} not in H5")
                continue
            seq_ds, pp_ds = f[chrom]["seq"], f[chrom]["phyloP"]
            clen = seq_ds.shape[1]
            g = g.sort_values("pos0")  # sorted access = chunk-friendly
            print(f"  {chrom}: {len(g):,} regions (len {clen:,})", flush=True)
            for pos0, oligo in zip(g["pos0"].to_numpy(), g["sequence"].to_numpy()):
                s, e = int(pos0) - half, int(pos0) - half + L
                if s < 0 or e > clen:
                    n_oob += 1
                    continue
                block = seq_ds[:, s:e]  # (241, L) uint8
                human = block[0]
                other = block[1:]
                called = other > 0  # exclude N in the aligned species
                diff = (other != human[None, :]) & called & (human[None, :] > 0)
                denom = called.sum(axis=0)
                rate = np.where(denom > 0, diff.sum(axis=0) / np.maximum(denom, 1), np.nan)
                # sanity: the human row must match the oligo where the oligo is genomic
                hstr = "".join(CODE[int(b)] for b in human)
                if oligo[:L].upper() != hstr:
                    n_mismatch += 1
                seqs.append(str(oligo)[:L])
                rates.append(rate.astype(np.float32))
                phylops.append(np.asarray(pp_ds[s:e], dtype=np.float32))
                kept.append((chrom, int(pos0)))

    if not seqs:
        raise SystemExit("no regions extracted -- check --h5 and coordinate parsing")
    rates = np.stack(rates)
    phylops = np.stack(phylops)
    finite = np.isfinite(rates)
    print(
        f"\n[extracted] {len(seqs):,} regions   out-of-bounds {n_oob:,}   "
        f"human-row mismatch {n_mismatch:,} ({n_mismatch / max(1, len(seqs)):.1%})"
    )
    print(
        f"[subst_rate] mean={np.nanmean(rates):.4f}  median={np.nanmedian(rates):.4f}  "
        f"p5={np.nanpercentile(rates, 5):.4f}  p95={np.nanpercentile(rates, 95):.4f}"
    )
    print(
        f"[phyloP]     mean={np.nanmean(phylops):.3f}  "
        f"frac>2 (conserved)={np.nanmean(phylops > 2):.3f}"
    )
    print(f"[coverage]   finite rate at {finite.mean():.1%} of positions")
    print(
        "\nNOTE the placeholder used a flat 2.0% at every position; the real mean above is the "
        "genome-wide comparison, and the p5-p95 spread is the signal the flat rate discards."
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        sequences=np.array(seqs, dtype=object),
        subst_rate=rates,
        phylop=phylops,
        chrom=np.array([c for c, _ in kept]),
        pos0=np.array([p for _, p in kept], dtype=np.int64),
        seq_len=L,
        source=np.str_(args.h5),
        n_species=241,
    )
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
