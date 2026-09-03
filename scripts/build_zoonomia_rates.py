"""Build REAL per-position substitution rates from the Zoonomia 241-species alignment.

Replaces the placeholder in `albench/reservoir/motif_planted_v2.py:PhylogeneticZoonomiaSampler`,
which applies position-INDEPENDENT uniform substitutions at a flat 2% and says so in its own
docstring ("For the proper version: load `data/zoonomia/per_position_rates.npz` (TODO)"). Every
sweep result labelled `phylogenetic_zoonomia` was therefore 2% uniform mutagenesis carrying no
conservation signal at all.

SOURCE OF REGIONS -- cCREs, not the Gosai oligos.
An earlier version tried to anchor windows on the Gosai variant IDs (`chr:pos:...`). That fails:
the ref allele matches the GRCh38 base at only ~18-20% of positions (chance level) and the 200 bp
oligo is absent from a +/-200 kb window in 16 of 20 cases, with the few hits at offsets scattered
from -30 kb to +41 kb. So the IDs are not GRCh38 coordinates and no arithmetic recovers them.
Drawing regions from the cCRE registry instead is both simpler and better posed: the BED and the H5
are the same assembly by construction, and a reservoir ought to generate NEW sequences rather than
inherit the existing library's coordinates.

    /grid/koo/home/shared/d3/data/zoonomia/
      zoonomia_241.h5    chrN/seq (241, len) uint8  0=N 1=A 2=C 3=G 4=T, row 0 = Homo_sapiens
                         chrN/phyloP (len,) float32
      GRCh38-cCREs.bed   chrom start end accD accE class   (2,348,854 regions, union of biosamples)

Emits per region: the human sequence, the per-position substitution rate (fraction of the 240
non-human species whose called base differs from human), and phyloP. The sampler mutates
proportionally to subst_rate, so unconserved positions vary and conserved ones are preserved --
which is the entire point of a phylogenetic reservoir.

LEAKAGE: --exclude_chroms drops the evaluation chromosomes so reservoir sequences cannot be
near-duplicates of test sequences. Note this does NOT protect against overlap with the Gosai
TRAINING sequences on other chromosomes; per Carl that needs a local-alignment homology filter,
which runs downstream on the emitted sequences.
"""

import argparse
import os

import numpy as np

ZOO_DIR = "/grid/koo/home/shared/d3/data/zoonomia"
CODE = np.array(list("NACGT"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=os.path.join(ZOO_DIR, "zoonomia_241.h5"))
    ap.add_argument("--ccre", default=os.path.join(ZOO_DIR, "GRCh38-cCREs.bed"))
    ap.add_argument("--out", default="data/zoonomia/per_position_rates.npz")
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--n_regions", type=int, default=200000)
    ap.add_argument("--exclude_chroms", nargs="*", default=["chr7", "chr13"])
    ap.add_argument(
        "--classes", nargs="*", default=None, help="restrict to cCRE classes, e.g. PLS pELS dELS"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max_n_frac",
        type=float,
        default=0.02,
        help="drop windows with more than this fraction of N in human",
    )
    args = ap.parse_args()

    import h5py
    import pandas as pd

    bed = pd.read_csv(
        args.ccre,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "accD", "accE", "cls"],
        low_memory=False,
    )
    keep = bed["chrom"].isin([f"chr{c}" for c in list(range(1, 23)) + ["X"]])
    bed = bed[keep & ~bed["chrom"].isin(args.exclude_chroms)]
    if args.classes:
        bed = bed[bed["cls"].isin(args.classes)]
    print(
        f"[cCRE] {len(bed):,} regions after filters; classes: {bed['cls'].value_counts().to_dict()}"
    )

    rng = np.random.default_rng(args.seed)
    if len(bed) > args.n_regions:
        bed = bed.iloc[rng.choice(len(bed), size=args.n_regions, replace=False)]
    bed = bed.sort_values(["chrom", "start"])  # sorted access is chunk-friendly
    L, half = args.seq_len, args.seq_len // 2

    seqs, rates, phylops, meta = [], [], [], []
    n_oob = n_masked = n_unaligned = 0
    with h5py.File(args.h5, "r") as f:
        for chrom, g in bed.groupby("chrom", sort=True):
            if chrom not in f:
                print(f"  [skip] {chrom} absent from H5")
                continue
            seq_ds, pp_ds = f[chrom]["seq"], f[chrom]["phyloP"]
            clen = seq_ds.shape[1]
            print(f"  {chrom}: {len(g):,} regions", flush=True)
            for st, en, cls in zip(g["start"].to_numpy(), g["end"].to_numpy(), g["cls"].to_numpy()):
                centre = (int(st) + int(en)) // 2  # centre-expand to a fixed L
                s, e = centre - half, centre - half + L
                if s < 0 or e > clen:
                    n_oob += 1
                    continue
                block = seq_ds[:, s:e]
                human = block[0]
                if (human == 0).mean() > args.max_n_frac:
                    n_masked += 1
                    continue
                other = block[1:]
                called = other > 0  # species with an aligned base
                diff = (other != human[None, :]) & called & (human[None, :] > 0)
                denom = called.sum(axis=0)
                rate = np.where(denom > 0, diff.sum(axis=0) / np.maximum(denom, 1), np.nan)
                # A window can have NO species aligned at any position (denom==0 everywhere),
                # which makes its per-region mean NaN and poisons downstream weighting. Require a
                # minimum fraction of positions with alignment coverage.
                if np.isfinite(rate).mean() < args.min_aligned_frac:
                    n_unaligned += 1
                    continue
                seqs.append("".join(CODE[human]))
                rates.append(rate.astype(np.float32))
                phylops.append(np.asarray(pp_ds[s:e], dtype=np.float32))
                meta.append((chrom, int(s), str(cls)))

    if not seqs:
        raise SystemExit("no regions extracted -- check --h5 / --ccre paths")
    rates, phylops = np.stack(rates), np.stack(phylops)
    print(
        f"\n[extracted] {len(seqs):,} regions   out-of-bounds {n_oob:,}   "
        f"N-masked {n_masked:,}   under-aligned {n_unaligned:,}"
    )
    print(
        f"[subst_rate] mean={np.nanmean(rates):.4f} median={np.nanmedian(rates):.4f} "
        f"p5={np.nanpercentile(rates, 5):.4f} p95={np.nanpercentile(rates, 95):.4f}"
    )
    print(f"[phyloP]     mean={np.nanmean(phylops):.3f} frac>2={np.nanmean(phylops > 2):.3f}")
    per_seq = np.nanmean(rates, axis=1)
    per_seq = per_seq[np.isfinite(per_seq)]
    print(
        f"[per-region] rate mean={per_seq.mean():.4f} sd={per_seq.std():.4f} "
        f"range=[{np.nanmin(per_seq):.3f}, {np.nanmax(per_seq):.3f}]"
    )
    print(
        f"\nThe placeholder used a FLAT 2.0% everywhere. The real per-position spread "
        f"(p5={np.nanpercentile(rates, 5):.3f} to p95={np.nanpercentile(rates, 95):.3f}) is "
        f"exactly the conservation signal the flat rate throws away."
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        sequences=np.array(seqs, dtype=object),
        subst_rate=rates,
        phylop=phylops,
        chrom=np.array([m[0] for m in meta]),
        start=np.array([m[1] for m in meta], dtype=np.int64),
        ccre_class=np.array([m[2] for m in meta]),
        seq_len=L,
        n_species=241,
        excluded_chroms=np.array(args.exclude_chroms),
        source=np.str_(args.h5),
    )
    print(f"[wrote] {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
