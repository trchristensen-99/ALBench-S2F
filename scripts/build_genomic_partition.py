"""Split the real-CRE pool into a DISJOINT baseline slice and reservoir slice.

WHY. The additive curves prepend a fixed genomic baseline and then add reservoir
sequences. For the synthetic arms that is unambiguous, because their sequences do not
exist in the baseline. For the GENOMIC arm both come from the same finite set of real
CREs, and drawing them independently makes the "added" data mostly re-add the baseline:
measured 28,564 of a 30,000 baseline (95.2%) already present in an independent 300k
genomic draw. The genomic curve would then look flat for a reason that has nothing to
do with genomic data being uninformative.

Partitioning once, deterministically, removes the ambiguity: the baseline is the first
`--baseline-n` of a seeded permutation, the reservoir pool is everything after it, and
the two cannot intersect.

THE CEILING IS REAL AND IS THE POINT. chr_train_ref_only.npz holds 314,981 distinct
sequences, so with a 30k baseline the genomic arm can offer at most ~285k additional
real CREs -- it cannot reach a +300k increment at all, while every synthetic arm can.
That asymmetry is a result, not a defect to engineer around.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline-n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="outputs/curves/pools")
    args = ap.parse_args()

    from albench.paths import resolve

    src = resolve("bg_cache")
    z = np.load(src, allow_pickle=True)
    seqs = np.array([str(s) for s in z["sequences"]], dtype=object)
    uniq, first_idx = np.unique(seqs, return_index=True)
    if len(uniq) != len(seqs):
        # Deduplicate up front; otherwise "disjoint" holds by index but not by content.
        seqs = seqs[np.sort(first_idx)]
    n = len(seqs)

    labels = None
    for k in ("oracle_labels", "oracle_mean", "labels"):
        if k in z.files:
            labels = np.asarray(z[k], dtype=np.float32)
            if len(labels) == len(z["sequences"]) and len(seqs) != len(z["sequences"]):
                labels = labels[np.sort(first_idx)]
            break

    if args.baseline_n >= n:
        raise SystemExit(f"baseline-n={args.baseline_n:,} but only {n:,} distinct sequences")

    perm = np.random.default_rng(args.seed).permutation(n)
    b_idx, r_idx = perm[: args.baseline_n], perm[args.baseline_n :]

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def write(path: Path, idx: np.ndarray) -> None:
        kw = {"sequences": seqs[idx], "strategy": "genomic", "params": "{}", "seed": args.seed}
        if labels is not None:
            kw["oracle_labels"] = labels[idx]
        np.savez_compressed(path, **kw)

    bp = out / f"genomic_baseline__n{args.baseline_n}__seed{args.seed}.npz"
    rp = out / f"genomic__n{len(r_idx)}__seed{args.seed}.npz"
    write(bp, b_idx)
    write(rp, r_idx)

    assert not (set(seqs[b_idx]) & set(seqs[r_idx])), "partition overlaps"
    print(f"source            : {src.name}  ({n:,} distinct)")
    print(f"baseline          : {bp.name}  n={len(b_idx):,}")
    print(f"genomic reservoir : {rp.name}  n={len(r_idx):,}")
    print(f"disjoint          : verified")
    print(
        f"\nCEILING: the genomic arm can add at most {len(r_idx):,} real CREs on top of a "
        f"{args.baseline_n:,} baseline,\nso its +300k curve point does not exist. Every "
        f"synthetic arm reaches +300k. Report that asymmetry."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
