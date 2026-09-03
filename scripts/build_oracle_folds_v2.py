"""Build the v2 oracle fold assignment: chromosome-based for ref/alt, random for designed.

WHY THIS REPLACES THE RANDOM SPLIT
The v1 oracle assigned all 856,252 pool sequences by a single random permutation. Three
consequences, all of which this fixes:

  1. NO HELD-OUT TEST FOLD. Each model had train and val only, so the reported number came from the
     fold used for model selection. Peter's point: a val fold is easy to overfit because it is what
     the hyperparameters were tuned on, so only a genuine test fold is worth reporting. Here each
     fold is TEST for exactly one model, VAL for one, and TRAIN for the other eight.
  2. REF AND ALT SPLIT APART. A random permutation puts a variant's two alleles in different folds
     ~90% of the time, so a variant-effect estimate needed both alleles held out by the same model
     and only ~1/10 of pairs qualified. Assigning by chromosome keeps every pair together, so ALL
     pairs become usable - roughly a tenfold increase in usable variant-effect n.
  3. DUPLICATED ALT BLOCK. v1 appended 35,226 alt sequences that were already present among the
     798,064 Table S2 rows, so those were trained on at double weight and were unusable
     out-of-fold. The pool here is Table S2 plus designed sequences, deduplicated.

Designed high-activity sequences have no genomic coordinates, so they are split randomly.

Rotation: model i trains on all folds except i and (i+1) % n, validates on (i+1) % n, tests on i.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

CELL = "K562"


def pack_chromosomes(sizes, n_folds, seed=42):
    """Greedy largest-first bin packing, so folds are balanced by SEQUENCE count.

    Balancing on chromosome count instead would leave wildly uneven folds, since chr1 carries
    several times the sequences of chr21.
    """
    order = sorted(sizes.items(), key=lambda kv: -kv[1])
    folds = {k: [] for k in range(n_folds)}
    totals = np.zeros(n_folds, dtype=np.int64)
    for chrom, n in order:
        k = int(np.argmin(totals))
        folds[k].append(chrom)
        totals[k] += n
    return folds, totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/k562/DATA-Table_S2__MPRA_dataset.txt")
    ap.add_argument("--designed", default="data/k562/test_sets/test_ood_designed_k562.tsv")
    ap.add_argument("--out", default="data/k562/oracle_folds_v2.json")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lab = f"{CELL}_log2FC"
    df = pd.read_csv(
        args.table, sep="\t", usecols=["IDs", "chr", "sequence", lab], low_memory=False
    )
    df = df.dropna(subset=[lab, "sequence"])
    df["chrom"] = df["chr"].astype(str).str.replace("chr", "", regex=False)
    keep = df["chrom"].isin([str(i) for i in range(1, 23)] + ["X", "Y"])
    dropped = int((~keep).sum())
    df = df[keep]
    sizes = df["chrom"].value_counts().to_dict()
    print(
        f"[table] {len(df):,} measured sequences over {len(sizes)} chromosomes "
        f"({dropped:,} dropped for unplaced/odd contigs)"
    )

    folds, totals = pack_chromosomes(sizes, args.n_folds, args.seed)
    print(f"\n[folds] chromosome assignment (balanced on sequence count)")
    for k in range(args.n_folds):
        chroms = sorted(folds[k], key=lambda c: (len(c), c))
        print(f"  fold {k}: {totals[k]:>7,} seqs  chr {','.join(chroms)}")
    print(
        f"  spread: min {totals.min():,}  max {totals.max():,}  "
        f"max/min {totals.max() / totals.min():.2f}"
    )

    chrom2fold = {c: k for k, cs in folds.items() for c in cs}

    # verify ref and alt of every variant land in the same fold
    p = df["IDs"].astype(str).str.split(":", expand=True)
    vk = p[0] + ":" + p[1] + ":" + p[2] + ":" + p[3]
    fold_of_row = df["chrom"].map(chrom2fold).to_numpy()
    g = pd.DataFrame({"vk": vk.to_numpy(), "f": fold_of_row}).groupby("vk")["f"].nunique()
    print(f"\n[check] variants whose alleles span >1 fold: {int((g > 1).sum()):,} of {len(g):,}")

    # designed sequences: random split, no coordinates available
    rng = np.random.default_rng(args.seed)
    nd = 0
    if os.path.exists(args.designed):
        dd = pd.read_csv(args.designed, sep="\t")
        nd = len(dd)
        d_fold = rng.integers(0, args.n_folds, size=nd)
        print(
            f"[designed] {nd:,} sequences split randomly: "
            f"{np.bincount(d_fold, minlength=args.n_folds).tolist()}"
        )
    else:
        d_fold = np.array([], dtype=int)
        print(f"[designed] {args.designed} not found - skipped")

    rot = [
        {
            "model": i,
            "test_fold": i,
            "val_fold": (i + 1) % args.n_folds,
            "train_folds": [f for f in range(args.n_folds) if f not in (i, (i + 1) % args.n_folds)],
        }
        for i in range(args.n_folds)
    ]
    print(f"\n[rotation] model i: test=fold i, val=fold (i+1)%{args.n_folds}, train=other 8")
    print(f"  e.g. model 0 -> test 0, val 1, train {rot[0]['train_folds']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "n_folds": args.n_folds,
                "seed": args.seed,
                "chrom_to_fold": chrom2fold,
                "fold_to_chroms": {str(k): sorted(v) for k, v in folds.items()},
                "fold_sizes_genomic": totals.tolist(),
                "designed_fold": d_fold.tolist(),
                "n_designed": nd,
                "rotation": rot,
                "note": (
                    "ref/alt assigned by chromosome so allele pairs stay together; designed "
                    "sequences assigned randomly. Each fold is test for one model, val for one, "
                    "train for eight."
                ),
            },
            f,
            indent=2,
        )
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
