"""Map the 71,103 DREAM eval sequences to their class, and stratify them into folds.

Two things come out of this:

  classes    which of the DREAM eval subsets each sequence belongs to -- needed to
             score the oracle PER CLASS, because "the oracle is good" is not a useful
             claim when the classes are as different as random 80-mers, planted
             motifs, GA-designed extremes and real promoters.

  folds      a stratified 10-fold assignment, so a variant of the oracle can train on
             80% of every class with 10% val and 10% test, matching the ratio the
             bulk random data already gets.

PAIRS MUST NOT BE SPLIT. SNVs, motif perturbation and motif tiling are ref/alt pairs,
and the quantity of interest is the DIFFERENCE between the two members. If one member
lands in train and the other in test, the "held-out" difference is half-memorised, and
the delta metric is meaningless. The human SNV analysis was corrupted by exactly this
class of pairing mistake, so pairs are assigned as a unit here and the check is
asserted rather than assumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# subset file -> (class name, columns holding sequences, is_paired)
SUBSETS = {
    "all_random_seqs.csv": ("random", ["sequence"], False),
    "yeast_seqs.csv": ("native_genomic", ["sequence"], False),
    "high_exp_seqs.csv": ("high_expression", ["sequence"], False),
    "low_exp_seqs.csv": ("low_expression", ["sequence"], False),
    "challenging_seqs.csv": ("challenging", ["sequence"], False),
    "all_SNVs_seqs.csv": ("snv", ["alt_sequence", "ref_sequence"], True),
    "motif_perturbation.csv": ("motif_perturbation", ["alt_sequence", "ref_sequence"], True),
    "motif_tiling_seqs.csv": ("motif_tiling", ["alt_sequence", "ref_sequence"], True),
}


def _read_subset(path: Path, cols: list[str]) -> list[list[str]]:
    """Return one row per record, each a list of the sequences in that record."""
    out = []
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        present = [c for c in cols if c in (r.fieldnames or [])]
        if not present:
            # Some subset files use 'sequence' where others use alt/ref; fall back to
            # whichever sequence-like column exists rather than silently emitting none.
            present = [c for c in (r.fieldnames or []) if "sequence" in c.lower()]
        for row in r:
            seqs = [(row.get(c) or "").strip().upper() for c in present]
            seqs = [s for s in seqs if s]
            if seqs:
                out.append(seqs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--test-file", default="data/yeast/filtered_test_data_with_MAUDE_expression.txt"
    )
    ap.add_argument("--subset-dir", default="data/yeast/test_subset_ids")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/yeast/eval_classes_v2.npz")
    args = ap.parse_args()

    tf = REPO / args.test_file
    seqs, labels = [], []
    with open(tf) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0]:
                seqs.append(p[0].strip().upper())
                labels.append(float(p[1]))
    n = len(seqs)
    print(f"eval sequences: {n:,}")
    pos = {s: i for i, s in enumerate(seqs)}
    if len(pos) != n:
        print(f"  note: {n - len(pos):,} duplicate sequences in the eval file")

    # ---- class membership ---------------------------------------------------
    cls = defaultdict(set)  # class -> set of eval-file indices
    pair_groups: list[list[int]] = []  # groups that must share a fold
    for fname, (cname, cols, paired) in SUBSETS.items():
        f = REPO / args.subset_dir / fname
        if not f.exists():
            print(f"  MISSING {fname} -- skipping {cname}")
            continue
        recs = _read_subset(f, cols)
        hit = miss = 0
        for rec in recs:
            idxs = [pos[s] for s in rec if s in pos]
            miss += len(rec) - len(idxs)
            hit += len(idxs)
            for i in idxs:
                cls[cname].add(i)
            if paired and len(idxs) > 1:
                pair_groups.append(idxs)
        print(f"  {cname:<20} records={len(recs):>6,}  matched={hit:>6,}  unmatched={miss:>6,}")

    unassigned = set(range(n)) - set().union(*cls.values()) if cls else set(range(n))
    if unassigned:
        cls["unassigned"] = unassigned
        print(f"  {'unassigned':<20} {len(unassigned):>6,} (in the eval file, in no subset)")

    # ---- stratified folds, pairs kept together ------------------------------
    rng = np.random.default_rng(args.seed)
    fold = np.full(n, -1, dtype=np.int8)

    # Union-find over pair groups so a sequence appearing in several pairs (an SNV
    # ref shared by many alts) drags its whole component into one fold.
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for g in pair_groups:
        for x in g[1:]:
            union(g[0], x)
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    print(f"  linked components (pairs kept together): {len(comps):,}")

    # Assign components to folds, balancing by SIZE rather than by count: the paired
    # sets have wildly uneven components (one motif_perturbation component holds the
    # entire class), so round-robin over components would put everything in one fold.
    #
    # A class is only SPLITTABLE if it has at least n_folds components. Below that no
    # stratified split exists that keeps pairs intact, and the class has to be all-in
    # or all-out of training. That is reported, not worked around, because the
    # alternative -- splitting a ref from its alt -- leaves the model having memorised
    # the reference and needing only the difference, which is precisely the quantity
    # the paired sets exist to measure.
    primary = {}
    for cname, idxs in cls.items():
        for i in idxs:
            primary.setdefault(i, cname)  # first class wins, deterministic by dict order
    by_class = defaultdict(list)
    for members in comps.values():
        cname = primary.get(members[0], "unassigned")
        by_class[cname].append(members)

    splittable, unsplittable = {}, {}
    for cname, groups in by_class.items():
        if len(groups) < args.n_folds:
            unsplittable[cname] = len(groups)
            for g in groups:  # keep them together in one fold, flagged for exclusion
                for i in g:
                    fold[i] = 0
            continue
        splittable[cname] = len(groups)
        load = np.zeros(args.n_folds, dtype=np.int64)
        for gi in rng.permutation(len(groups)):
            g = groups[gi]
            f = int(np.argmin(load))  # greedy: smallest current fold
            load[f] += len(g)
            for i in g:
                fold[i] = f

    print(
        f"\n  SPLITTABLE classes (>= {args.n_folds} components): "
        f"{ {k: v for k, v in sorted(splittable.items())} }"
    )
    if unsplittable:
        print(f"  UNSPLITTABLE classes (< {args.n_folds} components, all-in-or-all-out):")
        for k, v in sorted(unsplittable.items()):
            print(
                f"    {k}: only {v} component(s) -- cannot hold out a fold without "
                f"breaking ref/alt pairs"
            )

    assert (fold >= 0).all(), "some eval sequences were never assigned a fold"
    for g in pair_groups:
        assert len({int(fold[i]) for i in g}) == 1, "a pair was split across folds"
    print("  pair-integrity check: PASSED (no ref/alt pair spans two folds)")

    print("\n  per-class fold balance (count in each of 10 folds):")
    # Classes flagged unsplittable will show everything in fold 0 by design.
    for cname in sorted(cls):
        idx = np.array(sorted(cls[cname]))
        counts = np.bincount(fold[idx], minlength=args.n_folds)
        print(f"    {cname:<20} n={len(idx):>6,}  min={counts.min():>5,} max={counts.max():>5,}")

    out = REPO / args.out
    np.savez_compressed(
        out,
        sequences=np.array(seqs, dtype=object),
        labels=np.array(labels, dtype=np.float32),
        fold=fold,
        **{f"cls_{c}": np.array(sorted(v), dtype=np.int64) for c, v in cls.items()},
        splittable=np.array(sorted(splittable), dtype=object),
        unsplittable=np.array(sorted(unsplittable), dtype=object),
    )
    meta = {
        "n": n,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "classes": {c: len(v) for c, v in sorted(cls.items())},
        "n_pair_groups": len(pair_groups),
        "n_components": len(comps),
        "splittable": splittable,
        "unsplittable": unsplittable,
        "note": (
            "Unsplittable classes have fewer connected ref/alt components than folds, "
            "so no stratified split exists that keeps pairs intact. They must be "
            "entirely in or entirely out of oracle training."
        ),
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
