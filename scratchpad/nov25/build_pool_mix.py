"""Build a candidate pool with a chosen seed corpus and a chosen reservoir mixture.

`build_union_pool.py` unions exactly two reservoirs and always seeds from the first of
them. Two questions asked after the Nov 25 meeting need more than that:

*Where the model starts matters.* Every acquisition arm so far seeded from
motif_shared_core or from the 7-way union. "Run the rules from the 30k genomic setting"
is a different starting model, and the seed prefix is what sets it.

*The pool need not be an even mix.* acq52 handed every rule one equal 7-way union, so
"uncertainty takes 20.7% motif_shared_core" is a statement about that pool. If the pool is
already enriched for what the rule would have chosen, the rule may have nothing left to
add -- that is a measurement, not an opinion, and it needs a pool built to a stated
mixture.

CONSTRUCTION, and the two invariants that make arms comparable:

*The seed prefix is byte-identical to the single-reservoir arm's.* Rows ``[:seed_n]`` are
the seed reservoir's own nested prefix, so an arm started here and an Arm A cell at the
same size trained on the same data. Asserted, not assumed.

*No member reservoir contributes its own seed prefix.* Candidates are taken from
``[seed_n:]`` of each member. Otherwise a union arm would begin with k x seed_n labelled
rows available while the arms it is compared against begin with seed_n.

A ``source`` array records the reservoir of every row, which is what makes the realised
mixture readable afterwards, and a sidecar ``*_mix.json`` records what was asked for and
what was actually available -- a reservoir that cannot fill its share is reported, never
silently topped up from another.

    python scratchpad/nov25/build_pool_mix.py --seed-res genomic --seed-n 30000 \
        --members motif_shared_core:1 random:1 genomic:1 --cand-n 700000 --out p.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np

R = os.environ.get("ALBENCH_REPO", "/grid/wsbs/home_norepl/christen/ALBench-S2F")
NEST = f"{R}/outputs/grid_pools/k562_nested"

# The realised composition uncertainty chose when handed the equal 7-way union (fig18,
# ab54, 75k of 200k scored). Used by --weights uncertainty to ask the saturation question:
# hand the rule the mixture it would have built, and see whether it still beats random.
UNCERTAINTY_MIX = {
    "motif_shared_core": 0.207,
    "random": 0.205,
    "motif_ct_enriched": 0.167,
    "evoaug_paper2023": 0.115,
    "mutagenesis": 0.115,
    "genomic": 0.096,
    "zoonomia_orthologs": 0.095,
}
UNION7 = tuple(UNCERTAINTY_MIX)
# Distinct sequences a reservoir can ever supply. Only the real-CRE pool is capped; the
# synthetic reservoirs reseed, and zoonomia_orthologs' nominal capacity is not usable
# because 15.4% of its rows are exact duplicates.
CAPACITY = {"genomic": 315_000}


def largest_pool(res: str) -> str:
    """The biggest nested pool this reservoir has.

    Reservoirs top out at very different sizes -- genomic at ~315k because that is all the
    real CREs there are, motif_* at 1M -- so a fixed size argument would silently drop
    whichever members cannot reach it.
    """
    cands = []
    for p in glob.glob(f"{NEST}/{res}_d*.npz"):
        m = re.search(rf"{re.escape(res)}_d(\d+)\.npz$", os.path.basename(p))
        if m:
            cands.append((int(m.group(1)), p))
    if not cands:
        raise SystemExit(f"no nested pool for reservoir {res!r} under {NEST}")
    return max(cands)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-res", required=True, help="reservoir whose prefix seeds the arm")
    ap.add_argument("--seed-n", type=int, default=30_000)
    ap.add_argument(
        "--members",
        nargs="*",
        default=[],
        help="res[:weight] candidate reservoirs; default = the seed reservoir alone",
    )
    ap.add_argument(
        "--weights",
        default="",
        choices=["", "even", "uncertainty", "capacity"],
        help="shorthand for the 7-way union: even, fig18's uncertainty mix, or capped by "
        "how much of a real order each reservoir could actually fill",
    )
    ap.add_argument(
        "--order-size",
        type=int,
        default=3_200_000,
        help="library size that --weights capacity prices each reservoir's share against",
    )
    ap.add_argument("--cand-n", type=int, default=700_000)
    ap.add_argument(
        "--flat",
        action="store_true",
        help="no seed prefix: a from-scratch combination corpus, interleaved so it stays nested",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.weights == "capacity":
        # An even union offers genomic 14.3% of the candidates, but there are only ~315,000
        # distinct real CREs in existence -- under 10% of a 3.2M order, and less once the
        # seed has spent some. Offering a rule more real sequence than the order could ever
        # contain lets it build a mixture nobody can buy. Capped reservoirs are weighted by
        # what they could actually fill; the rest divide what is left.
        caps = {r: CAPACITY[r] / a.order_size for r in UNION7 if r in CAPACITY}
        rest = [r for r in UNION7 if r not in caps]
        spare = max(0.0, 1.0 - sum(caps.values()))
        members = {**caps, **{r: spare / max(1, len(rest)) for r in rest}}
    elif a.weights:
        members = {r: (1.0 if a.weights == "even" else UNCERTAINTY_MIX[r]) for r in UNION7}
    elif a.members:
        members = {}
        for m in a.members:
            res, _, w = m.partition(":")
            members[res] = float(w) if w else 1.0
    else:
        members = {a.seed_res: 1.0}
    tot = sum(members.values())
    members = {k: v / tot for k, v in members.items()}

    # Same atomicity guard as build_union_pool: two array tasks can race here, and a
    # partially written file that merely EXISTS was enough to make the second task skip and
    # then die on BadZipFile. Only a file that opens counts as present.
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        try:
            with np.load(a.out, allow_pickle=True) as z:
                _ = z["sequences"]
            print(f"SKIP already present: {a.out}")
            return
        except Exception as e:
            print(f"present but unreadable ({type(e).__name__}), rebuilding: {a.out}")

    seed_n = 0 if a.flat else a.seed_n
    zs = np.load(f"{NEST}/{a.seed_res}_d{a.seed_n}.npz", allow_pickle=True)
    oracle = zs["oracle_id"].item()
    if a.flat:
        seqs, labs, srcs = [], [], []
    else:
        seed_seq = zs["sequences"][: a.seed_n]
        seed_lab = zs["oracle_labels"][: a.seed_n]
        if len(seed_seq) != a.seed_n:
            raise SystemExit(f"{a.seed_res} d{a.seed_n} holds only {len(seed_seq)} rows")
        seqs = [seed_seq]
        labs = [seed_lab]
        srcs = [np.array([a.seed_res] * a.seed_n, dtype=object)]
    report = {}
    for res, w in members.items():
        p = largest_pool(res)
        z = np.load(p, allow_pickle=True)
        if z["oracle_id"].item() != oracle:
            raise SystemExit(
                f"refusing to mix pools labelled by different oracles: {oracle} vs "
                f"{z['oracle_id'].item()} -- the labels would not be on one scale and every "
                "selection would be comparing across calibrations"
            )
        # Skip rows only for the reservoir whose prefix IS the seed set -- those rows are
        # already labelled and trained on, so offering them back as candidates would let
        # the rule "acquire" data the model has seen. Other members are skipped past
        # nothing: their rows are unseen at any seed size, and the first rule (skip
        # seed_n everywhere) silently emptied five of seven reservoirs at a 300k seed,
        # because a pool that tops out at 300k has nothing after row 300,000.
        off = seed_n if res == a.seed_res else 0
        s, lb = z["sequences"][off:], z["oracle_labels"][off:]
        want = int(round(w * a.cand_n))
        take = min(want, len(s))
        seqs.append(s[:take])
        labs.append(lb[:take])
        srcs.append(np.array([res] * take, dtype=object))
        report[res] = {
            "weight": round(w, 4),
            "wanted": want,
            "taken": take,
            "pool": os.path.basename(p),
            "short_by": want - take,
        }
        if take < want:
            print(f"  WARNING {res}: wanted {want:,} candidates, pool holds {take:,}")

    if a.flat:
        # Round-robin by share, so EVERY prefix carries the mixture. A plain concatenation
        # would make the 30k prefix one reservoir and the 300k prefix three, and the size
        # axis would then vary the composition as well as the amount -- which is the exact
        # confound `k562_nested` exists to remove.
        frac = np.concatenate([np.linspace(0, 1, len(s), endpoint=False) for s in seqs])
        who = np.concatenate([np.full(len(s), i) for i, s in enumerate(seqs)])
        key = np.lexsort((who, frac))
        seq = np.concatenate(seqs)[key]
        lab = np.concatenate(labs)[key]
        src = np.concatenate(srcs)[key]
    else:
        seq = np.concatenate(seqs)
        lab = np.concatenate(labs)
        src = np.concatenate(srcs)
    assert len(seq) == len(lab) == len(src)

    # DEDUPE, keeping the first occurrence. Two reservoirs can offer the same sequence, and
    # zoonomia_orthologs offers the same sequence to itself: its pools are 15.4% exact
    # duplicates at every size. Left in, run_al refuses the arm outright ("the acquired pool
    # contains duplicate sequences"), and a duplicate that survived would be worse than the
    # refusal -- it would credit a reservoir for capacity it does not have and let the same
    # sequence be "acquired" twice. The seed prefix is first, so keeping first occurrences
    # preserves it exactly.
    _, first = np.unique(seq.astype(str), return_index=True)
    keep = np.sort(first)
    if len(keep) < len(seq):
        dropped = Counter(str(r) for r in src[np.setdiff1d(np.arange(len(seq)), keep)])
        print(f"  deduped {len(seq) - len(keep):,} of {len(seq):,} rows:")
        for r, n in dropped.most_common():
            share = n / max(1, int((src == r).sum()))
            print(f"    {r:22s} -{n:>8,}  ({share:.1%} of what it offered)")
        for r in dropped:
            if r in report:
                report[r]["dropped_duplicate"] = int(dropped[r])
                report[r]["taken"] -= int(dropped[r])
        seq, lab, src = seq[keep], lab[keep], src[keep]

    if not a.flat:
        assert np.array_equal(seq[: a.seed_n], seed_seq), "seed prefix is not the seed reservoir's"
    else:
        # Verify the construction did what it claims rather than trusting the sort: the
        # 30k prefix must already hold every member at roughly its share.
        head = src[: min(30_000, len(src))]
        got = {r: float(np.mean(head == r)) for r in members}
        bad = {r: got[r] for r in members if abs(got[r] - members[r]) > 0.02}
        if bad:
            raise SystemExit(f"interleave did not preserve the mixture in the 30k prefix: {bad}")
        print("  30k-prefix composition: " + ", ".join(f"{r} {got[r]:.3f}" for r in sorted(got)))

    tmp = f"{a.out}.tmp.{os.getpid()}"
    np.savez_compressed(
        tmp,
        sequences=seq,
        oracle_labels=lab,
        source=src,
        oracle_id=zs["oracle_id"],
        strategy=np.array(
            f"mix{len(members)}_flat" if a.flat else f"mix_{a.seed_res}_seed{seed_n}"
        ),
        seed=zs["seed"],
        n_base=np.array(seed_n),
        params=np.array(json.dumps({"seed_res": a.seed_res, "members": report})),
    )
    written = tmp if os.path.exists(tmp) else tmp + ".npz"
    os.replace(written, a.out)
    with open(a.out.replace(".npz", "_mix.json"), "w") as fh:
        json.dump(
            {"seed_res": a.seed_res, "seed_n": seed_n, "flat": a.flat, "members": report},
            fh,
            indent=1,
        )
    head = f"flat combination corpus" if a.flat else f"seed {seed_n:,} ({a.seed_res}) +"
    print(f"wrote {a.out}\n  {len(seq):,} rows: {head}")
    for res, d in sorted(report.items(), key=lambda kv: -kv[1]["taken"]):
        print(f"    {res:22s} {d['taken']:>9,}")


if __name__ == "__main__":
    main()
