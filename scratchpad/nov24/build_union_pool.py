"""Build a UNION candidate pool: one acquisition rule, two reservoirs to choose from.

Every acquisition result so far gave the rule a single reservoir and asked which rule is
best. That is not the question a library order poses. A 4M order draws from several
generators at once, and the interesting question is whether an uncertainty rule, handed
all of them, reinvents a good mixture by itself -- and what fraction it takes from each.

CONSTRUCTION. The first 30,000 rows are motif_shared_core's own 30k prefix, unchanged, so
the seed set is byte-identical to the one every Arm B arm started from and the union arm
is directly comparable to the single-reservoir rows rather than merely similar. After it
come shared_core[30k:] and then ct_enriched[30k:] as candidates. ct_enriched's first 30k
is dropped rather than appended: it is that reservoir's Arm B seed set, and including it
would let the union arm start from 60k of labelled data while the arms it is compared
against start from 30k.

A `source` array records which reservoir each row came from, which is the headline the arm
exists to produce -- "the rule took X% ct_enriched" is the answer to "what should the
mixture be", measured rather than hand-set.

    python scripts/../scratchpad/nov24/build_union_pool.py --out <path>
"""

from __future__ import annotations

import argparse
import os

import numpy as np

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
NEST = f"{R}/outputs/grid_pools/k562_nested"
SEED_N = 30_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="motif_shared_core", help="reservoir whose prefix seeds the arm")
    ap.add_argument("--b", default="motif_ct_enriched")
    ap.add_argument("--size", type=int, default=1_000_000, help="per-reservoir pool size to use")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # A size>0 check is NOT enough and this bit us on Nov 24: two array tasks called this
    # at once, the second saw the first's partially-written file, skipped, and then died
    # with BadZipFile. Only a file that actually opens counts as present, and the write
    # below is atomic so a visible file is always a complete one.
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        try:
            with np.load(a.out, allow_pickle=True) as z:
                _ = z["sequences"]
            print(f"SKIP already present: {a.out}")
            return
        except Exception as e:
            print(f"present but unreadable ({type(e).__name__}), rebuilding: {a.out}")

    za = np.load(f"{NEST}/{a.a}_d{a.size}.npz", allow_pickle=True)
    zb = np.load(f"{NEST}/{a.b}_d{a.size}.npz", allow_pickle=True)
    sa, la = za["sequences"], za["oracle_labels"]
    sb, lb = zb["sequences"], zb["oracle_labels"]
    if za["oracle_id"].item() != zb["oracle_id"].item():
        raise SystemExit(
            f"refusing to union pools labelled by different oracles: "
            f"{za['oracle_id'].item()} vs {zb['oracle_id'].item()} -- the labels would not "
            "be on one scale and every selection would be comparing across calibrations"
        )

    seqs = np.concatenate([sa[:SEED_N], sa[SEED_N:], sb[SEED_N:]])
    labs = np.concatenate([la[:SEED_N], la[SEED_N:], lb[SEED_N:]])
    src = np.array(
        [a.a] * SEED_N + [a.a] * (len(sa) - SEED_N) + [a.b] * (len(sb) - SEED_N),
        dtype=object,
    )
    assert len(seqs) == len(labs) == len(src)

    # The seed prefix must be IDENTICAL to the single-reservoir arm's, or the comparison
    # silently becomes "different starting data" rather than "more candidates".
    assert np.array_equal(seqs[:SEED_N], sa[:SEED_N]), "seed prefix is not shared_core's 30k"

    # Write to a per-process temp name and rename: os.replace is atomic within a
    # filesystem, so a concurrent task either sees no file or a complete one. Two tasks
    # may both build -- wasteful but correct, and the content is identical either way.
    tmp = f"{a.out}.tmp.{os.getpid()}"
    np.savez_compressed(
        tmp,
        sequences=seqs,
        oracle_labels=labs,
        source=src,
        oracle_id=za["oracle_id"],
        strategy=np.array(f"union_{a.a}_{a.b}"),
        seed=za["seed"],
        extended_from=np.array(f"{a.a}_d{a.size}.npz+{a.b}_d{a.size}.npz"),
        n_base=np.array(SEED_N),
    )
    # savez_compressed appends .npz when the name lacks it; a.out already ends in .npz so
    # tmp does not, and the real file on disk is tmp + ".npz".
    written = tmp if os.path.exists(tmp) else tmp + ".npz"
    os.replace(written, a.out)
    print(
        f"wrote {a.out}\n  {len(seqs):,} rows: seed {SEED_N:,} ({a.a}) + "
        f"{len(sa) - SEED_N:,} {a.a} + {len(sb) - SEED_N:,} {a.b} candidates"
    )


if __name__ == "__main__":
    main()
