"""Build the candidate pool the acquisition experiments select from.

An acquisition experiment needs far more candidates than it buys, and it needs them to span the reservoirs --
otherwise "which sequences should we add?" collapses into "which sequences of ONE reservoir should we add?",
and the acquisition function is denied the choice that rp47 showed matters most. So the pool is an equal mix
of every reservoir that can supply its share, drawn from the REPLICATE draws so it is a different sample from
the canonical pools the seed set comes from.

Every candidate is already oracle_v3-labelled. That is deliberate and it is not cheating, as long as the
labels are withheld from the acquisition function -- which `acq_ladder.py` enforces by never passing them to
the selector. Labelling upfront buys two things: the experiment costs training time only, and an `oracle_top`
arm becomes available as an upper bound on what any selector could achieve by chasing activity.

Sequences already in the seed pool are removed, so "select 75k new oligos" cannot quietly re-buy what the
pilot already measured. The genomic reservoir makes this essential rather than cosmetic: its replicate draws
share ~95% of their sequences with the canonical pool.

    python scripts/build_acq_candidates.py --seed-pool outputs/grid_pools/k562_nested/genomic_d30000.npz \
        --n 1000000 --out outputs/grid_pools/k562_acq52/cand_from_genomic_30k.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
REPS = REPO / "outputs/grid_pools/k562_reps"

# Every reservoir that reaches 300k in all three replicate draws. `random` is in deliberately: it is the null
# the whole study is measured against, and an acquisition function that prefers it over the motif reservoirs
# is telling us something worth hearing.
SOURCES = (
    "motif_shared_core",
    "motif_ct_enriched",
    "evoaug_paper2023",
    "mutagenesis",
    "genomic",
    "random",
    "zoonomia_orthologs",
)
SIZE = 300_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-pool", required=True, help="pool the experiment starts from; excluded")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", default="1,2,3")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    sp = REPO / a.seed_pool if not Path(a.seed_pool).is_absolute() else Path(a.seed_pool)
    z = np.load(sp, allow_pickle=True)
    seed_seqs = {str(s) for s in z["sequences"]}
    print(f"seed pool {sp.name}: {len(seed_seqs):,} sequences, excluded from the candidates")

    reps = tuple(int(r) for r in a.reps.split(","))
    parts: dict[str, tuple[list[str], np.ndarray]] = {}
    oids = set()
    for strat in SOURCES:
        seqs: list[str] = []
        ys: list[np.ndarray] = []
        for rep in reps:
            p = REPS / f"{strat}_r{rep}_d{SIZE}.npz"
            if not p.exists():
                continue
            zz = np.load(p, allow_pickle=True)
            seqs += [str(s) for s in zz["sequences"]]
            ys.append(zz["oracle_labels"].astype(np.float32))
            oids.add(str(zz["oracle_id"]))
        if not seqs:
            print(f"  {strat}: no replicate draws, skipped")
            continue
        parts[strat] = (seqs, np.concatenate(ys))
    assert len(oids) == 1, f"candidates labelled by different oracles: {oids}"
    oid = oids.pop()

    # Round-robin again, so a prefix of the candidate file is itself a balanced mixture. Nothing requires
    # that, but it makes a truncated run (or a cheaper variant) comparable with the full one.
    per = a.n // len(parts)
    names = list(parts)
    keep_seqs: list[str] = []
    keep_y: list[float] = []
    keep_src: list[str] = []
    seen = set(seed_seqs)
    cursor = dict.fromkeys(names, 0)
    taken = dict.fromkeys(names, 0)
    while len(keep_seqs) < a.n:
        progressed = False
        for nm in names:
            if taken[nm] >= per or len(keep_seqs) >= a.n:
                continue
            s_list, y_list = parts[nm]
            i = cursor[nm]
            while i < len(s_list) and s_list[i] in seen:
                i += 1
            if i >= len(s_list):
                cursor[nm] = i
                continue
            seen.add(s_list[i])
            keep_seqs.append(s_list[i])
            keep_y.append(float(y_list[i]))
            keep_src.append(nm)
            cursor[nm] = i + 1
            taken[nm] += 1
            progressed = True
        if not progressed:
            break

    y = np.asarray(keep_y, np.float32)
    counts = {nm: taken[nm] for nm in names}
    print(f"candidates: {len(keep_seqs):,} ({len(set(keep_seqs)):,} distinct)")
    for nm in names:
        kept = counts[nm]
        scanned = cursor[nm]
        print(
            f"  {nm:22s} {kept:>8,}  (scanned {scanned:,}; "
            f"{scanned - kept:,} were already in the seed pool or repeats)"
        )
    print(f"  label {y.mean():+.3f}+-{y.std():.3f}")
    if a.dry:
        return
    out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmpnew.npz")
    np.savez(
        tmp,
        sequences=np.asarray(keep_seqs, dtype=object),
        oracle_labels=y,
        source=np.asarray(keep_src, dtype=object),
        oracle_id=oid,
        strategy="acq_candidates",
        seed_pool=str(sp.name),
    )
    tmp.replace(out)
    (out.parent / f"{out.stem}_MANIFEST.json").write_text(
        json.dumps(
            {
                "n": len(keep_seqs),
                "per_source": counts,
                "sources": names,
                "replicates": list(reps),
                "seed_pool": str(sp),
                "seed_pool_n": len(seed_seqs),
                "oracle_id": oid,
                "label_mean": round(float(y.mean()), 4),
                "label_sd": round(float(y.std()), 4),
            },
            indent=1,
        )
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
