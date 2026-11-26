"""One round of acquisition: score the candidates with a trained bank, select, write a pool.

This is deliberately selection ONLY. The models are trained by the study's canonical
trainer, which already applies the tuned per-size recipe, the 240-bp input standard,
reverse-complement averaging and the 18 evaluation sets; a second training path here would
be a second place for those to drift, and the study has been burnt by exactly that before.
So an acquisition arm is a loop at the shell level: train, select, train, select, and the
models that get reported are the same objects Arm A produces.

The written pool is in nested-pool format -- ``sequences`` plus ``oracle_labels`` plus the
provenance stamps -- so the trainer consumes it with ``--pool`` and needs no special case,
and so the file says for ever which method and round produced it.

Two things that are not optional:

*The seed prefix is shared.* Every method starts from the SAME labelled set, so the only
difference between arms is the selection rule. Pools are prefix-nested, so the seed set is
the first ``--seed-size`` rows and the candidates are everything after it.

*Labels are revealed, not invented.* The candidate pool is already oracle-labelled; the
selection rule may not look at ``oracle_labels``, and this module never passes them to the
acquisition function. Only the selected rows' labels are copied into the output.

    python -m albench.acquisition.run_al --pool <big.npz> --bank <dir> --method epistemic \
        --seed-size 30000 --current <prev.npz> --select 10000 --out <next.npz>
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

METHODS = (
    "random",
    "epistemic",
    "badge",
    "badge_epistemic_only",
    "badge_kmeanspp_only",
    "batchbald",
    # Added 2026-11-24. The first three were registered as strategies but were never
    # reachable from the AL loop, so no arm had ever run them end to end.
    "activity_normalized",
    "diversity_kmeans",
    "expected_activity",
    # Added 2026-11-25. Coverage as a CONSTRAINT rather than a side effect of a ranking:
    # `coverage_random` is a pure prior over the predicted-activity range with no model
    # uncertainty in it, and is the control that says whether uncertainty adds anything
    # once the activity profile is matched.
    "coverage_random",
    "coverage_epistemic",
)


def build_acq(method: str, seed: int):
    """The method, and for BADGE the two controls that let a gain be attributed."""
    from albench.acquisition.badge import (
        BADGEAcquisition,
        EpistemicOnlyAcquisition,
        KMeansPPOnlyAcquisition,
    )
    from albench.acquisition.batchbald import BatchBALDAcquisition
    from albench.acquisition.random_acq import RandomAcquisition
    from albench.acquisition.uncertainty import UncertaintyAcquisition

    if method == "random":
        return RandomAcquisition(seed=seed)
    if method == "epistemic":
        return UncertaintyAcquisition()
    if method == "badge":
        return BADGEAcquisition(seed=seed)
    if method == "badge_epistemic_only":
        return EpistemicOnlyAcquisition(seed=seed)
    if method == "badge_kmeanspp_only":
        return KMeansPPOnlyAcquisition(seed=seed)
    if method == "batchbald":
        return BatchBALDAcquisition(seed=seed)
    if method == "activity_normalized":
        # The CONTROL for plain `epistemic`, and the reason it matters: activities are
        # log2FC, and in regression the absolute error grows with the magnitude of the
        # prediction, so ranking by raw ensemble spread partly re-ranks by activity.
        # Arm B is consistent with exactly that -- epistemic selected a label mean of
        # 2.38 against a pool mean of 1.21 on motif_shared_core. This scores each
        # candidate by how far its uncertainty sits ABOVE the trend at its own activity
        # level, so "genuinely ambiguous" is separated from "merely active". If it
        # matches random, epistemic's gain was an activity gain.
        from albench.acquisition.uncertainty_variants import (
            ActivityNormalisedUncertaintyAcquisition,
        )

        return ActivityNormalisedUncertaintyAcquisition(seed=seed)
    if method == "diversity_kmeans":
        from albench.acquisition.diversity import DiversityAcquisition

        return DiversityAcquisition(seed=seed)
    if method == "expected_activity":
        from albench.acquisition.ensemble_acq import ExpectedActivityAcquisition

        return ExpectedActivityAcquisition(seed=seed)
    if method in ("coverage_random", "coverage_epistemic"):
        from albench.acquisition.coverage import ActivityCoverageAcquisition

        return ActivityCoverageAcquisition(seed=seed, within=method.split("_", 1)[1])
    raise ValueError(f"unknown method {method!r}; have {METHODS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="the large candidate pool (nested, labelled)")
    ap.add_argument("--current", default="", help="pool already acquired; default = the prefix")
    ap.add_argument("--seed-size", type=int, default=30_000)
    ap.add_argument("--bank", required=True, help="directory of .pt checkpoints to score with")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--select", type=int, required=True)
    ap.add_argument("--cap", type=int, default=200_000, help="candidates scored per round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.pool, allow_pickle=True)
    seqs = np.array([str(s) for s in z["sequences"]], dtype=object)
    y = np.asarray(z["oracle_labels"], np.float32)
    if str(z.get("oracle_id", "")) and str(z["oracle_id"]) != "oracle_v3":
        raise SystemExit(f"pool was labelled by {z['oracle_id']}, not oracle_v3")

    if a.current and os.path.exists(a.current):
        cz = np.load(a.current, allow_pickle=True)
        have = {str(s) for s in cz["sequences"]}
    else:
        have = set(seqs[: a.seed_size].tolist())
    taken = np.array([s in have for s in seqs])
    cand_idx = np.flatnonzero(~taken)
    print(
        f"{len(seqs):,} in pool | {int(taken.sum()):,} already labelled | "
        f"{len(cand_idx):,} candidates",
        flush=True,
    )
    if len(cand_idx) < a.select:
        raise SystemExit(f"only {len(cand_idx):,} candidates left, need {a.select:,}")

    # Scoring every candidate of a 1M pool with a 3-member ensemble is the slow step and
    # buys little: a uniform subsample of the candidates is itself a valid candidate set,
    # and keeping it fixed per round keeps the arms comparable.
    rng = np.random.default_rng(a.seed)
    if len(cand_idx) > a.cap:
        cand_idx = np.sort(rng.choice(cand_idx, a.cap, replace=False))
        print(f"scoring a {a.cap:,} subsample of the candidates", flush=True)

    t0 = time.time()
    if a.method == "random":
        pick_local = rng.choice(len(cand_idx), a.select, replace=False)
        extra: dict = {}
    else:
        from albench.acquisition.legnet_bank import LegNetBank

        bank = LegNetBank(a.bank)
        print(f"bank: {len(bank)} members from {a.bank}", flush=True)
        if not bank.separates_uncertainty:
            raise SystemExit(
                "the bank has one member, so epistemic uncertainty is identically zero "
                "and every candidate would score the same; train more seeds first"
            )
        acq = build_acq(a.method, a.seed)
        cands = [str(s) for s in seqs[cand_idx]]
        pick_local = np.asarray(acq.select(bank, cands, a.select))
        extra = {"n_informative_": getattr(acq, "n_informative_", None)}
        # Coverage rules record what they actually achieved per stratum. Without it the
        # claim "this corpus covers the activity range" rests on the rule's intent rather
        # than on what the candidate pool allowed it to do.
        for k in ("realised_per_stratum_", "n_short_strata_"):
            if getattr(acq, k, None) is not None:
                extra[k] = getattr(acq, k)
    pick = cand_idx[pick_local]
    print(f"selected {len(pick):,} in {time.time() - t0:.0f}s", flush=True)

    keep = np.flatnonzero(taken)
    new_idx = np.concatenate([keep, pick])
    out_seqs, out_y = seqs[new_idx], y[new_idx]
    if len(set(out_seqs.tolist())) != len(out_seqs):
        raise SystemExit("the acquired pool contains duplicate sequences")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # A mixed candidate pool records which reservoir each row came from; carry that column onto
    # the selection. Without it the realised mixture -- the whole point of handing a rule
    # several reservoirs at once -- has to be reconstructed afterwards by matching sequence
    # strings back to the pool, which only works while that pool still exists.
    src = {"source": np.asarray(z["source"])[new_idx]} if "source" in z.files else {}
    np.savez(
        a.out,
        sequences=out_seqs,
        oracle_labels=out_y,
        **src,
        strategy=str(z.get("strategy", "")),
        params=str(z.get("params", "")),
        seed=int(z["seed"]) if "seed" in z.files else a.seed,
        oracle_id=str(z.get("oracle_id", "oracle_v3")),
        oracle_dir=str(z.get("oracle_dir", "")),
        acq_method=a.method,
        acq_seed=a.seed,
        acq_from=os.path.basename(a.pool),
        acq_selected=int(len(pick)),
    )
    meta = {
        "method": a.method,
        "selected": int(len(pick)),
        "total": int(len(out_seqs)),
        "candidates_scored": int(len(cand_idx)),
        "bank": a.bank,
        "bank_members": None if a.method == "random" else len(bank),
        "seconds": round(time.time() - t0, 1),
        "label_mean_selected": round(float(np.mean(y[pick])), 4),
        "label_mean_pool": round(float(np.mean(y)), 4),
        **extra,
    }
    with open(a.out.replace(".npz", "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"wrote {a.out} ({len(out_seqs):,} sequences)")


if __name__ == "__main__":
    main()
