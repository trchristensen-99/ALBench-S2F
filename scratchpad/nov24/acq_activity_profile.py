"""What activity does each acquisition rule actually buy, and is its uncertainty just activity?

Arm B's epistemic arm selected an oracle label mean of 2.38 against a pool mean of 1.21 on
motif_shared_core, and 1.94 against 0.785 on motif_ct_enriched. That is a large activity
shift, and it has an uninteresting explanation: activities are log2FC, and in regression
the absolute error grows with the magnitude of the prediction, so the spread across
ensemble members grows with predicted activity whether or not the sequence is genuinely
ambiguous. Ranking by raw ensemble spread then partly re-ranks by activity, and
"epistemic beats random" would partly mean "active beats random".

This separates them on the real pool, with no training:

  1. the trend itself -- uncertainty against predicted activity, by decile, and the
     correlation. If it is flat, the concern is unfounded and nothing below matters.
  2. what each rule SELECTS -- the oracle-label distribution of the top 30k under
     epistemic, activity-normalised epistemic, expected activity, diversity and random.
  3. the overlap between rules. If epistemic and expected_activity pick largely the same
     sequences, they are one rule with two names.

Normalising is the correction asked for on Nov 24: score each candidate by how far its
uncertainty sits ABOVE the trend at its own activity level, so what is selected is
"unusually uncertain FOR ITS ACTIVITY" rather than "active".

    python scratchpad/nov24/acq_activity_profile.py \
        --pool outputs/grid_pools/k562_nested/motif_shared_core_d1000000.npz \
        --bank outputs/wed41/bench38_ln_e500_wd2_30k_c20_motif_shared_core \
        --out outputs/acq43/activity_profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--seed-size", type=int, default=30_000)
    ap.add_argument("--select", type=int, default=30_000)
    ap.add_argument("--cap", type=int, default=200_000, help="candidates scored, as run_al does")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # NOTE: two classes share this name. albench.ensemble.bank.LegNetBank is the plain
    # predictor used to GUIDE generation; this one is the SequenceModel the acquisition
    # rules expect, with predict() and uncertainty(). Importing the wrong one fails late,
    # inside a rule, with a missing-attribute error that reads like a rule bug.
    from albench.acquisition.legnet_bank import LegNetBank

    z = np.load(a.pool, allow_pickle=True)
    seqs = [str(s) for s in z["sequences"]]
    y = np.asarray(z["oracle_labels"], dtype=float)

    # Candidates are everything past the seed prefix, subsampled exactly as run_al does,
    # so these numbers describe the selection the AL arms actually made.
    rng = np.random.default_rng(0)
    cand = np.arange(a.seed_size, len(seqs))
    if len(cand) > a.cap:
        cand = np.sort(rng.choice(cand, a.cap, replace=False))
    cs = [seqs[i] for i in cand]
    cy = y[cand]
    print(f"{len(cs):,} candidates from {Path(a.pool).name}", flush=True)

    student = LegNetBank(a.bank)
    pred = np.asarray(student.predict(cs), dtype=float).ravel()
    unc = np.asarray(student.uncertainty(cs), dtype=float).ravel()
    print(f"scored with {a.bank}", flush=True)

    res: dict = {
        "pool": Path(a.pool).name,
        "bank": Path(a.bank).name,
        "n_candidates": len(cs),
        "pool_label_mean": round(float(cy.mean()), 4),
    }

    # --- 1. is uncertainty just activity? -------------------------------------------
    order = np.argsort(pred)
    deciles = np.array_split(order, 10)
    res["trend"] = {
        "pearson_unc_vs_pred": round(float(np.corrcoef(pred, unc)[0, 1]), 4),
        "spearman_unc_vs_pred": round(
            float(
                np.corrcoef(
                    np.argsort(np.argsort(pred)).astype(float),
                    np.argsort(np.argsort(unc)).astype(float),
                )[0, 1]
            ),
            4,
        ),
        "by_decile": [
            {
                "decile": i + 1,
                "pred_mean": round(float(pred[d].mean()), 4),
                "unc_mean": round(float(unc[d].mean()), 4),
            }
            for i, d in enumerate(deciles)
        ],
    }

    # --- 2. what does each rule select? ----------------------------------------------
    from albench.acquisition.diversity import DiversityAcquisition
    from albench.acquisition.ensemble_acq import ExpectedActivityAcquisition
    from albench.acquisition.random_acq import RandomAcquisition
    from albench.acquisition.uncertainty import UncertaintyAcquisition
    from albench.acquisition.uncertainty_variants import (
        ActivityNormalisedUncertaintyAcquisition,
    )

    rules = {
        "random": RandomAcquisition(seed=0),
        "epistemic": UncertaintyAcquisition(),
        "activity_normalized": ActivityNormalisedUncertaintyAcquisition(seed=0),
        "expected_activity": ExpectedActivityAcquisition(seed=0),
        "diversity_kmeans": DiversityAcquisition(seed=0),
    }
    picks: dict[str, np.ndarray] = {}
    res["selected"] = {}
    for name, rule in rules.items():
        try:
            idx = np.asarray(rule.select(student, cs, a.select), dtype=int)
        except Exception as e:  # a rule that cannot run should say so, not vanish
            res["selected"][name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {name}: FAILED {e}", flush=True)
            continue
        picks[name] = idx
        sel = cy[idx]
        res["selected"][name] = {
            "label_mean": round(float(sel.mean()), 4),
            "label_sd": round(float(sel.std()), 4),
            # The decile profile matters more than the mean: a rule can match the pool
            # mean while buying only the two tails, which is a different corpus.
            "label_deciles": [
                round(float(q), 3) for q in np.quantile(sel, np.arange(0, 1.01, 0.1))
            ],
            "pred_mean": round(float(pred[idx].mean()), 4),
            "unc_mean": round(float(unc[idx].mean()), 4),
            "frac_above_pool_mean": round(float((sel > cy.mean()).mean()), 4),
        }
        print(
            "  %-20s label %.3f (pool %.3f)  unc %.4f  frac>mean %.2f"
            % (name, sel.mean(), cy.mean(), unc[idx].mean(), (sel > cy.mean()).mean()),
            flush=True,
        )

    # --- 3. are any two rules the same rule? -----------------------------------------
    res["overlap"] = {}
    names = sorted(picks)
    for i, A in enumerate(names):
        for B in names[i + 1 :]:
            j = len(np.intersect1d(picks[A], picks[B])) / a.select
            res["overlap"][f"{A}|{B}"] = round(float(j), 4)
    print("\noverlap (fraction of the 30k shared):")
    for k, v in sorted(res["overlap"].items(), key=lambda kv: -kv[1]):
        print(f"   {k:46s} {v:.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
