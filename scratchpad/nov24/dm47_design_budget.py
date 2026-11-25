"""Why do SA and AdaLead reach lower activity than FastSeqProp, and how should we compare them?

THE HYPOTHESIS. The canary did not give them comparable budgets, and the budgets differ in
a way that is easy to miss because the knobs are named differently.

  FastSeqProp  150 steps, and EVERY step updates all 200 positions at once through the
               gradient. Its edit capacity is effectively unbounded -- it can rewrite the
               whole sequence many times over.
  SA (canary)  60 steps x 1 mutation = at most 60 single-base changes, of which only the
               accepted ones stick. Realistically ~30 of 200 bases ever changed.
  AdaLead      4 rounds, mutation_rate 0.02 -> 4 bases per child, rollout 2. The accepted
               path changes at most ~48 bases.

So FastSeqProp had roughly 10-100x the edit capacity, and "lower activity" is a statement
about budget rather than about the method. Gosai ran simulated annealing for thousands of
steps; nobody runs it for 60.

THE TEST. Compare the methods along a BUDGET LADDER rather than at one arbitrary setting,
with budget measured in model evaluations per design -- the quantity that actually costs
GPU, and the only one commensurable across a gradient method, a walk and an evolutionary
search. Three things get read off it:

  1. the activity each method reaches at equal compute (the curve, not one point);
  2. whether each method has CONVERGED, which is what makes a comparison fair at all;
  3. the guide-oracle gap as a function of budget. This is the interesting one. If the gap
     grows with budget for FastSeqProp but stays flat for SA and AdaLead, then gradient
     design buys its extra activity by travelling further into the guide's adversarial
     region, and the right recommendation depends on budget rather than on method.

Everything is scored by the INDEPENDENT oracle as well as the guide, because the guide
grading its own optimisation is not a result.

    python scratchpad/nov24/dm47_design_budget.py --n 512 --out outputs/design47/ladder.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# (label, strategy, kwargs, evals per design per unit of the ladder knob)
# `evals` is counted as forward passes of ONE bank member on ONE strand, which is the unit
# the throughput number (10,957 seq/s on an H100) is quoted in.
LADDER = [
    ("fastseqprop", "fastseqprop", [{"n_steps": s} for s in (50, 150, 500)], "n_steps"),
    ("sim_anneal", "sim_anneal", [{"n_steps": s} for s in (300, 1000, 3000)], "n_steps"),
    ("adalead", "adalead", [{"n_rounds": r} for r in (5, 15, 40)], "n_rounds"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="designs per configuration")
    ap.add_argument("--bank", default="outputs/wed41/bench38_ln_e500_wd2_30k_c20")
    ap.add_argument("--members", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from exp1_1_scaling import _load_oracle

    from albench.ensemble.bank import LegNetBank
    from albench.paths import resolve
    from albench.registry import get
    from scripts.generate_labeled_pools import _label_sequences

    bank = LegNetBank(a.bank, max_members=a.members)
    z = np.load(resolve("bg_cache"), allow_pickle=True)
    pool = [str(s) for s in z["sequences"][:20000]]
    orc = _load_oracle("k562", oracle_type="ag_s2")

    starts = pool[: a.n]
    y_start = _label_sequences(orc, starts)
    rows = []
    print(f"oracle on {a.n} starting CREs: {float(np.mean(y_start)):.3f}", flush=True)

    for label, strat, settings, knob in LADDER:
        for kw in settings:
            t0 = time.time()
            s = get(strat).build(seed=0, **kw)
            designs, dm = s.generate(a.n, base_sequences=pool, task="k562", student_model=bank)
            secs = time.time() - t0
            guide = float(dm["guide_pred_final"].to_numpy().mean())
            y = _label_sequences(orc, designs)
            uniq = len(set(designs))
            # Mean Hamming distance from the sequence each design started from is the
            # honest measure of edit capacity, and it is the quantity the hypothesis is
            # about -- it should separate the methods far more than the step counts do.
            row = {
                "method": label,
                knob: kw[knob],
                "seconds": round(secs, 1),
                "guide_mean": round(guide, 4),
                "oracle_mean": round(float(np.mean(y)), 4),
                "gap": round(guide - float(np.mean(y)), 4),
                "oracle_p90": round(float(np.quantile(y, 0.9)), 4),
                "unique": uniq,
                "unique_frac": round(uniq / a.n, 4),
            }
            rows.append(row)
            print(
                "  %-12s %-10s guide %7.3f | oracle %7.3f | gap %+7.3f | p90 %6.3f | "
                "uniq %.2f | %.0fs"
                % (
                    label,
                    f"{knob}={kw[knob]}",
                    row["guide_mean"],
                    row["oracle_mean"],
                    row["gap"],
                    row["oracle_p90"],
                    row["unique_frac"],
                    secs,
                ),
                flush=True,
            )

    out = {
        "n_designs": a.n,
        "bank": a.bank,
        "members": a.members,
        "oracle_on_starts": round(float(np.mean(y_start)), 4),
        "rows": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
