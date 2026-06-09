"""Step-2 prep: distill the frozen DEPLOY RECIPE + global N* from the completed
Step-1 bake-off ablation reports.

Two axes, two questions (per the locked two-step design):
  - Part B (recipe / WHICH strategies): forward_selection curve -> diminishing-returns
    knee -> the strategy set worth keeping. Aggregated across all (cell x seed) reports.
  - Part A (N* / HOW MANY models): budget_sweep curve (oracle Pearson vs # models) ->
    knee -> ONE global pool size N*, matched across all D for scaling-law fairness.

Selection metric is the ORACLE landscape (test_oracle_pearson), never test_true.

Read-only: consumes outputs/hp_step1_bakeoff/*_d{D}/seed*/ablation/ablation_report.json,
writes outputs/hp_step1_bakeoff/deploy_recipe_d{D}.json. No training, no GPU.

The knee rule is "smallest x reaching `frac` of the attainable gain over the curve's
own baseline (x=min)", with `frac` default 0.90 -- a stable, parameter-light stand-in
for Kneedle that behaves well on these near-immediately-plateauing curves.
"""

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")


def knee_index(xs, ys, frac=0.90):
    """Index of the smallest x whose y reaches `frac` of the attainable gain.

    attainable gain = max(y) - y[0]. If the curve is flat (gain ~ 0), knee = 0.
    Returns (idx, x_at_knee, attainable_gain).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    y0 = ys[0]
    gain = float(ys.max() - y0)
    if gain <= 1e-6:
        return 0, float(xs[0]), gain
    thresh = y0 + frac * gain
    for i, y in enumerate(ys):
        if y >= thresh:
            return i, float(xs[i]), gain
    return len(ys) - 1, float(xs[-1]), gain


def pick_budget_curve(budget_sweep):
    """Prefer the realistic multi-strategy curve for N* (all_strategies > mixed6),
    falling back to whatever exists. Returns (key, list-of-rows)."""
    for key in ("all_strategies", "mixed6"):
        if key in budget_sweep and budget_sweep[key]:
            return key, budget_sweep[key]
    # fall back to first non-random_only key, else random_only
    for key in budget_sweep:
        if key != "random_only" and budget_sweep[key]:
            return key, budget_sweep[key]
    return "random_only", budget_sweep.get("random_only", [])


def _assert_single_regime(report_files):
    """Refuse to aggregate reports spanning >1 training/eval regime.

    Each report lives at <cell>_d{D}/seed*/ablation/<name>.json; the per-cell HP
    run stamps regime.json in the seed dir. Mixing regimes (e.g. epochs=15 vs 100,
    or a different oracle) across an aggregation makes the recipe/N* meaningless."""
    regimes = {}
    for f in report_files:
        seed_dir = Path(f).parents[1]
        rp = seed_dir / "regime.json"
        regimes[str(seed_dir)] = rp.read_text() if rp.exists() else "MISSING"
    distinct = set(regimes.values())
    if len(distinct) > 1:
        lines = "\n".join(f"  {k}: {v[:120]}" for k, v in sorted(regimes.items()))
        raise SystemExit(
            f"Reports span {len(distinct)} distinct regimes — refusing to aggregate "
            f"across regimes:\n{lines}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ds", default="30000", help="comma-sep D tier(s) to pool, e.g. 30000 or 30000,300000"
    )
    ap.add_argument(
        "--frac", type=float, default=0.90, help="fraction-of-attainable-gain knee threshold"
    )
    ap.add_argument(
        "--majority",
        type=float,
        default=0.5,
        help="min fraction of reports a strategy must appear in (at/before its recipe knee) to enter the recipe",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds_list = [d.strip() for d in args.ds.split(",") if d.strip()]
    reports = []
    for d in ds_list:
        reports += sorted(
            glob.glob(
                str(REPO / f"outputs/hp_step1_bakeoff/*_d{d}/seed*/ablation/ablation_report.json")
            )
        )
    if not reports:
        raise SystemExit(f"no ablation_report.json found for D in {ds_list}")
    _assert_single_regime(reports)

    per_report = []
    recipe_votes = Counter()  # strategy -> # reports where it's at/before knee
    recipe_positions = defaultdict(list)  # strategy -> list of 1-based positions
    nstar_knees = []
    budget_curves = []  # (budgets, pearsons) for aggregate

    for f in reports:
        d = json.load(open(f))
        cell = Path(f).parts[-4]
        seed = Path(f).parts[-3]

        # --- Part B: recipe knee on forward_selection ---
        fsel = d["forward_selection"]
        ks = [r["k"] for r in fsel]
        ps = [r["test_oracle_pearson"] for r in fsel]
        kidx, k_at_knee, fgain = knee_index(ks, ps, args.frac)
        recipe_at_knee = fsel[kidx]["chosen"]
        for pos, s in enumerate(recipe_at_knee, start=1):
            recipe_votes[s] += 1
            recipe_positions[s].append(pos)

        # --- Part A: N* knee on budget_sweep ---
        bkey, brows = pick_budget_curve(d["budget_sweep"])
        budgets = [r["n_total_budget"] for r in brows]
        bpears = [r["test_oracle_pearson"] for r in brows]
        bidx, n_at_knee, bgain = knee_index(budgets, bpears, args.frac)
        nstar_knees.append(n_at_knee)
        budget_curves.append((budgets, bpears))

        per_report.append(
            {
                "cell": cell,
                "seed": seed,
                "recipe_knee_k": int(k_at_knee),
                "recipe_at_knee": recipe_at_knee,
                "forward_gain": round(fgain, 4),
                "budget_curve_key": bkey,
                "nstar_knee": int(n_at_knee),
                "budget_gain": round(bgain, 4),
            }
        )

    n_rep = len(reports)
    min_votes = math.ceil(args.majority * n_rep)
    recipe = sorted(
        [s for s, v in recipe_votes.items() if v >= min_votes],
        key=lambda s: (-recipe_votes[s], np.mean(recipe_positions[s])),
    )

    # global N*: median of per-report knees, rounded to nearest tested budget grid value
    nstar_median = float(np.median(nstar_knees))
    global_nstar = int(round(nstar_median))

    out = {
        "ds": ds_list,
        "n_reports": n_rep,
        "method": {
            "knee_frac": args.frac,
            "recipe_majority": args.majority,
            "min_votes": min_votes,
            "selection_metric": "test_oracle_pearson",
        },
        "recipe": recipe,
        "recipe_votes": dict(recipe_votes.most_common()),
        "recipe_mean_position": {
            s: round(float(np.mean(p)), 2) for s, p in sorted(recipe_positions.items())
        },
        "global_nstar": global_nstar,
        "nstar_knees": nstar_knees,
        "nstar_median": nstar_median,
        "per_report": per_report,
        "note": "PROVISIONAL until D=300k bake-off completes; re-run with --ds 30000,300000 then.",
    }

    out_path = (
        Path(args.out)
        if args.out
        else REPO / f"outputs/hp_step1_bakeoff/deploy_recipe_d{'_'.join(ds_list)}.json"
    )
    out_path.write_text(json.dumps(out, indent=2))

    # --- console summary ---
    print(f"=== Deploy-recipe determination | D={ds_list} | {n_rep} reports ===")
    print(
        f"\nPart B — frozen RECIPE (strategies in >= {min_votes}/{n_rep} reports at/before knee):"
    )
    for s in recipe:
        print(
            f"  {s:20s} votes={recipe_votes[s]}/{n_rep}  mean_pos={np.mean(recipe_positions[s]):.2f}"
        )
    print("\n  full vote table:")
    for s, v in recipe_votes.most_common():
        print(f"    {s:20s} {v}/{n_rep}")
    print(
        f"\nPart A — global N* (deploy pool size): {global_nstar}  (median of {[int(x) for x in nstar_knees]})"
    )
    print("\nper-report knees:")
    for r in per_report:
        print(
            f"  {r['cell']:30s} {r['seed']:10s} recipe_k={r['recipe_knee_k']:2d} "
            f"({'+'.join(r['recipe_at_knee'])})  N*_knee={r['nstar_knee']} [{r['budget_curve_key']}]"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
