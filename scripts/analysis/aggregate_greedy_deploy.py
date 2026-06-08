"""Step-2 prep, stage 2: aggregate the 9 per-pool greedy_deploy.json curves
(3 reservoirs x 3 seeds, D=30k) into ONE global N* + a frozen deploy RECIPE.

Inputs: outputs/hp_step1_bakeoff/<reservoir>_d30000/seed*/ablation/greedy_deploy.json
written by greedy_deploy_select.py. Each holds a per-n curve of held-out
test_oracle_pearson grown by greedy per-MODEL forward selection.

Two outputs, mirroring the locked two-step design:
  - global N* (HOW MANY distinct configs the deploy pool holds): absolute oracle r
    differs by reservoir, so we normalise each pool's curve to its own attainable
    gain, average across pools, and take the knee of that mean fraction-of-gain
    curve. We also report the median of the per-pool integer knees as a check.
  - RECIPE (WHICH strategies are worth keeping): vote each strategy that appears
    at/before its pool's knee; keep those reaching a majority of pools, ranked by
    votes then mean pick-position.

The per-pool chosen model ids are pool-specific and are NOT frozen here -- at
deploy time the N* configs are re-selected per (reservoir x acquisition x D).
What we freeze is the COUNT (N*) and the strategy recipe.

Read-only over JSON; safe on the login node.
"""

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")


def knee_idx(ys, frac=0.90):
    """Smallest index whose y reaches `frac` of attainable gain over y[0]."""
    ys = np.asarray(ys, dtype=float)
    y0 = ys[0]
    gain = float(ys.max() - y0)
    if gain <= 1e-6:
        return 0
    thresh = y0 + frac * gain
    for i, y in enumerate(ys):
        if y >= thresh:
            return i
    return len(ys) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="30000", help="D tier to aggregate")
    ap.add_argument("--frac", type=float, default=0.90)
    ap.add_argument("--majority", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(
        glob.glob(
            str(REPO / f"outputs/hp_step1_bakeoff/*_d{args.d}/seed*/ablation/greedy_deploy.json")
        )
    )
    if not files:
        raise SystemExit(f"no greedy_deploy.json found for D={args.d}")

    per_pool = []
    recipe_votes = Counter()
    recipe_positions = defaultdict(list)
    # collect normalised fraction-of-gain curves on a common n-grid
    frac_curves = []  # each: (ns, fracs)
    pool_knees = []

    for f in files:
        d = json.load(open(f))
        curve = d["curve"]
        ns = [c["n"] for c in curve]
        rs = [c["test_oracle_pearson"] for c in curve]
        r0, rmax = rs[0], max(rs)
        gain = rmax - r0
        ki = knee_idx(rs, args.frac)
        knee_n = ns[ki]
        pool_knees.append(knee_n)
        frac_curves.append(
            (np.asarray(ns), (np.asarray(rs) - r0) / gain if gain > 1e-6 else np.ones(len(rs)))
        )

        # recipe votes: strategies added at/before this pool's knee
        for pos, c in enumerate(curve[: ki + 1], start=1):
            s = c["added_strategy"]
            recipe_votes[s] += 1
            recipe_positions[s].append(pos)

        cell = Path(f).parts[-4]
        seed = Path(f).parts[-3]
        per_pool.append(
            {
                "pool": f"{cell}/{seed}",
                "n_models_valid": d.get("n_models_valid"),
                "knee_n": int(knee_n),
                "r_at_n1": round(r0, 4),
                "r_at_knee": round(rs[ki], 4),
                "r_max": round(rmax, 4),
                "gain": round(gain, 4),
                "strategies_to_knee": [c["added_strategy"] for c in curve[: ki + 1]],
            }
        )

    # mean fraction-of-gain curve on the shared n-grid (all pools share n=1..max_n)
    max_common = min(int(c[0].max()) for c in frac_curves)
    grid = np.arange(1, max_common + 1)
    stack = np.vstack([np.interp(grid, ns, fr) for ns, fr in frac_curves])
    mean_frac = stack.mean(axis=0)
    global_knee = int(grid[knee_idx(mean_frac, args.frac)])
    median_knee = int(np.median(pool_knees))

    n_pool = len(files)
    min_votes = int(np.ceil(args.majority * n_pool))
    recipe = sorted(
        [s for s, v in recipe_votes.items() if v >= min_votes],
        key=lambda s: (-recipe_votes[s], float(np.mean(recipe_positions[s]))),
    )

    out = {
        "d": args.d,
        "n_pools": n_pool,
        "method": {
            "knee_frac": args.frac,
            "recipe_majority": args.majority,
            "min_votes": min_votes,
            "selection_metric": "test_oracle_pearson",
            "search": "greedy per-model forward",
        },
        "global_nstar": global_knee,
        "median_pool_knee": median_knee,
        "pool_knees": pool_knees,
        "mean_fraction_of_gain_curve": [round(float(x), 4) for x in mean_frac],
        "recipe": recipe,
        "recipe_votes": dict(recipe_votes.most_common()),
        "recipe_mean_position": {
            s: round(float(np.mean(p)), 2) for s, p in recipe_positions.items()
        },
        "per_pool": per_pool,
    }
    out_path = (
        Path(args.out)
        if args.out
        else REPO / f"outputs/hp_step1_bakeoff/deploy_spec_d{args.d}.json"
    )
    out_path.write_text(json.dumps(out, indent=2))

    print(f"=== greedy deploy aggregation | D={args.d} | {n_pool} pools ===")
    print(f"global N* (mean-frac knee): {global_knee}   median per-pool knee: {median_knee}")
    print(f"per-pool knees: {pool_knees}")
    print(f"\nRECIPE (strategies in >= {min_votes}/{n_pool} pools at/before knee):")
    for s in recipe:
        print(
            f"  {s:18s} votes={recipe_votes[s]}/{n_pool}  mean_pos={np.mean(recipe_positions[s]):.2f}"
        )
    print("\n  full vote table:")
    for s, v in recipe_votes.most_common():
        print(f"    {s:18s} {v}")
    print("\nper-pool:")
    for p in per_pool:
        print(
            f"  {p['pool']:42s} knee={p['knee_n']:2d}  r {p['r_at_n1']:.3f}->{p['r_at_knee']:.3f} (max {p['r_max']:.3f})"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
