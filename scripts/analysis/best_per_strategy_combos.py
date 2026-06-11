"""All-possible-combinations deploy analysis over the SINGLE BEST model per HP strategy.

Companion to greedy_deploy_select.py. Where greedy grows an ElasticNet ensemble one
config at a time over ALL ~600 search intermediates, this reduces each pool to one atom
per strategy -- the model with the highest solo val-oracle Pearson within that strategy
-- giving <=14 atoms, then EXHAUSTIVELY enumerates all 2^k - 1 non-empty subsets:
  - stack each subset with ElasticNetCV(positive=True) on validation predictions,
  - select the winning subset by VAL fit (val_mse), so test_oracle stays a clean
    held-out score (no selection leakage),
  - record the best-by-val subset at every subset size -> the size knee, plus the
    global winner and the top-K subsets.

Same fit + oracle-landscape metric as the greedy/recipe analyses, so the two N* views
(greedy-over-all vs best-per-strategy) are directly comparable.

Two modes:
  --pool_dir <pool>     run one pool, write <pool>/ablation/best_per_strategy_combos.json
  --aggregate_d <D[,D]> glob all pools for those D tiers, write a cross-pool summary

CPU/BLAS-bound (thousands of small ElasticNet fits) -> run via srun/sbatch on cpuq.
"""

import argparse
import glob
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from greedy_deploy_select import knee_n, load_pool_models
from scipy.stats import pearsonr

from albench.ensemble import fit_elasticnet_stack

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")


def fit_stack(val_X, val_y, test_X):
    return fit_elasticnet_stack(val_X, val_y, test_X)


def best_per_strategy(models, val_y):
    """One atom per strategy: highest solo val-oracle Pearson."""
    best: dict[str, dict] = {}
    for mdl in models:
        mm = np.isfinite(mdl["val"]) & np.isfinite(val_y)
        mdl["solo_val_r"] = float(pearsonr(mdl["val"][mm], val_y[mm])[0]) if mm.sum() > 3 else -1.0
        s = mdl["strategy"]
        if s not in best or mdl["solo_val_r"] > best[s]["solo_val_r"]:
            best[s] = mdl
    return sorted(best.values(), key=lambda m: -m["solo_val_r"])


def run_pool(pool_dir: str, out_dir: str | None) -> dict:
    models, labels = load_pool_models(pool_dir)
    val_y, oracle = labels["val_labels"], labels["test_oracle"]
    atoms = best_per_strategy(models, val_y)
    k = len(atoms)
    if k < 2:
        raise SystemExit(f"only {k} strategy atoms in {pool_dir}")

    best_by_size: dict[int, dict] = {}
    all_rows: list[dict] = []
    for size in range(1, k + 1):
        for combo in itertools.combinations(range(k), size):
            vX = np.vstack([atoms[i]["val"] for i in combo])
            tX = np.vstack([atoms[i]["test"] for i in combo])
            vpred, tpred = fit_stack(vX, val_y, tX)
            row = {
                "size": size,
                "strategies": [atoms[i]["strategy"] for i in combo],
                "val_mse": round(float(np.mean((vpred - val_y) ** 2)), 6),
                "test_oracle_pearson": round(float(pearsonr(tpred, oracle)[0]), 5),
                "test_oracle_mse": round(float(np.mean((tpred - oracle) ** 2)), 6),
            }
            all_rows.append(row)
            if size not in best_by_size or row["val_mse"] < best_by_size[size]["val_mse"]:
                best_by_size[size] = row

    size_curve = [
        {
            "n": s,
            **{
                kk: best_by_size[s][kk]
                for kk in ("strategies", "val_mse", "test_oracle_pearson", "test_oracle_mse")
            },
        }
        for s in sorted(best_by_size)
    ]
    nstar = knee_n(size_curve, 0.90)
    global_best = min(all_rows, key=lambda r: r["val_mse"])
    top = sorted(all_rows, key=lambda r: r["val_mse"])[:20]

    out = {
        "pool_dir": str(pool_dir),
        "n_models_valid": len(models),
        "n_strategy_atoms": k,
        "atoms": [
            {"strategy": a["strategy"], "id": a["id"], "solo_val_r": round(a["solo_val_r"], 5)}
            for a in atoms
        ],
        "n_subsets_evaluated": len(all_rows),
        "best_by_val": global_best,
        "size_knee": nstar,
        "size_curve_best_by_val": size_curve,
        "top20_by_val": top,
    }
    od = Path(out_dir) if out_dir else Path(pool_dir) / "ablation"
    od.mkdir(parents=True, exist_ok=True)
    (od / "best_per_strategy_combos.json").write_text(json.dumps(out, indent=2))

    print(f"=== best-per-strategy combos | {pool_dir} | {k} atoms, {len(all_rows)} subsets ===")
    print(f"size knee (90% of gain): {nstar}")
    print(
        f"best subset by val (size {global_best['size']}): "
        f"oracle_r={global_best['test_oracle_pearson']:.4f} "
        f"mse={global_best['test_oracle_mse']:.4f}  {global_best['strategies']}"
    )
    for c in size_curve:
        mark = " <- knee" if c["n"] == nstar else ""
        print(
            f"  size={c['n']:2d}  oracle_r={c['test_oracle_pearson']:.4f} "
            f"mse={c['test_oracle_mse']:.4f}{mark}"
        )
    print(f"wrote {od / 'best_per_strategy_combos.json'}")
    return out


def aggregate(ds: str) -> None:
    ds_list = [x.strip() for x in ds.split(",") if x.strip()]
    files: list[str] = []
    for dd in ds_list:
        files += sorted(
            glob.glob(
                str(
                    REPO
                    / f"outputs/hp_step1_bakeoff/*_d{dd}/seed*/ablation/best_per_strategy_combos.json"
                )
            )
        )
    if not files:
        raise SystemExit(f"no best_per_strategy_combos.json for D in {ds_list}")

    knees, strat_votes, per_pool = [], Counter(), []
    for f in files:
        d = json.loads(Path(f).read_text())
        knees.append(d["size_knee"])
        for s in d["best_by_val"]["strategies"]:
            strat_votes[s] += 1
        cell = Path(f).parts[-4]
        seed = Path(f).parts[-3]
        per_pool.append(
            {
                "pool": f"{cell}/{seed}",
                "n_atoms": d["n_strategy_atoms"],
                "size_knee": d["size_knee"],
                "best_size": d["best_by_val"]["size"],
                "best_oracle_r": d["best_by_val"]["test_oracle_pearson"],
                "best_strategies": d["best_by_val"]["strategies"],
            }
        )

    n_pool = len(files)
    out = {
        "d": ds_list,
        "n_pools": n_pool,
        "method": "best-model-per-strategy, exhaustive all-subsets ElasticNet, select on val",
        "median_size_knee": int(np.median(knees)),
        "size_knees": knees,
        "strategy_votes_in_best_subset": dict(strat_votes.most_common()),
        "per_pool": per_pool,
    }
    out_path = REPO / f"outputs/hp_step1_bakeoff/best_per_strategy_spec_d{'_'.join(ds_list)}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"=== best-per-strategy aggregate | D={ds_list} | {n_pool} pools ===")
    print(f"median size knee: {out['median_size_knee']}   knees: {knees}")
    print("strategies appearing in each pool's best subset:")
    for s, v in strat_votes.most_common():
        print(f"  {s:18s} {v}/{n_pool}")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--aggregate_d", default=None, help="comma-sep D tiers to aggregate")
    args = ap.parse_args()
    if args.aggregate_d:
        aggregate(args.aggregate_d)
    elif args.pool_dir:
        run_pool(args.pool_dir, args.out_dir)
    else:
        raise SystemExit("pass --pool_dir <pool> or --aggregate_d <D[,D]>")


if __name__ == "__main__":
    main()
