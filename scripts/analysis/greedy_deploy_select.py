"""Step-2 N* determination + deploy-config selection via GREEDY per-model forward
selection on the oracle landscape.

Exhaustive subset search over individual models is intractable (C(~600, N) subsets),
so we greedily grow an ElasticNet-stacked ensemble one CONFIG at a time:
  - candidate criterion = improvement in VAL oracle fit (val_labels), so test_oracle
    stays a clean held-out curve (no selection leakage).
  - distinct configs only (no replacement): the positive ElasticNet zeroes collinear
    duplicates anyway, so re-picking a config can never help.
  - optional HP/arch DIVERSITY guard: skip a candidate whose (block_class, optimizer,
    depth-band, width-band, lr-band) signature is already represented, so the pool
    spreads across architecture space rather than stacking near-twins.

The per-step held-out test_oracle curve plateaus once distinct configs stop adding
complementary signal -> that knee is the empirical N*. Run per (cell x seed x reservoir)
pool; aggregate the curves across pools to pick ONE global N*.

Reuses the pool loader contract from strategy_combination_ablation.py: each pool dir has
*/r*_meta.json + sibling .npz (val_pred, test_pred) and a labels.npz
(val_labels, test_oracle[, test_true]).

CPU/BLAS-bound (many small ElasticNet fits) -> run via srun/sbatch on cpuq, never the
login node.
"""

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")


def lr_band(lr):
    if lr is None:
        return "na"
    if lr < 3e-4:
        return "lo"
    if lr < 8e-4:
        return "mid"
    return "hi"


def width_band(w):
    if not w:
        return "na"
    return "w%d" % (int(w) // 128)


def depth_band(n):
    if not n:
        return "na"
    if n <= 5:
        return "shallow"
    if n <= 8:
        return "mid"
    return "deep"


def hp_signature(hp):
    return (
        hp.get("block_class", "na"),
        hp.get("optimizer", "na"),
        depth_band(hp.get("n_layers")),
        width_band(hp.get("width_base")),
        lr_band(hp.get("lr")),
    )


def load_pool_models(pool_dir):
    """Return (models, labels). models = list of dicts {id, strategy, hp, sig, val, test}."""
    pool_dir = Path(pool_dir)
    metas = sorted(pool_dir.glob("*/r*_meta.json")) or sorted(pool_dir.glob("**/r*_meta.json"))
    if not metas:
        raise SystemExit(f"no *_meta.json under {pool_dir}")
    lf = sorted(pool_dir.glob("labels.npz")) or sorted(pool_dir.glob("**/labels.npz"))
    if not lf:
        raise SystemExit(f"no labels.npz under {pool_dir}")
    lz = np.load(lf[0])
    labels = {
        "val_labels": lz["val_labels"].astype(np.float64),
        "test_oracle": lz["test_oracle"].astype(np.float64),
        "test_true": (lz["test_true"] if "test_true" in lz.files else lz["test_oracle"]).astype(
            np.float64
        ),
    }
    n_val = labels["val_labels"].shape[0]
    n_test = labels["test_oracle"].shape[0]
    models = []
    for m in metas:
        npz = m.with_name(m.name.replace("_meta.json", ".npz"))
        if not npz.exists():
            continue
        try:
            meta = json.loads(m.read_text())
            if "error" in meta and meta["error"]:
                continue
            d = np.load(npz)
            vp, tp = d["val_pred"], d["test_pred"]
        except Exception:
            continue
        if vp.shape[0] != n_val or tp.shape[0] != n_test:
            continue
        if not (np.isfinite(vp).all() and np.isfinite(tp).all()):
            continue
        hp = meta.get("hp", {})
        models.append(
            {
                "id": meta.get("model_id", npz.stem),
                "strategy": meta.get("strategy", "na"),
                "hp": hp,
                "sig": hp_signature(hp),
                "val": vp.astype(np.float64),
                "test": tp.astype(np.float64),
            }
        )
    return models, labels


def fit_stack(val_X, val_y, test_X):
    """ElasticNetCV(positive=True) on stacked val preds. val_X/test_X: (k, n)."""
    enet = ElasticNetCV(positive=True, cv=5, n_alphas=50, max_iter=5000, n_jobs=1)
    enet.fit(val_X.T, val_y)
    return enet.predict(val_X.T), enet.predict(test_X.T)


def greedy_select(models, labels, max_n, prefilter, diversity, rng):
    val_y = labels["val_labels"]
    oracle = labels["test_oracle"]

    # individual val-oracle pearson, for prefilter + ranking the first pick
    for mdl in models:
        mm = np.isfinite(mdl["val"]) & np.isfinite(val_y)
        mdl["solo_val_r"] = pearsonr(mdl["val"][mm], val_y[mm])[0] if mm.sum() > 3 else -1.0
    pool = sorted(models, key=lambda x: -x["solo_val_r"])
    if prefilter and len(pool) > prefilter:
        pool = pool[:prefilter]

    chosen = []
    chosen_sigs = set()
    curve = []
    remaining = list(range(len(pool)))

    while len(chosen) < max_n and remaining:
        best_j, best_val_mse, best_test = None, np.inf, None
        for j in remaining:
            cand = pool[j]
            if diversity and cand["sig"] in chosen_sigs and len(chosen) > 0:
                continue
            vX = np.vstack([pool[i]["val"] for i in chosen] + [cand["val"]])
            tX = np.vstack([pool[i]["test"] for i in chosen] + [cand["test"]])
            vpred, tpred = fit_stack(vX, val_y, tX)
            vmse = float(np.mean((vpred - val_y) ** 2))
            if vmse < best_val_mse:
                best_val_mse, best_j, best_test = vmse, j, tpred
        if best_j is None:  # diversity exhausted all remaining sigs
            break
        chosen.append(best_j)
        chosen_sigs.add(pool[best_j]["sig"])
        remaining.remove(best_j)
        r = pearsonr(best_test, oracle)[0]
        mse = float(np.mean((best_test - oracle) ** 2))
        curve.append(
            {
                "n": len(chosen),
                "added_id": pool[best_j]["id"],
                "added_strategy": pool[best_j]["strategy"],
                "added_sig": list(pool[best_j]["sig"]),
                "val_mse": round(best_val_mse, 5),
                "test_oracle_pearson": round(float(r), 5),
                "test_oracle_mse": round(mse, 5),
            }
        )
    return curve, [pool[i] for i in chosen]


def knee_n(curve, frac=0.90):
    if not curve:
        return 0
    ys = np.array([c["test_oracle_pearson"] for c in curve])
    y0 = ys[0]
    gain = float(ys.max() - y0)
    if gain <= 1e-6:
        return 1
    thresh = y0 + frac * gain
    for c, y in zip(curve, ys):
        if y >= thresh:
            return c["n"]
    return curve[-1]["n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_n", type=int, default=20)
    ap.add_argument(
        "--prefilter",
        type=int,
        default=120,
        help="keep top-K candidates by solo val-oracle r before greedy (0=all)",
    )
    ap.add_argument(
        "--diversity",
        action="store_true",
        help="skip candidates whose HP signature is already represented",
    )
    ap.add_argument("--frac", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    models, labels = load_pool_models(args.pool_dir)
    if len(models) < 2:
        raise SystemExit(f"only {len(models)} valid models in {args.pool_dir}")
    curve, chosen = greedy_select(models, labels, args.max_n, args.prefilter, args.diversity, rng)
    nstar = knee_n(curve, args.frac)

    out = {
        "pool_dir": str(args.pool_dir),
        "n_models_valid": len(models),
        "diversity": args.diversity,
        "prefilter": args.prefilter,
        "knee_frac": args.frac,
        "nstar_knee": nstar,
        "curve": curve,
        "chosen_at_knee": [
            {"id": c["added_id"], "strategy": c["added_strategy"], "sig": c["added_sig"]}
            for c in curve[:nstar]
        ],
    }
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.pool_dir) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "greedy_deploy_div.json" if args.diversity else "greedy_deploy.json"
    (out_dir / name).write_text(json.dumps(out, indent=2))

    print(f"=== greedy deploy select | {args.pool_dir} | {len(models)} valid models ===")
    print(f"N* knee (frac={args.frac}): {nstar}")
    for c in curve:
        mark = " <- knee" if c["n"] == nstar else ""
        print(
            f"  n={c['n']:2d}  +{c['added_strategy']:18s} oracle_r={c['test_oracle_pearson']:.4f} "
            f"mse={c['test_oracle_mse']:.4f}{mark}"
        )
    print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
