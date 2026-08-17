"""Slope-experiment analysis (val-selected ensembles on saved predictions).

Data layout (outputs/overnight/slope_<R>_d<D>_s<ds>_<tag>):
  R    in {genomic, motif_planted_v2, dinuc_shuffle, evoaug_heavy, gc_matched}
  D    in {10000, 30000, 100000}
  ds   in {42, 43}
  tag  in {gp, tpe, evos, evob, lexp, ldiv}  (personas lexp/ldiv share
          fullstrat=llm_autoresearch -> distinguished by the dir TAG, not meta.strategy)

Per model:  <model_id>.npz  (val_pred, test_pred, test_pred_<set> ...)
            <model_id>_meta.json (val_pearson, val_mse, per_set_metrics, round,
                                  best_epoch, early_stopped, epochs_trained ...)
Once/dir:   labels.npz (val_labels, oracle_<set> = test truth per battery set)

VERIFIED alignment facts (drive the design):
  * Within one reservoir cell (R,D,ds), ALL strategy dirs share the SAME
    val_labels and the SAME oracle_* test labels  ==> val_pred / test_pred rows
    are aligned ACROSS STRATEGIES -> pooling strategies into one candidate pool
    and fitting one ElasticNet on val is valid.
  * ACROSS RESERVOIRS (same D,ds) the test oracle_* labels are IDENTICAL (shared
    battery), but val_labels DIFFER (per-combo 10% held-out of each reservoir's
    own train).  => you CANNOT stack val across reservoirs into one design
    matrix.  Cross-reservoir pooling is therefore done on the SHARED TEST set:
    each reservoir builds its own val-selected ensemble, and the subset ensemble
    is the uniform mean of the per-reservoir ensembles' TEST predictions.
  * oracle_* are even identical across ds (test set is deterministic; the seed
    only moves the train/val split + init).

Design invariants:
  - GREEDY ensembles are VAL-SELECTED. Selection maximises ElasticNet-on-val
    pearson; TEST metrics are only reported AFTER selection (never selected on).
  - The scaling SLOPE uses a MATCHED round budget across D (--round_budget,
    default 50) so the candidate pool size is comparable at every D (D=10k has
    ~150 rounds, others ~50).

No GPU / no new inference: everything is computed from saved predictions.
Cap BLAS threads for reproducible CPU timing: OMP_NUM_THREADS=1.
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

# n_alphas is deprecated in sklearn>=1.7 but still functional; silence the spam.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# ----------------------------------------------------------------------------
# constants / config
# ----------------------------------------------------------------------------
DEFAULT_BASE = "outputs/overnight"
ALL_RESERVOIRS = [
    "genomic",
    "motif_planted_v2",
    "dinuc_shuffle",
    "evoaug_heavy",
    "gc_matched",
]
K4_TAGS = ["gp", "tpe", "evos", "evob"]  # algorithmic strategies (LLM paused)
LLM_TAGS = ["lexp", "ldiv"]  # persona dirs (both fullstrat=llm_autoresearch)
TAG_STRAT = {
    "gp": "optuna_gp",
    "tpe": "optuna_tpe",
    "evos": "evo_single",
    "evob": "evo_batch",
    "lexp": "llm_autoresearch",  # explore persona
    "ldiv": "llm_autoresearch",  # diverse persona
}
DEFAULT_DS = [10000, 30000, 100000]
DEFAULT_SEEDS = [42, 43]
TARGET_SET = "genomic"  # the "reference" test set for the headline slope

# ElasticNetCV config (kept modest so the O(pool*size) greedy stays tractable
# on CPU with OMP_NUM_THREADS=1). All faithfully ElasticNetCV as required.
EN_KW = dict(
    l1_ratio=[0.5, 0.9, 1.0],
    positive=True,
    cv=3,
    n_alphas=25,
    max_iter=20000,
    n_jobs=1,
)


# ----------------------------------------------------------------------------
# data structures
# ----------------------------------------------------------------------------
class Model:
    __slots__ = (
        "model_id",
        "tag",
        "strat",
        "round",
        "val_pearson",
        "val_mse",
        "best_epoch",
        "early_stopped",
        "epochs_trained",
        "val_pred",
        "test",  # dict set -> np.ndarray
        "per_set",  # dict set -> {pearson,mse,n}
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class Cell:
    """One reservoir cell (R,D,ds): shared val + test labels + its models."""

    def __init__(self, R, D, ds, val_labels, oracle, models):
        self.R = R
        self.D = D
        self.ds = ds
        self.val_labels = val_labels
        self.oracle = oracle  # dict set -> truth
        self.models = models  # list[Model]

    def sets(self):
        return list(self.oracle.keys())


# ----------------------------------------------------------------------------
# loader / adapter
# ----------------------------------------------------------------------------
def _cell_dir(base, R, D, ds, tag):
    return os.path.join(base, f"slope_{R}_d{D}_s{ds}_{tag}")


def load_cell(base, R, D, ds, tags, round_budget=None, verbose=False):
    """Load & pool all models across the given strategy `tags` for one (R,D,ds).

    Returns a Cell, or None if no usable models were found.  Uses the shared
    labels.npz (identical across the pooled strategy dirs).  Applies the matched
    round budget (keep round < round_budget) if given.  Silently skips missing
    or empty strategy dirs and drops models with shape-mismatched / non-finite
    predictions.
    """
    val_labels = None
    oracle = None
    models = []
    quirks = []
    for tag in tags:
        d = _cell_dir(base, R, D, ds, tag)
        if not os.path.isdir(d):
            continue
        lab_p = os.path.join(d, "labels.npz")
        if not os.path.exists(lab_p):
            continue
        lab = np.load(lab_p, allow_pickle=True)
        vy = lab["val_labels"]
        ora = {k[len("oracle_"):]: lab[k] for k in lab.files if k.startswith("oracle_")}
        if val_labels is None:
            val_labels, oracle = vy, ora
        else:
            # sanity: pooled strategy dirs must share the same val + test labels
            if vy.shape != val_labels.shape or not np.array_equal(vy, val_labels):
                quirks.append(f"{R} d{D} s{ds} {tag}: val_labels mismatch across tags")
        metas = sorted(glob.glob(os.path.join(d, "*_meta.json")))
        for mp in metas:
            try:
                meta = json.load(open(mp))
            except Exception:
                continue
            rnd = int(meta.get("round", -1))
            if round_budget is not None and rnd >= round_budget:
                continue
            npz_p = mp.replace("_meta.json", ".npz")
            try:
                z = np.load(npz_p)
            except Exception:
                continue
            if "val_pred" not in z.files:
                continue
            vp = z["val_pred"]
            if vp.shape != val_labels.shape or not np.all(np.isfinite(vp)):
                continue
            test = {}
            ok = True
            for s, truth in oracle.items():
                key = f"test_pred_{s}"
                if key not in z.files:
                    ok = False
                    break
                tp = z[key]
                if tp.shape != truth.shape or not np.all(np.isfinite(tp)):
                    ok = False
                    break
                test[s] = tp
            if not ok:
                continue
            vpear = meta.get("best_val_pearson", meta.get("val_pearson"))
            if vpear is None or not np.isfinite(vpear):
                continue
            models.append(
                Model(
                    model_id=meta.get("model_id", os.path.basename(npz_p)[:-4]),
                    tag=tag,
                    strat=meta.get("strategy", TAG_STRAT.get(tag, tag)),
                    round=rnd,
                    val_pearson=float(vpear),
                    val_mse=float(meta.get("val_mse", np.nan)),
                    best_epoch=meta.get("best_epoch"),
                    early_stopped=meta.get("early_stopped"),
                    epochs_trained=meta.get("epochs_trained"),
                    val_pred=vp.astype(np.float64),
                    test={s: t.astype(np.float64) for s, t in test.items()},
                    per_set=meta.get("per_set_metrics", {}),
                )
            )
    if verbose and quirks:
        for q in quirks:
            print("  [quirk]", q)
    if val_labels is None or not models:
        return None
    return Cell(R, D, ds, val_labels.astype(np.float64), {s: o.astype(np.float64) for s, o in oracle.items()}, models)


# ----------------------------------------------------------------------------
# greedy VAL-selected ElasticNet ensemble
# ----------------------------------------------------------------------------
def _fit_en(V, y):
    en = ElasticNetCV(**EN_KW)
    en.fit(V, y)
    return en


class Ensemble:
    def __init__(self, cell, selected, en, curve):
        self.cell = cell
        self.selected = selected  # list[Model]
        self.en = en  # ElasticNetCV fit on val for the final selection
        self.curve = curve  # list of dicts (per greedy step)

    def predict(self, set_name):
        """Ensemble prediction on a test set (post-selection ElasticNet)."""
        T = np.column_stack([m.test[set_name] for m in self.selected])
        return self.en.predict(T)


def greedy_ensemble(cell, target_set=TARGET_SET, max_pool=40, max_size=12, min_delta=1e-4):
    """Forward-greedy VAL selection.

    At each step add the candidate that most improves the ElasticNet-on-VAL
    pearson (weights fit on val_labels, scored on val).  Test metrics on
    `target_set` are recorded AFTER each selection purely as the reported
    outcome -- selection never touches test.  Candidate pool is pre-capped to
    the top `max_pool` models by their own val_pearson for tractability.
    """
    vy = cell.val_labels
    truth = cell.oracle[target_set]
    pool = sorted(cell.models, key=lambda m: -m.val_pearson)[:max_pool]
    selected, curve = [], []
    best_val = -np.inf
    remaining = list(pool)
    while remaining and len(selected) < max_size:
        best = None
        for cand in remaining:
            trial = selected + [cand]
            V = np.column_stack([m.val_pred for m in trial])
            en = _fit_en(V, vy)
            vp = pearsonr(en.predict(V), vy)[0]
            if best is None or vp > best[0]:
                best = (vp, cand, en)
        vp, cand, en = best
        if vp <= best_val + min_delta and selected:
            break  # plateau: no meaningful val gain
        selected.append(cand)
        remaining.remove(cand)
        best_val = vp
        T = np.column_stack([m.test[target_set] for m in selected])
        pred = en.predict(T)
        curve.append(
            dict(
                size=len(selected),
                added=cand.model_id,
                added_tag=cand.tag,
                val_pearson=float(vp),
                test_pearson=float(pearsonr(pred, truth)[0]),
                test_mse=float(np.mean((pred - truth) ** 2)),
            )
        )
    # refit the final selection so .en corresponds exactly to `selected`
    if selected:
        Vf = np.column_stack([m.val_pred for m in selected])
        en = _fit_en(Vf, vy)
    else:
        en = None
    return Ensemble(cell, selected, en, curve)


# ----------------------------------------------------------------------------
# helpers for slope: build per-cell ensembles and pool across reservoirs
# ----------------------------------------------------------------------------
def build_cell_ensembles(base, reservoirs, D, seeds, tags, round_budget, target_set, max_pool, max_size, cache):
    """Return {reservoir: {ds: Ensemble}} for one D, using a load/build cache."""
    out = {}
    for R in reservoirs:
        out[R] = {}
        for ds in seeds:
            key = (R, D, ds)
            if key not in cache:
                cell = load_cell(base, R, D, ds, tags, round_budget=round_budget)
                ens = greedy_ensemble(cell, target_set, max_pool, max_size) if cell is not None else None
                cache[key] = ens
            if cache[key] is not None:
                out[R][ds] = cache[key]
    return out


def pooled_test_pred(ens_by_R_ds, reservoirs, seeds, target_set):
    """Uniform mean over reservoirs x seeds of each cell's ensemble test pred.

    Only cells that exist contribute. Returns (mean_pred, n_cells) or (None, 0).
    """
    preds = []
    for R in reservoirs:
        for ds in seeds:
            ens = ens_by_R_ds.get(R, {}).get(ds)
            if ens is not None:
                preds.append(ens.predict(target_set))
    if not preds:
        return None, 0
    return np.mean(preds, axis=0), len(preds)


# ----------------------------------------------------------------------------
# (3) PRIMARY: log-log MSE scaling slope, all-R vs leave-one-reservoir-out
# ----------------------------------------------------------------------------
def slope_analysis(base, reservoirs, Ds, seeds, tags, round_budget, target_set, max_pool, max_size):
    cache = {}
    # per-D pooled ensembles per reservoir
    perD = {}
    for D in Ds:
        perD[D] = build_cell_ensembles(base, reservoirs, D, seeds, tags, round_budget, target_set, max_pool, max_size, cache)

    def subset_slope(subset):
        pts = []  # (D, mse, pearson, n_cells)
        for D in Ds:
            truth = None
            # find any available oracle for this D (shared across reservoirs/seeds)
            for R in subset:
                for ds in seeds:
                    ens = perD[D].get(R, {}).get(ds)
                    if ens is not None:
                        truth = ens.cell.oracle[target_set]
                        break
                if truth is not None:
                    break
            if truth is None:
                continue
            pred, n = pooled_test_pred(perD[D], subset, seeds, target_set)
            if pred is None:
                continue
            mse = float(np.mean((pred - truth) ** 2))
            pear = float(pearsonr(pred, truth)[0])
            pts.append((D, mse, pear, n))
        slope = intercept = np.nan
        if len(pts) >= 2:
            lx = np.log(np.array([p[0] for p in pts], float))
            ly = np.log(np.array([p[1] for p in pts], float))
            slope, intercept = np.polyfit(lx, ly, 1)
        return dict(points=pts, slope=float(slope), intercept=float(intercept))

    results = {}
    results["all"] = subset_slope(reservoirs)
    for held in reservoirs:
        sub = [r for r in reservoirs if r != held]
        results[f"loro:-{held}"] = subset_slope(sub)
    return results


# ----------------------------------------------------------------------------
# (4a) rounds-to-plateau
# ----------------------------------------------------------------------------
def rounds_to_plateau(base, reservoirs, D, seeds, tags, round_budget, target_set, max_pool, max_size, round_grid):
    """Mean (over reservoirs x seeds) ensemble test pearson vs #rounds included."""
    # preload full cells once (up to round_budget), then rebuild ensembles per cutoff
    curve = []
    # cache loaded models per cell
    loaded = {}
    for R in reservoirs:
        for ds in seeds:
            c = load_cell(base, R, D, ds, tags, round_budget=round_budget)
            if c is not None:
                loaded[(R, ds)] = c
    for r in round_grid:
        pears = []
        for (R, ds), c in loaded.items():
            sub_models = [m for m in c.models if m.round < r]
            if not sub_models:
                continue
            sub_cell = Cell(c.R, c.D, c.ds, c.val_labels, c.oracle, sub_models)
            ens = greedy_ensemble(sub_cell, target_set, max_pool, max_size)
            if ens.selected:
                pred = ens.predict(target_set)
                pears.append(pearsonr(pred, c.oracle[target_set])[0])
        if pears:
            curve.append((r, float(np.mean(pears)), len(pears)))
    return curve


# ----------------------------------------------------------------------------
# (4b) ensemble-size knee (from val-selected greedy curve)
# ----------------------------------------------------------------------------
def size_knee(base, reservoirs, D, seeds, tags, round_budget, target_set, max_pool, max_size, knee_delta=0.002):
    """Mean greedy test-pearson curve vs size; report knee (first size whose
    marginal mean gain < knee_delta)."""
    curves = []
    for R in reservoirs:
        for ds in seeds:
            c = load_cell(base, R, D, ds, tags, round_budget=round_budget)
            if c is None:
                continue
            ens = greedy_ensemble(c, target_set, max_pool, max_size)
            if ens.curve:
                curves.append([step["test_pearson"] for step in ens.curve])
    if not curves:
        return [], None
    maxlen = max(len(c) for c in curves)
    mean_curve = []
    for i in range(maxlen):
        vals = [c[i] for c in curves if len(c) > i]
        mean_curve.append((i + 1, float(np.mean(vals)), len(vals)))
    knee = None
    for i in range(1, len(mean_curve)):
        if mean_curve[i][1] - mean_curve[i - 1][1] < knee_delta:
            knee = mean_curve[i - 1][0]
            break
    if knee is None and mean_curve:
        knee = mean_curve[-1][0]
    return mean_curve, knee


# ----------------------------------------------------------------------------
# (4c) val-vs-test overfitting gap (per reservoir, per-model)
# ----------------------------------------------------------------------------
def overfit_gap(base, reservoirs, D, seeds, tags, round_budget, gap_sets):
    """Per reservoir: mean(val_pearson) vs mean(per-model test pearson) on each
    set in gap_sets.  Uses per-model meta (best_val_pearson & per_set_metrics)."""
    rows = {}
    for R in reservoirs:
        vps, sets = [], {s: [] for s in gap_sets}
        for ds in seeds:
            c = load_cell(base, R, D, ds, tags, round_budget=round_budget)
            if c is None:
                continue
            for m in c.models:
                vps.append(m.val_pearson)
                for s in gap_sets:
                    if s in m.per_set and m.per_set[s].get("pearson") is not None:
                        sets[s].append(m.per_set[s]["pearson"])
        if not vps:
            continue
        vmean = float(np.mean(vps))
        row = {"n": len(vps), "val_pearson": vmean}
        for s in gap_sets:
            if sets[s]:
                tmean = float(np.mean(sets[s]))
                row[f"test_{s}"] = tmean
                row[f"gap_{s}"] = vmean - tmean
        rows[R] = row
    return rows


# ----------------------------------------------------------------------------
# (4d) epoch check
# ----------------------------------------------------------------------------
def epoch_check(base, reservoirs, Ds, seeds, tags, round_budget):
    be, et, es, n = [], [], 0, 0
    for R in reservoirs:
        for D in Ds:
            for ds in seeds:
                c = load_cell(base, R, D, ds, tags, round_budget=round_budget)
                if c is None:
                    continue
                for m in c.models:
                    n += 1
                    if m.best_epoch is not None:
                        be.append(m.best_epoch)
                    if m.epochs_trained is not None:
                        et.append(m.epochs_trained)
                    if m.early_stopped:
                        es += 1
    if not n:
        return None

    def q(a):
        a = np.array(a, float)
        return dict(mean=float(a.mean()), median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)))

    return dict(
        n=n,
        best_epoch=q(be) if be else None,
        epochs_trained=q(et) if et else None,
        early_stopped_frac=es / n,
    )


# ----------------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------------
def print_slope(results, Ds):
    print("\n=== (3) PRIMARY: log-log MSE scaling slope (VAL-selected ensembles, matched round budget) ===")
    print(f"    fit = log(test MSE on '{TARGET_SET}') vs log(D);  D in {Ds}")
    print(f"    {'subset':<22s} {'slope':>8s} {'intercept':>10s}   points (D: MSE / pearson / n_cells)")
    for name, r in results.items():
        pts = "  ".join(f"{D//1000}k:{mse:.4f}/{pe:.3f}/{n}" for (D, mse, pe, n) in r["points"])
        print(f"    {name:<22s} {r['slope']:>8.4f} {r['intercept']:>10.4f}   {pts}")
    base = results.get("all", {}).get("slope")
    if base is not None and np.isfinite(base):
        print(f"\n    all-R slope = {base:+.4f}")
        for name, r in results.items():
            if name.startswith("loro") and np.isfinite(r["slope"]):
                print(f"      {name:<20s} slope={r['slope']:+.4f}  d(slope)={r['slope']-base:+.4f}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--reservoirs", default=",".join(ALL_RESERVOIRS))
    ap.add_argument("--tags", default=",".join(K4_TAGS), help="strategy dir tags to pool (default K4 algo)")
    ap.add_argument("--include_llm", action="store_true", help="also pool lexp,ldiv persona dirs")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--Ds", default=",".join(map(str, DEFAULT_DS)))
    ap.add_argument("--round_budget", type=int, default=50, help="matched round budget across D (keep round<budget)")
    ap.add_argument("--target_set", default=TARGET_SET)
    ap.add_argument("--max_pool", type=int, default=40, help="cap candidate models (top-N by val_pearson) for greedy")
    ap.add_argument("--max_size", type=int, default=12, help="max greedy ensemble size")
    ap.add_argument("--knee_D", type=int, default=None, help="D for the rounds/knee/overfit secondary analyses (default: largest available)")
    ap.add_argument(
        "--which",
        default="slope,rounds,knee,overfit,epochs",
        help="comma list of analyses to run",
    )
    ap.add_argument("--out", default=None, help="optional JSON dump of results")
    args = ap.parse_args()

    reservoirs = [r for r in args.reservoirs.split(",") if r]
    tags = [t for t in args.tags.split(",") if t]
    if args.include_llm:
        tags = tags + [t for t in LLM_TAGS if t not in tags]
    seeds = [int(s) for s in args.seeds.split(",")]
    Ds = [int(d) for d in args.Ds.split(",")]
    which = set(w.strip() for w in args.which.split(","))
    knee_D = args.knee_D or max(Ds)

    print("=" * 88)
    print("SLOPE-EXPERIMENT ANALYSIS")
    print(f"  base={args.base}  reservoirs={reservoirs}")
    print(f"  tags={tags}  seeds={seeds}  Ds={Ds}")
    print(f"  round_budget={args.round_budget} (matched across D)  target_set={args.target_set}")
    print(f"  greedy: VAL-selected ElasticNetCV | max_pool={args.max_pool} max_size={args.max_size}")
    print("=" * 88)

    dump = {"config": vars(args)}

    if "slope" in which:
        res = slope_analysis(args.base, reservoirs, Ds, seeds, tags, args.round_budget, args.target_set, args.max_pool, args.max_size)
        print_slope(res, Ds)
        dump["slope"] = res

    if "rounds" in which:
        grid = [5, 10, 20, 30, 50, 75, 100, 150]
        grid = [r for r in grid if r <= max(150, args.round_budget)]
        curve = rounds_to_plateau(args.base, reservoirs, knee_D, seeds, tags, max(grid), args.target_set, args.max_pool, args.max_size, grid)
        print(f"\n=== (4a) rounds-to-plateau (D={knee_D}, mean ensemble test pearson over reservoirs x seeds) ===")
        for r, p, n in curve:
            print(f"    rounds<{r:<4d}  test_pearson={p:.4f}  (n_cells={n})")
        dump["rounds_to_plateau"] = curve

    if "knee" in which:
        mean_curve, knee = size_knee(args.base, reservoirs, knee_D, seeds, tags, args.round_budget, args.target_set, args.max_pool, args.max_size)
        print(f"\n=== (4b) ensemble-size knee (D={knee_D}, val-selected greedy, mean test pearson vs size) ===")
        for size, p, n in mean_curve:
            print(f"    size={size:<3d} test_pearson={p:.4f}  (n_cells={n})")
        print(f"    -> knee at size ~{knee}")
        dump["size_knee"] = {"curve": mean_curve, "knee": knee}

    if "overfit" in which:
        gap_sets = [s for s in ["genomic", "snv_delta", "snv_ref", "ood"]]
        rows = overfit_gap(args.base, reservoirs, knee_D, seeds, tags, args.round_budget, gap_sets)
        print(f"\n=== (4c) val-vs-test overfitting gap (D={knee_D}, per reservoir; val is per-combo/own held-out) ===")
        for R, row in rows.items():
            extra = "  ".join(f"{k}={v:.3f}" for k, v in row.items() if k.startswith("test_") or k.startswith("gap_"))
            print(f"    {R:<18s} n={row['n']:<4d} val_pearson={row['val_pearson']:.3f}  {extra}")
        dump["overfit_gap"] = rows

    if "epochs" in which:
        ec = epoch_check(args.base, reservoirs, Ds, seeds, tags, args.round_budget)
        print("\n=== (4d) epoch check (over all loaded models) ===")
        if ec:
            print(f"    n_models={ec['n']}  early_stopped_frac={ec['early_stopped_frac']:.3f}")
            for k in ("best_epoch", "epochs_trained"):
                if ec[k]:
                    q = ec[k]
                    print(f"    {k}: mean={q['mean']:.1f} median={q['median']:.1f} p10={q['p10']:.0f} p90={q['p90']:.0f}")
        dump["epoch_check"] = ec

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dump, f, indent=2, default=str)
        print(f"\n[wrote {args.out}]")


if __name__ == "__main__":
    main()
