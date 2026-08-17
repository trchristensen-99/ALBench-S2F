"""Multi-seed weight-init comparison on saved multiseed predictions.

Layout (outputs/overnight/multiseed_<R>_d30000/):
  r<NN>_random_<i>_s<k>.npz      preds: val_pred, test_pred, test_pred_<set>
  r<NN>_random_<i>_s<k>_meta.json  val_pearson, init_seed, hp, per_set_metrics
  labels.npz                     val_labels, oracle_<set>  (shared across models)

Config = (round NN, config index i); the 3 siblings _s0/_s1/_s2 are the same HP
config trained from 3 different weight-init seeds.  All models in a dir share the
same val split + test labels.

Computes, per reservoir, on the genomic/reference test set (and OOD):
  (1) per-config weight-init seed variance (test-Pearson spread across s0/s1/s2)
  (2) 1-seed (s0 only) vs 3-seed (seed-averaged) val-selected greedy ElasticNet
      ensemble test-Pearson, and the delta.

Reuses the greedy VAL-selected ElasticNet logic from slope_analysis.py.
CPU only; set OMP_NUM_THREADS=1.
"""

import argparse
import glob
import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

EN_KW = dict(l1_ratio=[0.5, 0.9, 1.0], positive=True, cv=3, n_alphas=25, max_iter=20000, n_jobs=1)

CFG_RE = re.compile(r"^(r\d+_random_\d+)_s(\d+)$")


def _fit_en(V, y):
    en = ElasticNetCV(**EN_KW)
    en.fit(V, y)
    return en


class Model:
    __slots__ = ("model_id", "cfg", "seed", "val_pearson", "val_pred", "test")


def load_dir(d, sets):
    """Return (val_labels, oracle{set->truth}, models[list[Model]])."""
    lab = np.load(os.path.join(d, "labels.npz"), allow_pickle=True)
    val_labels = lab["val_labels"].astype(np.float64)
    oracle = {s: lab[f"oracle_{s}"].astype(np.float64) for s in sets}
    models = []
    for mp in sorted(glob.glob(os.path.join(d, "*_meta.json"))):
        base = os.path.basename(mp)[: -len("_meta.json")]
        m = CFG_RE.match(base)
        if not m:
            continue
        cfg, seed = m.group(1), int(m.group(2))
        try:
            meta = json.load(open(mp))
            z = np.load(mp.replace("_meta.json", ".npz"))
        except Exception:
            continue
        if "val_pred" not in z.files:
            continue
        vp = z["val_pred"].astype(np.float64)
        if vp.shape != val_labels.shape or not np.all(np.isfinite(vp)):
            continue
        test = {}
        ok = True
        for s, truth in oracle.items():
            key = f"test_pred_{s}"
            if key not in z.files:
                ok = False
                break
            tp = z[key].astype(np.float64)
            if tp.shape != truth.shape or not np.all(np.isfinite(tp)):
                ok = False
                break
            test[s] = tp
        if not ok:
            continue
        vpear = meta.get("best_val_pearson", meta.get("val_pearson"))
        if vpear is None or not np.isfinite(vpear):
            continue
        mo = Model()
        mo.model_id, mo.cfg, mo.seed = base, cfg, seed
        mo.val_pearson = float(vpear)
        mo.val_pred, mo.test = vp, test
        models.append(mo)
    return val_labels, oracle, models


def per_config_seed_variance(models, oracle, target_set):
    """Test-Pearson spread across seeds within each config. Returns list of dicts."""
    truth = oracle[target_set]
    by_cfg = defaultdict(list)
    for m in models:
        by_cfg[m.cfg].append(m)
    rows = []
    for cfg, sibs in sorted(by_cfg.items()):
        if len(sibs) < 2:
            continue
        pears = [pearsonr(m.test[target_set], truth)[0] for m in sibs]
        rows.append(dict(cfg=cfg, n=len(sibs), mean=float(np.mean(pears)),
                         std=float(np.std(pears, ddof=0)), vals=pears))
    return rows


def greedy_ensemble(cand_val, cand_test, val_labels, truth, max_pool, max_size, min_delta=1e-4):
    """Forward-greedy VAL-selected ElasticNet. cand_* are lists aligned by candidate.

    cand_val[i] = val_pred vector; cand_test[i] = test_pred vector (target set);
    val_pearson used only to cap the pool. Returns (test_pearson, size, sel_idx)."""
    val_pear = [pearsonr(v, val_labels)[0] for v in cand_val]
    order = sorted(range(len(cand_val)), key=lambda i: -val_pear[i])[:max_pool]
    selected, best_val = [], -np.inf
    remaining = list(order)
    final_en = None
    while remaining and len(selected) < max_size:
        best = None
        for ci in remaining:
            trial = selected + [ci]
            V = np.column_stack([cand_val[j] for j in trial])
            en = _fit_en(V, val_labels)
            vp = pearsonr(en.predict(V), val_labels)[0]
            if best is None or vp > best[0]:
                best = (vp, ci, en)
        vp, ci, en = best
        if vp <= best_val + min_delta and selected:
            break
        selected.append(ci)
        remaining.remove(ci)
        best_val = vp
        final_en = en
    if not selected:
        return float("nan"), 0, []
    Vf = np.column_stack([cand_val[j] for j in selected])
    final_en = _fit_en(Vf, val_labels)
    T = np.column_stack([cand_test[j] for j in selected])
    pred = final_en.predict(T)
    return float(pearsonr(pred, truth)[0]), len(selected), selected


def build_candidates_1seed(models, target_set, seed0=0):
    """One model per config: only _s0."""
    cv, ct = [], []
    for m in sorted(models, key=lambda m: (m.cfg, m.seed)):
        if m.seed == seed0:
            cv.append(m.val_pred)
            ct.append(m.test[target_set])
    return cv, ct


def build_candidates_3seed(models, target_set):
    """One seed-averaged model per config (mean over available siblings)."""
    by_cfg = defaultdict(list)
    for m in models:
        by_cfg[m.cfg].append(m)
    cv, ct = [], []
    for cfg, sibs in sorted(by_cfg.items()):
        cv.append(np.mean([m.val_pred for m in sibs], axis=0))
        ct.append(np.mean([m.test[target_set] for m in sibs], axis=0))
    return cv, ct


def analyze_reservoir(name, d, target_sets, max_pool, max_size):
    val_labels, oracle, models = load_dir(d, target_sets)
    out = {"name": name, "dir": d, "n_models": len(models),
           "n_configs": len(set(m.cfg for m in models))}
    for ts in target_sets:
        truth = oracle[ts]
        var_rows = per_config_seed_variance(models, oracle, ts)
        stds = [r["std"] for r in var_rows]
        means = [r["mean"] for r in var_rows]
        cv1, ct1 = build_candidates_1seed(models, ts)
        cv3, ct3 = build_candidates_3seed(models, ts)
        p1, n1, _ = greedy_ensemble(cv1, ct1, val_labels, truth, max_pool, max_size)
        p3, n3, _ = greedy_ensemble(cv3, ct3, val_labels, truth, max_pool, max_size)
        out[ts] = dict(
            n_configs=len(var_rows),
            per_config_mean_pearson=float(np.mean(means)),
            median_seed_std=float(np.median(stds)),
            mean_seed_std=float(np.mean(stds)),
            max_seed_std=float(np.max(stds)),
            ens_1seed=p1, ens_1seed_size=n1,
            ens_3seed=p3, ens_3seed_size=n3,
            delta=p3 - p1,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/overnight")
    ap.add_argument("--sets", default="genomic,ood")
    ap.add_argument("--max_pool", type=int, default=22)
    ap.add_argument("--max_size", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sets = [s for s in args.sets.split(",") if s]
    dirs = {
        "genomic": os.path.join(args.base, "multiseed_genomic_d30000"),
        "evoaug_heavy": os.path.join(args.base, "multiseed_evoaug_heavy_d30000"),
    }
    print("=" * 90)
    print("MULTI-SEED WEIGHT-INIT COMPARISON (D=30000, saved preds, CPU)")
    print(f"  sets={sets}  greedy: VAL-selected ElasticNetCV  max_pool={args.max_pool} max_size={args.max_size}")
    print("=" * 90)

    results = []
    for name, d in dirs.items():
        r = analyze_reservoir(name, d, sets, args.max_pool, args.max_size)
        results.append(r)
        print(f"\n### {name}   ({r['n_models']} models, {r['n_configs']} configs)")
        for ts in sets:
            s = r[ts]
            print(f"  [{ts}]  per-config mean test-pearson={s['per_config_mean_pearson']:.4f}")
            print(f"        seed-std across siblings: median={s['median_seed_std']:.4f} "
                  f"mean={s['mean_seed_std']:.4f} max={s['max_seed_std']:.4f}  (n_configs={s['n_configs']})")
            print(f"        ensemble test-pearson: 1-seed={s['ens_1seed']:.4f} (size {s['ens_1seed_size']})  "
                  f"3-seed={s['ens_3seed']:.4f} (size {s['ens_3seed_size']})  delta={s['delta']:+.4f}")

    print("\n" + "=" * 90)
    print("SUMMARY TABLE (genomic/reference test set)")
    print(f"  {'reservoir':<16s} {'median_seed_std':>15s} {'ens_1seed':>10s} {'ens_3seed':>10s} {'delta':>9s}")
    for r in results:
        s = r["genomic"]
        print(f"  {r['name']:<16s} {s['median_seed_std']:>15.4f} {s['ens_1seed']:>10.4f} "
              f"{s['ens_3seed']:>10.4f} {s['delta']:>+9.4f}")
    if "ood" in sets:
        print("\n  (OOD test set)")
        print(f"  {'reservoir':<16s} {'median_seed_std':>15s} {'ens_1seed':>10s} {'ens_3seed':>10s} {'delta':>9s}")
        for r in results:
            s = r["ood"]
            print(f"  {r['name']:<16s} {s['median_seed_std']:>15.4f} {s['ens_1seed']:>10.4f} "
                  f"{s['ens_3seed']:>10.4f} {s['delta']:>+9.4f}")

    max_delta = max(abs(r["genomic"]["delta"]) for r in results)
    verdict = ("MATERIAL: seed-averaging improves the ensemble by >~0.005 test Pearson"
               if max_delta > 0.005 else
               "WITHIN NOISE: seed-averaging delta <~0.005 test Pearson; 1 seed/config is fine for the main grid")
    print(f"\nVERDICT: {verdict}  (max |delta genomic| = {max_delta:.4f})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
