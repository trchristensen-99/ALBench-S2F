"""Slope-variance analysis: per-reservoir & per-eval-set log-log MSE scaling.

Reuses scripts/analysis/slope_analysis.py (load_cell, greedy_ensemble,
Ensemble.predict). Only D=3 points per fit -> honest CIs.

(1) Per-reservoir slope on the GENOMIC eval set. Ensemble built per (R,ds) with
    greedy VAL selection (target_set=genomic). For each reservoir we get 2 seed
    points at each D. We fit the slope in two complementary ways and report both:
      - per-seed OLS slope (fit each seed's 3-D curve), then mean +- SEM over 2 seeds
      - pooled OLS over all 6 (D,ds) points with the polyfit standard error
    CI = mean +- 1.96*SE (report both flavors; headline uses the pooled polyfit SE
    since 2 seeds gives only 1 dof for the seed-SEM).

(2) Per-eval-set slope for the POOLED (all-reservoir) ensemble: each reservoir
    builds its own greedy VAL-selected ensemble whose SELECTION uses the genomic
    target (the deployed ensemble is a single object per cell), then the pooled
    TEST prediction (uniform mean over R x seeds) is scored on EACH eval set.
    Slope fit over 3 D with polyfit SE. NOTE: ensemble is genomic-selected; we
    evaluate its transfer error scaling on each set (this matches deployment).

Significance: two slopes are "distinguishable" if their 95% CIs do not overlap
(a conservative screen) AND via a 2-sample z-test on |d|/sqrt(se1^2+se2^2).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "analysis"))
sys.path.insert(0, "scripts/analysis")
import slope_analysis as SA  # noqa: E402

BASE = "outputs/overnight"
RESERVOIRS = SA.ALL_RESERVOIRS
DS = [10000, 30000, 100000]
SEEDS = [42, 43]
TAGS = SA.K4_TAGS  # K4 algorithmic (matches existing slope_allD.json)
ROUND_BUDGET = 50
MAX_POOL = 40
MAX_SIZE = 12
GENOMIC = "genomic"

# eval sets to profile for per-eval-set scaling
EVAL_SETS = [
    "genomic", "ood", "snv_ref", "snv_alt", "random_32k", "dinuc_shuffle",
    "sub_low", "sub_med", "sub_high", "ins_med", "del_med",
    "translocation", "inversion",
]

OUTDIR = os.path.join(BASE, "final_analysis")
os.makedirs(OUTDIR, exist_ok=True)


def fit_loglog(Ds, mses):
    """OLS log(mse) ~ log(D). Return slope, intercept, slope_se (polyfit).
    slope_se uses the residual-based standard error of the OLS slope.
    With n points and 2 params dof=n-2 (n=3 -> dof=1)."""
    lx = np.log(np.asarray(Ds, float))
    ly = np.log(np.asarray(mses, float))
    n = len(lx)
    if n < 2:
        return np.nan, np.nan, np.nan
    coef, cov = np.polyfit(lx, ly, 1, cov=True)
    slope, intercept = coef
    # polyfit cov (scaled by residual variance) -> slope SE = sqrt(cov[0,0])
    slope_se = float(np.sqrt(cov[0, 0])) if np.all(np.isfinite(cov)) else np.nan
    return float(slope), float(intercept), slope_se


# ----- build cache of ensembles: cache[(R,D,ds)] = Ensemble (genomic-selected) -----
print("Building per-cell greedy VAL-selected ensembles (genomic target)...", flush=True)
cache = {}
for D in DS:
    for R in RESERVOIRS:
        for ds in SEEDS:
            cell = SA.load_cell(BASE, R, D, ds, TAGS, round_budget=ROUND_BUDGET)
            if cell is None:
                cache[(R, D, ds)] = None
                print(f"  MISSING cell R={R} D={D} ds={ds}", flush=True)
                continue
            ens = SA.greedy_ensemble(cell, GENOMIC, MAX_POOL, MAX_SIZE)
            cache[(R, D, ds)] = ens if ens.selected else None
    print(f"  D={D} done", flush=True)


def cell_mse(R, D, ds, eval_set):
    ens = cache.get((R, D, ds))
    if ens is None:
        return None
    if eval_set not in ens.cell.oracle:
        return None
    pred = ens.predict(eval_set)
    truth = ens.cell.oracle[eval_set]
    return float(np.mean((pred - truth) ** 2))


def pooled_mse(D, eval_set, subset=None):
    """Uniform mean of per-cell ensemble TEST preds over subset reservoirs x seeds,
    scored on eval_set. Returns (mse, n_cells) or (None,0)."""
    subset = subset or RESERVOIRS
    preds = []
    truth = None
    for R in subset:
        for ds in SEEDS:
            ens = cache.get((R, D, ds))
            if ens is None or eval_set not in ens.cell.oracle:
                continue
            preds.append(ens.predict(eval_set))
            truth = ens.cell.oracle[eval_set]
    if not preds or truth is None:
        return None, 0
    mp = np.mean(preds, axis=0)
    return float(np.mean((mp - truth) ** 2)), len(preds)


# =========================================================================
# (1) PER-RESERVOIR slope on genomic
# =========================================================================
print("\n(1) per-reservoir genomic slope", flush=True)
per_res = {}
for R in RESERVOIRS:
    # pooled fit over all (D,ds) points (6 pts)
    pool_D, pool_mse = [], []
    # per-seed fits
    seed_slopes, seed_ints = [], []
    per_seed_detail = {}
    for ds in SEEDS:
        Dv, Mv = [], []
        for D in DS:
            m = cell_mse(R, D, ds, GENOMIC)
            if m is not None:
                Dv.append(D); Mv.append(m)
                pool_D.append(D); pool_mse.append(m)
        if len(Dv) >= 2:
            s, i, se = fit_loglog(Dv, Mv)
            seed_slopes.append(s); seed_ints.append(i)
            per_seed_detail[ds] = dict(D=Dv, mse=Mv, slope=s, intercept=i)
    # pooled fit
    ps, pi, pse = fit_loglog(pool_D, pool_mse) if len(pool_D) >= 2 else (np.nan, np.nan, np.nan)
    seed_slopes = np.array(seed_slopes, float)
    seed_mean = float(np.nanmean(seed_slopes)) if seed_slopes.size else np.nan
    seed_sd = float(np.nanstd(seed_slopes, ddof=1)) if seed_slopes.size >= 2 else np.nan
    seed_sem = seed_sd / np.sqrt(len(seed_slopes)) if seed_slopes.size >= 2 else np.nan
    per_res[R] = dict(
        pooled_slope=ps, pooled_intercept=pi, pooled_slope_se=pse,
        seed_mean_slope=seed_mean, seed_sd_slope=seed_sd, seed_sem_slope=float(seed_sem) if np.isfinite(seed_sem) else np.nan,
        per_seed=per_seed_detail,
        n_points=len(pool_D),
    )
    print(f"  {R:<18s} pooled slope={ps:+.4f} (se={pse:.4f})  seed-mean={seed_mean:+.4f} (sem={seed_sem if np.isfinite(seed_sem) else float('nan'):.4f})  int={pi:+.4f}", flush=True)


# =========================================================================
# (2) PER-EVAL-SET slope for pooled all-reservoir ensemble
# =========================================================================
print("\n(2) per-eval-set pooled slope", flush=True)
per_eval = {}
for es in EVAL_SETS:
    Dv, Mv, ncells = [], [], []
    for D in DS:
        m, n = pooled_mse(D, es)
        if m is not None:
            Dv.append(D); Mv.append(m); ncells.append(n)
    if len(Dv) < 2:
        print(f"  {es:<14s} SKIP (only {len(Dv)} D pts)", flush=True)
        continue
    s, i, se = fit_loglog(Dv, Mv)
    per_eval[es] = dict(D=Dv, mse=Mv, n_cells=ncells, slope=s, intercept=i, slope_se=se)
    print(f"  {es:<14s} slope={s:+.4f} (se={se:.4f})  int={i:+.4f}  mse@[{','.join('%.4f'%x for x in Mv)}]", flush=True)


# =========================================================================
# (3) significance: pairwise slope distinguishability
# =========================================================================
def pairwise(d, key_slope, key_se):
    names = list(d.keys())
    out = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            sa, sb = d[na][key_slope], d[nb][key_slope]
            ea, eb = d[na][key_se], d[nb][key_se]
            if not all(np.isfinite([sa, sb, ea, eb])):
                continue
            dse = np.sqrt(ea**2 + eb**2)
            z = (sa - sb) / dse if dse > 0 else np.nan
            # CI overlap check (95%)
            lo_a, hi_a = sa - 1.96*ea, sa + 1.96*ea
            lo_b, hi_b = sb - 1.96*eb, sb + 1.96*eb
            ci_overlap = not (hi_a < lo_b or hi_b < lo_a)
            out[f"{na} vs {nb}"] = dict(
                d_slope=float(sa - sb), z=float(z),
                significant_z=bool(abs(z) > 1.96),
                ci_overlap=bool(ci_overlap),
            )
    return out

res_pairs = pairwise(per_res, "pooled_slope", "pooled_slope_se")
eval_pairs = pairwise(per_eval, "slope", "slope_se")

# spread relative to shared mean
res_slopes = np.array([per_res[R]["pooled_slope"] for R in RESERVOIRS], float)
res_shared_mean = float(np.nanmean(res_slopes))
eval_slopes = {es: per_eval[es]["slope"] for es in per_eval}

any_res_sig = any(v["significant_z"] and not v["ci_overlap"] for v in res_pairs.values())
any_eval_sig = any(v["significant_z"] and not v["ci_overlap"] for v in eval_pairs.values())

print("\n(3) significance", flush=True)
print(f"  reservoir slopes: shared mean={res_shared_mean:+.4f}; any pair CI-disjoint & z>1.96? {any_res_sig}", flush=True)
print(f"  eval-set slopes:  any pair CI-disjoint & z>1.96? {any_eval_sig}", flush=True)

# =========================================================================
# FIGURE
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# (a) per-reservoir slopes
ax = axes[0]
Rn = RESERVOIRS
ys = [per_res[R]["pooled_slope"] for R in Rn]
es = [per_res[R]["pooled_slope_se"] for R in Rn]
x = np.arange(len(Rn))
ax.errorbar(x, ys, yerr=[1.96*e for e in es], fmt="o", capsize=5, color="C0", ms=8, label="pooled slope ±95% (polyfit SE)")
# overlay per-seed slopes
for i, R in enumerate(Rn):
    for ds, det in per_res[R]["per_seed"].items():
        ax.plot(i + (0.12 if ds == SEEDS[-1] else -0.12), det["slope"], "x", color="gray", ms=7, alpha=0.7)
ax.axhline(res_shared_mean, ls="--", color="k", alpha=0.6, label=f"shared mean {res_shared_mean:+.3f}")
ax.set_xticks(x); ax.set_xticklabels([r.replace("_planted_v2","").replace("_shuffle","") for r in Rn], rotation=30, ha="right")
ax.set_ylabel("log-log MSE scaling slope (genomic eval)")
ax.set_title("(a) Per-reservoir slope\n(3 D pts/fit; x = per-seed slopes)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (b) per-eval-set slopes
ax = axes[1]
En = list(per_eval.keys())
ys = [per_eval[e]["slope"] for e in En]
es = [per_eval[e]["slope_se"] for e in En]
x = np.arange(len(En))
colors = ["C3" if e in ("ood","snv_ref","snv_alt") else "C2" for e in En]
ax.errorbar(x, ys, yerr=[1.96*e for e in es], fmt="s", capsize=4, ms=7, ecolor="gray",
            mfc="none", linestyle="none")
for xi, yi, c in zip(x, ys, colors):
    ax.plot(xi, yi, "s", color=c, ms=8)
gmean = eval_slopes.get("genomic")
if gmean is not None:
    ax.axhline(gmean, ls="--", color="C0", alpha=0.6, label=f"genomic slope {gmean:+.3f}")
ax.set_xticks(x); ax.set_xticklabels(En, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("log-log MSE scaling slope (pooled all-R ensemble)")
ax.set_title("(b) Per-eval-set slope\n(red=OOD/SNV, green=other; 3 D pts/fit)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
figpath = os.path.join(OUTDIR, "slope_variance.png")
plt.savefig(figpath, dpi=140)
print(f"\n[wrote {figpath}]", flush=True)

# =========================================================================
# JSON dump
# =========================================================================
dump = dict(
    config=dict(base=BASE, reservoirs=RESERVOIRS, Ds=DS, seeds=SEEDS, tags=TAGS,
                round_budget=ROUND_BUDGET, max_pool=MAX_POOL, max_size=MAX_SIZE,
                target_set_for_selection=GENOMIC, eval_sets=EVAL_SETS,
                n_D_points_per_fit=len(DS),
                caveat="Only 3 D per fit (dof=1 for pooled polyfit SE, seed-SEM has 1 dof). CIs are wide; treat slope-difference claims cautiously."),
    per_reservoir=per_res,
    per_eval_set=per_eval,
    reservoir_shared_mean_slope=res_shared_mean,
    significance=dict(
        reservoir_pairs=res_pairs,
        eval_pairs=eval_pairs,
        any_reservoir_pair_significant=any_res_sig,
        any_eval_pair_significant=any_eval_sig,
        criterion="z=|d_slope|/sqrt(se_a^2+se_b^2) > 1.96 AND 95% CIs disjoint",
    ),
)
jpath = os.path.join(OUTDIR, "slope_variance.json")
with open(jpath, "w") as f:
    json.dump(dump, f, indent=2, default=str)
print(f"[wrote {jpath}]", flush=True)
print("DONE", flush=True)
