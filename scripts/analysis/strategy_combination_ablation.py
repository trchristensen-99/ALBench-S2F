"""Strategy-combination ablation: empirical justification for the mixed6 HP-search
composition + ElasticNet ensembling, with matched-budget controls and bootstrap CIs.

The existing combinations_report.json enumerates many combos but reports bare point
estimates. This script adds what a reviewer actually needs to believe the choices:

  1. Control ladder (matched train budget): random-only -> best-single -> mixed6 -> all.
     Answers "does the search machinery beat random, and does combining help?"
  2. Leave-one-out from the full strategy set, with PAIRED test-set bootstrap on the
     delta. Answers "does each included strategy earn its seat?" (drop X -> ensemble
     degrades by Delta with a CI that excludes 0).
  3. Forward-selection curve (greedy add by val MSE) with bootstrap CI band. Shows the
     plateau -> empirical justification for "why stop at ~6 strategies, not all 16".
  4. Matched-budget parity (mixed6 vs all vs 2each at equal N_train). Answers "we don't
     lose accuracy by trimming the trained pool to 6 strategies."

Every number carries a bootstrap 95% CI so claims are "beats X with non-overlapping
CIs", not "0.002 lower MSE".

Pool layout (works for both Phase 2 and the live hp_search cells): glob */r*_meta.json
under --pool_dir; each model's strategy comes from meta['strategy']; predictions from
the sibling .npz (val_pred, test_pred); labels from labels.npz (val_labels,
test_oracle, test_true). Models are grouped by strategy regardless of directory.

CPU-only; safe to run alongside GPU jobs.

Usage:
    python scripts/analysis/strategy_combination_ablation.py \
        --pool_dir outputs/phase2_unified/D30000 \
        --out_dir  outputs/phase2_unified/D30000/ablation \
        --n_boot 500
"""

from __future__ import annotations

import argparse
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

# Canonical mixed6 composition. Phase 2 uses the *_opus/*_sonnet names; the live
# hp_search cells use bare llm_default/llm_diverse/llm_exploit. We match whichever set
# of names is actually present in the pool.
# Map deprecated autoresearch_* strategy names to canonical evo_* (terminology fix:
# "AutoResearch" is reserved for the LLM-iterative search; these are evolutionary).
_STRAT_ALIASES = {
    "autoresearch_single": "evo_single",
    "autoresearch_batch": "evo_batch",
    "autoresearch_explore": "evo_explore",
    "autoresearch_exploit": "evo_exploit",
    "autoresearch_massive": "evo_massive",
    "autoresearch_adaptive": "evo_adaptive",
    "autoresearch_knowledgeable": "evo_knowledgeable",
}


def _canon_strat(name: str) -> str:
    return _STRAT_ALIASES.get(name, name)


MIXED6_PHASE2 = [
    "llm_default_opus",
    "llm_diverse_sonnet",
    "llm_exploit_sonnet",
    "evo_batch",
    "evo_massive",
    "random",
]
MIXED6_HPSEARCH = [
    "llm_default",
    "llm_diverse",
    "llm_exploit",
    "evo_batch",
    "evo_massive",
    "random",
]

L1_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pool(pool_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Return (val_preds_by_strategy, test_preds_by_strategy, labels).

    val_preds_by_strategy[s] is an (n_models_s, n_val) matrix; test analogously.
    labels has val_labels, test_oracle, test_true.
    """
    metas = sorted(pool_dir.glob("*/r*_meta.json"))
    if not metas:
        metas = sorted(pool_dir.glob("**/r*_meta.json"))
    if not metas:
        raise SystemExit(f"no *_meta.json under {pool_dir}")

    # Labels are shared across models; load the first labels.npz we find and verify.
    labels_files = sorted(pool_dir.glob("**/labels.npz"))
    if not labels_files:
        raise SystemExit(f"no labels.npz under {pool_dir}")
    lz = np.load(labels_files[0])
    labels = {
        "val_labels": lz["val_labels"],
        "test_oracle": lz["test_oracle"],
        "test_true": lz["test_true"] if "test_true" in lz.files else lz["test_oracle"],
    }
    n_val = labels["val_labels"].shape[0]
    n_test = labels["test_oracle"].shape[0]

    val_by: dict[str, list[np.ndarray]] = {}
    test_by: dict[str, list[np.ndarray]] = {}
    n_skip = 0
    for m in metas:
        npz = m.with_name(m.name.replace("_meta.json", ".npz"))
        if not npz.exists():
            n_skip += 1
            continue
        meta = json.loads(m.read_text())
        # The meta 'strategy' field is the generic driver name "llm_autoresearch" for
        # every LLM variant; the real variant (llm_default_opus, llm_diverse_sonnet, ...)
        # is the directory name. For the hp_search 'algo' dir, which genuinely bundles
        # autoresearch_batch/massive/random, the meta field IS the right label. So: use
        # the meta strategy unless it is the generic LLM driver name, else the dir name.
        strat = meta.get("strategy")
        if not strat or strat == "llm_autoresearch":
            strat = m.parent.name
        strat = _canon_strat(strat)
        try:
            d = np.load(npz)
            vp, tp = d["val_pred"], d["test_pred"]
        except (KeyError, OSError, ValueError, EOFError, zipfile.BadZipFile):
            # Truncated / partially-written / corrupt npz (e.g. interrupted job).
            n_skip += 1
            continue
        # Only stack models whose predictions align to the shared val/test labels.
        if vp.shape[0] != n_val or tp.shape[0] != n_test:
            n_skip += 1
            continue
        if not np.all(np.isfinite(vp)) or not np.all(np.isfinite(tp)):
            n_skip += 1
            continue
        val_by.setdefault(strat, []).append(vp.astype(np.float64))
        test_by.setdefault(strat, []).append(tp.astype(np.float64))

    val_mats = {s: np.vstack(v) for s, v in val_by.items()}
    test_mats = {s: np.vstack(v) for s, v in test_by.items()}
    counts = {s: m.shape[0] for s, m in val_mats.items()}
    print(
        f"loaded {sum(counts.values())} models across {len(counts)} strategies "
        f"(skipped {n_skip}); n_val={n_val} n_test={n_test}"
    )
    for s in sorted(counts):
        print(f"  {s:28s} {counts[s]:4d}")
    return val_mats, test_mats, labels


# ---------------------------------------------------------------------------
# Ensembling + metrics
# ---------------------------------------------------------------------------
def fit_ensemble(
    val_X: np.ndarray, val_y: np.ndarray, test_X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """ElasticNetCV(positive=True) stack on val preds.

    Returns (test_pred, val_insample_pred, info). val_X / test_X are
    (n_models, n_points); transposed to (n_points, n_models) for sklearn.
    """
    Xv, Xt = val_X.T, test_X.T
    enet = ElasticNetCV(l1_ratio=L1_GRID, positive=True, cv=5, n_jobs=1, max_iter=5000)
    enet.fit(Xv, val_y)
    info = {
        "n_models": int(val_X.shape[0]),
        "n_kept": int(np.sum(enet.coef_ > 0)),
        "alpha": float(enet.alpha_),
        "l1_ratio": float(enet.l1_ratio_),
        "coef": enet.coef_.tolist(),
    }
    return enet.predict(Xt), enet.predict(Xv), info


def metrics(pred: np.ndarray, oracle: np.ndarray, true: np.ndarray) -> dict:
    return {
        "test_oracle_pearson": float(pearsonr(pred, oracle)[0]),
        "test_oracle_mse": float(np.mean((pred - oracle) ** 2)),
        "test_true_pearson": float(pearsonr(pred, true)[0]),
        "test_true_mse": float(np.mean((pred - true) ** 2)),
    }


def stack(mats: dict[str, np.ndarray], strategies: list[str]) -> np.ndarray:
    return np.vstack([mats[s] for s in strategies if s in mats])


def subsample_indices(
    mats: dict[str, np.ndarray], strategies: list[str], n_total: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Draw exactly min(n_total, total_available) models, spread as evenly across
    `strategies` as availability allows (round-robin water-filling). Exact totals are
    what make the matched-budget comparison fair across combos with different strategy
    counts.
    """
    present = [s for s in strategies if s in mats]
    avail = {s: mats[s].shape[0] for s in present}
    target = min(n_total, sum(avail.values()))
    alloc = {s: 0 for s in present}
    i, n = 0, len(present)
    while target > 0 and any(alloc[s] < avail[s] for s in present):
        s = present[i % n]
        if alloc[s] < avail[s]:
            alloc[s] += 1
            target -= 1
        i += 1
    return {s: rng.choice(avail[s], size=alloc[s], replace=False) for s in present if alloc[s] > 0}


def stack_subset(mats: dict[str, np.ndarray], sel: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([mats[s][idx] for s, idx in sel.items()])


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def ci(vals: list[float]) -> tuple[float, float, float]:
    a = np.asarray(vals, dtype=np.float64)
    return float(np.median(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def matched_budget_eval(val_mats, test_mats, labels, strategies, n_total, n_boot, rng):
    """Fit ENT over n_boot matched-budget model subsamples; CI over test metrics."""
    o_mse, o_r, actual_n = [], [], 0
    for _ in range(n_boot):
        sel = subsample_indices(val_mats, strategies, n_total, rng)
        actual_n = sum(len(idx) for idx in sel.values())
        vX = stack_subset(val_mats, sel)
        tX = stack_subset(test_mats, sel)
        pred, _, _ = fit_ensemble(vX, labels["val_labels"], tX)
        mm = metrics(pred, labels["test_oracle"], labels["test_true"])
        o_mse.append(mm["test_oracle_mse"])
        o_r.append(mm["test_oracle_pearson"])
    med, lo, hi = ci(o_mse)
    rmed, rlo, rhi = ci(o_r)
    return {
        "strategies": [s for s in strategies if s in val_mats],
        "n_total_budget": n_total,
        "n_models_drawn": actual_n,
        "test_oracle_mse": med,
        "test_oracle_mse_lo": lo,
        "test_oracle_mse_hi": hi,
        "test_oracle_pearson": rmed,
        "test_oracle_pearson_lo": rlo,
        "test_oracle_pearson_hi": rhi,
    }


def paired_test_bootstrap(pred_a, pred_b, oracle, n_boot, rng):
    """CI on MSE(b) - MSE(a) via paired test-set resampling (positive => b worse)."""
    n = oracle.shape[0]
    deltas = []
    ea = (pred_a - oracle) ** 2
    eb = (pred_b - oracle) ** 2
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(float(np.mean(eb[idx]) - np.mean(ea[idx])))
    med, lo, hi = ci(deltas)
    return {"delta_mse": med, "delta_mse_lo": lo, "delta_mse_hi": hi, "significant": bool(lo > 0)}


# ---------------------------------------------------------------------------
# The four framed analyses
# ---------------------------------------------------------------------------
def best_single_strategy(val_mats, test_mats, labels):
    """Strategy whose own-models ensemble has the lowest test_oracle MSE."""
    best, best_mse = None, np.inf
    for s in val_mats:
        pred, _, _ = fit_ensemble(val_mats[s], labels["val_labels"], test_mats[s])
        mse = metrics(pred, labels["test_oracle"], labels["test_true"])["test_oracle_mse"]
        if mse < best_mse:
            best, best_mse = s, mse
    return best


def analysis_control_ladder(val_mats, test_mats, labels, mixed6, n_total, n_boot, rng):
    """random -> best-single -> mixed6 -> all, all at a genuinely matched budget.

    The budget is auto-capped to the largest N every rung can supply (single-strategy
    rungs have the fewest models), so each rung draws the *same* number of models. This
    is the control that defeats "you just trained more". Returns (rungs, budget_used).
    """
    all_strats = sorted(val_mats)
    best_single = best_single_strategy(val_mats, test_mats, labels)
    rungs = {
        "random_only": ["random"] if "random" in val_mats else [best_single],
        f"best_single({best_single})": [best_single],
        "mixed6": [s for s in mixed6 if s in val_mats],
        "all_strategies": all_strats,
    }
    feasible = min(sum(val_mats[s].shape[0] for s in strats) for strats in rungs.values())
    budget = min(n_total, feasible)
    out = {
        name: matched_budget_eval(val_mats, test_mats, labels, strats, budget, n_boot, rng)
        for name, strats in rungs.items()
    }
    return out, budget


def analysis_leave_one_out(val_mats, test_mats, labels, n_boot, rng):
    """Full-set ensemble vs full-set-minus-X, with paired test-set bootstrap on Delta."""
    all_strats = sorted(val_mats)
    full_pred, _, full_info = fit_ensemble(
        stack(val_mats, all_strats), labels["val_labels"], stack(test_mats, all_strats)
    )
    full_m = metrics(full_pred, labels["test_oracle"], labels["test_true"])
    rows = []
    for drop in all_strats:
        kept = [s for s in all_strats if s != drop]
        loo_pred, _, _ = fit_ensemble(
            stack(val_mats, kept), labels["val_labels"], stack(test_mats, kept)
        )
        boot = paired_test_bootstrap(full_pred, loo_pred, labels["test_oracle"], n_boot, rng)
        rows.append({"dropped": drop, **boot})
    rows.sort(key=lambda r: -r["delta_mse"])
    return {"full_set_metrics": full_m, "full_set_info": full_info, "leave_one_out": rows}


def analysis_forward_selection(val_mats, test_mats, labels, n_boot, rng):
    """Greedily add the strategy that most improves VAL ensemble MSE; record test curve."""
    remaining = sorted(val_mats)
    chosen: list[str] = []
    curve = []
    while remaining:
        best_s, best_val, best_pred = None, np.inf, None
        for s in remaining:
            trial = chosen + [s]
            test_pred, val_pred, _ = fit_ensemble(
                stack(val_mats, trial), labels["val_labels"], stack(test_mats, trial)
            )
            val_mse = float(np.mean((val_pred - labels["val_labels"]) ** 2))
            if val_mse < best_val:
                best_s, best_val, best_pred = s, val_mse, test_pred
        chosen.append(best_s)
        remaining.remove(best_s)
        # Test-set bootstrap CI on this k-strategy ensemble's MSE.
        n = labels["test_oracle"].shape[0]
        err = (best_pred - labels["test_oracle"]) ** 2
        boot = [float(np.mean(err[rng.integers(0, n, size=n)])) for _ in range(n_boot)]
        med, lo, hi = ci(boot)
        m = metrics(best_pred, labels["test_oracle"], labels["test_true"])
        curve.append(
            {
                "k": len(chosen),
                "added": best_s,
                "chosen": list(chosen),
                "test_oracle_mse": med,
                "test_oracle_mse_lo": lo,
                "test_oracle_mse_hi": hi,
                "test_oracle_pearson": m["test_oracle_pearson"],
            }
        )
    return curve


def analysis_matched_parity(val_mats, test_mats, labels, mixed6, n_total, n_boot, rng):
    """mixed6 vs all vs 2each at equal N_train (parity, not 'more is better')."""
    all_strats = sorted(val_mats)
    combos = {
        "mixed6": [s for s in mixed6 if s in val_mats],
        "all_strategies": all_strats,
    }
    return {
        name: matched_budget_eval(val_mats, test_mats, labels, strats, n_total, n_boot, rng)
        for name, strats in combos.items()
    }


def analysis_budget_sweep(val_mats, test_mats, labels, mixed6, n_grid, n_boot, rng):
    """random / best-single / mixed6 / all across a grid of matched budgets N.

    Shows the crossover: at small N concentrating on the best strategy can win; the
    diversity benefit of mixed6/all only emerges (and plateaus) as N grows. Each combo's
    curve stops at its own pool size (random/best-single saturate early).
    """
    best_single = best_single_strategy(val_mats, test_mats, labels)
    combos = {
        "random_only": ["random"] if "random" in val_mats else [best_single],
        f"best_single({best_single})": [best_single],
        "mixed6": [s for s in mixed6 if s in val_mats],
        "all_strategies": sorted(val_mats),
    }
    out = {}
    for name, strats in combos.items():
        avail = sum(val_mats[s].shape[0] for s in strats if s in val_mats)
        points = []
        seen = set()
        for n in n_grid:
            n_cap = min(n, avail)
            if n_cap in seen:  # don't repeat the saturated point
                continue
            seen.add(n_cap)
            points.append(
                matched_budget_eval(val_mats, test_mats, labels, strats, n_cap, n_boot, rng)
            )
        out[name] = points
    return out


def analysis_all_subsets(
    val_mats,
    test_mats,
    labels,
    n_boot,
    rng,
    max_k: int = 9,
    knee_eps: float = 0.002,
):
    """Exhaustive recipe search: evaluate EVERY non-empty subset of strategies at a
    common matched budget, so subsets of different sizes are directly comparable.

    Answers "which strategies work best TOGETHER, and how many are enough?" — the
    Step-1 Phase-B recipe-composition curve. Cost is 2^K * n_boot ElasticNetCV fits,
    so K is capped at max_k (restrict to the top-max_k single strategies, always
    keeping 'random' if present) and n_boot defaults low.
    """
    strategies = sorted(val_mats)
    note = None
    if len(strategies) > max_k:
        # Rank singletons by oracle Pearson; keep the strongest max_k (+ random).
        ranked = sorted(
            strategies,
            key=lambda s: metrics(
                *(fit_ensemble(val_mats[s], labels["val_labels"], test_mats[s])[:1]),
                labels["test_oracle"],
                labels["test_true"],
            )["test_oracle_pearson"],
            reverse=True,
        )
        keep = ranked[:max_k]
        if "random" in strategies and "random" not in keep:
            keep[-1] = "random"
        strategies = sorted(keep)
        note = f"restricted to top-{max_k} strategies (2^K too large): {strategies}"

    # Common budget so every subset (down to a singleton) can supply the same N.
    budget = min(val_mats[s].shape[0] for s in strategies)

    results = []
    for m in range(1, len(strategies) + 1):
        for combo in itertools.combinations(strategies, m):
            ev = matched_budget_eval(val_mats, test_mats, labels, list(combo), budget, n_boot, rng)
            results.append(
                {
                    "m": m,
                    "strategies": list(combo),
                    "test_oracle_pearson": ev["test_oracle_pearson"],
                    "test_oracle_pearson_lo": ev["test_oracle_pearson_lo"],
                    "test_oracle_pearson_hi": ev["test_oracle_pearson_hi"],
                    "test_oracle_mse": ev["test_oracle_mse"],
                    "n_models_drawn": ev["n_models_drawn"],
                }
            )

    # Best subset per size + diminishing-returns knee on the best-of-size curve.
    best_by_m = {}
    for m in range(1, len(strategies) + 1):
        same = [r for r in results if r["m"] == m]
        best = max(same, key=lambda r: r["test_oracle_pearson"])
        rs = [r["test_oracle_pearson"] for r in same]
        best_by_m[m] = {
            "best_strategies": best["strategies"],
            "best_pearson": best["test_oracle_pearson"],
            "median_pearson": float(np.median(rs)),
            "min_pearson": float(np.min(rs)),
            "max_pearson": float(np.max(rs)),
            "n_subsets": len(same),
        }

    ms = sorted(best_by_m)
    knee = ms[-1]
    for i in range(1, len(ms)):
        if best_by_m[ms[i]]["best_pearson"] - best_by_m[ms[i - 1]]["best_pearson"] < knee_eps:
            knee = ms[i - 1]
            break
    overall_best = max(results, key=lambda r: r["test_oracle_pearson"])

    return {
        "note": note,
        "budget_per_subset": int(budget),
        "n_boot": n_boot,
        "knee_eps": knee_eps,
        "knee_size": int(knee),
        "recipe_at_knee": best_by_m[knee]["best_strategies"],
        "overall_best": {
            "strategies": overall_best["strategies"],
            "test_oracle_pearson": overall_best["test_oracle_pearson"],
        },
        "best_by_size": best_by_m,
        "all_subsets": sorted(results, key=lambda r: -r["test_oracle_pearson"]),
    }


def plot_all_subsets(allsub: dict, out_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_m = allsub["best_by_size"]
    ms = sorted(by_m)
    best = [by_m[m]["best_pearson"] for m in ms]
    med = [by_m[m]["median_pearson"] for m in ms]
    lo = [by_m[m]["min_pearson"] for m in ms]
    hi = [by_m[m]["max_pearson"] for m in ms]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.fill_between(ms, lo, hi, alpha=0.15, color="#d97b29", label="all subsets (min–max)")
    ax.plot(ms, med, "--o", ms=4, color="#9aa0a6", label="median subset")
    ax.plot(ms, best, "-o", ms=5, color="#d97b29", label="best subset of size m")
    knee = allsub["knee_size"]
    ax.axvline(knee, color="crimson", ls=":", lw=1.5)
    ax.annotate(
        f"knee = {knee}\n{', '.join(allsub['recipe_at_knee'])}",
        xy=(knee, by_m[knee]["best_pearson"]),
        xytext=(knee + 0.2, min(best) + 0.4 * (max(best) - min(best))),
        fontsize=8,
        color="crimson",
    )
    ax.set_xlabel("# strategies in recipe (m)")
    ax.set_ylabel("test oracle Pearson R (matched budget)")
    ax.set_title("Exhaustive recipe search: best ensemble vs # strategies")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(report: dict, out_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes2d = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes2d.ravel()

    # Panel A: control ladder
    ax = axes[0]
    ladder = report["control_ladder"]
    names = list(ladder)
    med = [ladder[n]["test_oracle_mse"] for n in names]
    lo = [ladder[n]["test_oracle_mse"] - ladder[n]["test_oracle_mse_lo"] for n in names]
    hi = [ladder[n]["test_oracle_mse_hi"] - ladder[n]["test_oracle_mse"] for n in names]
    ax.bar(range(len(names)), med, yerr=[lo, hi], capsize=4, color="#4477aa")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("test oracle MSE")
    ax.set_title(f"A. Control ladder (matched N={report['control_ladder_budget']})")

    # Panel B: leave-one-out delta
    ax = axes[1]
    loo = report["leave_one_out"]["leave_one_out"]
    labs = [r["dropped"] for r in loo]
    dm = [r["delta_mse"] for r in loo]
    dlo = [r["delta_mse"] - r["delta_mse_lo"] for r in loo]
    dhi = [r["delta_mse_hi"] - r["delta_mse"] for r in loo]
    colors = ["#cc6677" if r["significant"] else "#bbbbbb" for r in loo]
    ax.barh(range(len(labs)), dm, xerr=[dlo, dhi], capsize=3, color=colors)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("MSE increase when dropped (>0 = strategy helps)")
    ax.set_title("B. Leave-one-out (red = significant)")

    # Panel C: forward selection curve
    ax = axes[2]
    curve = report["forward_selection"]
    ks = [c["k"] for c in curve]
    cm = [c["test_oracle_mse"] for c in curve]
    clo = [c["test_oracle_mse_lo"] for c in curve]
    chi = [c["test_oracle_mse_hi"] for c in curve]
    ax.plot(ks, cm, "-o", color="#228833", ms=4)
    ax.fill_between(ks, clo, chi, alpha=0.2, color="#228833")
    ax.set_xlabel("# strategies (greedy, best-first by val)")
    ax.set_ylabel("test oracle MSE")
    ax.set_title("C. Forward selection (plateau = enough)")

    # Panel D: budget sweep (crossover of concentrate-vs-diversify)
    ax = axes[3]
    palette = {"random_only": "#999999", "mixed6": "#ee6677", "all_strategies": "#4477aa"}
    for name, pts in report["budget_sweep"].items():
        if not pts:
            continue
        xs = [p["n_models_drawn"] for p in pts]
        ys = [p["test_oracle_mse"] for p in pts]
        ylo = [p["test_oracle_mse_lo"] for p in pts]
        yhi = [p["test_oracle_mse_hi"] for p in pts]
        color = palette.get(name, "#228833" if name.startswith("best_single") else "#000000")
        ax.plot(xs, ys, "-o", ms=4, label=name, color=color)
        ax.fill_between(xs, ylo, yhi, alpha=0.15, color=color)
    ax.set_xlabel("matched train budget N (models)")
    ax.set_ylabel("test oracle MSE")
    ax.set_title("D. Budget sweep (where diversity starts to pay)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--n_total_budget", type=int, default=48)
    # Two bootstrap counts: ENet-subsample fits (ladder/parity/sweep) are expensive, so
    # default lower; test-set resampling (LOO, forward-selection) is cheap, default higher.
    ap.add_argument("--n_boot_fit", type=int, default=200)
    ap.add_argument("--n_boot_test", type=int, default=1000)
    ap.add_argument(
        "--budget_grid",
        default="6,12,24,48,96,192,301",
        help="comma-separated matched budgets N for the budget-sweep panel",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--all_subsets",
        action="store_true",
        help="run the exhaustive recipe search (2^K subsets); heavy, off by default",
    )
    ap.add_argument("--n_boot_subsets", type=int, default=15)
    ap.add_argument("--max_subset_k", type=int, default=9)
    args = ap.parse_args()
    budget_grid = [int(x) for x in args.budget_grid.split(",")]

    pool_dir = Path(args.pool_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pool_dir / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    val_mats, test_mats, labels = load_pool(pool_dir)
    present = set(val_mats)
    mixed6 = MIXED6_PHASE2 if set(MIXED6_PHASE2) & present else MIXED6_HPSEARCH

    ladder, ladder_budget = analysis_control_ladder(
        val_mats, test_mats, labels, mixed6, args.n_total_budget, args.n_boot_fit, rng
    )
    report = {
        "pool_dir": str(pool_dir),
        "n_total_budget": args.n_total_budget,
        "control_ladder_budget": ladder_budget,
        "n_boot_fit": args.n_boot_fit,
        "n_boot_test": args.n_boot_test,
        "mixed6": [s for s in mixed6 if s in present],
        "strategy_counts": {s: int(m.shape[0]) for s, m in val_mats.items()},
        "control_ladder": ladder,
        "leave_one_out": analysis_leave_one_out(val_mats, test_mats, labels, args.n_boot_test, rng),
        "forward_selection": analysis_forward_selection(
            val_mats, test_mats, labels, args.n_boot_test, rng
        ),
        "matched_parity": analysis_matched_parity(
            val_mats, test_mats, labels, mixed6, args.n_total_budget, args.n_boot_fit, rng
        ),
        "budget_sweep": analysis_budget_sweep(
            val_mats, test_mats, labels, mixed6, budget_grid, args.n_boot_fit, rng
        ),
    }

    if args.all_subsets:
        allsub = analysis_all_subsets(
            val_mats,
            test_mats,
            labels,
            args.n_boot_subsets,
            rng,
            max_k=args.max_subset_k,
        )
        report["all_subsets"] = allsub
        plot_all_subsets(allsub, out_dir / "all_subsets_recipe.png")

    (out_dir / "ablation_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_dir / 'ablation_report.json'}")
    make_figure(report, out_dir / "combination_ablation.png")


if __name__ == "__main__":
    main()
