"""Selection-regret figure: does the VAL-selected ElasticNet ensemble approximate
the best-possible single model from our search?

For every cell (reservoir x D, both seeds) and for EACH eval-set that exists:
  1. val-selected ensemble test-Pearson  -- greedy ElasticNet selected on VAL
     (the pipeline's normal output), scored on that eval-set's test oracle.
  2. oracle-best single model test-Pearson -- the single model in the cell with
     the HIGHEST test-Pearson on that eval-set (hindsight/oracle selection).
  3. val-best single model test-Pearson  -- the single model with highest
     val-Pearson (what naive val-selection would pick), scored on that eval-set.

  selection_regret = (val-selected ensemble) - (oracle-best single)
  ensemble_gain    = (val-selected ensemble) - (val-best single)

Reuses scripts/analysis/slope_analysis.py: load_cell, greedy_ensemble,
Ensemble.predict(set).  No GPU / no inference: saved predictions only.

CUDA-OOM stub models (NaN / missing val or test preds) are already dropped by
load_cell (shape / finite checks).  Cells with <1 usable model are skipped.
"""

import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slope_analysis import (  # noqa: E402
    ALL_RESERVOIRS,
    K4_TAGS,
    LLM_TAGS,
    greedy_ensemble,
    load_cell,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = "outputs/overnight"
OUTDIR = os.path.join(BASE, "final_analysis")
DS = [10000, 30000, 100000]
SEEDS = [42, 43]
TAGS = K4_TAGS + LLM_TAGS  # pool all algorithmic + persona strategies
MAX_POOL = 40
MAX_SIZE = 12


def safe_pearson(pred, truth):
    if pred is None or truth is None:
        return None
    if len(pred) != len(truth):
        return None
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(truth))):
        return None
    if np.std(pred) == 0 or np.std(truth) == 0:
        return None
    r = pearsonr(pred, truth)[0]
    return float(r) if np.isfinite(r) else None


def analyze_cell(cell):
    """Return {eval_set: {ens, oracle_best, val_best, regret, gain, ...}}."""
    out = {}
    sets = cell.sets()
    # val-best single model (chosen ONCE by val_pearson; scored per set)
    val_best_model = max(cell.models, key=lambda m: m.val_pearson)

    # val-selected ensemble is SELECTION-INVARIANT to target_set: greedy_ensemble
    # selects purely on val (target_set only affects recorded test metrics, never the
    # chosen models / en). Build the ensemble ONCE and predict on every eval set —
    # identical numbers to per-set rebuilds, ~12x faster.
    ens = greedy_ensemble(cell, max_pool=MAX_POOL, max_size=MAX_SIZE)
    ens_ok = bool(ens.selected) and ens.en is not None

    for s in sets:
        truth = cell.oracle[s]
        # (1) val-selected ensemble, evaluated on this eval set
        ens_pear = None
        if ens_ok:
            try:
                ens_pear = safe_pearson(ens.predict(s), truth)
            except Exception:
                ens_pear = None

        # (2) oracle-best single model on this eval set (hindsight)
        best_single = None
        best_single_id = None
        for m in cell.models:
            r = safe_pearson(m.test.get(s), truth)
            if r is None:
                continue
            if best_single is None or r > best_single:
                best_single = r
                best_single_id = m.model_id

        # (3) val-best single model, scored on this eval set
        val_best_pear = safe_pearson(val_best_model.test.get(s), truth)

        if ens_pear is None or best_single is None or val_best_pear is None:
            out[s] = dict(
                ens=ens_pear,
                oracle_best_single=best_single,
                val_best_single=val_best_pear,
                regret=None,
                gain=None,
                n_models=len(cell.models),
                ens_size=len(ens.selected) if ens.selected else 0,
                oracle_best_id=best_single_id,
                val_best_id=val_best_model.model_id,
                note="incomplete (missing a required metric)",
            )
            continue

        out[s] = dict(
            ens=ens_pear,
            oracle_best_single=best_single,
            val_best_single=val_best_pear,
            regret=ens_pear - best_single,
            gain=ens_pear - val_best_pear,
            n_models=len(cell.models),
            ens_size=len(ens.selected),
            oracle_best_id=best_single_id,
            val_best_id=val_best_model.model_id,
        )
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    per_cell = []
    by_set = defaultdict(list)  # eval_set -> list of (regret, gain, ens, oracle_best)

    for R in ALL_RESERVOIRS:
        for D in DS:
            for ds in SEEDS:
                cell = load_cell(BASE, R, D, ds, TAGS)
                if cell is None:
                    print(f"[skip] {R} d{D} s{ds}: no usable cell")
                    continue
                res = analyze_cell(cell)
                cell_rec = dict(reservoir=R, D=D, seed=ds, n_models=len(cell.models), sets={})
                for s, v in res.items():
                    cell_rec["sets"][s] = v
                    if v.get("regret") is not None:
                        by_set[s].append(
                            (v["regret"], v["gain"], v["ens"], v["oracle_best_single"])
                        )
                per_cell.append(cell_rec)
                n_ok = sum(1 for v in res.values() if v.get("regret") is not None)
                print(f"[ok]   {R} d{D} s{ds}: {len(cell.models)} models, {n_ok}/{len(res)} sets scored")

    # aggregate
    agg = {}
    for s, rows in by_set.items():
        reg = np.array([r[0] for r in rows], float)
        gn = np.array([r[1] for r in rows], float)
        agg[s] = dict(
            n_cells=len(rows),
            mean_regret=float(np.mean(reg)),
            median_regret=float(np.median(reg)),
            mean_gain=float(np.mean(gn)),
            median_gain=float(np.median(gn)),
            frac_regret_ge0=float(np.mean(reg >= -1e-9)),
            frac_gain_gt0=float(np.mean(gn > 1e-9)),
            min_regret=float(np.min(reg)),
        )

    dump = dict(
        config=dict(base=BASE, Ds=DS, seeds=SEEDS, tags=TAGS, max_pool=MAX_POOL, max_size=MAX_SIZE,
                    reservoirs=ALL_RESERVOIRS),
        per_cell=per_cell,
        aggregate=agg,
    )
    jpath = os.path.join(OUTDIR, "selection_regret.json")
    with open(jpath, "w") as f:
        json.dump(dump, f, indent=2, default=str)
    print(f"\n[wrote {jpath}]")

    # ---- print aggregate table ----
    # order eval-sets by n_cells desc then name
    order = sorted(agg.keys(), key=lambda s: (-agg[s]["n_cells"], s))
    print("\n=== AGGREGATE: selection-regret & ensemble-gain per eval-set ===")
    print(f"{'eval_set':<16s} {'n':>4s} {'mean_regret':>12s} {'med_regret':>11s} "
          f"{'mean_gain':>10s} {'med_gain':>9s} {'reg>=0':>7s} {'gain>0':>7s} {'min_reg':>8s}")
    for s in order:
        a = agg[s]
        print(f"{s:<16s} {a['n_cells']:>4d} {a['mean_regret']:>12.4f} {a['median_regret']:>11.4f} "
              f"{a['mean_gain']:>10.4f} {a['median_gain']:>9.4f} {a['frac_regret_ge0']:>7.2f} "
              f"{a['frac_gain_gt0']:>7.2f} {a['min_regret']:>8.4f}")

    # ---- figure ----
    make_figure(by_set, order, OUTDIR)


def make_figure(by_set, order, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    colors = {s: cmap(i % 20) for i, s in enumerate(order)}

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # ---- panel A: box+strip of regret (blue) and gain (orange) per eval-set ----
    ax = axes[0]
    positions = np.arange(len(order))
    reg_data = [[r[0] for r in by_set[s]] for s in order]
    gain_data = [[r[1] for r in by_set[s]] for s in order]
    w = 0.34

    bp1 = ax.boxplot(reg_data, positions=positions - w / 1.8, widths=w, patch_artist=True,
                     showfliers=False, manage_ticks=False)
    bp2 = ax.boxplot(gain_data, positions=positions + w / 1.8, widths=w, patch_artist=True,
                     showfliers=False, manage_ticks=False)
    for b in bp1["boxes"]:
        b.set(facecolor="#4C72B0", alpha=0.55)
    for b in bp2["boxes"]:
        b.set(facecolor="#DD8452", alpha=0.55)
    for bp in (bp1, bp2):
        for med in bp["medians"]:
            med.set(color="black", linewidth=1.4)

    rng = np.random.default_rng(0)
    for i, s in enumerate(order):
        rv = reg_data[i]
        gv = gain_data[i]
        ax.scatter(np.full(len(rv), i - w / 1.8) + rng.uniform(-0.06, 0.06, len(rv)), rv,
                   s=14, color="#1f3b6f", alpha=0.6, zorder=3)
        ax.scatter(np.full(len(gv), i + w / 1.8) + rng.uniform(-0.06, 0.06, len(gv)), gv,
                   s=14, color="#8c4a1f", alpha=0.6, zorder=3)
    ax.axhline(0, color="red", lw=1.2, ls="--", zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Pearson difference")
    ax.set_title("Selection regret (blue = ens - oracle-best single)\n"
                 "Ensemble gain (orange = ens - val-best single)", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#4C72B0", alpha=0.55, label="selection regret"),
                       Patch(facecolor="#DD8452", alpha=0.55, label="ensemble gain")],
              loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ---- panel B: scatter ensemble-test (y) vs oracle-best-single-test (x) ----
    ax = axes[1]
    lo, hi = 1.0, 0.0
    for s in order:
        xs = [r[3] for r in by_set[s]]  # oracle-best single
        ys = [r[2] for r in by_set[s]]  # ensemble test
        ax.scatter(xs, ys, s=28, color=colors[s], label=s, alpha=0.8, edgecolor="k", linewidth=0.3)
        if xs:
            lo = min(lo, min(xs), min(ys))
            hi = max(hi, max(xs), max(ys))
    pad = 0.03 * (hi - lo + 1e-6)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("oracle-best single model  test-Pearson")
    ax.set_ylabel("val-selected ensemble  test-Pearson")
    ax.set_title("Ensemble vs oracle-best single (points above y=x = ensemble wins)", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle("Selection-regret: does val-selected ensemble approximate the best-possible model?",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fpath = os.path.join(outdir, "selection_regret.png")
    fig.savefig(fpath, dpi=140)
    print(f"[wrote {fpath}]")


if __name__ == "__main__":
    main()
