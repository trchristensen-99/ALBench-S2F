"""Empirically characterize the HP-search ROUNDS knee across ALL existing per-round
curves, to set the early-stopping criteria (eps / persist / warm-up floor / horizon).

Per-round curves come from two sources:
  A. outputs/hp_step1_bakeoff/<cell>/seed*/<strategy>/ -- up to 100 trained models each
     (r*_meta.json + r*.npz holding val_pred/test_pred; labels.npz holding
     val_labels/test_oracle). One cumulative-ensemble curve per (cell x seed x strategy).
  B. outputs/hp_rounds_scaling/<cell>/algo/rounds_curve.json -- cached long-horizon
     curves (the dedicated plateau study; may exceed the 50-round bake-off horizon).

For each curve we build the cumulative-ensemble VALIDATION Pearson vs round -- the honest
deployable stopping signal, since at deploy time we don't see the oracle test set -- and
locate the diminishing-returns knee under three criteria:
  - marginal-gain (production knee_marginal_gain: persist consecutive sub-eps steps,
    floored at warmup_rounds),
  - relative fraction-of-attainable-gain (90%, warm-up floored),
  - Kneedle (geometric).
We then flag RIGHT-CENSORING (curve still climbing at the horizon) and sweep the
criteria so the stopping rule is chosen from the data, not assumed.

Read-heavy (thousands of npz) -> run on a compute node, not the login node.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_hp_rounds_and_ensemble import knee_kneedle, knee_marginal_gain  # production knees
from scipy.stats import pearsonr

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")


def _r(pred: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(pearsonr(pred[m], y[m])[0])


def bakeoff_curve(strat_dir: Path, cache_dir: Path) -> list[dict] | None:
    """Cumulative mean-ensemble val + oracle-test Pearson per round for one
    (cell x seed x strategy) dir. Caches to cache_dir keyed by path so reruns are fast."""
    tag = "__".join(strat_dir.parts[-3:])
    cache = cache_dir / f"{tag}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    lab = strat_dir / "labels.npz"
    if not lab.exists():
        return None
    lz = np.load(lab)
    val_y, test_oracle = lz["val_labels"], lz["test_oracle"]

    by_round: dict[int, list[Path]] = {}
    for m in sorted(strat_dir.glob("r*_meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("error"):
            continue
        npz = m.with_name(m.name.replace("_meta.json", ".npz"))
        if npz.exists():
            by_round.setdefault(int(meta["round"]), []).append(npz)
    if len(by_round) < 5:
        return None

    vps: list[np.ndarray] = []
    tps: list[np.ndarray] = []
    rows: list[dict] = []
    for rnd in sorted(by_round):
        for npz in by_round[rnd]:
            d = np.load(npz)
            vps.append(d["val_pred"])
            tps.append(d["test_pred"])
        rows.append(
            {
                "round": int(rnd),
                "n_models": len(vps),
                "ensemble_val_r": _r(np.mean(np.stack(vps), axis=0), val_y),
                "ensemble_test_oracle_r": _r(np.mean(np.stack(tps), axis=0), test_oracle),
            }
        )
    cache.write_text(json.dumps(rows, indent=2))
    return rows


def knee_relgain(rounds, ys, frac: float = 0.90, warmup: int = 8) -> int:
    """First round reaching `frac` of attainable gain over round-0, warm-up floored."""
    ys = np.asarray(ys, float)
    rounds = np.asarray(rounds)
    y0 = ys[0]
    gain = float(ys.max() - y0)
    if gain <= 1e-9:
        return int(rounds[0])
    thr = y0 + frac * gain
    hit = next((i for i, y in enumerate(ys) if y >= thr), len(ys) - 1)
    return int(max(int(rounds[hit]), warmup))


def is_censored(rounds, ys, tail: int = 5, eps: float = 0.002) -> bool:
    """Curve still climbing at the horizon: mean per-step gain over the last `tail`
    rounds still exceeds eps (so the true knee may lie beyond the observed horizon)."""
    ys = np.asarray(ys, float)
    if len(ys) <= tail:
        return True
    return (ys[-1] - ys[-1 - tail]) > eps * tail


def round_reaching(rounds, ys, frac: float) -> int:
    ys = np.asarray(ys, float)
    rounds = np.asarray(rounds)
    y0 = ys[0]
    gain = float(ys.max() - y0)
    if gain <= 1e-9:
        return int(rounds[0])
    thr = y0 + frac * gain
    return int(rounds[next((i for i, y in enumerate(ys) if y >= thr), len(ys) - 1)])


def collect_curves(out_dir: Path) -> list[dict]:
    cache_dir = out_dir / "curve_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    curves: list[dict] = []

    # Source A: Step-1 bake-off, one curve per (cell x seed x strategy)
    for strat_dir in sorted((REPO / "outputs/hp_step1_bakeoff").glob("*_d*/seed*/*")):
        if not strat_dir.is_dir() or strat_dir.name == "ablation":
            continue
        rows = bakeoff_curve(strat_dir, cache_dir)
        if not rows:
            continue
        cell = strat_dir.parts[-3]
        mm = re.search(r"_d(\d+)", cell)
        curves.append(
            {
                "source": "bakeoff",
                "cell": cell,
                "reservoir": cell.replace("k562_", "").split("_d")[0],
                "D": int(mm.group(1)) if mm else 0,
                "seed": strat_dir.parts[-2],
                "strategy": strat_dir.name,
                "rounds": [r["round"] for r in rows],
                "val": [r["ensemble_val_r"] for r in rows],
                "oracle": [r["ensemble_test_oracle_r"] for r in rows],
            }
        )

    # Source B: long-horizon rounds-scaling cells (pre-aggregated, mixed strategies)
    for f in sorted((REPO / "outputs/hp_rounds_scaling").glob("*/algo/rounds_curve.json")):
        rows = json.loads(f.read_text())
        if len(rows) < 5:
            continue
        cell = f.parts[-3]
        mm = re.search(r"_d(\d+)", cell)
        curves.append(
            {
                "source": "rounds_scaling",
                "cell": cell,
                "reservoir": cell.replace("k562_", "").split("_d")[0],
                "D": int(mm.group(1)) if mm else 0,
                "seed": "longhorizon",
                "strategy": "mixed_all",
                "rounds": [r["round"] for r in rows],
                "val": [r["ensemble_val_r"] for r in rows],
                "oracle": [r.get("ensemble_test_oracle_r") for r in rows],
            }
        )
    return curves


def analyze(curves: list[dict], eps: float, warmup: int, persist: int) -> list[dict]:
    recs = []
    for c in curves:
        rnd, val = c["rounds"], c["val"]
        horizon = int(max(rnd))
        k_mg = int(
            round(
                knee_marginal_gain(
                    rnd, val, eps=eps, rounds=rnd, warmup_rounds=warmup, persist=persist
                )
            )
        )
        k_rel = knee_relgain(rnd, val, 0.90, warmup)
        k_kd = int(round(knee_kneedle(rnd, val)))
        cens = is_censored(rnd, val, tail=5, eps=eps)
        recs.append(
            {
                **{k: c[k] for k in ("source", "cell", "reservoir", "D", "seed", "strategy")},
                "horizon": horizon,
                "knee_mg": k_mg,
                "knee_rel90": k_rel,
                "knee_kneedle": k_kd,
                "round_95pct": round_reaching(rnd, val, 0.95),
                "round_99pct": round_reaching(rnd, val, 0.99),
                "censored": bool(cens),
                "knee_at_ceiling": bool(k_mg >= horizon - persist),
            }
        )
    return recs


def mean_fraction_curve(curves: list[dict]):
    """Mean normalized fraction-of-gain (val) on a common round grid across all curves."""
    norm = []
    for c in curves:
        rnd = np.asarray(c["rounds"], float)
        y = np.asarray(c["val"], float)
        y0, gain = y[0], float(y.max() - y[0])
        norm.append((rnd, (y - y0) / gain if gain > 1e-9 else np.ones_like(y)))
    # Grid to the MEDIAN horizon (not min): a single short/incomplete curve must
    # not truncate the common grid. np.interp clamps shorter curves to their final
    # fraction (1.0) past their last round, so the mean still converges to ~1.0.
    gmax = int(np.median([r.max() for r, _ in norm]))
    grid = np.arange(0, gmax + 1)
    stack = np.vstack([np.interp(grid, r, f) for r, f in norm])
    return grid, stack.mean(axis=0), stack.std(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(REPO / "outputs/analysis_figures/rounds_knee"))
    ap.add_argument("--eps", type=float, default=0.002)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--persist", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    curves = collect_curves(out)
    print(f"collected {len(curves)} per-round curves", flush=True)
    if not curves:
        raise SystemExit("no per-round curves found")
    recs = analyze(curves, args.eps, args.warmup, args.persist)

    bo = [r for r in recs if r["source"] == "bakeoff"]
    mg = np.array([r["knee_mg"] for r in bo])
    rel = np.array([r["knee_rel90"] for r in bo])
    kd = np.array([r["knee_kneedle"] for r in bo])
    cens_frac = float(np.mean([r["censored"] for r in bo]))
    ceil_frac = float(np.mean([r["knee_at_ceiling"] for r in bo]))
    horizon = int(np.median([r["horizon"] for r in bo]))

    # sensitivity sweep over (eps, persist) for the marginal-gain rule
    eps_grid = [0.001, 0.002, 0.003, 0.005]
    persist_grid = [2, 3, 4, 5]
    sens_med = np.zeros((len(eps_grid), len(persist_grid)))
    sens_cens = np.zeros_like(sens_med)
    for i, e in enumerate(eps_grid):
        for j, p in enumerate(persist_grid):
            rr = analyze(curves, e, args.warmup, p)
            rb = [x for x in rr if x["source"] == "bakeoff"]
            sens_med[i, j] = float(np.median([x["knee_mg"] for x in rb]))
            sens_cens[i, j] = float(np.mean([x["censored"] for x in rb]))

    grid, mfrac, sfrac = mean_fraction_curve([c for c in curves if c["source"] == "bakeoff"])
    r90 = int(grid[next((i for i, v in enumerate(mfrac) if v >= 0.90), len(grid) - 1)])
    r95 = int(grid[next((i for i, v in enumerate(mfrac) if v >= 0.95), len(grid) - 1)])
    r99 = int(grid[next((i for i, v in enumerate(mfrac) if v >= 0.99), len(grid) - 1)])

    summary = {
        "n_curves": len(curves),
        "n_bakeoff_curves": len(bo),
        "horizon_median": horizon,
        "stopping_criteria_tested": {
            "eps": args.eps,
            "warmup": args.warmup,
            "persist": args.persist,
        },
        "knee_round_marginal_gain": {
            "median": float(np.median(mg)),
            "p90": float(np.percentile(mg, 90)),
            "max": int(mg.max()),
            "mean": float(mg.mean()),
        },
        "knee_round_rel90": {
            "median": float(np.median(rel)),
            "p90": float(np.percentile(rel, 90)),
            "max": int(rel.max()),
        },
        "knee_round_kneedle": {
            "median": float(np.median(kd)),
            "p90": float(np.percentile(kd, 90)),
        },
        "censored_fraction": cens_frac,
        "knee_at_ceiling_fraction": ceil_frac,
        "mean_curve_round_reaching": {"90pct": r90, "95pct": r95, "99pct": r99},
        "sensitivity_eps_grid": eps_grid,
        "sensitivity_persist_grid": persist_grid,
        "sensitivity_median_knee": sens_med.tolist(),
        "sensitivity_censored_fraction": sens_cens.tolist(),
        "per_curve": recs,
    }
    (out / "rounds_knee_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- fig 1: knee-round distributions vs the horizon ceiling ----
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, horizon + 2)
    ax.hist(
        mg, bins=bins, alpha=0.55, label=f"marginal-gain (eps={args.eps}, persist={args.persist})"
    )
    ax.hist(rel, bins=bins, alpha=0.55, label="relative 90%-gain")
    ax.hist(kd, bins=bins, alpha=0.45, label="Kneedle")
    ax.axvline(horizon, color="black", ls="--", lw=1.5, label=f"horizon ceiling (r{horizon})")
    ax.set_xlabel("knee round")
    ax.set_ylabel("# bake-off curves")
    ax.set_title(
        f"Where the rounds knee falls ({len(bo)} curves) — "
        f"censored {cens_frac:.0%}, at-ceiling {ceil_frac:.0%}"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"fig1_knee_distribution.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 2: mean fraction-of-gain curve with diminishing-returns markers ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, mfrac, "-o", ms=3, color="navy", label="mean fraction of attainable gain")
    ax.fill_between(grid, mfrac - sfrac, mfrac + sfrac, alpha=0.18, color="navy", label="±1 SD")
    for frac, rr, col in [(0.90, r90, "green"), (0.95, r95, "orange"), (0.99, r99, "crimson")]:
        ax.axhline(frac, color=col, ls=":", lw=1)
        ax.axvline(rr, color=col, ls=":", lw=1)
        ax.annotate(
            f"{frac:.0%} @ r{rr}",
            xy=(rr, frac),
            xytext=(rr + 0.5, frac - 0.06),
            fontsize=8,
            color=col,
        )
    ax.set_xlabel("HP-search round")
    ax.set_ylabel("fraction of attainable val-gain")
    ax.set_title("Diminishing returns: mean normalized gain vs round (bake-off curves)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"fig2_mean_fraction_of_gain.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 3: sensitivity heatmaps (median knee + censored fraction) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, M, title, fmt in [
        (axes[0], sens_med, "median knee round", "{:.0f}"),
        (axes[1], sens_cens, "censored fraction", "{:.0%}"),
    ]:
        im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(persist_grid)), persist_grid)
        ax.set_yticks(range(len(eps_grid)), eps_grid)
        ax.set_xlabel("persist (consecutive sub-eps steps)")
        ax.set_ylabel("eps")
        ax.set_title(title)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(
                    j, i, fmt.format(M[i, j]), ha="center", va="center", color="white", fontsize=8
                )
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Marginal-gain stopping rule — sensitivity to eps × persist")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"fig3_sensitivity.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 4: knee round by strategy (does evo need longer than random/llm?) ----
    str016 = sorted({r["strategy"] for r in bo})
    data = [[r["knee_mg"] for r in bo if r["strategy"] == s] for s in str016]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, tick_labels=str016, showmeans=True)
    ax.axhline(horizon, color="black", ls="--", lw=1, label=f"horizon r{horizon}")
    ax.set_ylabel("knee round (marginal-gain)")
    ax.set_title("Rounds-to-knee by search strategy (D=30k bake-off)")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"fig4_knee_by_strategy.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("=== rounds-knee analysis ===", flush=True)
    print(f"curves: {len(curves)} ({len(bo)} bake-off)  median horizon r{horizon}")
    print(
        f"marginal-gain knee: median r{np.median(mg):.0f}  p90 r{np.percentile(mg, 90):.0f}  max r{mg.max()}"
    )
    print(f"relative-90% knee:  median r{np.median(rel):.0f}  p90 r{np.percentile(rel, 90):.0f}")
    print(f"censored fraction (still climbing at horizon): {cens_frac:.1%}")
    print(f"knee-at-ceiling fraction: {ceil_frac:.1%}")
    print(f"mean curve reaches 90%/95%/99% of gain at rounds {r90}/{r95}/{r99}")
    print(f"wrote summary + 4 figs to {out}")


if __name__ == "__main__":
    main()
