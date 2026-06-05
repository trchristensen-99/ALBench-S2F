"""Aggregate + plot the long-horizon HP-optimization (rounds) study and the
ensemble-composition (control-ladder) study.

Produces, under --fig_dir:
  fig1_rounds_plateau.{png,pdf}     50-round cumulative-ensemble curve per cell
  fig2_strategy_rounds.{png,pdf}    per-search-strategy efficiency vs round (D=30k)
  fig3_composition_vs_D.{png,pdf}   control-ladder compositions vs dataset size
  fig4_composition_at_D30k.{png,pdf} per-reservoir composition at D=30k

Also writes rounds_curve.json next to each rounds cell it aggregates.
Run on a compute node (loads a few hundred prediction npz).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def _r(pred: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(pearsonr(pred[m], y[m])[0])


def knee_marginal_gain(xs, ys, eps: float = 0.002):
    """First x where the per-step gain in y drops below eps (cost-aware stop).

    xs, ys are monotone-in-x sample points (e.g. cumulative GPU-seconds vs ensemble R).
    Returns the x at the knee (or the last x if the gain never falls below eps).
    """
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for i in range(1, len(xs)):
        if ys[i] - ys[i - 1] < eps:
            return float(xs[i - 1])
    return float(xs[-1])


def knee_kneedle(xs, ys):
    """Geometric (Kneedle) knee: point of max distance to the chord between the
    first and last points of the normalized increasing curve. Returns x at knee."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 3:
        return float(xs[-1])
    xn = (xs - xs.min()) / (np.ptp(xs) or 1.0)
    yn = (ys - ys.min()) / (np.ptp(ys) or 1.0)
    dist = yn - xn  # distance above the y=x chord for a concave increasing curve
    return float(xs[int(np.argmax(dist))])


def aggregate_rounds_curve(cell_algo: Path) -> list[dict]:
    """Cumulative mean-ensemble val/test Pearson per round for one cell."""
    lz = np.load(cell_algo / "labels.npz")
    val_y, test_oracle, test_true = lz["val_labels"], lz["test_oracle"], lz["test_true"]

    by_round: dict[int, list[tuple[Path, float]]] = {}
    for m in sorted(cell_algo.glob("r*_meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("error"):
            continue
        npz = m.with_name(m.name.replace("_meta.json", ".npz"))
        if not npz.exists():
            continue
        t = float(meta.get("train_time_sec", 0.0) or 0.0)
        by_round.setdefault(int(meta["round"]), []).append((npz, t))

    val_preds: list[np.ndarray] = []
    test_preds: list[np.ndarray] = []
    best_single = float("-inf")
    cum_gpu_sec = 0.0
    rows: list[dict] = []
    for rnd in sorted(by_round):
        for npz, t in by_round[rnd]:
            d = np.load(npz)
            vp = d["val_pred"]
            val_preds.append(vp)
            test_preds.append(d["test_pred"])
            best_single = max(best_single, _r(vp, val_y))
            cum_gpu_sec += t
        ens_val = np.mean(np.stack(val_preds), axis=0)
        ens_test = np.mean(np.stack(test_preds), axis=0)
        rows.append(
            {
                "round": rnd,
                "n_models": len(val_preds),
                "cum_gpu_sec": cum_gpu_sec,
                "ensemble_val_r": _r(ens_val, val_y),
                "ensemble_test_oracle_r": _r(ens_test, test_oracle),
                "ensemble_test_true_r": _r(ens_test, test_true),
                "best_single_val_r": best_single,
            }
        )
    (cell_algo / "rounds_curve.json").write_text(json.dumps(rows, indent=2))
    return rows


def fig1_rounds_plateau(cells: dict[str, list[dict]], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(cells), figsize=(6.2 * len(cells), 4.6), squeeze=False)
    for ax, (name, rows) in zip(axes[0], cells.items()):
        rnd = [r["round"] for r in rows]
        ax.plot(rnd, [r["ensemble_val_r"] for r in rows], "-o", ms=3, label="ensemble (val)")
        ax.plot(
            rnd,
            [r["ensemble_test_true_r"] for r in rows],
            "-s",
            ms=3,
            label="ensemble (test, real MPRA)",
        )
        ax.plot(
            rnd,
            [r["best_single_val_r"] for r in rows],
            "--",
            color="gray",
            label="best single model (val)",
        )
        # knee marker: first round within 0.005 of the final ensemble val R
        final = rows[-1]["ensemble_val_r"]
        knee = next((r for r in rows if r["ensemble_val_r"] >= final - 0.005), rows[-1])
        ax.axvline(knee["round"], color="crimson", ls=":", lw=1)
        ax.annotate(
            f"knee r{knee['round']}\n({knee['n_models']} models)",
            xy=(knee["round"], knee["ensemble_val_r"]),
            xytext=(knee["round"] + 2, final - 0.04),
            fontsize=8,
            color="crimson",
        )
        ax.set_title(name)
        ax.set_xlabel("HP-search round")
        ax.set_ylabel("Pearson R")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Long-horizon HP optimization: ensemble quality vs. rounds (D=30k, 50 rounds)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig1_rounds_plateau.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig1b_efficiency_gpusec(cells: dict[str, list[dict]], fig_dir: Path) -> None:
    """Ensemble quality vs cumulative GPU-seconds (the fair cost axis), with the
    marginal-gain and Kneedle knees marked — the basis for 'optimal rounds'."""
    fig, axes = plt.subplots(1, len(cells), figsize=(6.2 * len(cells), 4.6), squeeze=False)
    for ax, (name, rows) in zip(axes[0], cells.items()):
        gpu_h = [r["cum_gpu_sec"] / 3600.0 for r in rows]
        yv = [r["ensemble_val_r"] for r in rows]
        ax.plot(gpu_h, yv, "-o", ms=3, label="ensemble (val)")
        ax.plot(
            gpu_h,
            [r["ensemble_test_true_r"] for r in rows],
            "-s",
            ms=3,
            label="ensemble (test, real MPRA)",
        )
        if any(g > 0 for g in gpu_h):
            k_mg = knee_marginal_gain(gpu_h, yv) / 1.0
            k_kd = knee_kneedle(gpu_h, yv)
            ax.axvline(k_mg, color="crimson", ls=":", lw=1.2, label="marginal-gain knee")
            ax.axvline(k_kd, color="purple", ls="--", lw=1.0, label="Kneedle knee")
            # round + n_models at the marginal-gain knee
            knee_row = min(rows, key=lambda r: abs(r["cum_gpu_sec"] / 3600.0 - k_mg))
            ax.annotate(
                f"knee r{knee_row['round']} ({knee_row['n_models']} models)",
                xy=(k_mg, knee_row["ensemble_val_r"]),
                xytext=(k_mg, min(yv) + 0.02),
                fontsize=8,
                color="crimson",
            )
        ax.set_title(name)
        ax.set_xlabel("cumulative GPU-hours")
        ax.set_ylabel("Pearson R")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("HP-search efficiency vs compute (fair cost axis): where to stop")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig1b_efficiency_gpusec.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_strategy_rounds(rounds_summary: Path, fig_dir: Path) -> None:
    data = json.loads(rounds_summary.read_text())
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for strat, rows in sorted(data.items()):
        if not isinstance(rows, list) or len(rows) < 2:
            continue
        ax.plot(
            [r["round"] for r in rows],
            [r["ensemble_val_r"] for r in rows],
            "-o",
            ms=3,
            label=strat,
        )
    ax.set_title("Search-strategy efficiency vs. round (D=30k)")
    ax.set_xlabel("HP-search round")
    ax.set_ylabel("ensemble val Pearson R")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig2_strategy_rounds.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ladder_value(cl: dict, prefix: str, field: str = "test_oracle_pearson"):
    for k, v in cl.items():
        if k.startswith(prefix):
            return v.get(field)
    return None


def load_ablations(hp_search: Path) -> list[dict]:
    out = []
    for f in glob.glob(str(hp_search / "*" / "ablation" / "ablation_report.json")):
        d = json.loads(Path(f).read_text())
        cell = Path(f).parts[-3]
        mm = re.search(r"_d(\d+)", cell)
        D = int(mm.group(1)) if mm else 0
        res = cell.replace("k562_", "").split("_d")[0]
        cl = d.get("control_ladder", {})
        out.append(
            {
                "D": D,
                "reservoir": res,
                "random_only": _ladder_value(cl, "random_only"),
                "best_single": _ladder_value(cl, "best_single"),
                "mixed6": _ladder_value(cl, "mixed6"),
                "all_strategies": _ladder_value(cl, "all_strategies"),
            }
        )
    return out


def fig3_composition_vs_D(rows: list[dict], fig_dir: Path, max_D: int = 300000) -> None:
    comps = ["random_only", "best_single", "mixed6", "all_strategies"]
    Ds = sorted({r["D"] for r in rows if r["D"] <= max_D})
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for comp in comps:
        means, los, his, xs = [], [], [], []
        for D in Ds:
            vals = [r[comp] for r in rows if r["D"] == D and r[comp] is not None]
            if not vals:
                continue
            xs.append(D)
            means.append(np.mean(vals))
            los.append(np.mean(vals) - np.min(vals))
            his.append(np.max(vals) - np.mean(vals))
        ax.errorbar(xs, means, yerr=[los, his], marker="o", capsize=3, label=comp)
    ax.set_xscale("log")
    ax.set_title(
        "Ensemble composition vs. dataset size (matched budget; mean±range over 5 reservoirs)"
    )
    ax.set_xlabel("training set size D")
    ax.set_ylabel("test Pearson R (vs oracle)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig3_composition_vs_D.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_composition_at_D(rows: list[dict], fig_dir: Path, D: int = 30000) -> None:
    comps = ["random_only", "best_single", "mixed6", "all_strategies"]
    sub = sorted([r for r in rows if r["D"] == D], key=lambda r: r["reservoir"])
    if not sub:
        return
    reservoirs = [r["reservoir"] for r in sub]
    x = np.arange(len(reservoirs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, comp in enumerate(comps):
        ax.bar(x + (i - 1.5) * w, [r[comp] for r in sub], w, label=comp)
    ax.set_xticks(x)
    ax.set_xticklabels(reservoirs, rotation=20, ha="right", fontsize=8)
    ax.set_title(f"Ensemble composition at D={D:,} (matched budget, per reservoir)")
    ax.set_ylabel("test Pearson R (vs oracle)")
    ax.set_ylim(0.6, None)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig4_composition_at_D{D}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds_root", default="outputs/hp_rounds_scaling")
    ap.add_argument(
        "--phase2_rounds_summary", default="outputs/phase2_longrounds/D30000/rounds_summary.json"
    )
    ap.add_argument("--hp_search", default="outputs/hp_search")
    ap.add_argument("--fig_dir", default="outputs/analysis_figures")
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[str, list[dict]] = {}
    for algo in sorted(Path(args.rounds_root).glob("*/algo")):
        name = algo.parent.name.replace("k562_", "")
        print(f"[rounds] aggregating {name} ...", flush=True)
        cells[name] = aggregate_rounds_curve(algo)
    if cells:
        fig1_rounds_plateau(cells, fig_dir)
        fig1b_efficiency_gpusec(cells, fig_dir)

    p2 = Path(args.phase2_rounds_summary)
    if p2.exists():
        fig2_strategy_rounds(p2, fig_dir)

    abl = load_ablations(Path(args.hp_search))
    if abl:
        fig3_composition_vs_D(abl, fig_dir)
        fig4_composition_at_D(abl, fig_dir, D=30000)
    print(f"figures written to {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
