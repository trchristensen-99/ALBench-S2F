#!/usr/bin/env python3
"""Scaling-law figures with both log-log MSE and log-log Pearson R panels.

Generates 4 figures in ``results/scaling/``:

  - Exp 0 (AG vs LegNet × real vs oracle labels)
      ``exp0_scaling_3panel_mse.{png,pdf}``     — log-log MSE
      ``exp0_scaling_3panel_pearson.{png,pdf}`` — log-log Pearson R

  - Exp 1 (LegNet across reservoir-sampling strategies, AG-S2 oracle labels)
      ``exp1_scaling_3panel_mse.{png,pdf}``
      ``exp1_scaling_3panel_pearson.{png,pdf}``

The underlying ``result.json`` files store both ``pearson_r`` and ``mse`` for
each test set (in_dist, ood, snv_delta), so this script re-walks the same
source directories and pulls whichever field is requested.

Error bars use a robust IQR-style band rather than a parametric 95 % CI:

    n  = 1 or 2 → no band (single point or scatter only)
    n == 3     → low = mean(2 lowest), high = mean(2 highest)
    n  ≥ 4     → low = 25th percentile, high = 75th percentile

This keeps the band tight to the body of the seed distribution and is less
sensitive to a single outlier seed than ±1.96·σ.

Usage:
    python scripts/analysis/plot_scaling_mse.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP0_TARGET_SIZES = {6395, 15987, 31974, 63949, 159871, 296382}


# ── Robust error-band helper ─────────────────────────────────────────────
def robust_band(vals: list[float]) -> tuple[float, float]:
    """Return (low, high) for the error band — see module docstring."""
    n = len(vals)
    if n <= 2:
        m = float(np.mean(vals)) if n else float("nan")
        return m, m
    if n == 3:
        s = sorted(vals)
        return (s[0] + s[1]) / 2.0, (s[1] + s[2]) / 2.0
    return float(np.percentile(vals, 25)), float(np.percentile(vals, 75))


# ── Exp 0: AG vs LegNet (real vs oracle labels) ──────────────────────────
EXP0_CURVES = {
    "AG (real labels)": {
        "model": "alphagenome_k562_s1_ground_truth",
        "color": "#2980B9",
        "ls": "-",
        "lw": 2.5,
        "marker": "s",
    },
    "AG (oracle labels)": {
        "model": "alphagenome_k562_s1",
        "color": "#2980B9",
        "ls": "--",
        "lw": 1.8,
        "marker": "s",
    },
    "LegNet (real labels)": {
        "model": "legnet_ground_truth",
        "color": "#D4A017",
        "ls": "-",
        "lw": 2.5,
        "marker": "o",
    },
    "LegNet (oracle labels)": {
        "model": "exp0_legnet_ag_s2_redo/k562/legnet_ag_s2",
        "color": "#D4A017",
        "ls": "--",
        "lw": 1.8,
        "marker": "o",
    },
}


def load_exp0_model(model_name: str, max_n: int = 500_000):
    """Load result.json files for one model, snapping n_train to TARGET_SIZES.

    Selection rule (matches plot_scaling_3panel.py): for each n, group by
    hp_config and pick the HP with the highest mean ``val_pearson_r``."""
    if "/" in model_name:
        base = REPO / "outputs" / model_name
    else:
        base = REPO / "outputs" / "exp0_oracle_scaling_v4" / "k562" / model_name
    by_n = defaultdict(lambda: defaultdict(list))
    for rj in base.rglob("result.json"):
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        n = d.get("n_train", 0)
        if n > max_n:
            continue
        if n not in EXP0_TARGET_SIZES:
            closest = min(EXP0_TARGET_SIZES, key=lambda t: abs(t - n))
            if abs(n - closest) / closest < 0.1:
                n = closest
            else:
                continue
        hp = json.dumps(d.get("hp_config", {}), sort_keys=True)
        by_n[n][hp].append((d.get("val_pearson_r", 0), d.get("test_metrics", {})))
    out = {}
    for n in sorted(by_n):
        best_hp = max(by_n[n], key=lambda k: np.mean([v[0] for v in by_n[n][k]]))
        out[n] = [v[1] for v in by_n[n][best_hp]]
    return out


_MIN_PANEL_PEARSON_FOR_MSE = 0.05


def extract_metric(data, ts_key: str, metric: str = "mse"):
    """Pull ``metric`` from ``data[n][i][<test_set_alias>]`` and aggregate.

    Returns (sizes, means, low_band, high_band) as numpy arrays. The
    test-set keys vary across older/newer runs (``in_dist`` vs
    ``in_distribution``, optional ``_real`` suffix when both leaked + real
    metrics are saved).

    For MSE on panels where the model has no predictive signal (Pearson
    ≈ 0), MSE is meaningless — it just reflects how close the constant
    "mean prediction" lands to E[y_true]. We drop *individual reps* whose
    panel-specific Pearson is below ``_MIN_PANEL_PEARSON_FOR_MSE`` so a
    near-zero-correlation rep can't masquerade as low MSE."""
    aliases = {
        "in_dist": ["in_dist", "in_distribution", "in_dist_real"],
        "ood": ["ood", "ood_real"],
        "snv_delta": ["snv_delta", "snv_delta_real"],
    }[ts_key]
    sizes, means, lows, highs = [], [], [], []
    for n in sorted(data):
        vals = []
        for tm in data[n]:
            for alias in aliases:
                if not (alias in tm and isinstance(tm[alias], dict)):
                    continue
                if metric not in tm[alias]:
                    break
                # For MSE, require the panel's Pearson to be > threshold.
                # This drops "no-signal" reps where MSE is uninformative.
                if metric == "mse":
                    p = tm[alias].get("pearson_r")
                    if p is None or p < _MIN_PANEL_PEARSON_FOR_MSE:
                        break
                vals.append(tm[alias][metric])
                break
        if vals:
            lo, hi = robust_band(vals)
            sizes.append(n)
            means.append(float(np.mean(vals)))
            lows.append(lo)
            highs.append(hi)
    return np.array(sizes), np.array(means), np.array(lows), np.array(highs)


_METRIC_LABELS = {
    "mse": ("Test MSE  (lower = better)", "MSE"),
    "pearson_r": ("Test Pearson R  (higher = better)", "Pearson R"),
}


def _clip_for_log(arr: np.ndarray, floor: float) -> np.ndarray:
    """Replace values <= 0 with `floor` so they fit on a log axis."""
    out = arr.astype(float, copy=True)
    out[out <= 0] = floor
    return out


def make_exp0(out_path: Path, metric: str = "mse", curves: dict | None = None):
    curves = curves or EXP0_CURVES
    ylabel, metric_short = _METRIC_LABELS[metric]
    model_label = " vs ".join(sorted({n.split(" (")[0] for n in curves}))
    print(f"Loading Exp 0 ({model_label}) result.json files [{metric}]…")
    model_data = {name: load_exp0_model(cfg["model"]) for name, cfg in curves.items()}
    for name, d in model_data.items():
        print(f"  {name}: sizes={sorted(d.keys())}")

    # Floor for log-Pearson is below the noise floor for n~1k Pearson estimates;
    # individual seeds can dip below 0 on hard panels (OOD at small N) and the
    # mean is sometimes legitimately ~0.02. We drop *points* whose mean ≤ 0
    # (no signal) and clip the IQR band edges at this floor so the band is
    # bounded on a log axis without inflating the central tendency.
    LOG_PEARSON_FLOOR = 0.001
    panels = [
        ("in_dist", "A. Genomic Sequences"),
        ("ood", "B. High-Activity Designed Sequences"),
        ("snv_delta", "C. SNV Effect (Genomic Sequence − SNV Sequence)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (ts_key, title) in zip(axes, panels):
        for name, cfg in curves.items():
            sizes, means, lows, highs = extract_metric(model_data[name], ts_key, metric)
            if not len(sizes):
                continue
            if metric == "pearson_r":
                # Drop points where the mean is non-positive (no correlation
                # to plot on a log axis). Clip band edges to floor.
                mask = means > 0
                sizes, means, lows, highs = sizes[mask], means[mask], lows[mask], highs[mask]
                if not len(sizes):
                    continue
                lows = _clip_for_log(lows, LOG_PEARSON_FLOOR)
                highs = _clip_for_log(highs, LOG_PEARSON_FLOOR)
            ax.plot(
                sizes,
                means,
                color=cfg["color"],
                label=name if "A." in title else "",
                linewidth=cfg["lw"],
                linestyle=cfg["ls"],
                marker=cfg["marker"],
                markersize=5,
            )
            if (highs - lows).any():
                ax.fill_between(sizes, lows, highs, alpha=0.15, color=cfg["color"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N Training Sequences")
        if metric == "pearson_r":
            ax.set_ylim(0.01, 1.0)
        if "A." in title:
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=12, loc="upper right" if metric == "mse" else "lower right")
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.grid(alpha=0.3, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ``sharey=True`` hides tick labels on the right panels by default — re-show
        # them so each panel is independently readable.
        ax.tick_params(axis="y", labelleft=True)

    suptitle_model = model_label.replace("AG", "AlphaGenome")
    fig.suptitle(
        f"Data Scaling — {metric_short}: {suptitle_model} (K562 MPRA)\n"
        "shaded band = IQR-style across seeds (3 seeds: avg-of-2 lowest…avg-of-2 highest; "
        "≥4 seeds: 25th–75th percentile)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}.{{png,pdf}}")


# ── Exp 1: Strategy scaling ─────────────────────────────────────────────
# Match the Stowers poster data sources exactly. The poster combined three
# trees with deduplication by (strategy, n, seed):
#
#   - exp1_1_definitive: consistent-HP main runs (Stowers main set)
#   - exp1_1_final: extra reps at small N where definitive had only 3
#   - exp1_1_new_oracle: the 200k genomic point (newer-oracle re-runs)
#
# We also drop runs that failed to converge (val_pearson_r < 0.2). At small
# N (≤2k), LegNet with the consistent lr=5e-4 occasionally diverges and
# produces near-random predictions — those runs pollute IQR bands.
EXP1_GLOBS = [
    "outputs/exp1_1_definitive/k562/legnet_ag_s2/*/n*/rep*/result.json",
    "outputs/exp1_1_definitive/k562/legnet_ag_s2/*/n*/hp*/seed*/result.json",
    "outputs/exp1_1_final/k562/legnet_ag_s2/*/n*/rep*/result.json",
    "outputs/exp1_1_final/k562/legnet_ag_s2/*/n*/hp*/seed*/result.json",
    "outputs/exp1_1_new_oracle/k562/legnet_ag_s2/*/n*/rep*/result.json",
    "outputs/exp1_1_new_oracle/k562/legnet_ag_s2/*/n*/hp*/seed*/result.json",
]

# Drop runs whose val_pearson_r is below this — these are training failures
# that produce near-random predictions and pollute IQR bands at small N.
VAL_PEARSON_FAILURE_THRESHOLD = 0.2

EXP1_STRATS = {
    "random": ("Random", "#888888", "--", 2.5),
    "genomic": ("Genomic", "#1f77b4", "-", 2.0),
    "prm_1pct": ("PRM 1%", "#d62728", "-", 2.0),
    "prm_20pct": ("PRM 20%", "#8c564b", "-", 2.0),
    "motif_grammar": ("Motif Grammar", "#2ca02c", "-", 2.0),
    "evoaug_heavy": ("EvoAug", "#9467bd", "-", 2.0),
}


def load_exp1(max_n: int = 500_000):
    """Combine result.json files from definitive + final + new_oracle trees,
    dedupe by (strategy, n, seed), and drop runs with val_pearson_r below
    VAL_PEARSON_FAILURE_THRESHOLD (training failures)."""
    by_strat = defaultdict(lambda: defaultdict(list))
    seen = set()
    n_dropped = 0
    for pattern in EXP1_GLOBS:
        for f in REPO.glob(pattern):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            n = d.get("n_train", 0)
            if n == 0 or n > max_n:
                continue
            # Strategy from path
            strategy = ""
            parts = f.parts
            for s in EXP1_STRATS:
                if s in parts:
                    strategy = s
                    break
            if not strategy:
                continue
            seed = d.get("seed", str(f))
            key = (strategy, n, seed)
            if key in seen:
                continue
            # Drop training failures
            val = d.get("val_pearson_r")
            if val is not None and val < VAL_PEARSON_FAILURE_THRESHOLD:
                n_dropped += 1
                continue
            seen.add(key)
            by_strat[strategy][n].append(d.get("test_metrics", {}))
    if n_dropped:
        print(f"  filtered {n_dropped} runs with val_pearson_r < {VAL_PEARSON_FAILURE_THRESHOLD}")
    return by_strat


def make_exp1(out_path: Path, metric: str = "mse", min_n: int = 5000):
    ylabel, metric_short = _METRIC_LABELS[metric]
    print(f"Loading Exp 1 (strategy scaling) result.json files [{metric}]…")
    raw = load_exp1()
    if not raw:
        print("WARNING: no exp1_1 results found, skipping.")
        return
    for strat, ns in raw.items():
        n_seeds = sum(len(v) for v in ns.values())
        print(f"  {strat}: {len(ns)} sizes, {n_seeds} result.jsons")

    # See note in make_exp0: drop points where mean Pearson ≤ 0; clip IQR
    # band edges at the floor so the band stays bounded on a log axis.
    LOG_PEARSON_FLOOR = 0.001
    panels = [
        ("in_dist", "A. Genomic Sequences"),
        ("ood", "B. High-Activity Designed Sequences"),
        ("snv_delta", "C. SNV Effect (Genomic Sequence − SNV Sequence)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (ts_key, title) in zip(axes, panels):
        for strat, (label, color, ls, lw) in EXP1_STRATS.items():
            if strat not in raw:
                continue
            sizes, means, lows, highs = extract_metric(raw[strat], ts_key, metric)
            if not len(sizes):
                continue
            mask = sizes >= min_n
            sizes, means, lows, highs = sizes[mask], means[mask], lows[mask], highs[mask]
            if not len(sizes):
                continue
            if metric == "pearson_r":
                # Drop points where the mean is non-positive (no correlation
                # to plot on a log axis). Clip band edges to floor.
                mask2 = means > 0
                sizes, means, lows, highs = (
                    sizes[mask2],
                    means[mask2],
                    lows[mask2],
                    highs[mask2],
                )
                if not len(sizes):
                    continue
                lows = _clip_for_log(lows, LOG_PEARSON_FLOOR)
                highs = _clip_for_log(highs, LOG_PEARSON_FLOOR)
            ax.plot(
                sizes,
                means,
                color=color,
                label=label if "A." in title else "",
                linewidth=lw,
                linestyle=ls,
                marker="o",
                markersize=4,
            )
            if (highs - lows).any():
                ax.fill_between(sizes, lows, highs, alpha=0.18, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N Training Sequences")
        if metric == "pearson_r":
            ax.set_ylim(0.01, 1.0)
        if "A." in title:
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=12, loc="upper right" if metric == "mse" else "lower right")
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.grid(alpha=0.3, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ``sharey=True`` hides tick labels on the right panels by default — re-show
        # them so each panel is independently readable.
        ax.tick_params(axis="y", labelleft=True)

    fig.suptitle(
        f"Strategy Scaling — {metric_short}: LegNet (K562, AG-S2 Oracle Labels)\n"
        "shaded band = IQR-style across seeds (3 seeds: avg-of-2 lowest…avg-of-2 highest; "
        "≥4 seeds: 25th–75th percentile)\n"
        f"runs filtered if val_pearson_r < {VAL_PEARSON_FAILURE_THRESHOLD} (training failures)",
        fontsize=11,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}.{{png,pdf}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--legnet_only",
        action="store_true",
        help="Exp-0: plot only the two LegNet curves (real vs oracle labels); drop AlphaGenome.",
    )
    args = ap.parse_args()
    out_dir = REPO / "results" / "scaling"
    curves = (
        {k: v for k, v in EXP0_CURVES.items() if k.startswith("LegNet")}
        if args.legnet_only
        else None
    )
    tag = "_legnet" if args.legnet_only else ""
    for metric in ("mse", "pearson_r"):
        suffix = "mse" if metric == "mse" else "pearson"
        make_exp0(out_dir / f"exp0_scaling_3panel{tag}_{suffix}", metric=metric, curves=curves)
        if not args.legnet_only:
            make_exp1(out_dir / f"exp1_scaling_3panel_{suffix}", metric=metric)


if __name__ == "__main__":
    main()
