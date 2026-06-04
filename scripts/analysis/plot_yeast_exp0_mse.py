#!/usr/bin/env python3
"""Yeast Exp 0 scaling figures — Pearson + MSE, for the oracle / student
candidates we have data for.

Yeast uses two different result.json schemas depending on when the run was
launched:

  legacy ("scaling_v2"-style): test_metrics keyed by ``random / genomic /
                                snv_abs``; n_train inferred from ``n_samples``;
                                ``label_source`` field tells real vs oracle.
  new ("ag_s2_warm"-style):    test_metrics keyed by ``in_dist / ood /
                                snv_delta`` and parallel ``*_real`` keys for
                                the same metric eval'd against real MPRA labels.

The plot bridges them via the alias map below — both schemas produce the same
3-panel layout (Genomic Sequences / High-Activity Designed / SNV Effect).

Saves to ``results/scaling/yeast/`` so the layout matches K562:
  - panel2_yeast_oracle_comparison_pearson.{png,pdf}
  - panel2_yeast_oracle_comparison_mse.{png,pdf}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]


# Aliases per test set — first-match wins. ``*_real`` keys (when present)
# evaluate against real MPRA labels and are preferred over the same key
# without ``_real`` (which evaluates against oracle predictions).
ALIASES = {
    "in_dist": ["in_dist_real", "random", "in_dist", "in_distribution"],
    "ood": ["ood_real", "genomic", "ood"],
    "snv_delta": ["snv_delta_real", "snv_delta", "snv", "snv_abs"],
}


def robust_band(vals):
    n = len(vals)
    if n <= 2:
        m = float(np.mean(vals)) if n else float("nan")
        return m, m
    if n == 3:
        s = sorted(vals)
        return (s[0] + s[1]) / 2.0, (s[1] + s[2]) / 2.0
    return float(np.percentile(vals, 25)), float(np.percentile(vals, 75))


def load_dir(top: Path, get_n=None):
    """Walk a directory of result.json files. ``get_n(d)`` extracts n_train —
    falls back to ``d['n_train']`` then ``d['n_samples']``."""
    by_n = defaultdict(list)
    for rj in top.rglob("result.json"):
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        n = (get_n(d) if get_n else None) or d.get("n_train") or d.get("n_samples")
        if not n:
            continue
        by_n[int(n)].append(d.get("test_metrics", {}) or {})
    return by_n


def extract(by_n, ts_key: str, metric: str):
    aliases = ALIASES[ts_key]
    sizes, means, lows, highs = [], [], [], []
    for n in sorted(by_n):
        vals = []
        for tm in by_n[n]:
            for alias in aliases:
                if alias in tm and isinstance(tm[alias], dict) and metric in tm[alias]:
                    vals.append(tm[alias][metric])
                    break
        if vals:
            lo, hi = robust_band(vals)
            sizes.append(n)
            means.append(float(np.mean(vals)))
            lows.append(lo)
            highs.append(hi)
    return np.array(sizes), np.array(means), np.array(lows), np.array(highs)


# ── Yeast oracle/student candidates ────────────────────────────────────
# Real-vs-oracle assignment is inferred from val_pearson_r: when validation is
# scored against the model's own oracle predictions you get val ≈ 0.95+, so any
# dir consistently above that bar is treated as oracle-label-trained.
#
#   exp0_s2_warm/yeast       val=0.87→0.91  ← AG S2 warmstart on REAL labels
#   exp0_yeast_scaling_v2    val=0.48→0.62  ← DREAM-RNN-style on REAL labels
#   exp0_yeast_scaling_ag_v2 val=0.38→0.53  ← AG variant on REAL labels (older)
#   exp0_yeast_ag_s1_reps    val=0.94→0.98  ← AG S1 student on ORACLE labels
#   exp0_yeast_dream_oracle  val=0.89→0.98  ← DREAM-RNN student on ORACLE labels
CURVES = {
    "AG S2 warmstart (real labels)": {
        "dir": REPO / "outputs" / "exp0_s2_warm" / "yeast",
        "color": "#2980B9",
        "ls": "-",
        "lw": 2.5,
        "marker": "s",
    },
    "DREAM-RNN (real labels)": {
        "dir": REPO / "outputs" / "exp0_yeast_scaling_v2",
        "color": "#E74C3C",
        "ls": "-",
        "lw": 2.5,
        "marker": "o",
    },
    "AG variant (real labels, older)": {
        "dir": REPO / "outputs" / "exp0_yeast_scaling_ag_v2",
        "color": "#8E44AD",
        "ls": "-",
        "lw": 2.0,
        "marker": "s",
    },
    "AG S1 student (oracle labels)": {
        "dir": REPO / "outputs" / "exp0_yeast_ag_s1_reps",
        "color": "#5DADE2",
        "ls": "--",
        "lw": 1.8,
        "marker": "v",
    },
    "DREAM-RNN student (oracle labels)": {
        "dir": REPO / "outputs" / "exp0_yeast_dream_oracle",
        "color": "#D4A017",
        "ls": "--",
        "lw": 1.8,
        "marker": "o",
    },
}


def make_plot(metric: str, ylabel: str, title: str, out_path: Path, log_y: bool = False):
    panels = [
        ("in_dist", "A. Genomic Sequences"),
        ("ood", "B. High-Activity Designed Sequences"),
        ("snv_delta", "C. SNV Effect"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (ts_key, panel_title) in zip(axes, panels):
        for label, cfg in CURVES.items():
            if not cfg["dir"].is_dir():
                continue
            by_n = load_dir(cfg["dir"])
            if not by_n:
                continue
            sizes, means, lows, highs = extract(by_n, ts_key, metric)
            if not len(sizes):
                continue
            ax.plot(
                sizes,
                means,
                color=cfg["color"],
                label=label if "A." in panel_title else "",
                linewidth=cfg["lw"],
                linestyle=cfg["ls"],
                marker=cfg["marker"],
                markersize=5,
            )
            if (highs - lows).any():
                ax.fill_between(sizes, lows, highs, alpha=0.15, color=cfg["color"])
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("N Training Sequences")
        if "A." in panel_title:
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=10, loc="best")
        ax.set_title(panel_title, fontweight="bold", fontsize=14)
        ax.grid(alpha=0.3, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelleft=True)

    fig.suptitle(
        title
        + "\nshaded band = IQR-style across seeds (3 seeds: avg-of-2 lowest…avg-of-2 highest; ≥4 seeds: 25th–75th percentile)",
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


def main():
    out_dir = REPO / "results" / "scaling" / "yeast"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample inventory
    print("=== Yeast Exp 0 inventory ===")
    for label, cfg in CURVES.items():
        if not cfg["dir"].is_dir():
            print(f"  [missing dir] {label} — {cfg['dir']}")
            continue
        by_n = load_dir(cfg["dir"])
        sizes = sorted(by_n)
        seeds = [len(by_n[n]) for n in sizes]
        print(
            f"  {label:<32s} n_sizes={len(sizes)}  sizes_max={max(sizes) if sizes else 0:>10d}  seeds_per_n={seeds}"
        )

    make_plot(
        metric="pearson_r",
        ylabel="Test Pearson R",
        title="Yeast Data Scaling — Pearson R: Oracle / Student Candidates",
        out_path=out_dir / "panel2_yeast_oracle_comparison_pearson",
        log_y=False,
    )
    make_plot(
        metric="mse",
        ylabel="Test MSE  (lower = better)",
        title="Yeast Data Scaling — MSE: Oracle / Student Candidates",
        out_path=out_dir / "panel2_yeast_oracle_comparison_mse",
        log_y=True,
    )


if __name__ == "__main__":
    main()
