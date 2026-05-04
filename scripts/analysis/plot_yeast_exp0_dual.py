"""Plot yeast Exp 0 dual scaling curves (real labels vs ensemble pseudolabels).

For each of the two arms, produces a 3-panel figure (in_dist / ood / snv)
in two metric variants (Pearson R and MSE), all log-x and log-y where
sensible.

Outputs (results/scaling/):
    yeast_exp0_dual_pearson.{png,pdf}
    yeast_exp0_dual_mse.{png,pdf}

Real-arm schema (from exp0_yeast_scaling.py):
    test_metrics.random      → in_dist
    test_metrics.genomic     → ood
    test_metrics.snv         → snv (variant-effect delta)
Oracle-arm schema (from exp0_yeast_oracle_scaling.py):
    test_metrics.in_dist     → in_dist (against pseudolabels)
    (no ood/snv keys — pseudolabel arm only evaluates against pseudo
    in-dist.)
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


def robust_band(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n <= 2:
        m = float(np.mean(vals)) if n else float("nan")
        return m, m
    if n == 3:
        s = sorted(vals)
        return (s[0] + s[1]) / 2.0, (s[1] + s[2]) / 2.0
    return float(np.percentile(vals, 25)), float(np.percentile(vals, 75))


def _aggregate_real() -> dict[str, dict[int, dict[str, list[float]]]]:
    """Walk real-arm results. Returns {panel_key: {n: {metric: [vals]}}}."""
    out: dict[str, dict[int, dict[str, list[float]]]] = {
        "in_dist": defaultdict(lambda: {"pearson": [], "mse": []}),
        "ood": defaultdict(lambda: {"pearson": [], "mse": []}),
        "snv": defaultdict(lambda: {"pearson": [], "mse": []}),
    }
    for f in (REPO / "outputs" / "exp0_yeast_real_scaling").rglob("result.json"):
        d = json.loads(f.read_text())
        n = d.get("n_samples", 0)
        if n == 0:
            continue
        tm = d.get("test_metrics", {})
        for panel_key, src in (("in_dist", "random"), ("ood", "genomic"), ("snv", "snv")):
            sub = tm.get(src, {})
            p = sub.get("pearson_r")
            m = sub.get("mse")
            if p is not None:
                out[panel_key][n]["pearson"].append(float(p))
            if m is not None:
                out[panel_key][n]["mse"].append(float(m))
    return out


def _aggregate_oracle() -> dict[int, dict[str, list[float]]]:
    """Walk oracle-arm results. Returns {n: {metric: [vals]}}."""
    out: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"pearson": [], "mse": []})
    for f in (REPO / "outputs" / "exp0_yeast_oracle_scaling").rglob("result.json"):
        d = json.loads(f.read_text())
        n = d.get("n_samples", 0)
        if n == 0:
            continue
        sub = d.get("test_metrics", {}).get("in_dist", {})
        p = sub.get("pearson_r")
        m = sub.get("mse")
        if p is not None:
            out[n]["pearson"].append(float(p))
        if m is not None:
            out[n]["mse"].append(float(m))
    return out


def _curve(by_n: dict, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sizes, means, lows, highs = [], [], [], []
    for n in sorted(by_n):
        vals = by_n[n][metric]
        if not vals:
            continue
        sizes.append(n)
        means.append(float(np.mean(vals)))
        lo, hi = robust_band(vals)
        lows.append(lo)
        highs.append(hi)
    return np.array(sizes), np.array(means), np.array(lows), np.array(highs)


def _clip(arr: np.ndarray, floor: float) -> np.ndarray:
    out = arr.astype(float, copy=True)
    out[out <= 0] = floor
    return out


def make_panel(metric: str, out_path: Path):
    real = _aggregate_real()
    oracle = _aggregate_oracle()

    panels = [
        ("in_dist", "A. In-distribution (random subset)"),
        ("ood", "B. OOD (native genomic promoters)"),
        ("snv", "C. SNV effect (delta alt − ref)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    LOG_PEARSON_FLOOR = 0.01

    for ax, (panel_key, title) in zip(axes, panels):
        # Real arm (DRNN, real labels)
        s_r, m_r, lo_r, hi_r = _curve(real[panel_key], metric)
        if len(s_r):
            if metric == "pearson":
                m_r, lo_r, hi_r = (_clip(a, LOG_PEARSON_FLOOR) for a in (m_r, lo_r, hi_r))
            ax.plot(
                s_r,
                m_r,
                color="#D4A017",
                lw=2.5,
                marker="o",
                ms=5,
                label="Real labels (DREAM→MAUDE)" if panel_key == "in_dist" else None,
            )
            if (hi_r - lo_r).any():
                ax.fill_between(s_r, lo_r, hi_r, alpha=0.18, color="#D4A017")

        # Oracle arm only has the in_dist panel
        if panel_key == "in_dist":
            s_o, m_o, lo_o, hi_o = _curve(oracle, metric)
            if len(s_o):
                if metric == "pearson":
                    m_o, lo_o, hi_o = (_clip(a, LOG_PEARSON_FLOOR) for a in (m_o, lo_o, hi_o))
                ax.plot(
                    s_o,
                    m_o,
                    color="#2980B9",
                    lw=2.5,
                    marker="s",
                    ms=5,
                    ls="--",
                    label="Ensemble pseudolabels (5 DRNN + 5 DCNN)",
                )
                if (hi_o - lo_o).any():
                    ax.fill_between(s_o, lo_o, hi_o, alpha=0.18, color="#2980B9")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N Training Sequences")
        if metric == "pearson":
            ax.set_ylim(LOG_PEARSON_FLOOR, 1.0)
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.grid(alpha=0.3, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelleft=True)

    ylabel = (
        "Test Pearson R (higher = better)" if metric == "pearson" else "Test MSE (lower = better)"
    )
    metric_short = "Pearson R" if metric == "pearson" else "MSE"
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=11, loc="lower right" if metric == "pearson" else "upper right")

    fig.suptitle(
        f"Yeast Exp 0 — {metric_short}: DREAM-RNN scaling on real DREAM labels vs 10-model ensemble pseudolabels\n"
        "shaded band = IQR-style across seeds (3 reps per size).  Real-arm test labels = MAUDE (cross-assay).  "
        "Oracle-arm test labels = ensemble pseudolabels (DREAM-scale, internal).",
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
    out_dir = REPO / "results" / "scaling"
    make_panel("pearson", out_dir / "yeast_exp0_dual_pearson")
    make_panel("mse", out_dir / "yeast_exp0_dual_mse")


if __name__ == "__main__":
    main()
