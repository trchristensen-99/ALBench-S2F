#!/usr/bin/env python3
"""AG vs LegNet × real vs oracle scaling on chr-split data, 3 panels.

Panels: Genomic Reference / SNV Effect / High-Activity Designed
Each model evaluated against the labels it was trained on (real→real, oracle→oracle).

Matches the wide 3-panel layout used in exp0_pearson_vs_mse with a narrow right
legend strip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LEGNET = REPO / "outputs/legnet_chrsplit_scaling"
AG = REPO / "outputs/ag_chrsplit_scaling"
OUT = REPO / "outputs/chrsplit_ag_vs_legnet"

NS = [3197, 6395, 15987, 31974, 63949, 159871, 296382]

CURVES = [
    {
        "key": "legnet_real",
        "base": LEGNET,
        "label_source": "real",
        "label": "LegNet\n(real labels)",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "--",
    },
    {
        "key": "legnet_orac",
        "base": LEGNET,
        "label_source": "oracle",
        "label": "LegNet\n(oracle labels)",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "key": "ag_real",
        "base": AG,
        "label_source": "real",
        "label": "AG\n(real labels)",
        "color": "#D55E00",
        "marker": "D",
        "linestyle": "--",
    },
    {
        "key": "ag_orac",
        "base": AG,
        "label_source": "oracle",
        "label": "AG\n(oracle labels)",
        "color": "#D55E00",
        "marker": "D",
        "linestyle": "-",
    },
]

# Each panel: (summary key suffix matching summary.json fields, display title)
PANELS = [
    ("test", "Genomic Reference (held-out chromosomes)"),
    ("snv_delta", "SNV Effect (Δ log2FC)"),
    ("ood", "High-Activity Designed"),
]


def load_curve(base_dir: Path, label_source: str) -> dict[int, list[dict]]:
    out = {}
    for N in NS:
        runs = []
        for seed_dir in (base_dir / label_source).glob(f"n{N}/seed*"):
            sj = seed_dir / "summary.json"
            if sj.exists():
                try:
                    runs.append(json.loads(sj.read_text()))
                except Exception:
                    pass
        if runs:
            out[N] = runs
    return out


def aggregate(runs, panel_key, target, use_calibrated_mse=True):
    """panel_key in {'test','snv_delta','ood'}; target in {'oracle','real'}.

    For MSE: uses calibration[*_calibrated] if available (affine-recalibrated on val).
    Pearson is affine-invariant so always the same.
    """
    rs, ms = [], []
    field = f"{panel_key}_vs_{target}"
    cal_field = f"{panel_key}_vs_{target}_calibrated"
    for d in runs:
        m = d.get(field)
        if m is None:
            continue
        pr = m.get("pearson_r")
        # Prefer calibrated MSE if available
        cal = d.get("calibration", {}).get(cal_field)
        if use_calibrated_mse and cal and cal.get("mse") is not None:
            mse = cal["mse"]
        else:
            mse = m.get("mse")
        if pr is None or mse is None:
            continue
        if not (np.isfinite(pr) and np.isfinite(mse)):
            continue
        rs.append(pr)
        ms.append(mse)
    if not rs:
        return None
    rs, ms = np.array(rs), np.array(ms)
    # 1-R² = fraction of variance unexplained (robust to scale shifts)
    one_minus_r2 = 1.0 - rs**2

    def _band(vals):
        """Median-consistent band edges. n=3: mean of 2 lowest / 2 highest;
        n>=4: 25th/75th percentile; n<=1: degenerate (lo=hi=value)."""
        v = np.sort(vals)
        if len(v) <= 1:
            return float(v[0]), float(v[0])
        if len(v) == 2:
            return float(v[0]), float(v[1])
        if len(v) == 3:
            return float(v[:2].mean()), float(v[1:].mean())
        return float(np.percentile(v, 25)), float(np.percentile(v, 75))

    pr_lo, pr_hi = _band(rs)
    mse_lo, mse_hi = _band(ms)
    r2_lo, r2_hi = _band(one_minus_r2)
    return {
        "pearson_median": float(np.median(rs)),
        "pearson_lo": pr_lo,
        "pearson_hi": pr_hi,
        "mse_median": float(np.median(ms)),
        "mse_lo": mse_lo,
        "mse_hi": mse_hi,
        "one_minus_r2_median": float(np.median(one_minus_r2)),
        "one_minus_r2_lo": r2_lo,
        "one_minus_r2_hi": r2_hi,
        "n_seeds": len(rs),
    }


# LegNet-only variant: real vs oracle as distinct hues (red vs blue), both solid.
LEGNET_ONLY_CURVES = [
    {**CURVES[0], "label": "LegNet\n(real labels)", "color": "#D62728", "linestyle": "-"},
    {**CURVES[1], "label": "LegNet\n(oracle labels)", "color": "#1F77B4", "linestyle": "-"},
]


def plot_panel(ax, curves_data, panel_key, panel_label, metric, curves=None):
    for c in curves or CURVES:
        by_n = curves_data.get(c["key"], {})
        xs, ys, los, his = [], [], [], []
        for N in sorted(by_n.keys()):
            # evaluate against the labels the model trained on
            agg = aggregate(by_n[N], panel_key, target=c["label_source"])
            if agg is None:
                continue
            xs.append(N)
            ys.append(agg[f"{metric}_median"])
            los.append(agg[f"{metric}_lo"])
            his.append(agg[f"{metric}_hi"])
        if not xs:
            continue
        xs, ys, los, his = np.array(xs), np.array(ys), np.array(los), np.array(his)
        ax.fill_between(xs, los, his, color=c["color"], alpha=0.15, edgecolor="none")
        ax.plot(
            xs,
            ys,
            color=c["color"],
            marker=c["marker"],
            label=c["label"],
            markersize=8,
            linewidth=2,
            alpha=0.92,
            linestyle=c["linestyle"],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training set size (log scale)", fontsize=15)
    if metric in ("mse", "one_minus_r2"):
        ax.set_yscale("log")
    # y-axis label set on leftmost panel only in main()
    ax.set_title(panel_label, fontsize=16, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xticks([3_000, 10_000, 30_000, 100_000, 300_000])
    ax.set_xticklabels(
        ["3,000", "10,000", "30,000", "100,000", "300,000"], rotation=30, ha="right", fontsize=12
    )
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.tick_params(axis="y", which="minor", labelsize=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--legnet_only",
        action="store_true",
        help="Plot only the two LegNet curves, real vs oracle as red vs blue.",
    )
    args = ap.parse_args()
    curves = LEGNET_ONLY_CURVES if args.legnet_only else CURVES
    name_tag = "_legnet" if args.legnet_only else ""

    OUT.mkdir(parents=True, exist_ok=True)
    curves_data = {}
    for c in curves:
        curves_data[c["key"]] = load_curve(c["base"], c["label_source"])
        n_total = sum(len(v) for v in curves_data[c["key"]].values())
        print(
            f"  {c['label'].replace(chr(10), ' '):<30}  {len(curves_data[c['key']])} N values, {n_total} total runs"
        )

    for metric in ["pearson", "mse", "one_minus_r2"]:
        fig = plt.figure(figsize=(21, 6))
        gs = fig.add_gridspec(1, 3, wspace=0.12)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        # Compress plot area; legend placed manually outside on the right
        fig.subplots_adjust(left=0.055, right=0.90, bottom=0.18, top=0.90)
        if metric == "pearson":
            for a in axes[1:]:
                a.sharey(axes[0])
        for i, (panel_key, panel_label) in enumerate(PANELS):
            plot_panel(axes[i], curves_data, panel_key, panel_label, metric, curves=curves)
        ylabel_map = {
            "pearson": "Pearson R",
            "mse": "MSE (log scale)",
            "one_minus_r2": "1 − R² (log scale)",
        }
        axes[0].set_ylabel(ylabel_map[metric], fontsize=15)
        if metric == "pearson":
            axes[0].set_ylim(0, 1.0)
        # Legend placed just to the right of the rightmost panel (small gap)
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels_,
            loc="center left",
            bbox_to_anchor=(0.905, 0.5),
            fontsize=10.5,
            framealpha=0.92,
            handlelength=3.2,
            labelspacing=1.0,
            borderpad=0.6,
            handletextpad=0.5,
        )
        suffix_map = {"pearson": "", "mse": "_mse", "one_minus_r2": "_1minusR2"}
        suffix = suffix_map[metric]
        fig.savefig(OUT / f"main{name_tag}{suffix}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"main{name_tag}{suffix}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved main{name_tag}{suffix}")


if __name__ == "__main__":
    main()
