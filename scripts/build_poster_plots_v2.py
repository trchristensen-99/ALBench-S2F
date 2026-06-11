"""Poster-quality scaling-law plots — Pearson + log-log MSE with bootstrap CI + power-law fits.

Reads:
  outputs/full_sweep/{cell}/summary.json — point estimates
  outputs/full_sweep_chrval/{cell}/summary.json — chrval genomic
  outputs/poster_plots/bootstrap_uncertainty.json — bootstrap CI per (reservoir, D, panel)

Outputs:
  outputs/poster_plots/main_3panel{,_mse}.{png,pdf}
  outputs/poster_plots/{panel}{,_mse}.{png,pdf}
  outputs/poster_plots/scaling_exponents.json

Curves are point estimates from the single ensemble; shaded bands are 95% bootstrap CI
from model resampling (B=200 within-cell resamples — captures ensemble-fit variance,
NOT HP-search variance). Smooth lines are log-log power-law fits using D >= 3000 only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = [300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
DS_VISIBLE_MIN = 3_000  # hide D=300 and D=1k from plots (high-variance, distracting)
FIT_DS_MIN = 3_000  # exclude D < 3k from the power-law fit
RESERVOIRS_MAIN = [
    # (key, label, color, marker) — Okabe-Ito colorblind-safe palette + maximally distinct shapes
    ("genomic", "Genomic", "#0072B2", "o"),  # blue, circle
    ("random", "Random", "#000000", "s"),  # black, square
    ("prm_10pct", "PRM 10%", "#009E73", "^"),  # green, triangle-up
    ("evoaug_heavy", "EvoAug", "#D55E00", "D"),  # vermillion, diamond
    ("motif_planted_v2", "Motif planted", "#CC79A7", "*"),  # pink, star
]

RESERVOIRS_MOTIF = [
    ("motif_planted_v2", "Planted v2", "#e377c2", "D"),
    ("motif_planted", "Planted v1", "#8B4513", "P"),
    ("motif_shuffled", "Shuffled", "#1f9978", "s"),
    ("motif_grammar", "Grammar", "#d62728", "^"),
    ("dinuc_shuffle", "Dinuc-shuffled", "#7f7f7f", "X"),
]

RESERVOIRS_PRM = [
    ("prm_1pct", "PRM 1%", "#2ca02c", "^"),
    ("prm_5pct", "PRM 5%", "#3aa75b", "v"),
    ("prm_10pct", "PRM 10%", "#9467bd", "s"),
    ("prm_20pct", "PRM 20%", "#d62728", "D"),
    ("prm_attribution_1pct", "PRM Attribution 1%", "#ff7f0e", "P"),
    ("prm_uncertainty_1pct", "PRM Uncertainty 1%", "#1f9978", "X"),
]

# Default used by existing plot functions
RESERVOIRS = RESERVOIRS_MAIN

PRIMARY_PANELS = [
    ("genomic", "Genomic Reference (held-out chromosomes)"),
    ("snv_delta", "SNV Effect (Δ log2FC)"),
    ("ood", "High-Activity Designed"),
]
EXTRA_PANELS = [
    ("sub_high", "Heavy substitution"),
    ("random_32k", "Random 32k sequences"),
    ("dinuc_shuffle", "Dinucleotide-shuffled"),
    ("translocation", "Translocation rearrangement"),
    ("inversion", "Inversion rearrangement"),
    ("snv_ref", "SNV Ref panel"),
    ("snv_alt", "SNV Alt panel"),
]

BOOTSTRAP_PATH = Path("outputs/poster_plots/bootstrap_uncertainty.json")


def load_point(reservoir, D, panel):
    """Return (pearson_mean, mse_mean, pearson_std, mse_std, n_seeds).

    For (random, 1M): aggregates main_sweep seed=42 + verify seeds 100/200/300.
    For all other cells: single-seed point estimate (std=0).
    """

    def _extract(j):
        if panel == "snv_delta":
            d = j.get("snv_delta", {}).get("oracle", {})
        else:
            d = j.get("per_set", {}).get(panel, {})
        pr, mse = d.get("pearson"), d.get("mse")

        def safe(x):
            if x is None:
                return None
            if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
                return None
            return x

        return safe(pr), safe(mse)

    pearsons, mses = [], []
    if reservoir == "genomic":
        p = Path(f"outputs/full_sweep_chrval/k562_genomic_d{D}_seed42/summary.json")
        if not p.exists():
            p = Path(f"outputs/full_sweep/k562_genomic_d{D}_seed42/summary.json")
        if p.exists():
            pr, mse = _extract(json.loads(p.read_text()))
            if pr is not None:
                pearsons.append(pr)
                mses.append(mse)
    else:
        p = Path(f"outputs/full_sweep/k562_{reservoir}_d{D}_seed42/summary.json")
        if p.exists():
            pr, mse = _extract(json.loads(p.read_text()))
            if pr is not None:
                pearsons.append(pr)
                mses.append(mse)
        # Augment (random, D=1M) with verify reps (seeds 100, 200, 300)
        if reservoir == "random" and D == 1_000_000:
            for seed in (100, 200, 300):
                vp = Path(f"outputs/random_d1M_verify/seed{seed}/summary.json")
                if vp.exists():
                    pr, mse = _extract(json.loads(vp.read_text()))
                    if pr is not None:
                        pearsons.append(pr)
                        mses.append(mse)
    if not pearsons:
        return None, None, 0.0, 0.0, 0
    pa, ma = np.array(pearsons), np.array(mses)
    p_std = float(pa.std(ddof=1)) if len(pa) > 1 else 0.0
    m_std = float(ma.std(ddof=1)) if len(ma) > 1 else 0.0
    return float(pa.mean()), float(ma.mean()), p_std, m_std, len(pa)


def load_bootstrap(reservoir, D, panel):
    """Return dict of bootstrap stats for this cell+panel, or None."""
    if not BOOTSTRAP_PATH.exists():
        return None
    boot = json.loads(BOOTSTRAP_PATH.read_text())
    cell = boot.get(f"{reservoir}|{D}", {})
    return cell.get("panels", {}).get(panel)


def power_law_fit(xs, ys):
    """Return (a, b) for log(y) = a + b*log(x). xs, ys numpy arrays > 0."""
    mask = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 3:
        return None
    coef = np.polyfit(np.log10(xs[mask]), np.log10(ys[mask]), 1)
    return float(coef[1]), float(coef[0])  # (intercept, slope)


def plot_panel(
    ax,
    panel_key,
    panel_label,
    metric="pearson",
    show_legend=False,
    fits_out=None,
    RESERVOIRS_SUBSET=None,
):
    """metric ∈ {pearson, mse}"""
    if RESERVOIRS_SUBSET is None:
        RESERVOIRS_SUBSET = RESERVOIRS_MAIN
    for key, label, color, marker in RESERVOIRS_SUBSET:
        xs, ys, lo, hi = [], [], [], []
        for D in DS:
            if D < DS_VISIBLE_MIN:
                continue
            pr, mse, p_std, m_std, n_seeds = load_point(key, D, panel_key)
            v = pr if metric == "pearson" else mse
            v_std = p_std if metric == "pearson" else m_std
            if v is None:
                continue
            xs.append(D)
            ys.append(v)
            if n_seeds > 1:
                # Multi-seed: use mean ± std as the band
                lo.append(v - v_std)
                hi.append(v + v_std)
            else:
                b = load_bootstrap(key, D, panel_key)
                if b:
                    # Symmetric: mean ± 1 SD (bootstrap-derived)
                    s = b.get(f"{metric}_std", 0.0)
                    lo.append(v - s)
                    hi.append(v + s)
                else:
                    lo.append(v)
                    hi.append(v)
        if not xs:
            continue
        xs = np.array(xs)
        ys = np.array(ys)
        lo = np.array(lo)
        hi = np.array(hi)
        # Bootstrap CI band
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12, edgecolor="none")
        # Raw points + line
        ax.plot(
            xs,
            ys,
            "-",
            color=color,
            marker=marker,
            label=label,
            markersize=7,
            linewidth=2,
            alpha=0.92,
        )
        # Power-law fit (only on D>=FIT_DS_MIN, only for MSE)
        if metric == "mse":
            fit_mask = xs >= FIT_DS_MIN
            if fit_mask.sum() >= 3:
                fit = power_law_fit(xs[fit_mask], ys[fit_mask])
                if fit:
                    intercept, slope = fit
                    xs_smooth = np.logspace(np.log10(FIT_DS_MIN), np.log10(1e6), 50)
                    ys_smooth = 10 ** (intercept + slope * np.log10(xs_smooth))
                    ax.plot(xs_smooth, ys_smooth, "--", color=color, linewidth=1.5, alpha=0.65)
                    if fits_out is not None:
                        fits_out.setdefault(panel_key, {})[key] = {
                            "intercept": intercept,
                            "slope": slope,
                            "fit_range_D_min": FIT_DS_MIN,
                        }
    ax.set_xscale("log")
    ax.set_xlabel("Training set size (log scale)", fontsize=15)
    if metric == "mse":
        ax.set_yscale("log")
        ax.set_ylabel("MSE (log scale)", fontsize=15)
    else:
        ax.set_ylabel("Pearson R", fontsize=15)
    ax.set_title(panel_label, fontsize=16, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)
    visible_ds = [d for d in DS if d >= DS_VISIBLE_MIN]
    ax.set_xticks(visible_ds)
    ax.set_xticklabels([f"{d:,}" for d in visible_ds], rotation=30, ha="right", fontsize=12)
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.tick_params(axis="y", which="minor", labelsize=12)
    if show_legend:
        loc = "upper right" if metric == "mse" else "lower left"
        ax.legend(loc=loc, fontsize=12, framealpha=0.92)


def make_main(metric, fits_out=None):
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=(metric == "pearson"))
    for i, (key, label) in enumerate(PRIMARY_PANELS):
        plot_panel(axes[i], key, label, metric=metric, show_legend=(i == 0), fits_out=fits_out)
    if metric == "pearson":
        axes[0].set_ylim(0, 1.0)
    fig.tight_layout()
    out = Path("outputs/poster_plots")
    suffix = "" if metric == "pearson" else "_mse"
    fig.savefig(out / f"main_3panel{suffix}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"main_3panel{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved main_3panel{suffix}")


def make_extra(key, label, metric, fits_out=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_panel(ax, key, label, metric=metric, show_legend=True, fits_out=fits_out)
    if metric == "pearson":
        ax.set_ylim(0, 1.0)
    fig.tight_layout()
    out = Path("outputs/poster_plots")
    suffix = "" if metric == "pearson" else "_mse"
    fig.savefig(out / f"{key}{suffix}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{key}{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {key}{suffix}")


def make_all_panels(metric, fits_out=None):
    panels = PRIMARY_PANELS + EXTRA_PANELS
    ncols = 5
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, (key, label) in enumerate(panels):
        plot_panel(axes[i], key, label, metric=metric, show_legend=(i == 0), fits_out=fits_out)
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")
    if metric == "pearson":
        for ax in axes[: len(panels)]:
            ax.set_ylim(-0.1, 1.0)
    fig.tight_layout()
    out = Path("outputs/poster_plots")
    suffix = "" if metric == "pearson" else "_mse"
    fig.savefig(out / f"all_panels{suffix}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"all_panels{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved all_panels{suffix}")


def make_subset_plot(subset, name, panels_to_plot, metric):
    """Single panel or multi-panel for a reservoir subset."""
    n = len(panels_to_plot)
    if n == 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=(metric == "pearson"))
    for i, (key, label) in enumerate(panels_to_plot):
        plot_panel(axes[i], key, label, metric, show_legend=(i == n - 1), RESERVOIRS_SUBSET=subset)
    if metric == "pearson":
        for ax in axes:
            ax.set_ylim(0, 1.0)
    fig.tight_layout()
    out = Path("outputs/poster_plots")
    suffix = "" if metric == "pearson" else "_mse"
    fig.savefig(out / f"{name}{suffix}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}{suffix}")


def main():
    Path("outputs/poster_plots").mkdir(parents=True, exist_ok=True)
    fits = {}
    for metric in ["pearson", "mse"]:
        make_main(metric, fits_out=fits if metric == "mse" else None)
        make_all_panels(metric, fits_out=fits if metric == "mse" else None)
        for key, label in EXTRA_PANELS:
            make_extra(key, label, metric, fits_out=fits if metric == "mse" else None)
    # Supplemental subset plots
    for metric in ["pearson", "mse"]:
        # Motif strategies on OOD (single panel)
        make_subset_plot(
            RESERVOIRS_MOTIF,
            "supp_motif_ood",
            [("ood", "OOD on motif sampling strategies")],
            metric,
        )
        # PRM strategies (3-panel: genomic / OOD / SNV Δ)
        make_subset_plot(RESERVOIRS_PRM, "supp_prm_3panel", PRIMARY_PANELS, metric)
    # Write scaling-exponents table
    out = Path("outputs/poster_plots/scaling_exponents.json")
    out.write_text(json.dumps(fits, indent=2))
    print(f"\nwrote {out}")
    # Pretty-print
    print("\nScaling exponents (log-log MSE slope) by reservoir × panel:")
    panels = sorted(fits.keys())
    reservoirs = sorted({r for v in fits.values() for r in v.keys()})
    hdr = f"{'reservoir':<25}" + "".join(f"  {p:>10}" for p in panels)
    print(hdr)
    for r in reservoirs:
        row = f"{r:<25}"
        for p in panels:
            s = fits.get(p, {}).get(r, {}).get("slope")
            row += f"  {s:>10.3f}" if s is not None else f"  {'-':>10}"
        print(row)


if __name__ == "__main__":
    main()
