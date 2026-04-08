#!/usr/bin/env python3
"""Compare oracle ensemble predictions across models: AG, DREAM-RNN, LegNet, AG S2.

Analyses:
  1. Scatter of oracle_mean vs true_label for each oracle type
  2. Residual (oracle_mean - true_label) distributions by oracle
  3. Prediction uncertainty (oracle_std) distributions by oracle
  4. Inter-oracle residual correlations (are biases shared?)
  5. Pairwise oracle prediction scatters
  6. Summary metrics (bias, variance, correlation, inter-oracle correlation)

Run from repo root::

    python scripts/analysis/analyze_oracle_landscapes.py

All outputs go to ``results/oracle_landscape_analysis/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]

ORACLE_SOURCES = {
    "AG": "outputs/oracle_pseudolabels_k562_ag",
    "DREAM-RNN": "outputs/oracle_pseudolabels_k562_dream",
    "LegNet": "outputs/oracle_legnet_k562_ensemble",
    "AG S2": "outputs/oracle_pseudolabels_stage2_k562_ag",
}

# Consistent color palette across all plots
ORACLE_COLORS = {
    "AG": "#1B5E20",
    "DREAM-RNN": "#7B2D8E",
    "LegNet": "#E8602C",
    "AG S2": "#3A86C8",
}

# Publication style defaults
SCATTER_KW = dict(s=3, alpha=0.20, rasterized=True, edgecolors="none")
FIGDPI = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def _set_pub_style(ax: plt.Axes) -> None:
    """Remove gridlines, clean up spines for publication look."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4)


def _load_pseudolabels(base_dir: Path, fname: str) -> dict[str, np.ndarray] | None:
    p = base_dir / fname
    if not p.exists():
        return None
    return dict(np.load(p))


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_oracle_vs_true_scatter(
    oracles: dict[str, dict[str, np.ndarray]],
    out_png: Path,
) -> None:
    """Grid of oracle_mean vs true_label scatters, one column per oracle."""
    names = [n for n in oracles if oracles[n] is not None]
    n = len(names)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    rng = np.random.default_rng(42)

    for i, name in enumerate(names):
        ax = axes[0, i]
        d = oracles[name]
        true = d["true_label"].astype(np.float64)
        pred = d["oracle_mean"].astype(np.float64)
        mask = np.isfinite(true) & np.isfinite(pred)
        yt, yp = true[mask], pred[mask]

        n_plot = min(8000, len(yt))
        idx = rng.choice(len(yt), n_plot, replace=False)

        ax.scatter(yt[idx], yp[idx], color=ORACLE_COLORS[name], **SCATTER_KW)
        lim = [
            min(yt.min(), yp.min()) - 0.3,
            max(yt.max(), yp.max()) + 0.3,
        ]
        ax.plot(lim, lim, "k-", linewidth=0.8, alpha=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("True K562 log2FC", fontsize=10)
        if i == 0:
            ax.set_ylabel("Oracle mean prediction", fontsize=10)
        r = _safe_pearson(yt, yp)
        bias = float(np.mean(yp - yt))
        ax.set_title(
            f"{name}\nr = {r:.4f}   bias = {bias:+.3f}   n = {mask.sum():,}",
            fontsize=10,
        )
        _set_pub_style(ax)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout(w_pad=2.5)
    fig.savefig(out_png, dpi=FIGDPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_png.name}")


def plot_residual_distributions(
    oracles: dict[str, dict[str, np.ndarray]],
    out_png: Path,
) -> None:
    """Overlay of residual (oracle_mean - true_label) histograms."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(-4, 4, 120)

    for name, d in oracles.items():
        if d is None:
            continue
        true = d["true_label"].astype(np.float64)
        pred = d["oracle_mean"].astype(np.float64)
        mask = np.isfinite(true) & np.isfinite(pred)
        residual = pred[mask] - true[mask]
        ax.hist(
            residual,
            bins=bins,
            alpha=0.45,
            density=True,
            label=f"{name} (bias={np.mean(residual):+.3f})",
            color=ORACLE_COLORS[name],
        )

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Residual (oracle mean $-$ true)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Oracle residual distributions (in-dist test)", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    _set_pub_style(ax)

    fig.tight_layout()
    fig.savefig(out_png, dpi=FIGDPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_png.name}")


def plot_uncertainty_distributions(
    oracles: dict[str, dict[str, np.ndarray]],
    out_png: Path,
) -> None:
    """Overlay of oracle_std distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: raw std distributions
    ax = axes[0]
    for name, d in oracles.items():
        if d is None or "oracle_std" not in d:
            continue
        std = d["oracle_std"].astype(np.float64)
        std = std[np.isfinite(std)]
        ax.hist(
            std,
            bins=60,
            alpha=0.45,
            density=True,
            label=f"{name} (med={np.median(std):.3f})",
            color=ORACLE_COLORS[name],
        )
    ax.set_xlabel("Ensemble std", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Prediction uncertainty (ensemble disagreement)", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    _set_pub_style(ax)

    # Right: std vs absolute error (calibration check)
    ax = axes[1]
    for name, d in oracles.items():
        if d is None or "oracle_std" not in d:
            continue
        true = d["true_label"].astype(np.float64)
        pred = d["oracle_mean"].astype(np.float64)
        std = d["oracle_std"].astype(np.float64)
        mask = np.isfinite(true) & np.isfinite(pred) & np.isfinite(std)
        abs_err = np.abs(pred[mask] - true[mask])
        std_m = std[mask]

        n_bins = 20
        quantiles = np.quantile(std_m, np.linspace(0, 1, n_bins + 1))
        bin_std, bin_err = [], []
        for lo, hi in zip(quantiles[:-1], quantiles[1:]):
            sel = (std_m >= lo) & (std_m < hi + 1e-9)
            if sel.sum() > 10:
                bin_std.append(float(np.mean(std_m[sel])))
                bin_err.append(float(np.mean(abs_err[sel])))
        ax.plot(
            bin_std,
            bin_err,
            "o-",
            color=ORACLE_COLORS[name],
            linewidth=1.5,
            markersize=4,
            label=name,
        )

    # Perfect calibration reference
    if bin_std:
        lim = [0, max(max(bin_std), max(bin_err)) * 1.1]
        ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.4, label="perfect calibration")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    ax.set_xlabel("Mean ensemble std (binned)", fontsize=10)
    ax.set_ylabel("Mean |prediction $-$ true|", fontsize=10)
    ax.set_title("Uncertainty calibration", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    _set_pub_style(ax)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(out_png, dpi=FIGDPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_png.name}")


def plot_residual_correlation_matrix(
    oracles: dict[str, dict[str, np.ndarray]],
    out_png: Path,
) -> dict:
    """Compute and plot pairwise Pearson correlation of residuals."""
    # Build aligned residual vectors
    names = []
    residuals = []
    for name, d in oracles.items():
        if d is None:
            continue
        true = d["true_label"].astype(np.float64)
        pred = d["oracle_mean"].astype(np.float64)
        mask = np.isfinite(true) & np.isfinite(pred)
        residuals.append(pred - true)  # keep full length for alignment
        names.append(name)

    if len(names) < 2:
        return {}

    n = len(names)
    corr_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            ri, rj = residuals[i], residuals[j]
            # They should be aligned (same test set), but guard with mask
            mask = np.isfinite(ri) & np.isfinite(rj)
            if mask.sum() < 3:
                continue
            corr_matrix[i, j] = _safe_pearson(ri[mask], rj[mask])

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=10, rotation=30, ha="right")
    ax.set_yticklabels(names, fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=color,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r of residuals", fontsize=10)
    ax.set_title("Inter-oracle residual correlation\n(shared vs independent bias)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=FIGDPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_png.name}")

    # Return as dict for JSON
    result = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{names[i]} vs {names[j]}"
            result[key] = float(corr_matrix[i, j])
    return result


def plot_pairwise_oracle_scatters(
    oracles: dict[str, dict[str, np.ndarray]],
    out_png: Path,
) -> None:
    """Pairwise scatter of oracle predictions between all model pairs."""
    names = [n for n in oracles if oracles[n] is not None]
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

    if not pairs:
        return

    n_cols = min(3, len(pairs))
    n_rows = (len(pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), squeeze=False)

    rng = np.random.default_rng(123)

    for k, (na, nb) in enumerate(pairs):
        row, col = divmod(k, n_cols)
        ax = axes[row, col]

        da, db = oracles[na], oracles[nb]
        pa = da["oracle_mean"].astype(np.float64)
        pb = db["oracle_mean"].astype(np.float64)
        mask = np.isfinite(pa) & np.isfinite(pb)
        xa, xb = pa[mask], pb[mask]

        n_plot = min(8000, len(xa))
        idx = rng.choice(len(xa), n_plot, replace=False)

        ax.scatter(xa[idx], xb[idx], color="#555555", **SCATTER_KW)
        lim = [
            min(xa.min(), xb.min()) - 0.3,
            max(xa.max(), xb.max()) + 0.3,
        ]
        ax.plot(lim, lim, "k-", linewidth=0.8, alpha=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{na} oracle mean", fontsize=10)
        ax.set_ylabel(f"{nb} oracle mean", fontsize=10)

        r = _safe_pearson(xa, xb)
        ax.set_title(f"{na} vs {nb}\nr = {r:.4f}", fontsize=10)
        _set_pub_style(ax)
        ax.set_aspect("equal", adjustable="box")

    # Hide unused subplots
    for k in range(len(pairs), n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("Pairwise oracle prediction comparison (in-dist test)", fontsize=12, y=1.01)
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    fig.savefig(out_png, dpi=FIGDPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_png.name}")


def plot_variance_vs_expression(
    oracles: dict[str, dict[str, np.ndarray]],
    out_png: Path,
) -> None:
    """Oracle std vs true expression level, binned."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for name, d in oracles.items():
        if d is None or "oracle_std" not in d:
            continue
        true = d["true_label"].astype(np.float64)
        std = d["oracle_std"].astype(np.float64)
        mask = np.isfinite(true) & np.isfinite(std)
        t, s = true[mask], std[mask]

        n_bins = 25
        quantiles = np.quantile(t, np.linspace(0, 1, n_bins + 1))
        bin_x, bin_y = [], []
        for lo, hi in zip(quantiles[:-1], quantiles[1:]):
            sel = (t >= lo) & (t < hi + 1e-9)
            if sel.sum() > 10:
                bin_x.append(float(np.mean(t[sel])))
                bin_y.append(float(np.mean(s[sel])))
        ax.plot(
            bin_x,
            bin_y,
            "o-",
            color=ORACLE_COLORS[name],
            linewidth=1.5,
            markersize=4,
            label=name,
        )

    ax.set_xlabel("True K562 log2FC (binned)", fontsize=10)
    ax.set_ylabel("Mean ensemble std", fontsize=10)
    ax.set_title("Prediction variance vs expression level", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    _set_pub_style(ax)

    fig.tight_layout()
    fig.savefig(out_png, dpi=FIGDPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_png.name}")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_summary_metrics(
    oracles: dict[str, dict[str, np.ndarray]],
) -> dict:
    """Compute per-oracle and inter-oracle metrics."""
    summary = {
        "per_oracle": {},
        "inter_oracle_prediction_correlation": {},
        "inter_oracle_residual_correlation": {},
    }

    # Per-oracle metrics
    for name, d in oracles.items():
        if d is None:
            continue
        true = d["true_label"].astype(np.float64)
        pred = d["oracle_mean"].astype(np.float64)
        mask = np.isfinite(true) & np.isfinite(pred)
        yt, yp = true[mask], pred[mask]

        residual = yp - yt
        m = {
            "n": int(mask.sum()),
            "pearson_r": _safe_pearson(yt, yp),
            "spearman_r": _safe_spearman(yt, yp),
            "mean_bias": float(np.mean(residual)),
            "std_bias": float(np.std(residual)),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "pred_mean": float(np.mean(yp)),
            "pred_std": float(np.std(yp)),
            "true_mean": float(np.mean(yt)),
            "true_std": float(np.std(yt)),
            "std_ratio": float(np.std(yp) / np.std(yt)) if np.std(yt) > 0 else float("nan"),
        }
        if "oracle_std" in d:
            std = d["oracle_std"].astype(np.float64)
            std_valid = std[np.isfinite(std)]
            m["mean_ensemble_std"] = float(np.mean(std_valid))
            m["median_ensemble_std"] = float(np.median(std_valid))
            m["max_ensemble_std"] = float(np.max(std_valid))

        summary["per_oracle"][name] = m

    # Inter-oracle correlations
    names = [n for n in oracles if oracles[n] is not None]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            da, db = oracles[na], oracles[nb]
            pa = da["oracle_mean"].astype(np.float64)
            pb = db["oracle_mean"].astype(np.float64)
            mask = np.isfinite(pa) & np.isfinite(pb)

            key = f"{na} vs {nb}"
            summary["inter_oracle_prediction_correlation"][key] = _safe_pearson(pa[mask], pb[mask])

            # Residual correlation (shared bias)
            ta = da["true_label"].astype(np.float64)
            tb = db["true_label"].astype(np.float64)
            mask2 = mask & np.isfinite(ta) & np.isfinite(tb)
            ra = pa[mask2] - ta[mask2]
            rb = pb[mask2] - tb[mask2]
            summary["inter_oracle_residual_correlation"][key] = _safe_pearson(ra, rb)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare oracle ensemble predictions across models."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO,
        help="Repository root (auto-detected).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/oracle_landscape_analysis/).",
    )
    parser.add_argument(
        "--test-split",
        default="test_in_dist",
        help="Which test split to analyze (default: test_in_dist).",
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out_dir = (
        args.out_dir.resolve() if args.out_dir else repo / "results" / "oracle_landscape_analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{args.test_split}_oracle_labels.npz"

    print(f"Repository root: {repo}")
    print(f"Output directory: {out_dir}")
    print(f"Test split file:  {fname}")
    print()

    # ── Load pseudolabels ────────────────────────────────────────────────────
    oracles: dict[str, dict[str, np.ndarray] | None] = {}
    for name, rel_dir in ORACLE_SOURCES.items():
        base = repo / rel_dir
        d = _load_pseudolabels(base, fname)
        if d is not None:
            print(f"  Loaded {name}: {base / fname}  ({d['oracle_mean'].shape[0]:,} sequences)")
        else:
            print(f"  MISSING {name}: {base / fname}")
        oracles[name] = d

    # Filter to available oracles
    available = {k: v for k, v in oracles.items() if v is not None}
    if not available:
        print("\nNo oracle pseudolabel files found. Nothing to analyze.")
        return

    print(f"\nAvailable oracles: {list(available.keys())}")
    print()

    # ── Generate plots ───────────────────────────────────────────────────────
    print("Generating plots...")

    plot_oracle_vs_true_scatter(available, out_dir / "oracle_vs_true_scatter.png")
    plot_residual_distributions(available, out_dir / "residual_distributions.png")
    plot_uncertainty_distributions(available, out_dir / "uncertainty_distributions.png")

    plot_residual_correlation_matrix(available, out_dir / "residual_correlation_matrix.png")

    plot_pairwise_oracle_scatters(available, out_dir / "pairwise_oracle_scatter.png")
    plot_variance_vs_expression(available, out_dir / "variance_vs_expression.png")

    # ── Compute and save metrics ─────────────────────────────────────────────
    print("\nComputing summary metrics...")
    metrics = compute_summary_metrics(available)

    metrics_path = out_dir / "summary_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Wrote: {metrics_path.name}")

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Oracle Landscape Summary ({})".format(args.test_split))
    print("=" * 72)

    print("\n  Per-oracle quality:")
    print(
        f"  {'Model':<12} {'r':>8} {'rho':>8} {'bias':>9} {'MAE':>8} {'std_ratio':>10} {'ens_std':>9}"
    )
    print("  " + "-" * 66)
    for name, m in metrics["per_oracle"].items():
        ens_std = m.get("mean_ensemble_std", float("nan"))
        print(
            f"  {name:<12} {m['pearson_r']:>8.4f} {m['spearman_r']:>8.4f} "
            f"{m['mean_bias']:>+9.4f} {m['mae']:>8.4f} {m['std_ratio']:>10.4f} "
            f"{ens_std:>9.4f}"
        )

    if metrics["inter_oracle_residual_correlation"]:
        print("\n  Inter-oracle residual correlation (shared bias):")
        for key, val in metrics["inter_oracle_residual_correlation"].items():
            print(f"    {key}: r = {val:.4f}")

    if metrics["inter_oracle_prediction_correlation"]:
        print("\n  Inter-oracle prediction correlation:")
        for key, val in metrics["inter_oracle_prediction_correlation"].items():
            print(f"    {key}: r = {val:.4f}")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
