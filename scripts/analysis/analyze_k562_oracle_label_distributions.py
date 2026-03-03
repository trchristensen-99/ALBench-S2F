#!/usr/bin/env python3
"""Analyse K562 oracle ensemble label distributions and prediction quality.

Loads pseudolabel npz files from `outputs/oracle_pseudolabels_k562_ag/` and
produces:

  summary.json                  — full stats + oracle-vs-true metrics
  distribution_stats.csv        — per-split summary statistics (true + oracle)
  oracle_vs_true_metrics.csv    — Pearson / Spearman / MAE / RMSE / Wasserstein

  true_train_val_id_ood_hist.png   — histogram overlay across all four splits
  oracle_vs_true_id_ood_ecdf.png   — ECDF: true vs oracle for in-dist & OOD
  oracle_scatter_panels.png        — scatter oracle_mean vs true (per test split)
  oracle_uncertainty.png           — oracle_std distribution + calibration

Run from repo root::

    python scripts/analysis/analyze_k562_oracle_label_distributions.py

All outputs go to ``outputs/analysis/k562_oracle_label_distributions/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wasserstein_distance

# ── helpers ───────────────────────────────────────────────────────────────────


def _safe_corr(a: np.ndarray, b: np.ndarray, fn) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(fn(a, b)[0])


def _summary_stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)].astype(np.float32)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q50": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x[np.isfinite(x)])
    ys = np.arange(1, xs.size + 1, dtype=np.float32) / float(xs.size)
    return xs, ys


def _compare_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, pred_std: np.ndarray | None = None
) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    result = {
        "n": int(yt.size),
        "pearson_r": _safe_corr(yt, yp, pearsonr),
        "spearman_r": _safe_corr(yt, yp, spearmanr),
        "mae": float(np.mean(np.abs(yt - yp))),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "wasserstein": float(wasserstein_distance(yt, yp)),
        "oracle_mean_bias": float(np.mean(yp) - np.mean(yt)),
        "oracle_std_ratio": float(np.std(yp) / np.std(yt)) if np.std(yt) > 0 else float("nan"),
    }
    if pred_std is not None:
        result["mean_oracle_std"] = float(np.mean(pred_std[mask]))
    return result


# ── plots ─────────────────────────────────────────────────────────────────────

SPLIT_COLORS = {
    "train+pool": "#4C72B0",
    "val": "#55A868",
    "in-dist test": "#C44E52",
    "OOD test": "#DD8452",
}


def plot_distribution_hist(
    train: np.ndarray,
    val: np.ndarray,
    in_dist: np.ndarray,
    ood: np.ndarray,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    # Left: train vs val vs in-dist (same distribution family)
    ax = axes[0]
    bins = np.linspace(-8, 12, 100)
    ax.hist(
        train,
        bins=bins,
        alpha=0.4,
        density=True,
        label="train+pool (319K)",
        color=SPLIT_COLORS["train+pool"],
    )
    ax.hist(val, bins=bins, alpha=0.5, density=True, label="val (41K)", color=SPLIT_COLORS["val"])
    ax.hist(
        in_dist,
        bins=bins,
        alpha=0.5,
        density=True,
        label="in-dist test (41K)",
        color=SPLIT_COLORS["in-dist test"],
    )
    ax.set_xlabel("K562 log2FC")
    ax.set_ylabel("Density")
    ax.set_title("In-distribution splits")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Right: in-dist vs OOD
    ax = axes[1]
    ax.hist(
        in_dist,
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"in-dist test (n={len(in_dist):,})",
        color=SPLIT_COLORS["in-dist test"],
    )
    ax.hist(
        ood,
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"OOD test (n={len(ood):,})",
        color=SPLIT_COLORS["OOD test"],
    )
    ax.set_xlabel("K562 log2FC")
    ax.set_title("In-dist vs OOD")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle("K562 MPRA: true expression label distributions by split", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf(
    true_id: np.ndarray,
    pred_id: np.ndarray,
    true_ood: np.ndarray,
    pred_ood: np.ndarray,
    true_val: np.ndarray,
    pred_val: np.ndarray,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (true, pred, label) in zip(
        axes,
        [
            (true_id, pred_id, "in-dist test"),
            (true_ood, pred_ood, "OOD test"),
            (true_val, pred_val, "val"),
        ],
    ):
        x_t, y_t = _ecdf(true)
        x_p, y_p = _ecdf(pred)
        ax.plot(x_t, y_t, label="true", linewidth=2)
        ax.plot(x_p, y_p, label="oracle mean", linewidth=2, linestyle="--")
        ax.set_xlabel("K562 log2FC")
        ax.set_ylabel("ECDF")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle("K562: oracle vs true label distributions (ECDF)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_panels(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: dict[str, dict],
    out_png: Path,
) -> None:
    """One scatter plot (true vs oracle mean) per test split."""
    n = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.8))
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    for ax, (name, (y_true, y_pred)) in zip(axes, splits.items()):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        # Subsample for scatter density
        n_plot = min(5000, len(yt))
        idx = rng.choice(len(yt), n_plot, replace=False)
        ax.scatter(yt[idx], yp[idx], s=4, alpha=0.3, rasterized=True, color="#4B7BE0")
        lim = [min(yt.min(), yp.min()) - 0.5, max(yt.max(), yp.max()) + 0.5]
        ax.plot(lim, lim, "k--", linewidth=1, alpha=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("True K562 log2FC")
        ax.set_ylabel("Oracle mean")
        m = metrics.get(name, {})
        r = m.get("pearson_r", float("nan"))
        rho = m.get("spearman_r", float("nan"))
        ax.set_title(f"{name}\nr={r:.3f}  ρ={rho:.3f}  n={m.get('n', 0):,}")
        ax.grid(alpha=0.2)

    fig.suptitle("K562 oracle mean vs true labels", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_oof_scatter(
    oof_oracle: np.ndarray,
    true_label: np.ndarray,
    metrics: dict,
    out_png: Path,
) -> None:
    """Scatter of out-of-fold predictions vs true labels on the train+pool set."""
    mask = np.isfinite(oof_oracle) & np.isfinite(true_label)
    yt, yp = true_label[mask], oof_oracle[mask]
    rng = np.random.default_rng(1)
    n_plot = min(10000, len(yt))
    idx = rng.choice(len(yt), n_plot, replace=False)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(yt[idx], yp[idx], s=3, alpha=0.25, rasterized=True, color="#55A868")
    lim = [min(yt.min(), yp.min()) - 0.5, max(yt.max(), yp.max()) + 0.5]
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True K562 log2FC (train+pool)")
    ax.set_ylabel("Oracle OOF prediction")
    r = metrics.get("pearson_r", float("nan"))
    rho = metrics.get("spearman_r", float("nan"))
    ax.set_title(
        f"Oracle out-of-fold (train+pool, n={mask.sum():,})\n"
        f"r={r:.3f}  ρ={rho:.3f}  MAE={metrics.get('mae', float('nan')):.3f}"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty(
    oracle_stds: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    out_png: Path,
) -> None:
    """Oracle std distribution per split + oracle_std vs |error| calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: std distribution across splits
    ax = axes[0]
    bins = 50
    for name, std in oracle_stds.items():
        std = std[np.isfinite(std)]
        ax.hist(std, bins=bins, alpha=0.5, density=True, label=name)
    ax.set_xlabel("Oracle std (ensemble disagreement)")
    ax.set_ylabel("Density")
    ax.set_title("Oracle uncertainty distribution by split")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Right: calibration — oracle_std vs |oracle_mean - true| for in-dist
    ax = axes[1]
    if "in-dist test" in oracle_stds and "in-dist test" in errors:
        std_id = oracle_stds["in-dist test"]
        err_id = errors["in-dist test"]
        mask = np.isfinite(std_id) & np.isfinite(err_id)
        if mask.sum() > 10:
            # Bin by std
            n_bins = 20
            quantiles = np.quantile(std_id[mask], np.linspace(0, 1, n_bins + 1))
            bin_std, bin_err = [], []
            for lo, hi in zip(quantiles[:-1], quantiles[1:]):
                sel = mask & (std_id >= lo) & (std_id < hi)
                if sel.sum() > 5:
                    bin_std.append(float(np.mean(std_id[sel])))
                    bin_err.append(float(np.mean(np.abs(err_id[sel]))))
            ax.plot(bin_std, bin_err, "o-", color="#C44E52", linewidth=2)
            ax.set_xlabel("Mean oracle std (binned)")
            ax.set_ylabel("Mean |oracle_mean − true|")
            ax.set_title("Oracle calibration: uncertainty vs error (in-dist test)")
            ax.grid(alpha=0.25)

    fig.suptitle("K562 oracle uncertainty", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_snv_delta(
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    out_png: Path,
) -> None:
    """Scatter of oracle delta (alt-ref) vs true delta log2FC."""
    mask = np.isfinite(true_delta) & np.isfinite(pred_delta)
    yt, yp = true_delta[mask], pred_delta[mask]
    rng = np.random.default_rng(2)
    n_plot = min(8000, len(yt))
    idx = rng.choice(len(yt), n_plot, replace=False)

    r = _safe_corr(yt, yp, pearsonr)
    rho = _safe_corr(yt, yp, spearmanr)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    ax.scatter(yt[idx], yp[idx], s=4, alpha=0.3, rasterized=True, color="#DD8452")
    lim = [min(yt.min(), yp.min()) - 0.1, max(yt.max(), yp.max()) + 0.1]
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("True Δlog2FC (alt − ref)")
    ax.set_ylabel("Oracle Δ (alt − ref)")
    ax.set_title(f"SNV delta predictions\nr={r:.3f}  ρ={rho:.3f}  n={mask.sum():,}")
    ax.grid(alpha=0.2)

    ax = axes[1]
    bins = np.linspace(-3, 3, 80)
    ax.hist(yt, bins=bins, alpha=0.5, density=True, label="true Δlog2FC")
    ax.hist(yp, bins=bins, alpha=0.5, density=True, label="oracle Δ")
    ax.set_xlabel("Δlog2FC")
    ax.set_ylabel("Density")
    ax.set_title("SNV delta distributions")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle("K562 oracle: SNV effect prediction (delta log2FC)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse K562 oracle label distributions and prediction quality."
    )
    parser.add_argument(
        "--pseudolabel-dir",
        type=Path,
        default=Path("outputs/oracle_pseudolabels_k562_ag"),
        help="Directory containing the oracle pseudolabel npz files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/k562_oracle_label_distributions"),
        help="Output directory for plots, CSV, and summary JSON.",
    )
    args = parser.parse_args()

    pl_dir = args.pseudolabel_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load oracle pseudolabels ──────────────────────────────────────────────
    def _load(fname: str) -> dict[str, np.ndarray] | None:
        p = pl_dir / fname
        if not p.exists():
            print(f"[WARN] Missing: {p}")
            return None
        return dict(np.load(p))

    train_pl = _load("train_oracle_labels.npz")
    val_pl = _load("val_oracle_labels.npz")
    id_pl = _load("test_in_dist_oracle_labels.npz")
    snv_pl = _load("test_snv_oracle_labels.npz")
    ood_pl = _load("test_ood_oracle_labels.npz")

    # ── true label arrays ─────────────────────────────────────────────────────
    true_train = train_pl["true_label"] if train_pl else np.array([])
    true_val = val_pl["true_label"] if val_pl else np.array([])
    true_id = id_pl["true_label"] if id_pl else np.array([])
    true_ood = ood_pl["true_label"] if ood_pl else np.array([])
    true_snv_alt = snv_pl["true_alt_label"] if snv_pl else np.array([])
    true_snv_delta = snv_pl["true_delta"] if snv_pl else np.array([])

    # ── oracle arrays ─────────────────────────────────────────────────────────
    pred_train = train_pl["oracle_mean"] if train_pl else np.array([])
    pred_val = val_pl["oracle_mean"] if val_pl else np.array([])
    pred_id = id_pl["oracle_mean"] if id_pl else np.array([])
    pred_ood = ood_pl["oracle_mean"] if ood_pl else np.array([])
    std_val = val_pl["oracle_std"] if val_pl else np.array([])
    std_id = id_pl["oracle_std"] if id_pl else np.array([])
    std_ood = ood_pl["oracle_std"] if ood_pl else np.array([])
    oof_train = train_pl["oof_oracle"] if train_pl else np.array([])
    pred_snv_delta = snv_pl["delta_oracle_mean"] if snv_pl else np.array([])
    pred_snv_alt = snv_pl["alt_oracle_mean"] if snv_pl else np.array([])
    std_snv_alt = snv_pl["alt_oracle_std"] if snv_pl else np.array([])

    # ── summary stats ─────────────────────────────────────────────────────────
    summary: dict = {
        "paths": {
            "pseudolabel_dir": str(pl_dir),
            "out_dir": str(out_dir),
        },
        "true": {
            "train+pool": _summary_stats(true_train),
            "val": _summary_stats(true_val),
            "in_dist_test": _summary_stats(true_id),
            "ood_test": _summary_stats(true_ood),
            "snv_alt": _summary_stats(true_snv_alt),
            "snv_delta": _summary_stats(true_snv_delta),
        },
    }

    # ── oracle vs true metrics ────────────────────────────────────────────────
    metrics: dict[str, dict] = {}
    if pred_id.size:
        metrics["in_dist_test"] = _compare_metrics(true_id, pred_id, std_id)
    if pred_val.size:
        metrics["val"] = _compare_metrics(true_val, pred_val, std_val)
    if pred_ood.size:
        metrics["ood_test"] = _compare_metrics(true_ood, pred_ood, std_ood)
    if pred_snv_alt.size:
        metrics["snv_abs"] = _compare_metrics(true_snv_alt, pred_snv_alt, std_snv_alt)
    if pred_snv_delta.size:
        metrics["snv_delta"] = _compare_metrics(true_snv_delta, pred_snv_delta)
    if oof_train.size:
        metrics["train_oof"] = _compare_metrics(true_train, oof_train)

    summary["oracle_mean"] = {
        "train+pool": _summary_stats(pred_train),
        "val": _summary_stats(pred_val),
        "in_dist_test": _summary_stats(pred_id),
        "ood_test": _summary_stats(pred_ood),
        "snv_delta": _summary_stats(pred_snv_delta),
    }
    summary["oracle_vs_true"] = metrics

    # ── plots ─────────────────────────────────────────────────────────────────
    if true_train.size and true_val.size and true_id.size and true_ood.size:
        plot_distribution_hist(
            true_train,
            true_val,
            true_id,
            true_ood,
            out_dir / "true_train_val_id_ood_hist.png",
        )
        print("Wrote: true_train_val_id_ood_hist.png")

    if pred_id.size and pred_ood.size and pred_val.size:
        plot_ecdf(
            true_id,
            pred_id,
            true_ood,
            pred_ood,
            true_val,
            pred_val,
            out_dir / "oracle_vs_true_id_ood_ecdf.png",
        )
        print("Wrote: oracle_vs_true_id_ood_ecdf.png")

    scatter_splits = {}
    if pred_id.size:
        scatter_splits["in-dist test"] = (true_id, pred_id)
    if pred_ood.size:
        scatter_splits["OOD test"] = (true_ood, pred_ood)
    if pred_val.size:
        scatter_splits["val"] = (true_val, pred_val)
    scatter_metrics = {
        "in-dist test": metrics.get("in_dist_test", {}),
        "OOD test": metrics.get("ood_test", {}),
        "val": metrics.get("val", {}),
    }
    if scatter_splits:
        plot_scatter_panels(scatter_splits, scatter_metrics, out_dir / "oracle_scatter_panels.png")
        print("Wrote: oracle_scatter_panels.png")

    if oof_train.size:
        plot_oof_scatter(
            oof_train,
            true_train,
            metrics.get("train_oof", {}),
            out_dir / "oracle_oof_scatter.png",
        )
        print("Wrote: oracle_oof_scatter.png")

    unc_stds = {}
    unc_errors = {}
    for name, pred, true, std in [
        ("in-dist test", pred_id, true_id, std_id),
        ("OOD test", pred_ood, true_ood, std_ood),
        ("val", pred_val, true_val, std_val),
    ]:
        if std.size and pred.size and true.size:
            unc_stds[name] = std
            unc_errors[name] = pred - true
    if unc_stds:
        plot_uncertainty(unc_stds, unc_errors, out_dir / "oracle_uncertainty.png")
        print("Wrote: oracle_uncertainty.png")

    if pred_snv_delta.size:
        plot_snv_delta(true_snv_delta, pred_snv_delta, out_dir / "oracle_snv_delta.png")
        print("Wrote: oracle_snv_delta.png")

    # ── write CSVs ────────────────────────────────────────────────────────────
    rows_dist = []
    for source, block in [
        ("true", summary["true"]),
        ("oracle_mean", summary.get("oracle_mean", {})),
    ]:
        for subset, stats in block.items():
            rows_dist.append({"source": source, "subset": subset, **stats})
    pd.DataFrame(rows_dist).to_csv(out_dir / "distribution_stats.csv", index=False)
    print("Wrote: distribution_stats.csv")

    if metrics:
        rows_m = [{"subset": k, **v} for k, v in metrics.items()]
        pd.DataFrame(rows_m).to_csv(out_dir / "oracle_vs_true_metrics.csv", index=False)
        print("Wrote: oracle_vs_true_metrics.csv")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote: summary.json")

    # ── print console summary ─────────────────────────────────────────────────
    print("\n=== Oracle ensemble quality (K562 hashFrag) ===")
    order = ["val", "in_dist_test", "snv_abs", "snv_delta", "ood_test", "train_oof"]
    for key in order:
        m = metrics.get(key)
        if m is None:
            continue
        r = m.get("pearson_r", float("nan"))
        rho = m.get("spearman_r", float("nan"))
        mae = m.get("mae", float("nan"))
        bias = m.get("oracle_mean_bias", float("nan"))
        std_ratio = m.get("oracle_std_ratio", float("nan"))
        n = m.get("n", 0)
        print(
            f"  {key:<18} r={r:.4f}  ρ={rho:.4f}  MAE={mae:.4f}  "
            f"bias={bias:+.4f}  std_ratio={std_ratio:.3f}  n={n:,}"
        )


if __name__ == "__main__":
    main()
