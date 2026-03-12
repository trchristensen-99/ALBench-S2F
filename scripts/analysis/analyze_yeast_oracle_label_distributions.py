#!/usr/bin/env python3
"""Analyse yeast DREAM-RNN oracle ensemble label distributions and prediction quality.

Loads pseudolabel npz files from the oracle pseudolabel directory and produces:

  summary.json                  — full stats + oracle-vs-true metrics
  distribution_stats.csv        — per-split summary statistics (true + oracle)
  oracle_vs_true_metrics.csv    — Pearson / Spearman / MAE / RMSE / Wasserstein

  true_train_test_id_ood_hist.png  — histogram overlay (train + ID + OOD test)
  oracle_vs_true_id_ood_ecdf.png   — ECDF: true vs oracle for ID & OOD
  oracle_scatter_panels.png        — scatter oracle_mean vs true (per test split)
  oracle_oof_scatter.png           — out-of-fold scatter (train set)
  oracle_uncertainty.png           — oracle std distribution + calibration
  oracle_snv_delta.png             — SNV delta (alt-ref) scatter + histograms

Run from repo root::

    python scripts/analysis/analyze_yeast_oracle_label_distributions.py

All outputs go to ``results/exp0_plots/yeast_oracle_dist/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wasserstein_distance

from data.yeast import YeastDataset
from evaluation.yeast_testsets import load_yeast_test_subsets

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


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    xm, ym = float(np.mean(x)), float(np.mean(y))
    xv = float(np.mean((x - xm) ** 2))
    if xv == 0.0:
        return 1.0, 0.0
    cov = float(np.mean((x - xm) * (y - ym)))
    return cov / xv, ym - (cov / xv) * xm


# ── plots ─────────────────────────────────────────────────────────────────────

SPLIT_COLORS = {
    "train": "#4C72B0",
    "val": "#55A868",
    "ID test (random)": "#C44E52",
    "OOD test (genomic)": "#DD8452",
}


def plot_distribution_hist(
    train: np.ndarray,
    id_test: np.ndarray,
    ood_test: np.ndarray,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    bins = 80

    ax = axes[0]
    ax.hist(train, bins=bins, alpha=0.4, density=True, label="train", color=SPLIT_COLORS["train"])
    ax.hist(
        id_test,
        bins=bins,
        alpha=0.5,
        density=True,
        label="ID test (random)",
        color=SPLIT_COLORS["ID test (random)"],
    )
    ax.set_xlabel("Expression label")
    ax.set_ylabel("Density")
    ax.set_title("Train vs ID test")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(
        id_test,
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"ID test (random, n={len(id_test):,})",
        color=SPLIT_COLORS["ID test (random)"],
    )
    ax.hist(
        ood_test,
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"OOD test (genomic, n={len(ood_test):,})",
        color=SPLIT_COLORS["OOD test (genomic)"],
    )
    ax.set_xlabel("Expression label")
    ax.set_title("ID vs OOD test")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle("Yeast: true expression label distributions by split", fontsize=12)
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
            (true_id, pred_id, "ID test (random)"),
            (true_ood, pred_ood, "OOD test (genomic)"),
            (true_val, pred_val, "val"),
        ],
    ):
        x_t, y_t = _ecdf(true)
        x_p, y_p = _ecdf(pred)
        ax.plot(x_t, y_t, label="true", linewidth=2)
        ax.plot(x_p, y_p, label="DREAM-RNN Ensemble mean", linewidth=2, linestyle="--")
        ax.set_xlabel("Expression")
        ax.set_ylabel("ECDF")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle("Yeast: DREAM-RNN Ensemble vs true label distributions (ECDF)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_panels(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: dict[str, dict],
    out_png: Path,
) -> None:
    n = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.8))
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    for ax, (name, (y_true, y_pred)) in zip(axes, splits.items()):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        n_plot = min(5000, len(yt))
        idx = rng.choice(len(yt), n_plot, replace=False)
        ax.scatter(yt[idx], yp[idx], s=4, alpha=0.3, rasterized=True, color="#4B7BE0")
        lim = [min(yt.min(), yp.min()) - 0.5, max(yt.max(), yp.max()) + 0.5]
        ax.plot(lim, lim, "k--", linewidth=1, alpha=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("True expression")
        ax.set_ylabel("DREAM-RNN Ensemble mean")
        m = metrics.get(name, {})
        r = m.get("pearson_r", float("nan"))
        rho = m.get("spearman_r", float("nan"))
        ax.set_title(f"{name}\nr={r:.3f}  rho={rho:.3f}  n={m.get('n', 0):,}")
        ax.grid(alpha=0.2)

    fig.suptitle("Yeast DREAM-RNN Ensemble mean vs true labels", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_oof_scatter(
    oof_oracle: np.ndarray,
    true_label: np.ndarray,
    metrics: dict,
    out_png: Path,
) -> None:
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
    ax.set_xlabel("True expression (train)")
    ax.set_ylabel("DREAM-RNN Ensemble OOF prediction")
    r = metrics.get("pearson_r", float("nan"))
    rho = metrics.get("spearman_r", float("nan"))
    ax.set_title(
        f"DREAM-RNN Ensemble out-of-fold (train, n={mask.sum():,})\n"
        f"r={r:.3f}  rho={rho:.3f}  MAE={metrics.get('mae', float('nan')):.3f}"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty(
    oracle_stds: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    true_values: dict[str, np.ndarray],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for name, std in oracle_stds.items():
        std = std[np.isfinite(std)]
        ax.hist(std, bins=50, alpha=0.5, density=True, label=name)
    ax.set_xlabel("Ensemble std (disagreement)")
    ax.set_ylabel("Density")
    ax.set_title("DREAM-RNN Ensemble uncertainty distribution by split")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    # Normalized calibration plot for ID (random) test split:
    # Divide error and std by max(|true|, eps) so the calibration is not
    # dominated by high-activity sequences.
    cal_key = next((k for k in oracle_stds if "random" in k.lower() or "id" in k.lower()), None)
    if cal_key and cal_key in errors and cal_key in true_values:
        std_arr = oracle_stds[cal_key]
        err_arr = errors[cal_key]
        true_arr = true_values[cal_key]
        eps = 0.1  # floor to avoid division by near-zero
        denom = np.maximum(np.abs(true_arr), eps)
        rel_std = std_arr / denom
        rel_err = np.abs(err_arr) / denom
        mask = np.isfinite(rel_std) & np.isfinite(rel_err)
        if mask.sum() > 10:
            n_bins = 20
            quantiles = np.quantile(rel_std[mask], np.linspace(0, 1, n_bins + 1))
            bin_std, bin_err = [], []
            for lo, hi in zip(quantiles[:-1], quantiles[1:]):
                sel = mask & (rel_std >= lo) & (rel_std < hi)
                if sel.sum() > 5:
                    bin_std.append(float(np.mean(rel_std[sel])))
                    bin_err.append(float(np.mean(rel_err[sel])))
            ax.plot(bin_std, bin_err, "o-", color="#C44E52", linewidth=2)
            ax.set_xlabel("Mean relative ensemble std (binned)")
            ax.set_ylabel("Mean relative |prediction - true|")
            ax.set_title(f"Normalized calibration ({cal_key})")
            ax.grid(alpha=0.25)

    fig.suptitle("Yeast DREAM-RNN Ensemble uncertainty", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_snv_delta(
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    out_png: Path,
) -> None:
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
    ax.set_xlabel("True delta expression (alt - ref)")
    ax.set_ylabel("DREAM-RNN Ensemble delta (alt - ref)")
    ax.set_title(f"SNV delta predictions\nr={r:.3f}  rho={rho:.3f}  n={mask.sum():,}")
    ax.grid(alpha=0.2)

    ax = axes[1]
    bins = np.linspace(yt.min() - 0.1, yt.max() + 0.1, 80)
    ax.hist(yt, bins=bins, alpha=0.5, density=True, label="true delta")
    ax.hist(yp, bins=bins, alpha=0.5, density=True, label="ensemble delta")
    ax.set_xlabel("Delta expression")
    ax.set_ylabel("Density")
    ax.set_title("SNV delta distributions")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle("Yeast DREAM-RNN Ensemble: SNV effect prediction (delta expression)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse yeast oracle label distributions and prediction quality."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/yeast"))
    parser.add_argument("--context-mode", type=str, default="dream150")
    parser.add_argument("--subset-dir", type=Path, default=Path("data/yeast/test_subset_ids"))
    parser.add_argument(
        "--pseudolabel-dir",
        type=Path,
        default=Path("outputs/oracle_pseudolabels/yeast_dream_oracle"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exp0_plots/yeast_oracle_dist"),
    )
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pl_dir = args.pseudolabel_dir.expanduser().resolve()

    # ── load datasets ────────────────────────────────────────────────────────
    ds_train = YeastDataset(
        data_path=str(args.data_path),
        split="train",
        context_mode=args.context_mode,
    )
    ds_test = YeastDataset(
        data_path=str(args.data_path),
        split="test",
        context_mode=args.context_mode,
    )
    train_labels = ds_train.labels.astype(np.float32)
    test_labels = ds_test.labels.astype(np.float32)

    subsets = load_yeast_test_subsets(args.subset_dir)
    idx_id = subsets["random_idx"].astype(int)
    idx_ood = subsets["genomic_idx"].astype(int)
    snv_pairs = subsets["snv_pairs"].astype(int)
    snv_ref = snv_pairs[:, 0]
    snv_alt = snv_pairs[:, 1]

    # ── load pseudolabels ────────────────────────────────────────────────────
    def _load(fname: str) -> dict[str, np.ndarray] | None:
        p = pl_dir / fname
        if not p.exists():
            print(f"[WARN] Missing: {p}")
            return None
        return dict(np.load(p))

    train_pl = _load("train_oracle_labels.npz")
    if train_pl is None:
        # Fall back to combined train+pool format.
        train_pl = _load("train_pool_oracle_labels.npz")
    val_pl = _load("val_oracle_labels.npz")
    test_pl = _load("test_oracle_labels.npz")

    # ── true label summary stats ─────────────────────────────────────────────
    true_data = {
        "train": train_labels,
        "id_random": test_labels[idx_id],
        "ood_genomic": test_labels[idx_ood],
        "snv_ref": test_labels[snv_ref],
        "snv_alt": test_labels[snv_alt],
        "snv_abs": np.concatenate([test_labels[snv_ref], test_labels[snv_alt]]),
        "test_all": test_labels,
    }

    summary: dict = {
        "paths": {
            "data_path": str(args.data_path),
            "subset_dir": str(args.subset_dir),
            "pseudolabel_dir": str(pl_dir),
        },
        "true": {k: _summary_stats(v) for k, v in true_data.items()},
    }

    # ── distribution histogram (train + ID + OOD) ───────────────────────────
    plot_distribution_hist(
        train_labels,
        test_labels[idx_id],
        test_labels[idx_ood],
        out_dir / "true_train_test_id_ood_hist.png",
    )
    print("Wrote: true_train_test_id_ood_hist.png")

    # ── oracle analysis (if pseudolabels available) ──────────────────────────
    metrics: dict[str, dict] = {}
    oracle_rows: list[dict] = []

    if test_pl is not None:
        pred_mean = test_pl["oracle_mean"].astype(np.float32)
        pred_std = test_pl["oracle_std"].astype(np.float32)

        pred_data = {
            "id_random": pred_mean[idx_id],
            "ood_genomic": pred_mean[idx_ood],
            "snv_ref": pred_mean[snv_ref],
            "snv_alt": pred_mean[snv_alt],
            "snv_abs": np.concatenate([pred_mean[snv_ref], pred_mean[snv_alt]]),
            "test_all": pred_mean,
        }
        std_data = {
            "id_random": pred_std[idx_id],
            "ood_genomic": pred_std[idx_ood],
            "snv_abs": np.concatenate([pred_std[snv_ref], pred_std[snv_alt]]),
            "test_all": pred_std,
        }

        summary["oracle_mean"] = {k: _summary_stats(v) for k, v in pred_data.items()}

        # Oracle vs true metrics per test subset
        for key in ["id_random", "ood_genomic", "snv_abs", "test_all"]:
            y_true = true_data[key]
            y_pred = pred_data[key]
            p_std = std_data.get(key)
            metrics[key] = _compare_metrics(y_true, y_pred, p_std)

        # SNV delta metrics
        true_snv_delta = test_labels[snv_alt] - test_labels[snv_ref]
        pred_snv_delta = pred_mean[snv_alt] - pred_mean[snv_ref]
        metrics["snv_delta"] = _compare_metrics(true_snv_delta, pred_snv_delta)

        # Affine calibration from val (val labels are on the same RAW scale as
        # oracle predictions, so this captures minor offsets within that scale)
        if val_pl is not None:
            val_pred = val_pl["oracle_mean"].astype(np.float32)
            val_true = val_pl["true_label"].astype(np.float32)
            a_val, b_val = _fit_affine(val_pred, val_true)
            summary["oracle_affine_map_val"] = {"scale_a": float(a_val), "bias_b": float(b_val)}

            metrics["val"] = _compare_metrics(
                val_true,
                val_pred,
                val_pl["oracle_std"].astype(np.float32) if "oracle_std" in val_pl else None,
            )

        # Affine calibration from test (maps oracle RAW scale → MAUDE-calibrated
        # test scale; used for visualization so plots are interpretable)
        a_test, b_test = _fit_affine(pred_mean, test_labels)
        pred_mean_cal = (a_test * pred_mean + b_test).astype(np.float32)
        # Also calibrate std: scale by |a_test| to preserve relative uncertainty
        pred_std_cal = (np.abs(a_test) * pred_std).astype(np.float32)
        summary["oracle_affine_map_test"] = {
            "scale_a": float(a_test),
            "bias_b": float(b_test),
            "note": "Maps oracle predictions → MAUDE-calibrated test label scale",
        }

        # Calibrated metrics
        pred_cal_sub = {
            "id_random": pred_mean_cal[idx_id],
            "ood_genomic": pred_mean_cal[idx_ood],
            "snv_abs": np.concatenate([pred_mean_cal[snv_ref], pred_mean_cal[snv_alt]]),
            "test_all": pred_mean_cal,
        }
        std_cal_sub = {
            "id_random": pred_std_cal[idx_id],
            "ood_genomic": pred_std_cal[idx_ood],
            "snv_abs": np.concatenate([pred_std_cal[snv_ref], pred_std_cal[snv_alt]]),
            "test_all": pred_std_cal,
        }
        cal_metrics: dict[str, dict] = {}
        for key in ["id_random", "ood_genomic", "snv_abs", "test_all"]:
            cal_metrics[key] = _compare_metrics(
                true_data[key], pred_cal_sub[key], std_cal_sub.get(key)
            )

        summary["oracle_vs_true"] = metrics
        summary["oracle_vs_true_affine_calibrated"] = cal_metrics

        # Build CSV rows
        for key in ["id_random", "ood_genomic", "snv_abs", "snv_delta", "test_all"]:
            m = metrics.get(key, {})
            if m:
                oracle_rows.append({"scale": "raw", "subset": key, **m})
        for key in ["id_random", "ood_genomic", "snv_abs", "test_all"]:
            m = cal_metrics.get(key, {})
            if m:
                oracle_rows.append({"scale": "affine_calibrated", "subset": key, **m})

        # ECDF plot (calibrated oracle for test splits; raw for val since val
        # true labels are on the same raw scale as oracle predictions)
        if val_pl is not None:
            plot_ecdf(
                true_data["id_random"],
                pred_cal_sub["id_random"],
                true_data["ood_genomic"],
                pred_cal_sub["ood_genomic"],
                val_pl["true_label"].astype(np.float32),
                val_pl["oracle_mean"].astype(np.float32),
                out_dir / "oracle_vs_true_id_ood_ecdf.png",
            )
            print("Wrote: oracle_vs_true_id_ood_ecdf.png")

        # Scatter panels (calibrated oracle predictions for test; raw for val
        # since val true labels are on the same raw scale as oracle)
        scatter_splits = {
            "ID (random)": (true_data["id_random"], pred_cal_sub["id_random"]),
            "OOD (genomic)": (true_data["ood_genomic"], pred_cal_sub["ood_genomic"]),
        }
        scatter_metrics = {
            "ID (random)": cal_metrics.get("id_random", {}),
            "OOD (genomic)": cal_metrics.get("ood_genomic", {}),
        }
        if val_pl is not None:
            scatter_splits["val"] = (
                val_pl["true_label"].astype(np.float32),
                val_pl["oracle_mean"].astype(np.float32),
            )
            scatter_metrics["val"] = metrics.get("val", {})
        plot_scatter_panels(scatter_splits, scatter_metrics, out_dir / "oracle_scatter_panels.png")
        print("Wrote: oracle_scatter_panels.png")

        # Uncertainty plot (use calibrated predictions + calibrated std for test)
        unc_stds = {
            "ID (random)": std_cal_sub["id_random"],
            "OOD (genomic)": std_cal_sub["ood_genomic"],
        }
        unc_errors = {
            "ID (random)": pred_cal_sub["id_random"] - true_data["id_random"],
            "OOD (genomic)": pred_cal_sub["ood_genomic"] - true_data["ood_genomic"],
        }
        unc_true = {
            "ID (random)": true_data["id_random"],
            "OOD (genomic)": true_data["ood_genomic"],
        }
        if val_pl is not None and "oracle_std" in val_pl:
            unc_stds["val"] = val_pl["oracle_std"].astype(np.float32)
            unc_errors["val"] = val_pl["oracle_mean"].astype(np.float32) - val_pl[
                "true_label"
            ].astype(np.float32)
            unc_true["val"] = val_pl["true_label"].astype(np.float32)
        plot_uncertainty(unc_stds, unc_errors, unc_true, out_dir / "oracle_uncertainty.png")
        print("Wrote: oracle_uncertainty.png")

        # SNV delta plot (calibrated: scale factor applies to delta too)
        if len(snv_pairs) > 0:
            pred_snv_delta_cal = pred_mean_cal[snv_alt] - pred_mean_cal[snv_ref]
            plot_snv_delta(true_snv_delta, pred_snv_delta_cal, out_dir / "oracle_snv_delta.png")
            print("Wrote: oracle_snv_delta.png")

    # ── OOF scatter (train set) ──────────────────────────────────────────────
    if train_pl is not None and "oof_oracle" in train_pl:
        oof_oracle = train_pl["oof_oracle"].astype(np.float32)
        oof_true = train_pl["true_label"].astype(np.float32)
        oof_metrics = _compare_metrics(oof_true, oof_oracle)
        metrics["train_oof"] = oof_metrics
        summary.setdefault("oracle_vs_true", {})["train_oof"] = oof_metrics

        plot_oof_scatter(oof_oracle, oof_true, oof_metrics, out_dir / "oracle_oof_scatter.png")
        print("Wrote: oracle_oof_scatter.png")

        # Train ensemble metrics
        train_pred = train_pl["oracle_mean"].astype(np.float32)
        train_std = train_pl["oracle_std"].astype(np.float32)
        train_ens_metrics = _compare_metrics(oof_true, train_pred, train_std)
        metrics["train_ensemble"] = train_ens_metrics
        summary.setdefault("oracle_vs_true", {})["train_ensemble"] = train_ens_metrics
        summary.setdefault("oracle_mean", {})["train"] = _summary_stats(train_pred)

    # ── write outputs ────────────────────────────────────────────────────────
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote: summary.json")

    rows_dist = []
    for source, block in [
        ("true", summary["true"]),
        ("oracle_mean", summary.get("oracle_mean", {})),
    ]:
        for subset, stats in block.items():
            rows_dist.append({"source": source, "subset": subset, **stats})
    pd.DataFrame(rows_dist).to_csv(out_dir / "distribution_stats.csv", index=False)
    print("Wrote: distribution_stats.csv")

    if oracle_rows:
        pd.DataFrame(oracle_rows).to_csv(out_dir / "oracle_vs_true_metrics.csv", index=False)
        print("Wrote: oracle_vs_true_metrics.csv")

    # ── console summary ──────────────────────────────────────────────────────
    print("\n=== Oracle ensemble quality (yeast DREAM-RNN) ===")
    order = [
        "val",
        "id_random",
        "ood_genomic",
        "snv_abs",
        "snv_delta",
        "test_all",
        "train_oof",
        "train_ensemble",
    ]
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
            f"  {key:<18} r={r:.4f}  rho={rho:.4f}  MAE={mae:.4f}  "
            f"bias={bias:+.4f}  std_ratio={std_ratio:.3f}  n={n:,}"
        )


if __name__ == "__main__":
    main()
