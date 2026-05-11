"""Activity-stratified evaluation: report Pearson/Spearman/MSE per label-percentile bin.

Why: Peter (May 11 2026) flagged that aggregate Pearson masks activity skew — if
most sequences are inactive, the headline R is dominated by the inactive bin's
correlation, which doesn't tell you whether the model picks signal in the
biologically interesting (high-activity) regime.

This script:
1. Loads predictions + real labels for a test set
2. Bins sequences by label percentile (default 5 bins, configurable)
3. Computes Pearson R, Spearman R, and MSE per bin + an "overall" reference
4. Writes a CSV and a PNG bar-plot to the output dir

Defaults point at the AG-S2 oracle ensemble's K562 test predictions.

Usage (default — runs on AG-S2 oracle):
    python -m scripts.preflight.activity_stratified_eval

Override predictions / labels:
    python -m scripts.preflight.activity_stratified_eval \\
        --preds outputs/foo/test_preds.npz --preds_key oracle_mean \\
        --labels outputs/foo/pool/test.parquet --labels_col K562_log2FC \\
        --output_dir results/eval/activity_stratified/my_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]


def stratified_metrics(preds: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Bin by label percentile, compute per-bin metrics + overall."""
    quantiles = np.quantile(labels, np.linspace(0, 1, n_bins + 1))
    rows: list[dict] = []
    for i in range(n_bins):
        lo, hi = quantiles[i], quantiles[i + 1]
        if i == n_bins - 1:
            mask = (labels >= lo) & (labels <= hi)
        else:
            mask = (labels >= lo) & (labels < hi)
        p, lbl = preds[mask], labels[mask]
        if len(p) < 2:
            continue
        rows.append(
            {
                "bin": f"{int(100 * i / n_bins)}-{int(100 * (i + 1) / n_bins)}%",
                "label_lo": float(lo),
                "label_hi": float(hi),
                "n": int(mask.sum()),
                "pearson_r": float(pearsonr(p, lbl)[0]),
                "spearman_r": float(spearmanr(p, lbl).correlation),
                "mse": float(np.mean((p - lbl) ** 2)),
                "mean_pred": float(p.mean()),
                "mean_label": float(lbl.mean()),
            }
        )
    rows.append(
        {
            "bin": "overall",
            "label_lo": float(labels.min()),
            "label_hi": float(labels.max()),
            "n": int(len(labels)),
            "pearson_r": float(pearsonr(preds, labels)[0]),
            "spearman_r": float(spearmanr(preds, labels).correlation),
            "mse": float(np.mean((preds - labels) ** 2)),
            "mean_pred": float(preds.mean()),
            "mean_label": float(labels.mean()),
        }
    )
    return pd.DataFrame(rows)


def plot_stratified(df: pd.DataFrame, title: str, out_path: Path):
    """Per-bin Pearson + MSE bar plot, with overall as a reference line."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bin_rows = df[df["bin"] != "overall"]
    overall = df[df["bin"] == "overall"].iloc[0]

    # Pearson
    ax = axes[0]
    ax.bar(bin_rows["bin"], bin_rows["pearson_r"], color="steelblue", edgecolor="black")
    ax.axhline(
        overall["pearson_r"],
        color="tomato",
        linestyle="--",
        linewidth=2,
        label=f"overall = {overall['pearson_r']:.3f}",
    )
    for i, r in enumerate(bin_rows["pearson_r"]):
        ax.text(i, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("Label-percentile bin")
    ax.set_ylabel("Pearson R")
    ax.set_title("Pearson R within activity bin")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(1.0, bin_rows["pearson_r"].max() * 1.1))

    # MSE
    ax = axes[1]
    ax.bar(bin_rows["bin"], bin_rows["mse"], color="darkorange", edgecolor="black")
    ax.axhline(
        overall["mse"],
        color="tomato",
        linestyle="--",
        linewidth=2,
        label=f"overall = {overall['mse']:.3f}",
    )
    for i, m in enumerate(bin_rows["mse"]):
        ax.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("Label-percentile bin")
    ax.set_ylabel("MSE")
    ax.set_title("MSE within activity bin")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _load_preds(path: Path, key: str | None) -> np.ndarray:
    """Load predictions from .npz (with key) or .npy or .csv (single column)."""
    if path.suffix == ".npz":
        npz = np.load(path)
        if key is None:
            keys = list(npz.keys())
            if len(keys) == 1:
                key = keys[0]
            else:
                raise ValueError(f"npz has multiple keys {keys}; pass --preds_key")
        return np.asarray(npz[key], dtype=np.float32)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix == ".csv":
        return pd.read_csv(path).iloc[:, 0].to_numpy(dtype=np.float32)
    raise ValueError(f"Unsupported preds format: {path.suffix}")


def _load_labels(path: Path, col: str) -> np.ndarray:
    """Load labels from parquet/csv."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported labels format: {path.suffix}")
    if col not in df.columns:
        raise ValueError(f"column {col!r} not found in {path}  (have {list(df.columns)})")
    return df[col].to_numpy(dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preds",
        default=str(REPO / "outputs/oracle_pseudolabels_k562_ag_s2_refalt/test_oracle_labels.npz"),
    )
    ap.add_argument("--preds_key", default="oracle_mean")
    ap.add_argument(
        "--labels",
        default=str(REPO / "outputs/oracle_pseudolabels_k562_ag_s2_refalt/pool/test.parquet"),
    )
    ap.add_argument("--labels_col", default="K562_log2FC")
    ap.add_argument(
        "--output_dir",
        default=str(REPO / "results/eval/activity_stratified/ag_s2_oracle"),
    )
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    preds = _load_preds(Path(args.preds), args.preds_key)
    labels = _load_labels(Path(args.labels), args.labels_col)

    # Align lengths if mismatched (oracle .npz may include subset rows)
    n = min(len(preds), len(labels))
    if len(preds) != len(labels):
        print(f"  WARN: preds={len(preds):,} labels={len(labels):,} — truncating to {n:,}")
    preds, labels = preds[:n], labels[:n]

    df = stratified_metrics(preds, labels, n_bins=args.n_bins)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics.csv", index=False)

    title = args.title or f"{args.preds_key} vs {args.labels_col}  (n={n:,}, {args.n_bins} bins)"
    plot_stratified(df, title, out_dir / "plot.png")

    summary = {
        "preds": args.preds,
        "preds_key": args.preds_key,
        "labels": args.labels,
        "labels_col": args.labels_col,
        "n": n,
        "n_bins": args.n_bins,
    }
    (out_dir / "config.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Activity-stratified metrics ({n:,} sequences, {args.n_bins} bins) ===\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {out_dir}/metrics.csv, plot.png, config.json")


if __name__ == "__main__":
    main()
