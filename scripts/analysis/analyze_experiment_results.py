#!/usr/bin/env python3
"""Generic experiment result aggregation and plotting from synced/local result.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def infer_seed(run_dir: str) -> int | None:
    if "seed" not in run_dir:
        return None
    tail = run_dir.split("seed", 1)[1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    return int(digits) if digits else None


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def load_records(input_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_root.rglob("result.json")):
        payload = json.loads(path.read_text())
        flat = flatten(payload)

        run_dir = ""
        fraction_dir = ""
        if len(path.parts) >= 3:
            run_dir = path.parts[-3]
            fraction_dir = path.parts[-2]

        row: dict[str, Any] = {
            "path": str(path),
            "run_dir": run_dir,
            "fraction_dir": fraction_dir,
            "seed": infer_seed(run_dir),
            **flat,
        }
        # Normalize common nested test metric names for stable CLI defaults.
        for src, dst in [
            ("test_metrics_random_pearson_r", "test_random_pearson_r"),
            ("test_metrics_random_spearman_r", "test_random_spearman_r"),
            ("test_metrics_snv_abs_pearson_r", "test_snv_pearson_r"),
            ("test_metrics_snv_abs_spearman_r", "test_snv_spearman_r"),
            ("test_metrics_snv_pearson_r", "test_snv_pearson_r"),  # fallback
            ("test_metrics_snv_spearman_r", "test_snv_spearman_r"),  # fallback
            ("test_metrics_genomic_pearson_r", "test_genomic_pearson_r"),
            ("test_metrics_genomic_spearman_r", "test_genomic_spearman_r"),
        ]:
            if src in row and dst not in row:
                row[dst] = row[src]

        source = "local"
        if "/citra/" in str(path):
            source = "citra"
        elif "/hpc/" in str(path):
            source = "hpc"
        row["source"] = source

        if "fraction" not in row and fraction_dir.startswith("fraction_"):
            try:
                row["fraction"] = float(fraction_dir.split("_", 1)[1])
            except Exception:
                pass

        rows.append(row)

    return pd.DataFrame(rows)


def deduplicate(
    df: pd.DataFrame, preferred_metric_col: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "fraction" not in df.columns:
        return df.copy(), pd.DataFrame(columns=df.columns)

    ranked = df.copy()
    if "num_epochs_run" in ranked.columns:
        ranked["_epochs"] = ranked["num_epochs_run"].fillna(-1)
    else:
        ranked["_epochs"] = -1

    metric = "best_val_pearson_r" if "best_val_pearson_r" in ranked.columns else None
    ranked["_metric"] = ranked[metric].fillna(-1e9) if metric else -1e9
    if preferred_metric_col and preferred_metric_col in ranked.columns:
        ranked["_preferred_metric"] = ranked[preferred_metric_col]
        ranked["_has_preferred_metric"] = ranked["_preferred_metric"].notna().astype(int)
    else:
        ranked["_preferred_metric"] = -1e9
        ranked["_has_preferred_metric"] = 0

    loss = "best_val_loss" if "best_val_loss" in ranked.columns else None
    ranked["_loss"] = ranked[loss].fillna(1e9) if loss else 1e9

    subset = [c for c in ["seed", "fraction"] if c in ranked.columns]
    if not subset:
        return df.copy(), pd.DataFrame(columns=df.columns)

    ranked = ranked.sort_values(
        subset + ["_has_preferred_metric", "_preferred_metric", "_epochs", "_metric", "_loss"],
        ascending=True,
    )
    keep = ranked.drop_duplicates(subset=subset, keep="last").drop(
        columns=["_epochs", "_metric", "_loss", "_preferred_metric", "_has_preferred_metric"]
    )
    drop = ranked[ranked.duplicated(subset=subset, keep="last")].drop(
        columns=["_epochs", "_metric", "_loss", "_preferred_metric", "_has_preferred_metric"]
    )
    return keep, drop


def maybe_plot(df: pd.DataFrame, metric_col: str, out_path: Path, title: str) -> bool:
    if "fraction" not in df.columns or metric_col not in df.columns:
        return False

    series = df.dropna(subset=["fraction", metric_col]).sort_values("fraction")
    if series.empty:
        return False

    plt.figure(figsize=(8, 4.5))
    plt.plot(series["fraction"], series[metric_col], marker="o", linewidth=1.8)
    plt.xscale("log")
    plt.xlabel("Fraction")
    plt.ylabel(metric_col)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="exp0_yeast_scaling")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo_root() / "outputs" / "analysis" / "synced",
        help="Root containing synced experiment outputs.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=repo_root() / "outputs" / "analysis" / "reports",
        help="Root for generated CSV/plots.",
    )
    parser.add_argument(
        "--metric-col",
        type=str,
        default="test_random_pearson_r",
        help="Metric column to plot when fraction is available.",
    )
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args()

    in_dir = args.input_root / args.experiment
    out_dir = args.out_root / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_records(in_dir)
    if df.empty:
        raise SystemExit(f"No result.json found under {in_dir}")

    if args.no_dedup:
        dedup_df = df.copy()
        dropped_df = pd.DataFrame(columns=df.columns)
    else:
        dedup_df, dropped_df = deduplicate(df, preferred_metric_col=args.metric_col)

    raw_csv = out_dir / "results_raw.csv"
    dedup_csv = out_dir / "results_dedup.csv"
    dropped_csv = out_dir / "results_dropped_duplicates.csv"
    plot_png = out_dir / f"scaling_{args.metric_col}.png"

    df.to_csv(raw_csv, index=False)
    dedup_df.to_csv(dedup_csv, index=False)
    dropped_df.to_csv(dropped_csv, index=False)

    plotted = maybe_plot(
        dedup_df,
        metric_col=args.metric_col,
        out_path=plot_png,
        title=f"{args.experiment} ({args.metric_col})",
    )

    print(f"Input dir: {in_dir}")
    print(f"Rows raw={len(df)} dedup={len(dedup_df)} dropped={len(dropped_df)}")
    print(f"Wrote: {raw_csv}")
    print(f"Wrote: {dedup_csv}")
    print(f"Wrote: {dropped_csv}")
    if plotted:
        print(f"Wrote: {plot_png}")
    else:
        print(f"Skipped plot (missing/empty metric: {args.metric_col})")


if __name__ == "__main__":
    main()
