#!/usr/bin/env python3
"""Fetch, aggregate, deduplicate, and plot exp0 yeast scaling results."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class HostSpec:
    name: str
    ssh_target: str
    remote_cmd: str


HOSTS = [
    HostSpec(
        name="citra",
        ssh_target="143.48.59.3",
        remote_cmd=(
            'cd "$HOME/ALBench-S2F" && '
            "find outputs/exp0_yeast_scaling -type f -name 'result.json' -print0 "
            "| tar --null -czf - --files-from -"
        ),
    ),
    HostSpec(
        name="hpc",
        ssh_target="bamdev4.cshl.edu",
        remote_cmd=(
            "bash -lc 'cd /grid/wsbs/home_norepl/christen/ALBench-S2F && "
            'find outputs/exp0_yeast_scaling -type f -name "result.json" -print0 '
            "| tar --null -czf - --files-from -'"
        ),
    ),
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def fetch_results(host: HostSpec, workdir: Path) -> Path:
    tar_path = workdir / f"{host.name}_results.tgz"
    with tar_path.open("wb") as f:
        proc = subprocess.Popen(["ssh", host.ssh_target, host.remote_cmd], stdout=f)
        if proc.wait() != 0:
            raise RuntimeError(f"Failed to fetch results from {host.name}")
    return tar_path


def extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["tar", "-xzf", str(tar_path), "-C", str(out_dir)])


def infer_seed(run_dir: str) -> int | None:
    if "seed" not in run_dir:
        return None
    tail = run_dir.split("seed", 1)[1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    return int(digits) if digits else None


def load_records(source_root: Path, source_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(source_root.rglob("result.json")):
        data = json.loads(path.read_text())
        run_dir = path.parts[-3]
        fraction_dir = path.parts[-2]
        fraction = float(data.get("fraction", fraction_dir.split("_")[-1]))
        rows.append(
            {
                "source": source_name,
                "run_dir": run_dir,
                "fraction_dir": fraction_dir,
                "path": str(path),
                "seed": infer_seed(run_dir),
                "fraction": fraction,
                "n_samples": data.get("n_samples"),
                "n_total": data.get("n_total"),
                "best_val_pearson_r": data.get("best_val_pearson_r"),
                "best_val_spearman_r": data.get("best_val_spearman_r"),
                "best_val_loss": data.get("best_val_loss"),
                "num_epochs_run": data.get("num_epochs_run"),
                "training_time_seconds": data.get("training_time_seconds"),
                "test_random_pearson_r": data.get("test_metrics", {})
                .get("random", {})
                .get("pearson_r"),
                "test_random_spearman_r": data.get("test_metrics", {})
                .get("random", {})
                .get("spearman_r"),
                "test_genomic_pearson_r": data.get("test_metrics", {})
                .get("genomic", {})
                .get("pearson_r"),
                "test_genomic_spearman_r": data.get("test_metrics", {})
                .get("genomic", {})
                .get("spearman_r"),
                "test_snv_pearson_r": data.get("test_metrics", {}).get("snv", {}).get("pearson_r"),
                "test_snv_spearman_r": data.get("test_metrics", {})
                .get("snv", {})
                .get("spearman_r"),
            }
        )
    return pd.DataFrame(rows)


def dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = df.copy()
    ranked["_epochs"] = ranked["num_epochs_run"].fillna(-1)
    ranked["_pearson"] = ranked["best_val_pearson_r"].fillna(-1e9)
    ranked["_loss"] = ranked["best_val_loss"].fillna(1e9)
    ranked = ranked.sort_values(
        ["seed", "fraction", "_epochs", "_pearson", "_loss"],
        ascending=[True, True, False, False, True],
    )
    keep = ranked.drop_duplicates(subset=["seed", "fraction"], keep="first").drop(
        columns=["_epochs", "_pearson", "_loss"]
    )
    drop = ranked[ranked.duplicated(subset=["seed", "fraction"], keep="first")].drop(
        columns=["_epochs", "_pearson", "_loss"]
    )
    return keep, drop


def plot_scaling(df: pd.DataFrame, out_path: Path, metric_col: str, title: str) -> None:
    series = df.dropna(subset=[metric_col]).sort_values("fraction")
    plt.figure(figsize=(8, 4.5))
    plt.plot(series["fraction"], series[metric_col], marker="o", linewidth=1.8)
    plt.xscale("log")
    plt.xlabel("Fraction of training set")
    plt.ylabel(metric_col)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/albench_results_agg"))
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument(
        "--metric-col",
        type=str,
        default="best_val_pearson_r",
        choices=[
            "best_val_pearson_r",
            "test_random_pearson_r",
            "test_snv_pearson_r",
            "test_genomic_pearson_r",
        ],
    )
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    extracted_root = args.workdir / "extracted"
    out_dir = args.workdir / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_fetch:
        for host in HOSTS:
            tar_path = fetch_results(host, args.workdir)
            extract_tar(tar_path, extracted_root / host.name)

    citra_df = load_records(extracted_root / "citra", "citra")
    hpc_df = load_records(extracted_root / "hpc", "hpc")
    full_df = pd.concat([citra_df, hpc_df], ignore_index=True)
    if full_df.empty:
        raise SystemExit("No result.json files found. Run without --skip-fetch first.")

    dedup_df, dropped_df = dedup(full_df)

    full_df.to_csv(out_dir / "exp0_yeast_results_raw.csv", index=False)
    dedup_df.sort_values("fraction").to_csv(out_dir / "exp0_yeast_results_dedup.csv", index=False)
    dropped_df.to_csv(out_dir / "exp0_yeast_results_dropped_duplicates.csv", index=False)
    plot_scaling(
        dedup_df,
        out_dir / "exp0_yeast_scaling_plot.png",
        metric_col=args.metric_col,
        title=f"Exp0 Yeast Scaling ({args.metric_col}, deduplicated Citra + HPC)",
    )

    print(f"Raw rows: {len(full_df)}")
    print(f"Deduplicated rows: {len(dedup_df)}")
    print(f"Dropped duplicates: {len(dropped_df)}")
    print("Summary:")
    print(
        dedup_df.sort_values("fraction")[
            [
                "source",
                "run_dir",
                "seed",
                "fraction",
                "n_samples",
                "best_val_pearson_r",
                "num_epochs_run",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
