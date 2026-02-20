#!/usr/bin/env python3
"""Exp0 convenience wrapper: sync remote results and produce standard analysis outputs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument(
        "--metric-col",
        default="test_random_pearson_r",
        choices=[
            "best_val_pearson_r",
            "test_random_pearson_r",
            "test_snv_pearson_r",
            "test_genomic_pearson_r",
        ],
    )
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=["citra", "hpc"],
        choices=["citra", "hpc"],
    )
    args = parser.parse_args()

    root = repo_root()
    if not args.skip_fetch:
        subprocess.run(
            [
                "python3",
                str(root / "scripts" / "analysis" / "sync_remote_results.py"),
                "--experiment",
                "exp0_yeast_scaling",
                "--hosts",
                *args.hosts,
            ],
            check=True,
        )

    subprocess.run(
        [
            "python3",
            str(root / "scripts" / "analysis" / "analyze_experiment_results.py"),
            "--experiment",
            "exp0_yeast_scaling",
            "--metric-col",
            args.metric_col,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
