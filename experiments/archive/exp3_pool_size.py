"""Experiment 3: reservoir candidate pool-size sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf, open_dict

from experiments.exp0_scaling import ExpRunArtifacts, run_exp0_scaling


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    """Create an independent mutable copy of a Hydra config object."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _best_metric_row(
    pool_size: int,
    artifacts: ExpRunArtifacts,
    selection_metric: str,
) -> dict[str, Any]:
    """Summarize best metric achieved for one pool size."""
    if artifacts.curve.empty:
        return {
            "pool_size": pool_size,
            "best_metric": 0.0,
            "best_round_idx": -1,
            "best_n_labeled": 0,
            "best_test_set": "",
        }
    if selection_metric not in artifacts.curve.columns:
        raise KeyError(f"selection_metric '{selection_metric}' missing from curve columns")
    best_row = artifacts.curve.sort_values(by=selection_metric, ascending=False).iloc[0]
    return {
        "pool_size": pool_size,
        "best_metric": float(best_row[selection_metric]),
        "best_round_idx": int(best_row["round_idx"]),
        "best_n_labeled": int(best_row["n_labeled"]),
        "best_test_set": str(best_row["test_set"]),
    }


def _sensitivity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple pool-size sensitivity metrics."""
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "n_pool_sizes": 0,
                    "metric_range": 0.0,
                    "metric_per_log10_pool_slope": 0.0,
                }
            ]
        )

    best = df.sort_values(by="pool_size")
    y = best["best_metric"].to_numpy(dtype=np.float64)
    metric_range = float(np.max(y) - np.min(y))

    if len(best) < 2:
        slope = 0.0
    else:
        x = np.log10(best["pool_size"].to_numpy(dtype=np.float64))
        # Fit y = m*x + b for rough sensitivity trend.
        m, _b = np.polyfit(x, y, deg=1)
        slope = float(m)

    return pd.DataFrame(
        [
            {
                "n_pool_sizes": int(len(best)),
                "metric_range": metric_range,
                "metric_per_log10_pool_slope": slope,
            }
        ]
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Experiment 3.

    Runs one or more pool sizes and writes explicit sensitivity summaries.
    """
    load_dotenv()
    selection_metric = str(cfg.experiment.get("selection_metric", "pearson_r"))

    if "exp3_pool_sizes" in cfg.experiment:
        sizes = [int(v) for v in cfg.experiment.exp3_pool_sizes]
    else:
        sizes = [int(cfg.experiment.n_reservoir_candidates)]
    sizes = sorted(set(sizes))
    if not sizes:
        raise ValueError("exp3_pool_sizes cannot be empty")

    base_output_dir = Path(str(cfg.experiment.output_dir))
    rows: list[dict[str, Any]] = []
    for pool_size in sizes:
        run_cfg = _clone_cfg(cfg)
        with open_dict(run_cfg):
            run_cfg.experiment.name = f"exp3_pool_size_{pool_size}"
            run_cfg.experiment.n_reservoir_candidates = int(pool_size)
            run_cfg.experiment.output_dir = str(base_output_dir / "exp3" / f"pool_{pool_size}")
        artifacts = run_exp0_scaling(run_cfg)
        rows.append(
            _best_metric_row(
                pool_size=pool_size, artifacts=artifacts, selection_metric=selection_metric
            )
        )

    out_dir = base_output_dir / "exp3"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows).sort_values(by="pool_size").reset_index(drop=True)
    summary_df["delta_from_min_pool"] = summary_df["best_metric"] - float(
        summary_df.iloc[0]["best_metric"]
    )
    summary_df.to_csv(out_dir / "exp3_pool_sensitivity.csv", index=False)

    sensitivity = _sensitivity_summary(summary_df)
    sensitivity.to_csv(out_dir / "exp3_pool_sensitivity_summary.csv", index=False)


if __name__ == "__main__":
    main()
