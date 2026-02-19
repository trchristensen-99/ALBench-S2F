"""Experiment 2: round-structure sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf, open_dict

from experiments.exp0_scaling import ExpRunArtifacts, run_exp0_scaling


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    """Create an independent mutable copy of a Hydra config object."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _protocol_row(
    name: str,
    n_rounds: int,
    batch_size: int,
    artifacts: ExpRunArtifacts,
    selection_metric: str,
) -> dict[str, Any]:
    """Summarize one Exp2 protocol result."""
    if artifacts.curve.empty:
        return {
            "protocol": name,
            "n_rounds": n_rounds,
            "batch_size": batch_size,
            "total_budget": n_rounds * batch_size,
            "best_metric": 0.0,
            "best_round_idx": -1,
            "best_n_labeled": 0,
        }
    if selection_metric not in artifacts.curve.columns:
        raise KeyError(f"selection_metric '{selection_metric}' missing from curve columns")

    best_row = artifacts.curve.sort_values(by=selection_metric, ascending=False).iloc[0]
    return {
        "protocol": name,
        "n_rounds": n_rounds,
        "batch_size": batch_size,
        "total_budget": n_rounds * batch_size,
        "best_metric": float(best_row[selection_metric]),
        "best_round_idx": int(best_row["round_idx"]),
        "best_n_labeled": int(best_row["n_labeled"]),
        "best_test_set": str(best_row["test_set"]),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Experiment 2.

    Runs two protocols with comparable budget:
    - multi-round: more rounds with smaller batch size
    - single-round: one round with larger batch size
    """
    load_dotenv()
    selection_metric = str(cfg.experiment.get("selection_metric", "pearson_r"))
    multi_rounds = int(cfg.experiment.get("exp2_multi_rounds", 3))
    multi_batch_size = int(cfg.experiment.get("exp2_multi_batch_size", 20))
    single_rounds = int(cfg.experiment.get("exp2_single_rounds", 1))
    single_batch_size = int(
        cfg.experiment.get("exp2_single_batch_size", multi_rounds * multi_batch_size)
    )

    protocols = [
        ("multi_round", multi_rounds, multi_batch_size),
        ("single_round", single_rounds, single_batch_size),
    ]
    rows: list[dict[str, Any]] = []
    base_output_dir = Path(str(cfg.experiment.output_dir))

    for protocol_name, n_rounds, batch_size in protocols:
        run_cfg = _clone_cfg(cfg)
        with open_dict(run_cfg):
            run_cfg.experiment.name = f"exp2_rounds_{protocol_name}"
            run_cfg.experiment.n_rounds = n_rounds
            run_cfg.experiment.batch_size = batch_size
            run_cfg.experiment.output_dir = str(base_output_dir / "exp2" / protocol_name)
        artifacts = run_exp0_scaling(run_cfg)
        rows.append(
            _protocol_row(
                name=protocol_name,
                n_rounds=n_rounds,
                batch_size=batch_size,
                artifacts=artifacts,
                selection_metric=selection_metric,
            )
        )

    out_dir = base_output_dir / "exp2"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "exp2_round_comparison.csv", index=False)

    if {"multi_round", "single_round"}.issubset(set(summary_df["protocol"])):
        multi = summary_df[summary_df["protocol"] == "multi_round"].iloc[0]
        single = summary_df[summary_df["protocol"] == "single_round"].iloc[0]
        delta_df = pd.DataFrame(
            [
                {
                    "selection_metric": selection_metric,
                    "multi_best_metric": float(multi["best_metric"]),
                    "single_best_metric": float(single["best_metric"]),
                    "delta_multi_minus_single": float(multi["best_metric"] - single["best_metric"]),
                }
            ]
        )
        delta_df.to_csv(out_dir / "exp2_round_delta.csv", index=False)


if __name__ == "__main__":
    main()
