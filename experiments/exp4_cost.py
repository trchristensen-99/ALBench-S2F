"""Experiment 4: cost-adjusted strategy ranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf, open_dict

from experiments.exp0_scaling import run_exp0_scaling


def _reservoir_strategy_key(cfg: DictConfig) -> str:
    """Return normalized key for current reservoir strategy."""
    target = str(cfg.reservoir._target_).split(".")[-1]
    return target.lower()


def _default_cost_map() -> dict[str, float]:
    """Provide placeholder per-sequence synthesis costs by reservoir strategy.

    Values are relative cost units and should be replaced with lab-validated values.
    """
    return {
        "randomsampler": 1.0,
        "fixedpoolsampler": 1.0,
        "genomicsampler": 0.4,
        "partialmutagenesissampler": 0.6,
        "tfmotifshufflesampler": 0.7,
        "evoaugsampler": 0.8,
    }


def _configured_cost_map(cfg: DictConfig) -> dict[str, float]:
    """Read synthesis cost map from config, falling back to defaults."""
    exp_cfg = cfg.get("experiment", {})
    if isinstance(exp_cfg, DictConfig) and "synthesis_cost_per_sequence" in exp_cfg:
        mapping: Any = OmegaConf.to_container(exp_cfg.synthesis_cost_per_sequence, resolve=True)
        if isinstance(mapping, dict):
            return {str(k).lower(): float(v) for k, v in mapping.items()}
    return _default_cost_map()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Experiment 4 and produce cost-adjusted rankings.

    Uses random subsampling acquisition and reports synthesis-cost-adjusted metrics.
    Synthesis costs are configurable placeholders until lab-specific values are known.
    """
    load_dotenv()
    with open_dict(cfg):
        cfg.experiment.name = "exp4_cost"
        cfg.acquisition = OmegaConf.create(
            {
                "_target_": "albench.acquisition.random_acq.RandomAcquisition",
                "seed": int(cfg.seed) if cfg.seed is not None else None,
            }
        )

    artifacts = run_exp0_scaling(cfg)
    if artifacts.curve.empty:
        return

    reservoir_key = _reservoir_strategy_key(cfg)
    cost_map = _configured_cost_map(cfg)
    cost_per_sequence = float(cost_map.get(reservoir_key, 1.0))

    round_time_map = {r.round_idx: r.round_wall_seconds for r in artifacts.results}
    selected_count_map = {r.round_idx: len(r.selected_sequences) for r in artifacts.results}
    round_cost_map = {k: float(v) * cost_per_sequence for k, v in selected_count_map.items()}
    cumulative_cost_map: dict[int, float] = {}
    running = 0.0
    for round_idx in sorted(round_cost_map):
        running += round_cost_map[round_idx]
        cumulative_cost_map[round_idx] = running

    cost_df = artifacts.curve.copy()
    cost_df["round_wall_seconds"] = cost_df["round_idx"].map(round_time_map).astype(float)
    cost_df["selected_in_round"] = cost_df["round_idx"].map(selected_count_map).astype(int)
    cost_df["synthesis_cost_per_sequence"] = cost_per_sequence
    cost_df["round_synthesis_cost"] = cost_df["round_idx"].map(round_cost_map).astype(float)
    cost_df["cumulative_synthesis_cost"] = (
        cost_df["round_idx"].map(cumulative_cost_map).astype(float)
    )
    cost_df["pearson_r_per_cost_unit"] = cost_df["pearson_r"] / cost_df[
        "cumulative_synthesis_cost"
    ].clip(lower=1e-6)
    cost_df["labeled_per_cost_unit"] = cost_df["n_labeled"] / cost_df[
        "cumulative_synthesis_cost"
    ].clip(lower=1e-6)
    cost_df["pearson_r_per_second"] = cost_df["pearson_r"] / cost_df["round_wall_seconds"].clip(
        lower=1e-6
    )

    out_dir = Path(artifacts.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_df.to_csv(out_dir / "exp4_cost_adjusted.csv", index=False)

    summary = (
        cost_df.sort_values(by="pearson_r_per_cost_unit", ascending=False)
        .groupby("test_set", as_index=False)
        .head(1)
    )
    summary.to_csv(out_dir / "exp4_top_cost_adjusted.csv", index=False)

    metadata = {
        "reservoir_strategy_key": reservoir_key,
        "synthesis_cost_per_sequence": cost_per_sequence,
        "cost_map_used": cost_map,
        "note": (
            "Synthesis costs are placeholder relative units unless overridden via "
            "experiment.synthesis_cost_per_sequence."
        ),
    }
    with (out_dir / "exp4_cost_model.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


if __name__ == "__main__":
    main()
