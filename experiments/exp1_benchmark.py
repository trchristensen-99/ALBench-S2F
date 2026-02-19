"""Experiment 1: benchmark matrix across strategies."""

from __future__ import annotations

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, open_dict

from experiments.exp0_scaling import run_exp0_scaling


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run benchmark combinations (Hydra multirun compatible).

    This entry point reuses Exp0 runtime logic while changing experiment
    grouping so each Hydra multirun job is tracked under Exp1 in W&B.
    """
    load_dotenv()
    with open_dict(cfg):
        cfg.experiment.name = "exp1_benchmark"
    run_exp0_scaling(cfg)


if __name__ == "__main__":
    main()
