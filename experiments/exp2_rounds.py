"""Experiment 2: round-structure sweep."""

from __future__ import annotations

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, open_dict

from experiments.exp0_scaling import run_exp0_scaling


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Experiment 2.

    This script reuses the Exp0 execution engine and is intended to be launched
    with Hydra multirun over ``experiment.n_rounds`` and/or ``experiment.batch_size``.
    """
    load_dotenv()
    with open_dict(cfg):
        cfg.experiment.name = "exp2_rounds"
    run_exp0_scaling(cfg)


if __name__ == "__main__":
    main()
