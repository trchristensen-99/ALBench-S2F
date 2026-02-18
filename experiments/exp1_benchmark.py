"""Experiment 1: benchmark matrix across strategies."""

from __future__ import annotations

import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run benchmark combinations (Hydra multirun compatible)."""
    load_dotenv()
    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.task.name}_exp1",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.task.name, "exp1", "benchmark"],
        group="exp1_benchmark",
        mode=cfg.wandb.mode,
    )
    wandb.log(
        {
            "status": 1,
            "task": cfg.task.name,
            "reservoir": cfg.reservoir._target_.split(".")[-1],
            "acquisition": cfg.acquisition._target_.split(".")[-1],
        }
    )
    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
