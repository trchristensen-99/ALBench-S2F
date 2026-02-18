"""Experiment 4 stub."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Experiment 4 (TODO)."""
    raise NotImplementedError("TODO: implement exp4_cost pipeline")


if __name__ == "__main__":
    main()
