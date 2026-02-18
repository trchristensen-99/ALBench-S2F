"""Experiment 5 stub."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Experiment 5 (TODO)."""
    raise NotImplementedError("TODO: implement exp5_best_student pipeline")


if __name__ == "__main__":
    main()
