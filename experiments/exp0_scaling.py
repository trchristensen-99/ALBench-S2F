"""Experiment 0: baseline scaling curve."""

from __future__ import annotations

import random
from pathlib import Path

import hydra
import numpy as np
import wandb
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from albench.evaluation import compute_scaling_curve
from albench.loop import RunConfig, run_al_loop
from albench.oracle.perfect_oracle import PerfectOracle
from albench.task import TaskConfig


def _set_seed(seed: int) -> None:
    """Set all experiment random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def _random_sequences(n: int, length: int, seed: int) -> list[str]:
    """Generate deterministic synthetic sequences for dry-run execution."""
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"))
    return ["".join(alphabet[rng.integers(0, 4, size=length)]) for _ in range(n)]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Experiment 0 scaling benchmark."""
    load_dotenv()
    _set_seed(int(cfg.seed))

    task = TaskConfig(
        name=cfg.task.name,
        organism=cfg.task.organism,
        sequence_length=int(cfg.task.sequence_length),
        data_root=cfg.task.data_root,
        test_set=OmegaConf.to_container(cfg.task.test_set, resolve=True),
    )

    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.task.name}_{cfg.reservoir._target_.split('.')[-1]}_{cfg.acquisition._target_.split('.')[-1]}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[
            cfg.task.name,
            cfg.reservoir._target_.split(".")[-1],
            cfg.acquisition._target_.split(".")[-1],
        ],
        group=cfg.experiment.name,
        mode=cfg.wandb.mode,
    )
    wandb.define_metric("test/pearson_r", step_metric="n_labeled")

    student = instantiate(cfg.student)
    reservoir = instantiate(cfg.reservoir)
    acquisition = instantiate(cfg.acquisition)

    if cfg.experiment.dry_run:
        if hasattr(student, "train_config"):
            student.train_config.epochs = 1
            student.train_config.batch_size = 64
        if hasattr(student, "models"):
            student.models = student.models[:1]
            student.ensemble_size = 1
        sequences = _random_sequences(256, int(cfg.task.sequence_length), int(cfg.seed))
        labels = np.random.default_rng(int(cfg.seed)).normal(size=len(sequences)).astype(np.float32)
        task.test_set = {
            "dry": {
                "sequences": sequences[:64],
                "labels": labels[:64].tolist(),
            }
        }
    else:
        raise NotImplementedError(
            "Non-dry-run dataset wiring will be added in full experiment integration"
        )

    oracle = PerfectOracle(dict(zip(sequences, labels, strict=False)))
    initial_labeled = sequences[: int(cfg.experiment.batch_size)]

    run_config = RunConfig(
        n_rounds=1 if cfg.experiment.dry_run else int(cfg.experiment.n_rounds),
        batch_size=int(cfg.experiment.batch_size),
        reservoir_schedule={"default": reservoir},
        acquisition_schedule={"default": acquisition},
        output_dir=str(Path(cfg.experiment.output_dir)),
        n_reservoir_candidates=int(cfg.experiment.n_reservoir_candidates),
    )

    results = run_al_loop(
        task=task,
        oracle=oracle,
        student=student,
        initial_labeled=initial_labeled,
        run_config=run_config,
    )
    curve = compute_scaling_curve(results)
    out = Path(cfg.experiment.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / "exp0_scaling_curve.csv", index=False)

    if run is not None:
        if not curve.empty:
            for row in curve.to_dict(orient="records"):
                wandb.log(
                    {
                        "n_labeled": row["n_labeled"],
                        "test/pearson_r": row.get("pearson_r", 0.0),
                        "test/spearman_r": row.get("spearman_r", 0.0),
                        "test/loss": row.get("loss", 0.0),
                    }
                )
        wandb.finish()


if __name__ == "__main__":
    main()
