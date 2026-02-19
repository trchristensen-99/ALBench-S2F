"""Experiment 0: baseline scaling curve."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from albench.evaluation import compute_scaling_curve
from albench.loop import RunConfig, run_al_loop
from albench.oracle.perfect_oracle import PerfectOracle
from albench.task import TaskConfig


@dataclass
class ExpRunArtifacts:
    """Container for experiment outputs needed by downstream scripts."""

    results: list[Any]
    curve: pd.DataFrame
    output_dir: Path


def _set_seed(seed: int) -> None:
    """Set all experiment random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def _random_sequences(n: int, length: int, seed: int) -> list[str]:
    """Generate deterministic synthetic sequences for dry-run execution."""
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"))
    return ["".join(alphabet[rng.integers(0, 4, size=length)]) for _ in range(n)]


def _metadata_rows(dataset: Any) -> list[dict[str, Any]] | None:
    """Convert dataset metadata dict-of-arrays into per-example rows."""
    metadata = getattr(dataset, "metadata", None)
    if not metadata:
        return None
    n_examples = len(dataset.sequences)
    rows: list[dict[str, Any]] = []
    for idx in range(n_examples):
        row: dict[str, Any] = {}
        for key, values in metadata.items():
            value = values[idx]
            if hasattr(value, "item"):
                value = value.item()
            row[str(key)] = value
        rows.append(row)
    return rows


def run_exp0_scaling(cfg: DictConfig) -> ExpRunArtifacts:
    """Execute Experiment 0 core logic with current Hydra config."""
    _set_seed(int(cfg.seed))

    task = TaskConfig(
        name=cfg.task.name,
        organism=cfg.task.organism,
        sequence_length=int(cfg.task.sequence_length),
        data_root=cfg.task.data_root,
        input_channels=int(cfg.task.input_channels),
        task_mode=cfg.task.task_mode,
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
    pool_metadata: list[dict[str, Any]] | None = None

    if cfg.experiment.dry_run:
        # --- Dry-run: synthetic data, 1 round, 1 ensemble member ---
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
        pool_sequences = sequences[64:]
    else:
        # --- Real data: load K562 or Yeast dataset ---
        if cfg.task.task_mode == "k562":
            from albench.data.k562 import K562Dataset

            ds_train = K562Dataset(data_path=cfg.task.data_root, split="train")
            ds_pool = K562Dataset(data_path=cfg.task.data_root, split="pool")
            ds_test = K562Dataset(data_path=cfg.task.data_root, split="test")
        elif cfg.task.task_mode == "yeast":
            from albench.data.yeast import YeastDataset

            ds_train = YeastDataset(data_path=cfg.task.data_root, split="train")
            ds_pool = YeastDataset(data_path=cfg.task.data_root, split="pool")
            ds_test = YeastDataset(data_path=cfg.task.data_root, split="test")
        else:
            raise ValueError(f"Unknown task_mode: {cfg.task.task_mode}")

        sequences = list(ds_train.sequences)
        labels = ds_train.labels.astype(np.float32)
        pool_sequences = list(ds_pool.sequences)
        pool_metadata = _metadata_rows(ds_pool)

        task.test_set = {
            "test": {
                "sequences": list(ds_test.sequences),
                "labels": ds_test.labels.tolist(),
            }
        }

    # Build oracle: lookup mapping from sequence → label
    if cfg.experiment.dry_run:
        label_map = dict(zip(sequences, labels.tolist(), strict=False))
    else:
        # For real data, build oracle from all labeled splits
        all_seqs = sequences + pool_sequences
        all_labels = np.concatenate([labels, ds_pool.labels.astype(np.float32)]).tolist()
        label_map = dict(zip(all_seqs, all_labels, strict=False))

    oracle = PerfectOracle(label_map)

    initial_size = min(int(cfg.experiment.batch_size), len(sequences))
    if initial_size == 0:
        raise ValueError("No training sequences available for initial labeled set")
    rng = np.random.default_rng(int(cfg.seed))
    initial_indices = rng.choice(len(sequences), size=initial_size, replace=False).tolist()
    initial_labeled = [sequences[i] for i in initial_indices]

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
        pool_sequences=pool_sequences,
        pool_metadata=pool_metadata,
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
    return ExpRunArtifacts(results=results, curve=curve, output_dir=out)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Experiment 0 scaling benchmark."""
    load_dotenv()
    run_exp0_scaling(cfg)


if __name__ == "__main__":
    main()
