"""Experiment 1: benchmark matrix across strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf, open_dict

from experiments.exp0_scaling import ExpRunArtifacts, run_exp0_scaling

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    """Create an independent mutable copy of a Hydra config object."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _load_component_cfg(group: str, name: str) -> DictConfig:
    """Load a component config file by group/name."""
    path = CONFIG_ROOT / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return OmegaConf.load(path)


def _best_metric_row(
    reservoir_name: str,
    acquisition_name: str,
    artifacts: ExpRunArtifacts,
    selection_metric: str,
) -> dict[str, Any]:
    """Summarize best metric for one strategy combination."""
    if artifacts.curve.empty:
        return {
            "reservoir": reservoir_name,
            "acquisition": acquisition_name,
            "best_metric": 0.0,
            "best_round_idx": -1,
            "best_n_labeled": 0,
            "best_test_set": "",
        }
    if selection_metric not in artifacts.curve.columns:
        raise KeyError(f"selection_metric '{selection_metric}' missing from curve columns")
    best = artifacts.curve.sort_values(by=selection_metric, ascending=False).iloc[0]
    return {
        "reservoir": reservoir_name,
        "acquisition": acquisition_name,
        "best_metric": float(best[selection_metric]),
        "best_round_idx": int(best["round_idx"]),
        "best_n_labeled": int(best["n_labeled"]),
        "best_test_set": str(best["test_set"]),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run benchmark combinations (Hydra multirun compatible).

    Runs one or more reservoir/acquisition combinations and writes a comparison
    table of best achieved metric per combination.
    """
    load_dotenv()
    selection_metric = str(cfg.experiment.get("selection_metric", "pearson_r"))

    if "exp1_reservoirs" in cfg.experiment:
        reservoir_names = [str(v) for v in cfg.experiment.exp1_reservoirs]
    else:
        reservoir_names = [
            str(cfg.reservoir._target_).split(".")[-1].replace("Sampler", "").lower()
        ]
        # Fallback to config shortname if available.
        if "reservoir_name" in cfg.experiment:
            reservoir_names = [str(cfg.experiment.reservoir_name)]

    if "exp1_acquisitions" in cfg.experiment:
        acquisition_names = [str(v) for v in cfg.experiment.exp1_acquisitions]
    else:
        acquisition_names = [
            str(cfg.acquisition._target_).split(".")[-1].replace("Acquisition", "").lower()
        ]
        if "acquisition_name" in cfg.experiment:
            acquisition_names = [str(cfg.experiment.acquisition_name)]

    # Normalize defaults to known config keys when introspection produced class-derived names.
    class_to_cfg_res = {
        "random": "random",
        "genomic": "genomic",
        "fixedpool": "fixed_pool",
        "partialmutagenesis": "partial_mutagenesis",
        "evoaug": "evoaug",
        "tfmotifshuffle": "tf_motif_shuffle",
        "insilicoevolution": "in_silico_evolution",
    }
    class_to_cfg_acq = {
        "random": "random",
        "uncertainty": "uncertainty",
        "diversity": "diversity",
        "combined": "combined",
        "priorknowledge": "prior_knowledge",
        "ensemble": "ensemble_acq",
    }
    reservoir_names = [
        class_to_cfg_res.get(name.replace("_", ""), name) for name in reservoir_names
    ]
    acquisition_names = [
        class_to_cfg_acq.get(name.replace("_", ""), name) for name in acquisition_names
    ]

    out_base = Path(str(cfg.experiment.output_dir))
    rows: list[dict[str, Any]] = []
    for reservoir_name in reservoir_names:
        for acquisition_name in acquisition_names:
            run_cfg = _clone_cfg(cfg)
            with open_dict(run_cfg):
                run_cfg.experiment.name = f"exp1_benchmark_{reservoir_name}_{acquisition_name}"
                run_cfg.experiment.output_dir = str(
                    out_base / "exp1" / reservoir_name / acquisition_name
                )
                run_cfg.reservoir = _load_component_cfg("reservoir", reservoir_name)
                run_cfg.acquisition = _load_component_cfg("acquisition", acquisition_name)

            artifacts = run_exp0_scaling(run_cfg)
            rows.append(
                _best_metric_row(
                    reservoir_name=reservoir_name,
                    acquisition_name=acquisition_name,
                    artifacts=artifacts,
                    selection_metric=selection_metric,
                )
            )

    out_dir = out_base / "exp1"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows).sort_values(
        by=["best_metric", "reservoir", "acquisition"], ascending=[False, True, True]
    )
    summary_df.to_csv(out_dir / "exp1_benchmark_summary.csv", index=False)


if __name__ == "__main__":
    main()
