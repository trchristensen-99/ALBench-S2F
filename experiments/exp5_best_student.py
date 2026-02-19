"""Experiment 5: select and export the best student checkpoint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, open_dict

from experiments.exp0_scaling import run_exp0_scaling


def _resolve_selection_metric(cfg: DictConfig) -> str:
    """Resolve metric used for best-checkpoint selection."""
    if "selection_metric" in cfg.experiment:
        return str(cfg.experiment.selection_metric)
    return "pearson_r"


def _pick_best_row(curve: pd.DataFrame, metric: str) -> pd.Series:
    """Return the best row by the requested metric."""
    if metric not in curve.columns:
        raise KeyError(
            f"Metric '{metric}' not present in scaling-curve columns: {list(curve.columns)}"
        )
    return curve.sort_values(by=metric, ascending=False).iloc[0]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Experiment 5 and select the best student by test Pearson R."""
    load_dotenv()
    with open_dict(cfg):
        cfg.experiment.name = "exp5_best_student"

    artifacts = run_exp0_scaling(cfg)
    if artifacts.curve.empty or not artifacts.results:
        return

    metric = _resolve_selection_metric(cfg)
    best_row = _pick_best_row(artifacts.curve, metric=metric)
    best_round = int(best_row["round_idx"])
    round_lookup = {r.round_idx: r for r in artifacts.results}
    if best_round not in round_lookup:
        raise KeyError(f"Best round {best_round} missing from round artifacts")

    best_result = round_lookup[best_round]
    source_ckpt = Path(best_result.checkpoint_path)
    out_dir = Path(artifacts.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_ckpt = out_dir / "best_student_checkpoint.pt"
    copied = False
    if source_ckpt.exists():
        shutil.copy2(source_ckpt, target_ckpt)
        copied = True

    summary = {
        "best_round_idx": best_round,
        "best_n_labeled": int(best_result.n_labeled),
        "best_test_set": str(best_row["test_set"]),
        "selection_metric": metric,
        "selection_metric_value": float(best_row[metric]),
        "source_checkpoint_path": str(source_ckpt),
        "best_checkpoint_path": str(target_ckpt),
        "checkpoint_copied": copied,
    }
    with (out_dir / "exp5_best_student_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
