"""Experiment 5: select and export the best student checkpoint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

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


def _write_fasta(path: Path, sequences: Iterable[str], prefix: str = "exp5_seq") -> int:
    """Write sequences to FASTA and return count written."""
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for i, seq in enumerate(sequences, start=1):
            handle.write(f">{prefix}_{i}\n")
            handle.write(f"{seq}\n")
            count += 1
    return count


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

    selected = list(best_result.selected_sequences)
    export_limit = int(cfg.experiment.get("exp5_export_limit", len(selected)))
    if export_limit < 0:
        raise ValueError("experiment.exp5_export_limit must be >= 0")
    selected = selected[: min(export_limit, len(selected))]

    selected_df = pd.DataFrame(
        [
            {"rank": i + 1, "sequence": seq, "round_idx": best_round}
            for i, seq in enumerate(selected)
        ]
    )
    selected_df.to_csv(out_dir / "exp5_selected_sequences.csv", index=False)

    fasta_count = _write_fasta(out_dir / "exp5_selected_sequences.fasta", selected, prefix="exp5")

    summary = {
        "best_round_idx": best_round,
        "best_n_labeled": int(best_result.n_labeled),
        "best_test_set": str(best_row["test_set"]),
        "selection_metric": metric,
        "selection_metric_value": float(best_row[metric]),
        "source_checkpoint_path": str(source_ckpt),
        "best_checkpoint_path": str(target_ckpt),
        "checkpoint_copied": copied,
        "selected_sequences_exported": int(len(selected)),
        "selected_sequences_fasta_records": int(fasta_count),
        "selected_sequences_csv": str(out_dir / "exp5_selected_sequences.csv"),
        "selected_sequences_fasta": str(out_dir / "exp5_selected_sequences.fasta"),
    }
    with (out_dir / "exp5_best_student_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
