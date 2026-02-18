"""Active-learning loop implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import wandb

from albench.evaluation import evaluate_on_test_sets
from albench.model import SequenceModel
from albench.task import TaskConfig


@dataclass
class RunConfig:
    """Runtime configuration for one AL run."""

    n_rounds: int
    batch_size: int
    reservoir_schedule: dict[int | str, Any]
    acquisition_schedule: dict[int | str, Any]
    output_dir: str
    n_reservoir_candidates: int = 10000


@dataclass
class RoundResult:
    """Metrics and artifacts for one AL round."""

    round_idx: int
    n_labeled: int
    selected_sequences: list[str]
    test_metrics: dict[str, dict[str, float]]
    checkpoint_path: str


def _scheduled(schedule: dict[int | str, Any], round_idx: int) -> Any:
    """Resolve a schedule item for the current round."""
    if round_idx in schedule:
        return schedule[round_idx]
    if "default" in schedule:
        return schedule["default"]
    raise KeyError("schedule must provide a round-specific entry or 'default'")


def run_al_loop(
    task: TaskConfig,
    oracle: SequenceModel,
    student: SequenceModel,
    initial_labeled: list[str],
    run_config: RunConfig,
) -> list[RoundResult]:
    """Run active learning rounds with schedule-based dispatch."""
    results: list[RoundResult] = []
    labeled = list(initial_labeled)
    labels = oracle.predict(labeled)

    student.fit(labeled, labels)

    for round_idx in range(run_config.n_rounds):
        sampler = _scheduled(run_config.reservoir_schedule, round_idx)
        acquirer = _scheduled(run_config.acquisition_schedule, round_idx)

        candidate_indices = sampler.sample(
            candidates=labeled,
            n_samples=min(run_config.n_reservoir_candidates, len(labeled)),
            metadata=None,
        )
        candidate_sequences = [labeled[idx] for idx in candidate_indices]

        selected_local_idx = acquirer.select(
            student=student,
            candidates=candidate_sequences,
            n_select=min(run_config.batch_size, len(candidate_sequences)),
        )
        selected = [candidate_sequences[int(idx)] for idx in selected_local_idx]

        if not selected:
            break

        new_labels = oracle.predict(selected)
        labeled.extend(selected)
        labels = np.concatenate([labels, new_labels], axis=0)

        student.fit(labeled, labels)

        out_dir = Path(run_config.output_dir) / f"round_{round_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_on_test_sets(student, task)

        checkpoint_path = str(out_dir / "student_checkpoint.pt")
        if hasattr(student, "save"):
            student.save(checkpoint_path)

        with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        log_payload: dict[str, float | int] = {
            "round": round_idx,
            "n_labeled": len(labeled),
        }
        for test_name, test_metrics in metrics.items():
            log_payload[f"test/{test_name}/pearson_r"] = test_metrics.get("pearson_r", 0.0)
        if wandb.run is not None:
            wandb.log(log_payload)

        results.append(
            RoundResult(
                round_idx=round_idx,
                n_labeled=len(labeled),
                selected_sequences=selected,
                test_metrics=metrics,
                checkpoint_path=checkpoint_path,
            )
        )

    return results
