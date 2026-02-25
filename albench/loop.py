"""Active-learning loop — ALLoop megaclass plus run_al_loop compatibility shim.

The primary API is :class:`ALLoop`: initialize it once with all configuration,
then call :meth:`ALLoop.run` to execute all rounds (or :meth:`ALLoop.step` to
run a single round and inspect state between rounds).

The functional ``run_al_loop(...)`` interface is kept as a thin backward-
compatible wrapper around :class:`ALLoop`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# wandb is an optional dependency — logging is skipped gracefully when absent.
try:
    import wandb as _wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

from albench.evaluation import evaluate_on_test_sets
from albench.model import SequenceModel
from albench.task import TaskConfig

# ---------------------------------------------------------------------------
# Config and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Runtime configuration for one AL run.

    Attributes:
        n_rounds: Number of active-learning rounds.
        batch_size: Number of sequences to select per round.
        reservoir_schedule: Mapping of ``round_idx → ReservoirSampler``
            (use key ``"default"`` to apply the same sampler every round).
        acquisition_schedule: Mapping of ``round_idx → AcquisitionFunction``
            (use key ``"default"`` to apply the same function every round).
        output_dir: Root directory for per-round checkpoints and metrics.
        n_reservoir_candidates: Candidate pool size fed to the acquisition
            function at each round (default 10 000).
    """

    n_rounds: int
    batch_size: int
    reservoir_schedule: dict[int | str, Any]
    acquisition_schedule: dict[int | str, Any]
    output_dir: str
    n_reservoir_candidates: int = 10_000


@dataclass
class RoundResult:
    """Metrics and artifacts produced by a single AL round.

    Attributes:
        round_idx: Zero-based round index.
        n_labeled: Total number of labeled sequences after this round.
        selected_sequences: Sequences chosen by the acquisition function.
        test_metrics: Per-test-set evaluation metrics.
        checkpoint_path: Path to the saved student checkpoint.
        round_wall_seconds: Wall-clock time for the round.
    """

    round_idx: int
    n_labeled: int
    selected_sequences: list[str]
    test_metrics: dict[str, dict[str, float]]
    checkpoint_path: str
    round_wall_seconds: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scheduled(schedule: dict[int | str, Any], round_idx: int) -> Any:
    """Resolve a schedule entry for *round_idx*, falling back to ``"default"``."""
    if round_idx in schedule:
        return schedule[round_idx]
    if "default" in schedule:
        return schedule["default"]
    raise KeyError(f"No schedule entry for round {round_idx} and no 'default' key.")


# ---------------------------------------------------------------------------
# ALLoop megaclass
# ---------------------------------------------------------------------------


class ALLoop:
    """Active-learning loop that tracks state across rounds.

    Initialize once with task, oracle, student, and config; then call
    :meth:`run` to execute all rounds or :meth:`step` to run one at a time.

    Attributes:
        task: Task configuration with test-set info.
        oracle: Oracle model used to label selected sequences.
        student: Student model trained on labeled data.
        run_config: AL loop hyperparameters.
        labeled: Current list of labeled sequences (grows each round).
        labels: Labels corresponding to ``labeled`` (grows each round).
        pool: Remaining unlabeled pool (shrinks each round, ``None`` if
            running in generative mode).
        metadata_pool: Optional metadata aligned with ``pool``.
        round_idx: Index of the *next* round to be executed.
        results: Accumulated :class:`RoundResult` objects from completed rounds.
    """

    def __init__(
        self,
        task: TaskConfig,
        oracle: Any,
        student: Any,
        initial_labeled: list[str],
        run_config: RunConfig,
        pool_sequences: list[str] | None = None,
        pool_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self.task = task
        self.oracle = oracle
        self.student = student
        self.run_config = run_config

        # Mutable state
        self.labeled: list[str] = list(initial_labeled)
        self.labels: np.ndarray = oracle.predict(self.labeled)
        self.pool: list[str] = list(pool_sequences) if pool_sequences is not None else []
        self.metadata_pool: list[dict[str, Any]] | None = (
            list(pool_metadata) if pool_metadata is not None else None
        )
        self.use_fixed_pool: bool = pool_sequences is not None
        self.round_idx: int = 0
        self.results: list[RoundResult] = []

        # Initial fit on seed data
        student.fit(self.labeled, self.labels)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(self) -> RoundResult | None:
        """Execute one AL round and return its :class:`RoundResult`.

        Returns ``None`` if the loop is already complete (exhausted
        ``n_rounds`` or ran out of pool sequences).
        """
        if self.round_idx >= self.run_config.n_rounds:
            return None
        if self.use_fixed_pool and len(self.pool) == 0:
            return None

        round_start = time.perf_counter()
        run_config = self.run_config

        sampler = _scheduled(run_config.reservoir_schedule, self.round_idx)
        acquirer = _scheduled(run_config.acquisition_schedule, self.round_idx)

        # Reservoir step
        if self.use_fixed_pool:
            candidate_indices: list[int] = sampler.sample(
                candidates=self.pool,
                n_samples=min(run_config.n_reservoir_candidates, len(self.pool)),
                metadata=self.metadata_pool,
            )
            candidate_sequences: list[str] = [self.pool[i] for i in candidate_indices]
        else:
            candidate_indices = sampler.sample(
                candidates=self.labeled,
                n_samples=min(run_config.n_reservoir_candidates, len(self.labeled)),
                metadata=None,
            )
            candidate_sequences = [self.labeled[i] for i in candidate_indices]

        # Acquisition step
        selected_local_idx: list[int] = acquirer.select(
            student=self.student,
            candidates=candidate_sequences,
            n_select=min(run_config.batch_size, len(candidate_sequences)),
        )
        selected: list[str] = [candidate_sequences[i] for i in selected_local_idx]

        if not selected:
            return None

        new_labels: np.ndarray = self.oracle.predict(selected)
        self.labeled.extend(selected)
        self.labels = np.concatenate([self.labels, new_labels], axis=0)

        # Remove selected sequences from unlabeled pool
        if self.use_fixed_pool:
            selected_pool_idx = {candidate_indices[i] for i in selected_local_idx}
            keep = [i for i in range(len(self.pool)) if i not in selected_pool_idx]
            self.pool = [self.pool[i] for i in keep]
            if self.metadata_pool is not None:
                self.metadata_pool = [self.metadata_pool[i] for i in keep]

        self.student.fit(self.labeled, self.labels)

        # Save checkpoint and metrics
        out_dir = Path(run_config.output_dir) / f"round_{self.round_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, dict[str, float]] = evaluate_on_test_sets(self.student, self.task)

        checkpoint_path = str(out_dir / "student_checkpoint.pt")
        if hasattr(self.student, "save"):
            self.student.save(checkpoint_path)

        with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        # W&B logging
        log_payload: dict[str, float | int] = {
            "round": self.round_idx,
            "n_labeled": len(self.labeled),
        }
        for test_name, test_metrics in metrics.items():
            log_payload[f"test/{test_name}/pearson_r"] = test_metrics.get("pearson_r", 0.0)
        round_duration = time.perf_counter() - round_start
        if _WANDB_AVAILABLE and _wandb is not None and _wandb.run is not None:
            log_payload["round/wall_seconds"] = round_duration
            _wandb.log(log_payload)

        result = RoundResult(
            round_idx=self.round_idx,
            n_labeled=len(self.labeled),
            selected_sequences=selected,
            test_metrics=metrics,
            checkpoint_path=checkpoint_path,
            round_wall_seconds=round_duration,
        )
        self.results.append(result)
        self.round_idx += 1
        return result

    def run(self) -> list[RoundResult]:
        """Execute all remaining rounds and return the accumulated results."""
        while self.round_idx < self.run_config.n_rounds:
            result = self.step()
            if result is None:
                break
        return self.results

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        """``True`` once all rounds have completed or the pool is exhausted."""
        if self.round_idx >= self.run_config.n_rounds:
            return True
        if self.use_fixed_pool and len(self.pool) == 0:
            return True
        return False

    @property
    def n_labeled(self) -> int:
        """Current number of labeled sequences."""
        return len(self.labeled)


# ---------------------------------------------------------------------------
# Backward-compatible functional API
# ---------------------------------------------------------------------------


def run_al_loop(
    task: TaskConfig,
    oracle: Any,
    student: Any,
    initial_labeled: list[str],
    run_config: RunConfig,
    pool_sequences: list[str] | None = None,
    pool_metadata: list[dict[str, Any]] | None = None,
) -> list[RoundResult]:
    """Run active learning rounds with schedule-based dispatch.

    This is a thin wrapper around :class:`ALLoop` kept for backward
    compatibility. Prefer instantiating :class:`ALLoop` directly when you
    need to inspect or resume state between rounds.

    Args:
        task: Task configuration with test-set info.
        oracle: Oracle model implementing ``predict(list[str]) -> ndarray``.
        student: Student model implementing ``predict``, ``fit``, and
            optionally ``uncertainty``, ``embed``, and ``save``.
        initial_labeled: Seed sequences for the first round.
        run_config: AL loop hyperparameters.
        pool_sequences: Optional fixed pool of unlabeled sequences.
        pool_metadata: Optional metadata aligned with ``pool_sequences``.

    Returns:
        List of per-round :class:`RoundResult` objects.
    """
    loop = ALLoop(
        task=task,
        oracle=oracle,
        student=student,
        initial_labeled=initial_labeled,
        run_config=run_config,
        pool_sequences=pool_sequences,
        pool_metadata=pool_metadata,
    )
    return loop.run()
