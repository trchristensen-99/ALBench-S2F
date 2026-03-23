"""Adaptive reservoir selection.

Trains a small student model on a pilot sample from EACH reservoir strategy,
evaluates which works best, then allocates the remaining budget to the
top-performing strategies.

This automates the process of "pick the best reservoir for this model"
based on a quick validation signal rather than running full scaling curves.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)


class AdaptiveSelectionSampler(ReservoirSampler):
    """Adaptively select the best reservoir strategy via pilot training.

    Phase 1 (pilot): Train a student on ``pilot_size`` sequences from each
    candidate reservoir. Evaluate on a held-out validation set.

    Phase 2 (allocation): Allocate the remaining budget to the top-k
    reservoirs proportional to their pilot performance.

    Parameters
    ----------
    candidate_reservoirs : list[str]
        Names of reservoir strategies to try.
    pilot_size : int
        Number of sequences per reservoir in the pilot phase.
    top_k : int
        Number of top reservoirs to use in the allocation phase.
    allocation_mode : str
        How to split budget among top-k:
        - ``"equal"``: equal split
        - ``"proportional"``: proportional to pilot val Pearson
        - ``"winner_take_all"``: 100% to the best
    seed : int | None
        RNG seed.
    """

    def __init__(
        self,
        candidate_reservoirs: list[str] | None = None,
        pilot_size: int = 1000,
        top_k: int = 3,
        allocation_mode: str = "proportional",
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.candidate_reservoirs = candidate_reservoirs or [
            "random",
            "genomic",
            "motif_planted",
            "prm_5pct",
            "recombination_uniform",
            "snv",
            "dinuc_shuffle",
        ]
        self.pilot_size = pilot_size
        self.top_k = min(top_k, len(self.candidate_reservoirs))
        self.allocation_mode = allocation_mode

    def generate(
        self,
        n_sequences: int,
        task: str = "k562",
        candidate_samplers: dict[str, Any] | None = None,
        oracle: Any = None,
        student_factory: Any = None,
        pool_sequences: list[str] | None = None,
        pool_labels: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate sequences via adaptive selection.

        Args:
            n_sequences: Total sequences to generate.
            task: Task name.
            candidate_samplers: Dict of reservoir name → sampler instance.
            oracle: Oracle model for labeling pilot sequences.
            student_factory: Callable that returns a fresh student model.
            pool_sequences: Pool for strategies that need it.
            pool_labels: Pool labels.
        """
        if candidate_samplers is None:
            raise ValueError("Must provide candidate_samplers dict")

        # Phase 1: Pilot evaluation
        logger.info(
            f"AdaptiveSelection: pilot phase ({self.pilot_size} seqs × "
            f"{len(self.candidate_reservoirs)} reservoirs)"
        )

        pilot_scores: dict[str, float] = {}
        pilot_sequences: dict[str, list[str]] = {}
        pilot_labels: dict[str, np.ndarray] = {}

        for res_name in self.candidate_reservoirs:
            if res_name not in candidate_samplers:
                logger.warning(f"Skipping '{res_name}' — no sampler provided")
                continue

            sampler = candidate_samplers[res_name]
            try:
                # Generate pilot sequences
                if res_name in ("random", "motif_planted", "motif_grammar"):
                    seqs, _ = sampler.generate(self.pilot_size, task=task)
                elif res_name in ("genomic", "gc_matched"):
                    seqs, _ = sampler.generate(
                        self.pilot_size,
                        pool_sequences=pool_sequences,
                        pool_labels=pool_labels,
                    )
                elif res_name.startswith("prm") or res_name == "snv":
                    seqs, _ = sampler.generate(
                        self.pilot_size,
                        base_sequences=pool_sequences,
                        task=task,
                    )
                else:
                    seqs, _ = sampler.generate(self.pilot_size, task=task)

                # Label with oracle
                if oracle is not None:
                    labels = oracle.predict(seqs)
                elif pool_labels is not None and res_name == "genomic":
                    labels = pool_labels[: len(seqs)]
                else:
                    logger.warning(f"Cannot label '{res_name}' — skipping")
                    continue

                pilot_sequences[res_name] = seqs
                pilot_labels[res_name] = labels

                # Train a quick student and evaluate
                if student_factory is not None:
                    student = student_factory()
                    # Split 80/20
                    n_train = int(0.8 * len(seqs))
                    student.fit(seqs[:n_train], labels[:n_train])
                    val_pred = student.predict(seqs[n_train:])
                    val_true = labels[n_train:]
                    from scipy.stats import pearsonr

                    val_r = float(pearsonr(val_pred, val_true)[0])
                    pilot_scores[res_name] = val_r
                    logger.info(f"  {res_name}: pilot val_r={val_r:.4f}")
                else:
                    # Without a student, score by label diversity
                    pilot_scores[res_name] = float(np.std(labels))
                    logger.info(f"  {res_name}: label std={np.std(labels):.4f}")

            except Exception as e:
                logger.error(f"  {res_name}: FAILED — {e}")
                continue

        if not pilot_scores:
            raise RuntimeError("All pilot reservoirs failed")

        # Phase 2: Allocation
        ranked = sorted(pilot_scores.items(), key=lambda x: -x[1])
        top_reservoirs = [name for name, _ in ranked[: self.top_k]]
        top_scores = [pilot_scores[name] for name in top_reservoirs]

        logger.info(
            f"AdaptiveSelection: top-{self.top_k} = {top_reservoirs} "
            f"(scores: {[f'{s:.3f}' for s in top_scores]})"
        )

        # Calculate allocation
        remaining = n_sequences - sum(len(pilot_sequences.get(r, [])) for r in top_reservoirs)
        remaining = max(0, remaining)

        if self.allocation_mode == "equal":
            allocations = {r: remaining // self.top_k for r in top_reservoirs}
        elif self.allocation_mode == "proportional":
            total_score = sum(max(s, 0.01) for s in top_scores)
            allocations = {
                r: int(remaining * max(pilot_scores[r], 0.01) / total_score) for r in top_reservoirs
            }
        elif self.allocation_mode == "winner_take_all":
            allocations = {top_reservoirs[0]: remaining}
            for r in top_reservoirs[1:]:
                allocations[r] = 0
        else:
            raise ValueError(f"Unknown allocation_mode: {self.allocation_mode}")

        # Phase 3: Generate additional sequences for top reservoirs
        all_seqs: list[str] = []
        all_meta: list[dict] = []

        for res_name in top_reservoirs:
            # Add pilot sequences
            if res_name in pilot_sequences:
                all_seqs.extend(pilot_sequences[res_name])
                for i in range(len(pilot_sequences[res_name])):
                    all_meta.append(
                        {
                            "method": f"adaptive_{res_name}",
                            "phase": "pilot",
                            "pilot_score": pilot_scores[res_name],
                        }
                    )

            # Generate additional allocation
            n_additional = allocations.get(res_name, 0)
            if n_additional > 0:
                sampler = candidate_samplers[res_name]
                try:
                    if res_name in ("random", "motif_planted", "motif_grammar"):
                        extra_seqs, _ = sampler.generate(n_additional, task=task)
                    elif res_name in ("genomic", "gc_matched"):
                        extra_seqs, _ = sampler.generate(
                            n_additional,
                            pool_sequences=pool_sequences,
                            pool_labels=pool_labels,
                        )
                    elif res_name.startswith("prm") or res_name == "snv":
                        extra_seqs, _ = sampler.generate(
                            n_additional,
                            base_sequences=pool_sequences,
                            task=task,
                        )
                    else:
                        extra_seqs, _ = sampler.generate(n_additional, task=task)

                    all_seqs.extend(extra_seqs)
                    for i in range(len(extra_seqs)):
                        all_meta.append(
                            {
                                "method": f"adaptive_{res_name}",
                                "phase": "allocation",
                                "pilot_score": pilot_scores[res_name],
                            }
                        )
                except Exception as e:
                    logger.error(f"  Additional generation for '{res_name}' failed: {e}")

        # Trim to exact size
        if len(all_seqs) > n_sequences:
            idx = self._rng.choice(len(all_seqs), size=n_sequences, replace=False)
            all_seqs = [all_seqs[i] for i in idx]
            all_meta = [all_meta[i] for i in idx]

        logger.info(
            f"AdaptiveSelection: {len(all_seqs):,} total sequences "
            f"from {len(top_reservoirs)} selected reservoirs"
        )

        return all_seqs, pd.DataFrame(all_meta)
