"""In-silico evolution reservoir sampler — student-guided sequence optimization.

Generates sequences by iteratively mutating pool sequences toward specified
activity targets using a student model's predictions as a fitness function.
This is analogous to directed evolution in the lab but using the student model
(NOT the oracle) to evaluate fitness.

Unlike oracle-based approaches, this sampler uses only the student model,
making it suitable for active learning settings where the oracle is expensive.

Supports multiple evolution modes:
- **maximize**: evolve toward high predicted activity
- **minimize**: evolve toward low predicted activity
- **target**: evolve toward a specific activity value
- **diverse_targets**: sample target activities from a specified distribution
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)

# Yeast flanking sequences
_YEAST_FLANK_5 = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG"
_YEAST_FLANK_3 = "GGTTACGGCTGTT"

_NUCLEOTIDES = "ACGT"
_NUC_TO_IDX = {c: i for i, c in enumerate(_NUCLEOTIDES)}


class Predictor(Protocol):
    """Minimal interface for student model predictions."""

    def predict(self, sequences: list[str]) -> np.ndarray: ...


class InSilicoEvolutionGenerativeSampler(ReservoirSampler):
    """Generate sequences via in-silico directed evolution using a student model.

    Starting from pool sequences, applies rounds of mutation + selection
    guided by the student model's predictions. The resulting sequences
    are biased toward regions of sequence space that the student predicts
    to have desired properties.

    This sampler requires a ``student_model`` with a ``predict()`` method.
    If no student is available (e.g., first AL round), falls back to
    random mutagenesis.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_evolution_rounds: int = 5,
        mutation_rate: float = 0.05,
        population_size: int = 100,
        elite_fraction: float = 0.2,
        evolution_mode: str = "maximize",
        target_activity: float | None = None,
        activity_distribution: str = "uniform",
        activity_range: tuple[float, float] = (-1.0, 2.0),
        temperature: float = 1.0,
    ) -> None:
        """Initialize sampler.

        Args:
            seed: Random seed.
            n_evolution_rounds: Number of mutation+selection rounds per lineage.
            mutation_rate: Per-position mutation probability.
            population_size: Population size per lineage during evolution.
            elite_fraction: Fraction of population kept each round.
            evolution_mode: ``"maximize"``, ``"minimize"``, ``"target"``,
                or ``"diverse_targets"``.
            target_activity: Target activity for ``"target"`` mode.
            activity_distribution: For ``"diverse_targets"``: ``"uniform"``
                or ``"normal"``.
            activity_range: Range for activity targets (min, max).
            temperature: Softmax temperature for selection (higher = more random).
        """
        self._rng = np.random.default_rng(seed)
        self.n_evolution_rounds = n_evolution_rounds
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.evolution_mode = evolution_mode
        self.target_activity = target_activity
        self.activity_distribution = activity_distribution
        self.activity_range = activity_range
        self.temperature = temperature

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Backward-compatible: random subset."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    def _extract_region(self, seq: str, task: str) -> str:
        if task == "yeast":
            s = str(seq)
            if s.startswith(_YEAST_FLANK_5) and s.endswith(_YEAST_FLANK_3):
                return s[len(_YEAST_FLANK_5) : -len(_YEAST_FLANK_3)]
            return s[:80] if len(s) >= 80 else s
        return str(seq)

    def _wrap_region(self, region: str, task: str) -> str:
        if task == "yeast":
            return _YEAST_FLANK_5 + region + _YEAST_FLANK_3
        return region

    def _mutate_region(self, region: str) -> str:
        """Apply random point mutations to a DNA region."""
        arr = list(region.upper())
        n = len(arr)
        n_mut = self._rng.binomial(n, self.mutation_rate)
        if n_mut == 0:
            return region
        positions = self._rng.choice(n, size=n_mut, replace=False)
        for pos in positions:
            old_idx = _NUC_TO_IDX.get(arr[pos], 0)
            shift = int(self._rng.integers(1, 4))
            arr[pos] = _NUCLEOTIDES[(old_idx + shift) % 4]
        return "".join(arr)

    def _fitness(
        self,
        predictions: np.ndarray,
        target: float | None = None,
    ) -> np.ndarray:
        """Compute fitness scores from predictions."""
        if self.evolution_mode == "maximize":
            return predictions
        elif self.evolution_mode == "minimize":
            return -predictions
        elif self.evolution_mode in ("target", "diverse_targets"):
            if target is None:
                target = self.target_activity or 0.0
            return -np.abs(predictions - target)
        else:
            return predictions

    def _sample_target(self) -> float:
        """Sample a target activity for diverse_targets mode."""
        lo, hi = self.activity_range
        if self.activity_distribution == "uniform":
            return float(self._rng.uniform(lo, hi))
        else:  # normal
            mean = (lo + hi) / 2
            std = (hi - lo) / 4
            return float(np.clip(self._rng.normal(mean, std), lo, hi))

    def _evolve_one_lineage(
        self,
        seed_region: str,
        student: Predictor,
        task: str,
        target: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Evolve one lineage from a seed sequence.

        Returns the best sequence and evolution metadata.
        """
        # Initialize population by mutating the seed
        population = [seed_region]
        for _ in range(self.population_size - 1):
            population.append(self._mutate_region(seed_region))

        best_seq = seed_region
        best_fitness = float("-inf")

        for round_idx in range(self.n_evolution_rounds):
            # Evaluate fitness
            full_seqs = [self._wrap_region(r, task) for r in population]
            predictions = student.predict(full_seqs)
            fitness = self._fitness(predictions, target)

            # Track best
            round_best_idx = int(np.argmax(fitness))
            if fitness[round_best_idx] > best_fitness:
                best_fitness = fitness[round_best_idx]
                best_seq = population[round_best_idx]

            # Selection: softmax-weighted sampling of elites
            n_elite = max(1, int(self.elite_fraction * len(population)))
            if self.temperature > 0:
                logits = fitness / max(self.temperature, 1e-8)
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                elite_idx = self._rng.choice(len(population), size=n_elite, replace=False, p=probs)
            else:
                elite_idx = np.argsort(fitness)[-n_elite:]

            elites = [population[i] for i in elite_idx]

            # Reproduce: mutate elites to fill population
            population = list(elites)
            while len(population) < self.population_size:
                parent = elites[self._rng.integers(0, len(elites))]
                population.append(self._mutate_region(parent))

        meta = {
            "final_fitness": float(best_fitness),
            "final_prediction": float(predictions[round_best_idx]),
            "n_rounds": self.n_evolution_rounds,
            "target_activity": target,
        }
        return best_seq, meta

    def generate(
        self,
        n_sequences: int,
        base_sequences: list[str] | np.ndarray,
        task: str = "k562",
        student_model: Predictor | None = None,
        batch_size: int = 1000,
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate evolved sequences using student model as fitness function.

        Args:
            n_sequences: Number of sequences to produce.
            base_sequences: Pool of seed sequences to evolve from.
            task: ``"k562"`` or ``"yeast"``.
            student_model: A model with ``predict(list[str]) -> np.ndarray``.
                If None, falls back to random mutagenesis.
            batch_size: Number of lineages to evolve in parallel.
                Larger = fewer predict() calls = faster on GPU.

        Returns:
            Tuple of (evolved_sequences, metadata_df).
        """
        n_pool = len(base_sequences)

        if student_model is None:
            logger.warning(
                "No student model provided for in-silico evolution. "
                "Falling back to random mutagenesis (5% rate)."
            )
            return self._fallback_mutagenesis(n_sequences, base_sequences, task)

        parent_indices = self._rng.choice(n_pool, size=n_sequences, replace=True)
        sequences: list[str] = []
        meta_records: list[dict[str, Any]] = []

        # Process lineages in batches for efficient GPU utilization.
        # Instead of 1 predict() call per lineage per round (very slow),
        # batch all populations across lineages into single predict() calls.
        for batch_start in range(0, n_sequences, batch_size):
            batch_end = min(batch_start + batch_size, n_sequences)
            n_batch = batch_end - batch_start

            # Determine targets for this batch
            targets: list[float | None] = []
            for _ in range(n_batch):
                if self.evolution_mode == "diverse_targets":
                    targets.append(self._sample_target())
                elif self.evolution_mode == "target":
                    targets.append(self.target_activity)
                else:
                    targets.append(None)

            # Initialize populations: n_batch lineages, each with population_size members
            # populations[i] = list of region strings for lineage i
            populations: list[list[str]] = []
            for i in range(n_batch):
                seed_region = self._extract_region(
                    base_sequences[parent_indices[batch_start + i]], task
                )
                pop = [seed_region]
                for _ in range(self.population_size - 1):
                    pop.append(self._mutate_region(seed_region))
                populations.append(pop)

            # Track best per lineage
            best_seqs = [p[0] for p in populations]
            best_fitness = np.full(n_batch, float("-inf"))

            for _round in range(self.n_evolution_rounds):
                # Flatten all populations into one predict() call
                all_seqs: list[str] = []
                for pop in populations:
                    all_seqs.extend(self._wrap_region(r, task) for r in pop)

                all_preds = student_model.predict(all_seqs)

                # Reshape back: (n_batch, population_size)
                preds_2d = all_preds.reshape(n_batch, self.population_size)

                # Process each lineage
                new_populations: list[list[str]] = []
                for i in range(n_batch):
                    fitness = self._fitness(preds_2d[i], targets[i])

                    # Track best
                    round_best = int(np.argmax(fitness))
                    if fitness[round_best] > best_fitness[i]:
                        best_fitness[i] = fitness[round_best]
                        best_seqs[i] = populations[i][round_best]

                    # Selection
                    n_elite = max(1, int(self.elite_fraction * self.population_size))
                    if self.temperature > 0:
                        logits = fitness / max(self.temperature, 1e-8)
                        logits -= logits.max()
                        probs = np.exp(logits)
                        probs /= probs.sum()
                        elite_idx = self._rng.choice(
                            self.population_size, size=n_elite, replace=False, p=probs
                        )
                    else:
                        elite_idx = np.argsort(fitness)[-n_elite:]

                    elites = [populations[i][j] for j in elite_idx]

                    # Reproduce
                    new_pop = list(elites)
                    while len(new_pop) < self.population_size:
                        parent = elites[self._rng.integers(0, len(elites))]
                        new_pop.append(self._mutate_region(parent))
                    new_populations.append(new_pop)

                populations = new_populations

            # Collect results for this batch
            for i in range(n_batch):
                full_seq = self._wrap_region(best_seqs[i], task)
                sequences.append(full_seq)
                meta_records.append(
                    {
                        "seq_idx": batch_start + i,
                        "method": f"in_silico_evolution_{self.evolution_mode}",
                        "source": "evolved",
                        "parent_idx": int(parent_indices[batch_start + i]),
                        "evolution_mode": self.evolution_mode,
                        "final_fitness": float(best_fitness[i]),
                        "target_activity": targets[i],
                    }
                )

            logger.info(
                f"  Evolved {batch_end}/{n_sequences} sequences "
                f"({n_batch} lineages x {self.population_size} pop x "
                f"{self.n_evolution_rounds} rounds)"
            )

        meta = pd.DataFrame(meta_records)
        if "final_fitness" in meta.columns:
            logger.info(
                f"In-silico evolution ({self.evolution_mode}): {n_sequences:,} sequences, "
                f"mean fitness={meta['final_fitness'].mean():.4f}"
            )
        return sequences, meta

    def _fallback_mutagenesis(
        self,
        n_sequences: int,
        base_sequences: list[str] | np.ndarray,
        task: str,
    ) -> tuple[list[str], pd.DataFrame]:
        """Fallback: random mutagenesis when no student model is available."""
        n_pool = len(base_sequences)
        parent_indices = self._rng.choice(n_pool, size=n_sequences, replace=True)
        sequences: list[str] = []

        for idx in range(n_sequences):
            region = self._extract_region(base_sequences[parent_indices[idx]], task)
            mutated = self._mutate_region(region)
            sequences.append(self._wrap_region(mutated, task))

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(n_sequences, dtype=np.int64),
                "method": "in_silico_evolution_fallback",
                "source": "mutagenized_fallback",
                "parent_idx": parent_indices,
            }
        )
        return sequences, meta
