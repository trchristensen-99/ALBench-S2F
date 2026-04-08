"""Mixed-pool reservoir: combines sequences from multiple reservoir strategies.

Generates sequences from multiple reservoirs in specified proportions,
then combines them into a single training pool. This tests whether
mixing complementary strategies produces better results than any single one.

For example, mixing motif-planted (good for AG) with SNV (good for DREAM-RNN)
might produce a more robust training set that works well across architectures.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

logger = logging.getLogger(__name__)


class MixedPoolSampler(ReservoirSampler):
    """Generate sequences by mixing multiple reservoir strategies.

    Parameters
    ----------
    component_configs : list[dict]
        Each dict has ``"name"`` (reservoir config name) and ``"fraction"``
        (proportion of total pool from this strategy). Fractions must sum to 1.
    seed : int | None
        RNG seed.
    """

    def __init__(
        self,
        component_configs: list[dict[str, Any]],
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.component_configs = component_configs
        total = sum(c["fraction"] for c in component_configs)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Component fractions must sum to 1.0, got {total}")

    def generate(
        self,
        n_sequences: int,
        task: str = "k562",
        component_samplers: dict[str, Any] | None = None,
        pool_sequences: list[str] | None = None,
        pool_labels: np.ndarray | None = None,
        student_model: Any = None,
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate mixed-pool sequences.

        Args:
            n_sequences: Total sequences to generate.
            task: Task name.
            component_samplers: Dict mapping component name to instantiated sampler.
            pool_sequences: Genomic pool sequences (for strategies that need them).
            pool_labels: Pool labels (for strategies that need them).
            student_model: Model for ISE strategies.
        """
        if component_samplers is None:
            raise ValueError("Must provide component_samplers dict")

        all_seqs: list[str] = []
        all_meta: list[dict] = []

        for config in self.component_configs:
            name = config["name"]
            fraction = config["fraction"]
            n_component = max(1, int(n_sequences * fraction))

            if name not in component_samplers:
                logger.warning(f"Sampler '{name}' not found, using random fallback")
                seqs = [
                    "".join(self._rng.choice(list("ACGT"), size=200)) for _ in range(n_component)
                ]
                meta_records = [{"method": "random_fallback", "source": name}] * n_component
            else:
                sampler = component_samplers[name]
                try:
                    if name in ("random", "motif_planted", "motif_grammar", "motif_grammar_tight"):
                        seqs, meta = sampler.generate(n_component, task=task)
                    elif name == "dinuc_shuffle":
                        seqs, meta = sampler.generate(
                            n_component,
                            task=task,
                            method="dinuc_shuffle",
                            reference_sequences=pool_sequences,
                        )
                    elif name in ("genomic", "gc_matched"):
                        seqs, meta = sampler.generate(
                            n_component, pool_sequences=pool_sequences, pool_labels=pool_labels
                        )
                    elif name.startswith("prm") or name == "snv":
                        seqs, meta = sampler.generate(
                            n_component, base_sequences=pool_sequences, task=task
                        )
                    elif name.startswith("recombination") or name.startswith("evoaug"):
                        seqs, meta = sampler.generate(
                            n_component, base_sequences=pool_sequences, task=task
                        )
                    elif name.startswith("ise"):
                        seqs, meta = sampler.generate(
                            n_component,
                            base_sequences=pool_sequences,
                            task=task,
                            student_model=student_model,
                        )
                    else:
                        seqs, meta = sampler.generate(n_component, task=task)
                    meta_records = (
                        meta.to_dict("records") if hasattr(meta, "to_dict") else [{}] * len(seqs)
                    )
                except Exception as e:
                    logger.error(f"Failed to generate from '{name}': {e}")
                    seqs = []
                    meta_records = []

            for m in meta_records:
                m["component"] = name
                m["component_fraction"] = fraction

            all_seqs.extend(seqs)
            all_meta.extend(meta_records)

        # Trim or pad to exact n_sequences
        if len(all_seqs) > n_sequences:
            idx = self._rng.choice(len(all_seqs), size=n_sequences, replace=False)
            all_seqs = [all_seqs[i] for i in idx]
            all_meta = [all_meta[i] for i in idx]
        elif len(all_seqs) < n_sequences:
            # Pad with random sequences
            deficit = n_sequences - len(all_seqs)
            for _ in range(deficit):
                all_seqs.append("".join(self._rng.choice(list("ACGT"), size=200)))
                all_meta.append({"method": "padding", "component": "random"})

        # Shuffle
        perm = self._rng.permutation(len(all_seqs))
        all_seqs = [all_seqs[i] for i in perm]
        all_meta = [all_meta[i] for i in perm]

        logger.info(
            f"MixedPool: {n_sequences:,} sequences from {len(self.component_configs)} components"
        )

        return all_seqs, pd.DataFrame(all_meta)
