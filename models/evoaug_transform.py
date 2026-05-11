"""Training-time EvoAug transformation for one-hot DNA batches.

Wraps EvoAugStructuralSampler for per-batch application during training.
Supports intensity levels (light, medium, heavy) and per-sample apply probability.

Based on Lee/Koo 2023 (EvoAug paper): apply random biologically-plausible
structural variations (deletions, insertions, inversions, translocations,
tandem duplications, point mutations) to a fraction of training samples.

Each intensity preset matches the existing config YAML files:
  light  ≈ evoaug.yaml          (default basic)
  medium ≈ evoaug_prior.yaml    (mild structural)
  heavy  ≈ evoaug_structural.yaml (aggressive)
  extreme ≈ evoaug_heavy.yaml    (most aggressive)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from albench.reservoir.evoaug_structural import EvoAugStructuralSampler

INTENSITY_PRESETS = {
    "light": dict(
        p_deletion=0.2, p_insertion=0.2, p_inversion=0.1,
        p_translocation=0.05, p_tandem_dup=0.05, p_point_mutation=0.2,
        max_indel_size=10, max_inversion_size=15,
        point_mutation_rate=0.01, min_events=1, max_events=1,
    ),
    "medium": dict(
        p_deletion=0.3, p_insertion=0.3, p_inversion=0.2,
        p_translocation=0.15, p_tandem_dup=0.1, p_point_mutation=0.3,
        max_indel_size=20, max_inversion_size=30,
        point_mutation_rate=0.02, min_events=1, max_events=3,
    ),
    "heavy": dict(
        p_deletion=0.4, p_insertion=0.4, p_inversion=0.3,
        p_translocation=0.2, p_tandem_dup=0.15, p_point_mutation=0.5,
        max_indel_size=30, max_inversion_size=50,
        point_mutation_rate=0.05, min_events=2, max_events=5,
    ),
}

_NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
_IDX_TO_NUC = "ACGT"


class EvoAugTransform:
    """Apply EvoAug operations to one-hot DNA batches during training.

    Args:
        intensity: One of "light", "medium", "heavy" (or custom params via init).
        apply_prob: Probability that EvoAug is applied to a given sample per batch.
            apply_prob=0.0 → never apply; 1.0 → always apply.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        intensity: str = "medium",
        apply_prob: float = 0.5,
        seed: int | None = None,
        target_length: int | None = None,
    ) -> None:
        if intensity not in INTENSITY_PRESETS:
            raise ValueError(f"intensity must be one of {list(INTENSITY_PRESETS)}")
        self.intensity = intensity
        self.apply_prob = apply_prob
        self.target_length = target_length
        params = INTENSITY_PRESETS[intensity]
        # Each batch uses a fresh sampler with shared seed (so different ops across batches)
        self._sampler = EvoAugStructuralSampler(seed=seed, **params)

    def _onehot_to_str(self, oh: np.ndarray) -> str:
        """(C, L) or (L, C) one-hot → ACGT string. Handles channels-first or last."""
        if oh.shape[0] == 4 and oh.shape[1] != 4:
            arr = oh.argmax(axis=0)
        else:
            arr = oh.argmax(axis=-1)
        return "".join(_IDX_TO_NUC[i] for i in arr.tolist())

    def _str_to_onehot(self, seq: str, target_len: int, channels_first: bool) -> np.ndarray:
        """ACGT string → (C, L) one-hot."""
        seq = seq.upper()
        if len(seq) < target_len:
            seq = seq + "N" * (target_len - len(seq))
        elif len(seq) > target_len:
            seq = seq[:target_len]
        oh = np.zeros((4, target_len), dtype=np.float32)
        for i, c in enumerate(seq):
            if c in _NUC_TO_IDX:
                oh[_NUC_TO_IDX[c], i] = 1.0
        if not channels_first:
            oh = oh.T  # (L, C)
        return oh

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply EvoAug to a batch of one-hot encoded sequences in place.

        batch shape: (N, C, L) or (N, L, C). N samples. Each sample has p=apply_prob
        chance of being augmented.
        """
        if self.apply_prob <= 0.0:
            return batch
        device = batch.device
        x = batch.detach().cpu().numpy()
        channels_first = x.shape[1] == 4
        target_len = x.shape[2] if channels_first else x.shape[1]
        target_len = self.target_length or target_len
        new_x = x.copy()
        for i in range(x.shape[0]):
            if np.random.rand() >= self.apply_prob:
                continue
            seq = self._onehot_to_str(x[i])
            # Apply EvoAug operations
            seq_list = list(seq)
            n_events = self._sampler._rng.integers(
                self._sampler.min_events, self._sampler.max_events + 1
            )
            for _ in range(int(n_events)):
                ops = [
                    (self._sampler.p_deletion, self._sampler._apply_deletion),
                    (self._sampler.p_insertion, self._sampler._apply_insertion),
                    (self._sampler.p_inversion, self._sampler._apply_inversion),
                    (self._sampler.p_translocation, self._sampler._apply_translocation),
                    (self._sampler.p_tandem_dup, self._sampler._apply_tandem_dup),
                ]
                # Pick one operation weighted by probability
                weights = np.array([p for p, _ in ops])
                if weights.sum() == 0:
                    break
                weights = weights / weights.sum()
                op_idx = self._sampler._rng.choice(len(ops), p=weights)
                seq_list, _ = ops[op_idx][1](seq_list)
            # Apply point mutations
            if self._sampler._rng.random() < self._sampler.p_point_mutation:
                seq_list, _ = self._sampler._apply_point_mutations(seq_list)
            new_seq = "".join(seq_list)
            new_x[i] = self._str_to_onehot(new_seq, target_len, channels_first)
        return torch.from_numpy(new_x).to(device)
