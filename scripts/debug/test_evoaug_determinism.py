"""Verify EvoAugTransform is deterministic per seed and diverse across seeds."""

from __future__ import annotations

import numpy as np
import torch

from models.evoaug_transform import EvoAugTransform


def _batch():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 4, size=(16, 200))
    oh = np.zeros((16, 4, 200), dtype=np.float32)
    for n in range(16):
        for i in range(200):
            oh[n, idx[n, i], i] = 1.0
    return torch.from_numpy(oh)


def main() -> None:
    b = _batch()
    a1 = EvoAugTransform(intensity="medium", apply_prob=0.5, seed=123)(b.clone()).numpy()
    a2 = EvoAugTransform(intensity="medium", apply_prob=0.5, seed=123)(b.clone()).numpy()
    a3 = EvoAugTransform(intensity="medium", apply_prob=0.5, seed=999)(b.clone()).numpy()
    same = np.array_equal(a1, a2)
    diff = not np.array_equal(a1, a3)
    print(f"same-seed identical: {same}")
    print(f"diff-seed differs:   {diff}")
    assert same, "same seed produced different augmentation"
    assert diff, "different seeds produced identical augmentation"
    print("PASS: EvoAug deterministic per seed, diverse across seeds")


if __name__ == "__main__":
    main()
