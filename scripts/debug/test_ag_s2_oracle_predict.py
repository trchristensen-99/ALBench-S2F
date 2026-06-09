"""Smoke-test the AG_S2 oracle loader end-to-end.

Loads the canonical full856k_clean ensemble via the production loader and runs
live inference on a handful of random 200bp sequences. Passes if predict()
returns finite floats of the right shape (i.e. the head-name fix resolved the
`Unable to retrieve parameter 'scale'` crash).
"""

from __future__ import annotations

import numpy as np

from experiments.exp1_1_scaling import _load_k562_ag_s2_oracle


def main() -> None:
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(8)]
    oracle = _load_k562_ag_s2_oracle()
    preds = oracle.predict(seqs)
    print(f"PRED SHAPE: {preds.shape}")
    print(f"PRED DTYPE: {preds.dtype}")
    print(f"PRED VALUES: {np.round(preds, 4).tolist()}")
    print(f"ALL FINITE: {bool(np.all(np.isfinite(preds)))}")
    assert preds.shape == (8,), preds.shape
    assert np.all(np.isfinite(preds)), "non-finite predictions"
    print("PASS: AG_S2 oracle predict() works")


if __name__ == "__main__":
    main()
