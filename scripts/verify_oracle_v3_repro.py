"""Confirm the canonical oracle_v3 default re-labels an existing grid cache to the
same values -- a guard that a freshly built cache is scored by the SAME ensemble
that produced the rest of the grid, not a drifted or wrong checkpoint.

Why this exists: label_pool.py only STAMPS oracle_id into the output, it does not use
it to pick a checkpoint. If the default checkpoint ever drifts, a pool can be labelled
by one oracle and stamped as another -- an audit this project has already had to run
once. This measures the stamp against reality before we trust a new cache.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)

cache = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

z = np.load(cache, allow_pickle=True)
seqs = [str(s) for s in z["sequences"][:n]]
stored = np.asarray(z["oracle_labels"][:n], dtype=np.float32)

os.environ.setdefault("AG_S2_ORACLE_DIR", f"{REPO}/outputs/oracle_v3/s2")
from experiments.exp1_1_scaling import _load_oracle  # noqa: E402
from scripts.generate_labeled_pools import _label_sequences  # noqa: E402

oracle = _load_oracle("k562", oracle_type="ag_s2")
got = np.asarray(_label_sequences(oracle, seqs), dtype=np.float32).ravel()

r = float(np.corrcoef(stored, got)[0, 1])
mad = float(np.abs(stored - got).max())
print(
    f"repro check on {cache} n={n}: pearson={r:.6f} max|diff|={mad:.5f} "
    f"stored_mean={stored.mean():.4f} got_mean={got.mean():.4f}",
    flush=True,
)
# Tolerant of GPU/cuDNN numerical noise, intolerant of a different ensemble.
assert r > 0.999 and mad < 0.05, f"oracle MISMATCH (pearson={r}, max|diff|={mad}) -- NOT oracle_v3"
print("ORACLE_V3_REPRO_OK", flush=True)
