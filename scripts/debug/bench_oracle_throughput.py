"""Measure AG_S2 oracle labeling throughput (seq/s) on the current GPU.

The oracle does 20 forward passes per sequence (10 folds x fwd+rev strand)
through the 551M-param AlphaGenome backbone, so labeling cost is the gating
factor for the master-pool relabel (#54). Run on an H100 to match production.
"""

import os
import time

import numpy as np

os.environ.setdefault(
    "AG_S2_ORACLE_DIR", os.path.join(os.getcwd(), "outputs/oracle_full856k_clean/s2")
)

from experiments.exp1_1_scaling import _load_oracle

t0 = time.perf_counter()
oracle = _load_oracle("k562", oracle_type="ag_s2")
print(f"[bench] oracle loaded in {time.perf_counter() - t0:.1f}s", flush=True)

rng = np.random.default_rng(0)
bases = np.array(list("ACGT"))


def rand_seqs(n: int) -> list[str]:
    return ["".join(rng.choice(bases, 200)) for _ in range(n)]


_ = oracle.predict(rand_seqs(128))  # warmup / JIT compile
print("[bench] warmup done", flush=True)

for n in [512, 2048, 8192]:
    seqs = rand_seqs(n)
    t = time.perf_counter()
    _ = oracle.predict(seqs)
    dt = time.perf_counter() - t
    rate = n / dt
    print(
        f"[bench] N={n:>6}  {dt:7.2f}s  {rate:8.1f} seq/s  "
        f"-> 1M takes {1_000_000 / rate / 3600:.2f}h",
        flush=True,
    )
