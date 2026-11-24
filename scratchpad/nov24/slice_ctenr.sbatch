#!/bin/bash
# Slice motif_ct_enriched to 400k/500k/600k from its nested 1M pool, for sc46.
#
# No generation and no oracle call: every canonical pool is nested, so a size below the
# one on disk is that pool's first N rows and still contains the 300k set a smaller run
# used. That is the property the scaling comparison needs -- only the amount of data
# changes between rungs, never which sequences were drawn.
#
# On the queue rather than the login node: each slice loads a 1M-row npz and writes a
# compressed one, which is minutes of CPU and several GB of memory.
#SBATCH --job-name=slice46
#SBATCH --partition=cpuq
#SBATCH --account=cshlsbs
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/grid_pools/logs/slice46_%j.log
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"

# --reps "" : only the canonical seed-42 pool has a 1M draw, so there is nothing to slice
# for the replicate pools and asking would just print three SKIPs.
.venv/bin/python scripts/slice_intermediate_pools.py \
  --reservoirs motif_ct_enriched \
  --sizes 400000,500000,600000 \
  --reps ""

# Verify nesting rather than assuming it: the whole point of the slice is that 300k stays
# a prefix of 400k, and a silent failure here would make the curve compare different
# draws at different sizes instead of different amounts of one draw.
.venv/bin/python - <<'PY'
import numpy as np
base = np.load(
    "outputs/grid_pools/k562_nested/motif_ct_enriched_d300000.npz", allow_pickle=True
)["sequences"]
for n in (400000, 500000, 600000, 1000000):
    s = np.load(
        f"outputs/grid_pools/k562_nested/motif_ct_enriched_d{n}.npz", allow_pickle=True
    )["sequences"]
    ok = len(s) == n and np.array_equal(s[:300000], base)
    print(f"  d{n}: n={len(s):,} 300k-prefix-intact={np.array_equal(s[:300000], base)}")
    assert ok, f"d{n} is not a superset of the 300k pool"
print("nesting verified")
PY
