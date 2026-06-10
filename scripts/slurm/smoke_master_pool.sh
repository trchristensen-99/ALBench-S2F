#!/bin/bash
# Smoke-test the #54 master-pool generator end-to-end on ONE tiny shard before the
# full 205-task array (620 GPU-hours). Generates + labels a few thousand seqs for a
# couple of strategies into an isolated _smoke out-root, so nothing real is touched.
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/smoke_master_pool.sh
#SBATCH --job-name=smoke_master_pool
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1
export AG_ORACLE_CHUNK="${AG_ORACLE_CHUNK:-32}"

ROOT="outputs/_smoke_master_pools"
rm -rf "${ROOT}"
echo "=== smoke_master_pool node=${SLURMD_NODENAME} chunk=${AG_ORACLE_CHUNK} $(date) ==="

# 1) seed-mode (genomic-derived, expandable): tiny prm_5pct shard.
uv run --no-sync python scripts/generate_master_pool.py \
    --task k562 --reservoir prm_5pct \
    --target 4000 --n-shards 2 --shard 0 --mode seed --seed 42 --out-root "${ROOT}"

# 2) index-mode (raw genomic, finite pool slice).
uv run --no-sync python scripts/generate_master_pool.py \
    --task k562 --reservoir genomic \
    --target 4000 --n-shards 2 --shard 0 --mode index --seed 42 --out-root "${ROOT}"

# 3) verify shard schema + materialize a seeded subset cache from the prm shard.
uv run --no-sync python - "${ROOT}" <<'PY'
import sys, numpy as np
from pathlib import Path
root = Path(sys.argv[1])
for sp in sorted(root.glob("k562/*/shard_*.npz")):
    z = np.load(sp, allow_pickle=True)
    assert {"sequences", "oracle_labels", "oracle_id"} <= set(z.files), z.files
    assert str(z["oracle_id"]) == "full856k_clean", z["oracle_id"]
    n = len(z["oracle_labels"])
    assert np.isfinite(z["oracle_labels"]).all(), "non-finite labels"
    assert len(z["sequences"]) == n
    print(f"OK {sp.parent.name}/{sp.name}  n={n}  mean={float(z['oracle_labels'].mean()):.3f}")
PY

uv run --no-sync python scripts/materialize_subset_cache.py \
    --task k562 --reservoir prm_5pct --D 1000 --seed 7 \
    --master-root "${ROOT}" --out "${ROOT}/_subset_prm5_d1000_s7.npz"

echo "=== SMOKE DONE $(date) ==="
