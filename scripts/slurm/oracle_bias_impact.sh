#!/bin/bash
# Test whether oracle CpG bias affects scaling law conclusions.
#
# Runs LegNet training on oracle labels WITH post-hoc CpG correction
# applied. Compares to uncorrected results to see if strategy rankings change.
#
# The CpG correction: pred_corrected = pred - 12.9 * (seq_cpg - 0.01)
# Applied to the oracle pseudolabels in the pool before training.
#
# 6 strategies × 3 sizes (10K, 50K, 200K) × 3 seeds = 54 jobs
# Array: strat_idx * 9 + size_idx * 3 + seed_idx
#
#SBATCH --job-name=bias_imp
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_heavy")
SIZES=(10000 50000 200000)
SEEDS=(42 1042 2042)

STRAT_IDX=$((T / 9))
SIZE_IDX=$(( (T % 9) / 3 ))
SEED_IDX=$((T % 3))

STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUT="outputs/exp1_1_cpg_corrected/k562/legnet_ag_s2"

[ -f "${OUT}/${STRAT}/n${SIZE}/hp0/seed${SEED}/result.json" ] && echo "SKIP" && exit 0

echo "=== CpG-corrected: ${STRAT} n=${SIZE} seed=${SEED} — $(date) ==="

# Step 1: Load pool, apply CpG correction to labels, save as temp pool
uv run --no-sync python << PYEOF
import numpy as np
from pathlib import Path

pool_2m = Path("outputs/labeled_pools_2m/k562/ag_s2/${STRAT}/pool.npz")
pool_618k = Path("outputs/labeled_pools/k562/ag_s2/${STRAT}/pool.npz")
pool_path = pool_2m if pool_2m.exists() else pool_618k

data = np.load(pool_path, allow_pickle=True)
seqs = data["sequences"]
labels = data["labels"].astype(np.float32)

# Apply CpG correction to labels
EXCESS_SLOPE = 12.9
BASELINE_CPG = 0.01

for i in range(len(seqs)):
    seq = str(seqs[i])[:200].upper()
    cpg_freq = seq.count("CG") / max(len(seq) - 1, 1)
    correction = EXCESS_SLOPE * (cpg_freq - BASELINE_CPG)
    labels[i] -= correction

out_path = Path("/tmp/cpg_corrected_pool_${STRAT}.npz")
np.savez_compressed(out_path, sequences=seqs, labels=labels, metadata=data.get("metadata", "cpg_corrected"))
print(f"Saved corrected pool: {out_path} ({len(seqs)} seqs)")
PYEOF

# Step 2: Train LegNet on corrected labels
POOL_DIR="/tmp"

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir "${POOL_DIR}/cpg_corrected_pool_${STRAT}.npz" \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${OUT}" \
    --training-sizes "${SIZE}" \
    --chr-split --lr 0.003 --batch-size 512 \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10

echo "=== DONE — $(date) ==="
