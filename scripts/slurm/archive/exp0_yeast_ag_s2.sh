#!/bin/bash
# Yeast AlphaGenome S2 scaling — all 10 fractions.
#
# Usage:
#   sbatch --array=0-9 scripts/slurm/exp0_yeast_ag_s2.sh
#
# Cross-oracle experiments are handled by exp0_oracle_parallel.sh with ORACLE env var:
#   TASK=yeast ORACLE=ag STUDENT=alphagenome_yeast_s2 sbatch --array=0-9 scripts/slurm/exp0_oracle_parallel.sh
#
#SBATCH --job-name=exp0_yeast_ag_s2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

# Yeast pool=6,065,324: 0.1% 0.2% 0.5% 1% 2% 5% 10% 20% 50% 100%
YEAST_SIZES=(6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324)

N_TRAIN="${YEAST_SIZES[$SLURM_ARRAY_TASK_ID]}"

# Fewer epochs for large N
if [[ "${N_TRAIN}" -ge 100000 ]]; then
    EPOCHS=50
else
    EPOCHS=80
fi

echo "=== Yeast AlphaGenome S2 Scaling ==="
echo "N_train: ${N_TRAIN}, Epochs: ${EPOCHS}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task yeast \
    --student alphagenome_yeast_s2 \
    --oracle default \
    --reservoir random \
    --n-replicates 3 \
    --seed 42 \
    --output-dir "outputs/exp0_oracle_scaling_v4/yeast/alphagenome_yeast_s2" \
    --training-sizes "${N_TRAIN}" \
    --epochs "${EPOCHS}" \
    --ensemble-size 3 \
    --early-stop-patience 10

echo "Done: $(date)"
