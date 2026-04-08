#!/bin/bash
# Complete missing DREAM-RNN scaling curve fractions for K562 and Yeast.
#
# K562: 4 missing sizes (array 0-3):  31974, 63949, 159871, 319742
# Yeast: 5 missing sizes (array 0-4): 303266, 606532, 1213065, 3032662, 6065324
#
# Usage:
#   TASK=k562 sbatch --array=0-3 scripts/slurm/exp0_dream_rnn_remaining.sh
#   TASK=yeast sbatch --array=0-4 scripts/slurm/exp0_dream_rnn_remaining.sh
#
# Cross-oracle experiments are handled by exp0_oracle_parallel.sh with ORACLE env var:
#   TASK=k562 ORACLE=dream_rnn STUDENT=dream_rnn sbatch --array=0-6 scripts/slurm/exp0_oracle_parallel.sh
#   TASK=yeast ORACLE=ag STUDENT=dream_rnn sbatch --array=0-9 scripts/slurm/exp0_oracle_parallel.sh
#
#SBATCH --job-name=exp0_drnn_rem
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

TASK="${TASK:-k562}"

# Missing fractions per task
K562_SIZES=(31974 63949 159871 319742)
YEAST_SIZES=(303266 606532 1213065 3032662 6065324)

if [[ "${TASK}" == "k562" ]]; then
    ALL_SIZES=("${K562_SIZES[@]}")
elif [[ "${TASK}" == "yeast" ]]; then
    ALL_SIZES=("${YEAST_SIZES[@]}")
else
    echo "ERROR: Unknown task ${TASK}. Use k562 or yeast."
    exit 1
fi

N_TRAIN="${ALL_SIZES[$SLURM_ARRAY_TASK_ID]}"

# Fewer epochs for large N
if [[ "${N_TRAIN}" -ge 100000 ]]; then
    EPOCHS=50
else
    EPOCHS=80
fi

echo "=== DREAM-RNN Remaining Fractions ==="
echo "Task: ${TASK}, N_train: ${N_TRAIN}, Epochs: ${EPOCHS}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student dream_rnn \
    --oracle default \
    --reservoir random \
    --n-replicates 3 \
    --seed 42 \
    --output-dir "outputs/exp0_oracle_scaling_v4/${TASK}/dream_rnn" \
    --training-sizes "${N_TRAIN}" \
    --epochs "${EPOCHS}" \
    --ensemble-size 3 \
    --early-stop-patience 10

echo "Done: $(date)"
