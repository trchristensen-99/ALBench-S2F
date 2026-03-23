#!/bin/bash
# Experiment 0: Oracle-label scaling — ONE FRACTION PER JOB for max parallelism.
#
# Array job: each task handles one downsample fraction.
# K562: 7 fractions (array 0-6), Yeast: 10 fractions (array 0-9).
#
# Usage:
#   TASK=k562 STUDENT=dream_cnn sbatch --array=0-6 scripts/slurm/exp0_oracle_parallel.sh
#   TASK=k562 STUDENT=dream_rnn sbatch --array=0-6 scripts/slurm/exp0_oracle_parallel.sh
#   TASK=k562 STUDENT=alphagenome_k562_s1 sbatch --array=0-6 scripts/slurm/exp0_oracle_parallel.sh
#   TASK=yeast STUDENT=dream_cnn sbatch --array=0-9 scripts/slurm/exp0_oracle_parallel.sh
#   TASK=yeast STUDENT=dream_rnn sbatch --array=0-9 scripts/slurm/exp0_oracle_parallel.sh
#   TASK=yeast STUDENT=alphagenome_yeast_s2 sbatch --array=0-9 scripts/slurm/exp0_oracle_parallel.sh
#
#SBATCH --job-name=exp0_par
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
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
STUDENT="${STUDENT:-dream_cnn}"
ORACLE="${ORACLE:-default}"
N_REPLICATES="${N_REPLICATES:-3}"
SEED="${SEED:-42}"

# Define fractions per task
# K562 pool=319,742: 1% 2% 5% 10% 20% 50% 100%
K562_SIZES=(3197 6395 15987 31974 63949 159871 319742)
# Yeast pool=6,065,324: 0.1% 0.2% 0.5% 1% 2% 5% 10% 20% 50% 100%
YEAST_SIZES=(6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324)

if [[ "${TASK}" == "k562" ]]; then
    ALL_SIZES=("${K562_SIZES[@]}")
elif [[ "${TASK}" == "yeast" ]]; then
    ALL_SIZES=("${YEAST_SIZES[@]}")
else
    echo "ERROR: Unknown task ${TASK}"
    exit 1
fi

N_TRAIN="${ALL_SIZES[$SLURM_ARRAY_TASK_ID]}"

echo "=== Experiment 0: Oracle-Label Scaling (parallel) ==="
echo "Task: ${TASK}, Student: ${STUDENT}, Oracle: ${ORACLE}"
echo "Array task ${SLURM_ARRAY_TASK_ID}: N_train=${N_TRAIN}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

OUT_DIR="outputs/exp0_oracle_scaling_v4/${TASK}/${STUDENT}"

# Fewer epochs for large N (val plateaus by epoch ~40-50)
if [[ "${N_TRAIN}" -ge 100000 ]]; then
    EPOCHS=50
else
    EPOCHS=80
fi

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir random \
    --n-replicates "${N_REPLICATES}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --training-sizes "${N_TRAIN}" \
    --epochs "${EPOCHS}" \
    --ensemble-size 3 \
    --early-stop-patience 10

echo "Done: $(date)"
