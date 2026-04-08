#!/bin/bash
# Re-run AG S2 yeast oracle scaling with softmax bug fix.
# The old runs all produced val_r=0.0 due to softmax gradient vanishing.
# Fix: replaced softmax→expected_bin with mean-pool across 18 tracks.
#
# Array 0-9: 10 yeast fractions (0.1% to 100%)
#
# Usage:
#   TASK=yeast STUDENT=alphagenome_yeast_s2 sbatch --array=0-9 scripts/slurm/rerun_yeast_ag_s2_fixed.sh
#   TASK=yeast ORACLE=ag STUDENT=alphagenome_yeast_s2 sbatch --array=0-9 scripts/slurm/rerun_yeast_ag_s2_fixed.sh
#
#SBATCH --job-name=yeast_s2_fix
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

TASK="${TASK:-yeast}"
STUDENT="${STUDENT:-alphagenome_yeast_s2}"
ORACLE="${ORACLE:-default}"

YEAST_SIZES=(6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324)
N_TRAIN="${YEAST_SIZES[$SLURM_ARRAY_TASK_ID]}"

echo "=== Yeast AG S2 fixed (mean-pool, no softmax) ==="
echo "Oracle: ${ORACLE}, N_train=${N_TRAIN}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

if [[ "${ORACLE}" == "default" ]]; then
    OUT_DIR="outputs/exp0_oracle_scaling_v4/${TASK}/${STUDENT}"
else
    OUT_DIR="outputs/exp0_oracle_scaling_v4/${TASK}/${STUDENT}_oracle_${ORACLE}"
fi

# Clear old broken results for this fraction
rm -rf "${OUT_DIR}/random/n${N_TRAIN}"

EPOCHS=50
if [[ "${N_TRAIN}" -lt 100000 ]]; then
    EPOCHS=80
fi

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir random \
    --n-replicates 3 \
    --seed 42 \
    --output-dir "${OUT_DIR}" \
    --training-sizes "${N_TRAIN}" \
    --epochs "${EPOCHS}" \
    --ensemble-size 3 \
    --early-stop-patience 10

echo "Done: $(date)"
