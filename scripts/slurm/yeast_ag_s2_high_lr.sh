#!/bin/bash
# Yeast AG S2 oracle scaling with HIGHER encoder LR (1e-3).
#
# The previous oracle scaling used encoder_lr=1e-4 (same as K562), but yeast
# needs ~10x higher LR because the AG encoder is pretrained on human/mouse
# data. The real-label S2 used lr=1e-3 and achieved 0.795 vs S1's 0.707.
#
# Array 0-9: 10 yeast fractions
#
#SBATCH --job-name=yeast_s2_hlr
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

YEAST_SIZES=(6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324)
N_TRAIN="${YEAST_SIZES[$SLURM_ARRAY_TASK_ID]}"

echo "=== Yeast AG S2 high-LR: N=${N_TRAIN} node=${SLURMD_NODENAME} date=$(date) ==="

# Use higher encoder LR (1e-3) for yeast — the default 1e-4 is too conservative
# Also unfreeze more blocks (0-5 instead of just 4,5) to give more capacity
OUT_DIR="outputs/exp0_oracle_scaling_v4/yeast/alphagenome_yeast_s2_hlr"

EPOCHS=50
if [[ "${N_TRAIN}" -lt 100000 ]]; then
    EPOCHS=80
fi

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task yeast \
    --student alphagenome_yeast_s2 \
    --oracle default \
    --reservoir random \
    --n-replicates 3 \
    --seed 42 \
    --output-dir "${OUT_DIR}" \
    --training-sizes "${N_TRAIN}" \
    --epochs "${EPOCHS}" \
    --ensemble-size 3 \
    --early-stop-patience 10 \
    --lr 0.001

echo "Done: $(date)"
