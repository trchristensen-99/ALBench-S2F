#!/bin/bash
# AG S1 yeast — LR sweep on real labels at full data, 1 seed each.
# We trained S1 with lr=3e-4 (the lower of the HP_GRIDS large-N options),
# but never validated lr=1e-3 (the higher option) on ground-truth labels.
# Picking the best S1 here gives us the strongest warmstart for the S2 sweep.
#
# 3-task array: lr ∈ {3e-4, 5e-4, 1e-3} × bs=256 × seed=42 × n=6065324.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/yeast_ag_s1_real_lr_sweep.sh
#
#SBATCH --job-name=yeast_ags1_real_lr
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --array=0-2

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

N=6065324

LRS=(0.0003 0.0005 0.001)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

OUT="outputs/exp0_s2_warm_real/yeast/alphagenome_yeast_s1_lr${LR}"
echo "=== AG S1 real-label sweep: lr=${LR} ==="

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task yeast \
    --student alphagenome_yeast_s1 \
    --oracle ground_truth --reservoir genomic \
    --n-replicates 1 --no-hp-sweep --seed 42 \
    --lr "${LR}" --batch-size 256 \
    --output-dir "${OUT}" \
    --training-sizes "${N}" \
    --epochs 50 \
    --early-stop-patience 7

echo "=== DONE — $(date) ==="
