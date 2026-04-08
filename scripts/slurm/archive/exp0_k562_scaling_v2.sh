#!/bin/bash
# Exp 0 v2: DREAM-RNN scaling curve on K562 with optimized batch_size=128.
# Re-runs the scaling curve with bs=128 (was 1024 in v1).
# 7 fractions × 3 seeds = 21 array tasks.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp0_k562_scaling_v2.sh
#
#SBATCH --job-name=exp0_k562_v2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-20

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH=$PWD/external/hashFrag/src:$PWD/external/hashFrag:$PATH
source scripts/slurm/setup_hpc_deps.sh

FRACTIONS=(0.01 0.02 0.05 0.10 0.20 0.50 1.00)
N_FRAC=${#FRACTIONS[@]}
FRAC_IDX=$((SLURM_ARRAY_TASK_ID % N_FRAC))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / N_FRAC))
FRACTION=${FRACTIONS[$FRAC_IDX]}

echo "K562 DREAM-RNN scaling v2 (bs=128): fraction=${FRACTION} seed_slot=${SEED_IDX}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp0_k562_scaling.py \
    fraction="$FRACTION" \
    data_path=data/k562 \
    output_dir=outputs/exp0_k562_scaling_v2 \
    batch_size=128 \
    lr=0.005 \
    lr_lstm=0.005 \
    wandb_mode=offline

echo "fraction=${FRACTION} seed_slot=${SEED_IDX} DONE — $(date)"
