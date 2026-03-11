#!/bin/bash
# Exp 0 v3: DREAM-RNN oracle-label scaling curve on K562.
# Fixed LR: v2 used lr=0.005 which caused training divergence at peak OneCycleLR.
# lr=0.001 is stable for oracle (smoother) labels.
#
# 7 fractions × 3 seeds = 21 array tasks.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp0_k562_scaling_oracle_labels_v3.sh
#
#SBATCH --job-name=exp0_k562_orc_v3
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
source scripts/slurm/setup_hpc_deps.sh

FRACTIONS=(0.01 0.02 0.05 0.10 0.20 0.50 1.00)
N_FRAC=${#FRACTIONS[@]}
FRAC_IDX=$((SLURM_ARRAY_TASK_ID % N_FRAC))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / N_FRAC))
FRACTION=${FRACTIONS[$FRAC_IDX]}

echo "K562 DREAM-RNN oracle-label scaling v3 (lr=0.001): fraction=${FRACTION} seed_slot=${SEED_IDX}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp0_k562_scaling_oracle_labels.py \
    ++fraction="$FRACTION" \
    ++output_dir=outputs/exp0_k562_scaling_oracle_labels_v3 \
    ++batch_size=128 \
    ++lr=0.001 \
    ++lr_lstm=0.001 \
    ++wandb_mode=offline

echo "fraction=${FRACTION} seed_slot=${SEED_IDX} DONE — $(date)"
