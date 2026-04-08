#!/bin/bash
# Exp 0: DREAM-RNN oracle-label (Stage 2) scaling curve on K562.
# Uses Stage 2 pseudolabels (better oracle: in_dist=0.9147 vs Stage 1's 0.9052).
# 3 replicates per fraction (array=0-20: 7 fractions × 3 seeds).
#
# Prerequisites:
#   Stage 2 pseudolabels: outputs/oracle_pseudolabels_stage2_k562_ag/
#
# Submit with dependency on pseudolabel generation:
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:PSEUDO_JOB \
#       scripts/slurm/exp0_k562_oracle_labels_s2_dream.sh
#
#SBATCH --job-name=exp0_k562_oracle_s2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-20

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

FRACTIONS=(0.01 0.02 0.05 0.10 0.20 0.50 1.00)
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
FRACTION=${FRACTIONS[$FRAC_IDX]}

echo "K562 DREAM-RNN oracle-label (S2) scaling: fraction=${FRACTION}, replicate=${SEED_IDX} (task ${SLURM_ARRAY_TASK_ID}/20)"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp0_k562_scaling_oracle_labels.py \
    ++fraction="$FRACTION" \
    ++pseudolabel_dir=outputs/oracle_pseudolabels_stage2_k562_ag \
    ++output_dir=outputs/exp0_k562_scaling_oracle_labels_s2 \
    ++wandb_mode=offline
