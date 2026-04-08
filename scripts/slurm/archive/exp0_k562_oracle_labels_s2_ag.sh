#!/bin/bash
# Exp 0: AlphaGenome cached-head oracle-label (Stage 2) scaling curve on K562.
# Uses Stage 2 pseudolabels (better oracle: in_dist=0.9147 vs Stage 1's 0.9052).
# Submits 3 replicates per fraction (array=0-20: 7 fractions × 3 seeds).
#
# Prerequisites:
#   1. Fixed embedding cache: outputs/ag_hashfrag/embedding_cache/ (pool + val rebuilt)
#   2. Stage 2 pseudolabels:  outputs/oracle_pseudolabels_stage2_k562_ag/
#
# Submit with dependency on cache rebuild + pseudolabel generation:
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:CACHE_JOB:PSEUDO_JOB \
#       scripts/slurm/exp0_k562_oracle_labels_s2_ag.sh
#
#SBATCH --job-name=exp0_ag_k562_oracle_s2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-20

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

FRACTIONS=(0.01 0.02 0.05 0.10 0.20 0.50 1.00)
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
FRACTION=${FRACTIONS[$FRAC_IDX]}

echo "AG cached oracle-label (S2) scaling: fraction=${FRACTION}, replicate=${SEED_IDX} (task ${SLURM_ARRAY_TASK_ID}/20)"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp0_k562_scaling_oracle_labels_ag.py \
    ++fraction="${FRACTION}" \
    ++pseudolabel_dir=outputs/oracle_pseudolabels_stage2_k562_ag \
    ++output_dir=outputs/exp0_k562_scaling_oracle_labels_s2_ag \
    ++wandb_mode=offline
