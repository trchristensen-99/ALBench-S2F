#!/bin/bash
# Exp 0: AlphaGenome cached-head oracle-label scaling curve on K562 (float32 cache + pseudolabels).
# 3 seeds × 7 fractions = 21 tasks.
#
# Submit (with dependency on pseudolabel generation):
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:$PL_JOB \
#       scripts/slurm/exp0_k562_scaling_oracle_ag_f32.sh
#
#SBATCH --job-name=exp0_ag_oracle_f32
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
N_FRACTIONS=${#FRACTIONS[@]}

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_FRACTIONS ))
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID % N_FRACTIONS ))
FRACTION=${FRACTIONS[$FRAC_IDX]}

echo "AG oracle-label scaling (f32): fraction=${FRACTION} seed_idx=${SEED_IDX} (task ${SLURM_ARRAY_TASK_ID})"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/exp0_k562_scaling_oracle_labels_ag.py \
    ++fraction="${FRACTION}" \
    ++cache_dir=outputs/ag_hashfrag/embedding_cache_f32 \
    ++pseudolabel_dir=outputs/oracle_pseudolabels_k562_ag_f32 \
    ++output_dir=outputs/exp0_k562_scaling_oracle_labels_ag_f32 \
    ++wandb_mode=offline

echo "fraction=${FRACTION} seed_idx=${SEED_IDX} DONE — $(date)"
