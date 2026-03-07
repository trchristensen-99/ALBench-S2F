#!/bin/bash
# Train 3 Malinois (BassetBranched) models on full K562 HashFrag train set.
# 3 random seeds, each evaluated on all 4 test metrics.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_malinois_k562_3seeds.sh
#
#SBATCH --job-name=malinois_k562
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "Malinois K562 training: seed_idx=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_malinois_k562.py \
    ++output_dir=outputs/malinois_k562_3seeds

echo "seed_idx=${SLURM_ARRAY_TASK_ID} DONE — $(date)"
