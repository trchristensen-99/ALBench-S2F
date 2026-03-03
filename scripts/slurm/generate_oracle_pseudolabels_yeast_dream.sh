#!/bin/bash
# Generate DREAM-RNN yeast oracle pseudo-labels for train/pool/val/test.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/generate_oracle_pseudolabels_yeast_dream.sh
#
#SBATCH --job-name=oracle_pseudolabels_yeast
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
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

echo "Generating yeast DREAM pseudo-labels at $(date)"
echo "Node: ${SLURMD_NODENAME}"

uv run --no-sync python experiments/generate_oracle_pseudolabels_yeast_dream.py \
  --config-name generate_oracle_pseudolabels_yeast_dream
