#!/bin/bash
# Re-evaluate OOD test set for K562 scaling models that used the old OOD file
# or have no OOD metrics. Updates result.json in-place.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/reeval_ood_k562.sh
#
#SBATCH --job-name=reeval_ood_k562
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

echo "Re-evaluating OOD for K562 scaling models — $(date)"
echo "Node: ${SLURMD_NODENAME}"

WEIGHTS=/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1
DATA=data/k562

# Full-encoder models (old OOD file, n=14086)
echo "=== Full-encoder models ==="
uv run --no-sync python experiments/reeval_ood.py \
    --scan-dir outputs/exp0_k562_scaling_alphagenome \
    --k562-data-path "$DATA" \
    --weights-path "$WEIGHTS"

echo "Done — $(date)"
